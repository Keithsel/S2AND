from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

from s2and.incremental_linking.artifact import (
    BOOSTER_FILENAME,
    METADATA_FILENAME,
    load_incremental_linking_artifact,
    save_incremental_linking_artifact,
)
from s2and.incremental_linking.contracts import validate_required_rust_capabilities
from s2and.incremental_linking.features import promoted_linker_feature_columns
from s2and.incremental_linking.logistic_gate import logistic_gate_config


def _tiny_booster() -> tuple[lgb.Booster, np.ndarray]:
    columns = promoted_linker_feature_columns()
    matrix = np.zeros((8, len(columns)), dtype=np.float32)
    matrix[:, columns.index("min_distance")] = np.linspace(1.0, 0.0, len(matrix), dtype=np.float32)
    labels = np.asarray([0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int8)
    dataset = lgb.Dataset(matrix, label=labels, free_raw_data=False)
    booster = lgb.train(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "num_threads": 1,
            "learning_rate": 0.3,
            "num_leaves": 3,
            "min_data_in_leaf": 1,
            "min_data_in_bin": 1,
            "force_col_wise": True,
        },
        dataset,
        num_boost_round=6,
    )
    return booster, matrix[:3]


def _logistic_gate_config(link: bool = True) -> dict[str, object]:
    return logistic_gate_config(
        feature_names=("chosen_probability",),
        weights=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        bias=np.asarray([0.0, 0.0, 10.0 if link else -10.0], dtype=np.float64),
        missing_values=np.asarray([0.0], dtype=np.float64),
        calibration_mode="test",
    )


def test_save_and_load_incremental_linking_artifact_round_trip(tmp_path: Path) -> None:
    booster, fixture = _tiny_booster()
    metadata = save_incremental_linking_artifact(
        booster,
        tmp_path,
        prediction_fixture_matrix=fixture,
        gate_config=_logistic_gate_config(),
        audit_metadata={"artifact_version": "v1.2", "pairwise_model": {"version": "1.2"}},
    )

    assert (tmp_path / BOOSTER_FILENAME).exists()
    assert (tmp_path / METADATA_FILENAME).exists()
    loaded = load_incremental_linking_artifact(tmp_path)

    assert loaded.metadata.feature_columns == promoted_linker_feature_columns()
    assert loaded.metadata.feature_schema_digest == metadata.feature_schema_digest
    assert loaded.metadata.audit_metadata["artifact_version"] == "v1.2"
    assert loaded.metadata.audit_metadata["pairwise_model"]["version"] == "1.2"
    np.testing.assert_allclose(
        loaded.predict_probabilities(fixture),
        np.asarray(metadata.prediction_fixture_expected_probabilities),
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize(
    ("field_name", "message"),
    (
        ("feature_schema_digest", "feature_schema_digest mismatch"),
        ("production_contract_digest", "production_contract_digest mismatch"),
        ("retrieval_stack_digest", "retrieval_stack_digest mismatch"),
    ),
)
def test_load_incremental_linking_artifact_rejects_digest_drift(
    tmp_path: Path,
    field_name: str,
    message: str,
) -> None:
    booster, fixture = _tiny_booster()
    save_incremental_linking_artifact(
        booster,
        tmp_path,
        prediction_fixture_matrix=fixture,
        gate_config=_logistic_gate_config(),
    )
    metadata_path = tmp_path / METADATA_FILENAME
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload[field_name] = "bad"
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_incremental_linking_artifact(tmp_path)


def test_validate_required_rust_capabilities_rejects_missing_names() -> None:
    with pytest.raises(RuntimeError, match="Missing required Rust capabilities"):
        validate_required_rust_capabilities(
            ("incremental_linking_pair_plan_v1",),
            available=(),
        )
