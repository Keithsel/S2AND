from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from s2and.model import _ensure_lightgbm_fitted, _selected_feature_indices
from s2and.production_bundle import finalize_production_bundle, write_pairwise_production_bundle
from s2and.production_model import NativeLightGBMBinaryClassifier, _config_choice, load_production_model
from s2and.serialization import load_pickle_with_verified_label_encoder_compat


def test_native_production_bundle_loads_as_mutable_clusterer() -> None:
    clusterer = load_production_model("s2and/data/production_model_v1.21")

    assert isinstance(clusterer.classifier, NativeLightGBMBinaryClassifier)
    assert isinstance(clusterer.nameless_classifier, NativeLightGBMBinaryClassifier)
    assert Path(clusterer.incremental_linker_artifact_dir).name == "incremental_linker"
    assert clusterer.production_model_bundle_version == "1.21"

    clusterer.n_jobs = 7
    clusterer.cluster_model.eps = 0.5

    assert clusterer.n_jobs == 7
    assert clusterer.classifier.n_jobs == 7
    assert clusterer.nameless_classifier.n_jobs == 7
    assert clusterer.cluster_model.eps == 0.5


def test_native_lightgbm_set_params_rejects_unknown_params() -> None:
    clusterer = load_production_model("s2and/data/production_model_v1.21")

    with pytest.raises(ValueError, match="Invalid parameter"):
        clusterer.classifier.set_params(learning_rate=0.1)


def test_native_lightgbm_deepcopy_does_not_require_model_path(tmp_path: Path) -> None:
    clusterer = load_production_model("s2and/data/production_model_v1.21")
    classifier = clusterer.classifier
    features = np.zeros((2, classifier.n_features_in_), dtype=np.float64)
    classifier.model_path = str(tmp_path / "missing_model.txt")

    copied = copy.deepcopy(classifier)

    np.testing.assert_allclose(copied.predict_proba(features), classifier.predict_proba(features))
    assert copied.model_path == classifier.model_path


def test_native_pairwise_models_match_v12_pickle_fixture() -> None:
    native_clusterer = load_production_model("s2and/data/production_model_v1.21")
    legacy_clusterer = load_pickle_with_verified_label_encoder_compat("s2and/data/production_model_v1.2.pickle")[
        "clusterer"
    ]
    _ensure_lightgbm_fitted(legacy_clusterer.classifier)
    _ensure_lightgbm_fitted(legacy_clusterer.nameless_classifier)

    rng = np.random.default_rng(921)
    main_width = len(_selected_feature_indices(legacy_clusterer.featurizer_info))
    nameless_width = len(_selected_feature_indices(legacy_clusterer.nameless_featurizer_info))
    main_features = rng.normal(size=(8, main_width))
    nameless_features = rng.normal(size=(8, nameless_width))

    np.testing.assert_allclose(
        native_clusterer.classifier.predict_proba(main_features),
        legacy_clusterer.classifier.predict_proba(main_features),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        native_clusterer.nameless_classifier.predict_proba(nameless_features),
        legacy_clusterer.nameless_classifier.predict_proba(nameless_features),
        rtol=1e-10,
        atol=1e-10,
    )


def test_pairwise_stage_finalizes_into_loadable_production_bundle(tmp_path: Path) -> None:
    source_bundle = Path("s2and/data/production_model_v1.21")
    source_clusterer = load_production_model(source_bundle)
    output_bundle = tmp_path / "production_model_v9.9"

    pairwise_summary = write_pairwise_production_bundle(
        source_clusterer,
        output_bundle,
        bundle_version="9.9",
        source_model_version="9.9",
        pairwise_training_config={"test": True},
        pairwise_training_summary={"rows": 1},
    )

    assert pairwise_summary.bundle_status == "pairwise_only"
    clusterer_config = json.loads((output_bundle / "clusterer.json").read_text(encoding="utf-8"))
    assert "incremental_phase_a_pair_batch_target_multiple" not in clusterer_config
    with pytest.raises(FileNotFoundError, match="pairwise-only"):
        load_production_model(output_bundle)

    pairwise_only_clusterer = load_production_model(output_bundle, require_incremental_linker=False)
    assert pairwise_only_clusterer.production_model_bundle_status == "pairwise_only"

    final_summary = finalize_production_bundle(
        pairwise_bundle_dir=output_bundle,
        output_bundle_dir=output_bundle,
        incremental_linker_artifact_dir=source_bundle / "incremental_linker",
        target_json=source_bundle / "reproducibility" / "incremental_linker_training_target.json",
        bundle_version="9.9",
        pairwise_model_version="9.9",
        incremental_linker_version="9.9",
    )

    assert final_summary.bundle_status == "complete"
    loaded = load_production_model(output_bundle)
    assert loaded.production_model_bundle_version == "9.9"
    assert Path(loaded.incremental_linker_artifact_dir) == output_bundle / "incremental_linker"

    with pytest.raises(ValueError, match="existing incremental linker artifacts"):
        write_pairwise_production_bundle(
            source_clusterer,
            output_bundle,
            bundle_version="9.9",
            source_model_version="9.9",
        )


def test_production_model_config_choice_rejects_unknown_literal() -> None:
    with pytest.raises(ValueError, match="incremental_seed_score_mode"):
        _config_choice(
            {"incremental_seed_score_mode": "unsupported"},
            "incremental_seed_score_mode",
            allowed=frozenset({"mean", "min", "mean_min_hybrid"}),
        )
