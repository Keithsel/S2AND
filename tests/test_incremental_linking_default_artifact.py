from __future__ import annotations

import json
from pathlib import Path

import pytest

from s2and.incremental_linking.artifact import load_incremental_linking_artifact
from s2and.incremental_linking.features import promoted_linker_feature_columns
from s2and.model import DEFAULT_INCREMENTAL_LINKER_ARTIFACT_DIR

s2and_rust = pytest.importorskip(
    "s2and_rust",
    reason="default incremental linker artifact requires the Rust extension",
)


def test_default_incremental_linker_artifact_loads_with_current_schema() -> None:
    artifact_dir = Path(DEFAULT_INCREMENTAL_LINKER_ARTIFACT_DIR)
    if not artifact_dir.exists():
        pytest.skip(f"default incremental linker artifact is not present: {artifact_dir}")
    target_path = artifact_dir / "training_target.json"
    if target_path.exists():
        target = json.loads(target_path.read_text(encoding="utf-8"))
        if str(target.get("status", "")).endswith("pending_retrain"):
            pytest.skip("default artifact is intentionally pending retrain for the promoted schema")

    artifact = load_incremental_linking_artifact(artifact_dir)

    assert artifact.metadata.feature_columns == promoted_linker_feature_columns()
    assert len(artifact.metadata.feature_columns) == 53
    assert artifact.metadata.retrieval_top_k == 25
