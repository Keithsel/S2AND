from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import scripts.eval_prod_models as eval_prod_models


def test_resolve_arrow_dataset_paths_includes_name_counts_index_from_manifest(tmp_path: Path) -> None:
    dataset_root = tmp_path / "arrow" / "dummy"
    dataset_root.mkdir(parents=True)
    name_counts_index = tmp_path / "name_counts_index"
    name_counts_index.mkdir()
    for filename in (
        "signatures.arrow",
        "papers.arrow",
        "paper_authors.arrow",
        "specter.arrow",
        "dummy_clusters.json",
    ):
        (dataset_root / filename).touch()
    (dataset_root / "manifest.json").write_text(
        json.dumps({"paths": {"name_counts_index": str(name_counts_index)}}),
        encoding="utf-8",
    )

    resolved = eval_prod_models.resolve_arrow_dataset_paths(str(tmp_path / "arrow"), "dummy", "_specter.pickle")

    assert resolved["name_counts_index"] == str(name_counts_index)


def test_resolve_arrow_dataset_paths_rejects_bad_manifest_name_counts_index(tmp_path: Path) -> None:
    dataset_root = tmp_path / "arrow" / "dummy"
    dataset_root.mkdir(parents=True)
    (tmp_path / "arrow" / "name_counts_index").mkdir()
    for filename in (
        "signatures.arrow",
        "papers.arrow",
        "paper_authors.arrow",
        "specter.arrow",
        "dummy_clusters.json",
    ):
        (dataset_root / filename).touch()
    (dataset_root / "manifest.json").write_text(
        json.dumps({"paths": {"name_counts_index": "missing/name_counts_index"}}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="names missing name_counts_index"):
        eval_prod_models.resolve_arrow_dataset_paths(str(tmp_path / "arrow"), "dummy", "_specter.pickle")


def test_cluster_eval_arrow_passes_name_counts_index_and_batch_indexes(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeClusterer:
        def predict_from_arrow_paths(self, block_dict, arrow_paths, **kwargs):
            captured["block_dict"] = dict(block_dict)
            captured["arrow_paths"] = dict(arrow_paths)
            captured["kwargs"] = dict(kwargs)
            return {"pred": ["s1"]}, None

    monkeypatch.setattr(eval_prod_models, "read_arrow_s2_blocks", lambda _path: {"block": ["s1"]})
    monkeypatch.setattr(
        eval_prod_models,
        "split_blocks_like_anddata",
        lambda blocks, *, random_seed: ({}, {}, dict(blocks)),
    )
    monkeypatch.setattr(eval_prod_models, "read_signature_to_cluster_id", lambda _path: {"s1": "truth"})

    arrow_paths = {
        "signatures": "signatures.arrow",
        "papers": "papers.arrow",
        "paper_authors": "paper_authors.arrow",
        "specter": "specter.arrow",
        "clusters": "clusters.json",
        "name_counts_index": "name_counts_index",
        "signatures_batch_index": "signatures.signatures_batch_index.bin",
    }
    eval_prod_models.cluster_eval_arrow(
        arrow_paths,
        SimpleNamespace(predict_from_arrow_paths=FakeClusterer().predict_from_arrow_paths),
        random_seed=42,
        n_jobs=1,
    )

    assert captured["block_dict"] == {"block": ["s1"]}
    assert captured["kwargs"]["load_name_counts"] is True
    assert captured["arrow_paths"]["name_counts_index"] == "name_counts_index"
    assert captured["arrow_paths"]["signatures_batch_index"] == "signatures.signatures_batch_index.bin"
    assert "clusters" not in captured["arrow_paths"]
