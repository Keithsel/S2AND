from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import scripts.convert_to_arrow as convert_to_arrow
from scripts.convert_to_arrow import RuntimeDatasetSources


def _fake_sources(tmp_path: Path, dataset: str) -> RuntimeDatasetSources:
    source_dir = tmp_path / "source" / dataset
    return RuntimeDatasetSources(
        dataset=dataset,
        source_dir=source_dir,
        signatures_path=source_dir / f"{dataset}_signatures.json",
        papers_path=source_dir / f"{dataset}_papers.json",
    )


def test_benchmark_main_overwrite_name_counts_index_rebuilds_shared_index_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_convert_runtime_dataset_to_arrow(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"dataset": kwargs["sources"].dataset}

    monkeypatch.setattr(
        convert_to_arrow, "benchmark_dataset_sources", lambda _source_root, dataset: _fake_sources(tmp_path, dataset)
    )
    monkeypatch.setattr(convert_to_arrow, "convert_runtime_dataset_to_arrow", fake_convert_runtime_dataset_to_arrow)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_to_arrow.py",
            "benchmark",
            "--source-root",
            str(tmp_path / "source"),
            "--output-root",
            str(tmp_path / "out"),
            "--name-counts-index-root",
            str(tmp_path / "index"),
            "--datasets",
            "first",
            "second",
            "--n-jobs",
            "1",
            "--overwrite",
            "--overwrite-name-counts-index",
        ],
    )

    convert_to_arrow.main()

    assert [call["overwrite_name_counts_index"] for call in calls] == [True, False]


def test_benchmark_main_overwrite_does_not_force_name_counts_index_rebuild(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_convert_runtime_dataset_to_arrow(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"dataset": kwargs["sources"].dataset}

    monkeypatch.setattr(
        convert_to_arrow, "benchmark_dataset_sources", lambda _source_root, dataset: _fake_sources(tmp_path, dataset)
    )
    monkeypatch.setattr(convert_to_arrow, "convert_runtime_dataset_to_arrow", fake_convert_runtime_dataset_to_arrow)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_to_arrow.py",
            "benchmark",
            "--source-root",
            str(tmp_path / "source"),
            "--output-root",
            str(tmp_path / "out"),
            "--name-counts-index-root",
            str(tmp_path / "index"),
            "--datasets",
            "first",
            "second",
            "--n-jobs",
            "1",
            "--overwrite",
        ],
    )

    convert_to_arrow.main()

    assert [call["overwrite_name_counts_index"] for call in calls] == [False, False]


def test_root_manifest_normalizes_existing_benchmark_reports(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "out"
    output_root.mkdir()
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "output_root": str(output_root),
                "reports": [{"dataset": "existing", "paths": {"manifest": "existing/manifest.json"}}],
            }
        ),
        encoding="utf-8",
    )

    dataset_dir = output_root / "new"
    dataset_dir.mkdir()
    convert_to_arrow._upsert_root_manifest(output_root, dataset_name="new", dataset_dir=dataset_dir)

    root_manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert root_manifest["schema"] == "inference_arrow_bundle_v1"
    assert "reports" not in root_manifest
    assert root_manifest["datasets"] == ["existing", "new"]
    assert [entry["dataset"] for entry in root_manifest["dataset_manifests"]] == ["existing", "new"]


def test_linker_replay_main_writes_datasets_under_release_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_linker_sources(_raw_root: Path, _embeddings_root: Path, dataset: str) -> RuntimeDatasetSources:
        return RuntimeDatasetSources(
            dataset=dataset,
            source_dir=tmp_path / "raw" / dataset,
            signatures_path=tmp_path / "raw" / dataset / "signatures.json",
            papers_path=tmp_path / "raw" / dataset / "papers.json",
            specter2_path=tmp_path / "embeddings" / dataset / "specter2.pkl",
        )

    def fake_convert_runtime_dataset_to_arrow(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"dataset": kwargs["sources"].dataset}

    monkeypatch.setattr(convert_to_arrow, "linker_replay_dataset_sources", fake_linker_sources)
    monkeypatch.setattr(convert_to_arrow, "convert_runtime_dataset_to_arrow", fake_convert_runtime_dataset_to_arrow)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_to_arrow.py",
            "linker-replay",
            "--raw-root",
            str(tmp_path / "raw"),
            "--embeddings-root",
            str(tmp_path / "embeddings"),
            "--output-root",
            str(tmp_path / "linker_replay_20260513"),
            "--datasets",
            "pubmed",
            "--skip-validation",
        ],
    )

    convert_to_arrow.main()

    assert calls[0]["output_dir"] == tmp_path / "linker_replay_20260513" / "datasets" / "pubmed"
    assert calls[0]["root_manifest_dir"] == tmp_path / "linker_replay_20260513"
    assert calls[0]["selected_embedding"] == "specter2"
