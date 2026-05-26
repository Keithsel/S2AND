from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import scripts.convert_to_arrow as convert_to_arrow
from s2and.incremental_linking.feature_block import write_arrow_ipc_table
from scripts.convert_to_arrow import RuntimeDatasetSources


def _fake_sources(tmp_path: Path, dataset: str) -> RuntimeDatasetSources:
    source_dir = tmp_path / "source" / dataset
    return RuntimeDatasetSources(
        dataset=dataset,
        source_dir=source_dir,
        signatures_path=source_dir / f"{dataset}_signatures.json",
        papers_path=source_dir / f"{dataset}_papers.json",
    )


def test_benchmark_parser_requires_explicit_dataset_selection(tmp_path: Path) -> None:
    parser = convert_to_arrow._build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(
            [
                "benchmark",
                "--source-root",
                str(tmp_path / "source"),
                "--output-root",
                str(tmp_path / "out"),
            ]
        )

    assert excinfo.value.code == 2


def test_linker_replay_parser_requires_explicit_dataset_selection(tmp_path: Path) -> None:
    parser = convert_to_arrow._build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(
            [
                "linker-replay",
                "--raw-root",
                str(tmp_path / "raw"),
                "--embeddings-root",
                str(tmp_path / "embeddings"),
                "--output-root",
                str(tmp_path / "out"),
            ]
        )

    assert excinfo.value.code == 2


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


def test_benchmark_main_run_full_discovers_datasets_only_when_explicit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_convert_runtime_dataset_to_arrow(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"dataset": kwargs["sources"].dataset}

    monkeypatch.setattr(convert_to_arrow, "discover_benchmark_datasets", lambda _source_root: ["first", "second"])
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
            "--run-full",
            "--skip-validation",
        ],
    )

    convert_to_arrow.main()

    assert [call["sources"].dataset for call in calls] == ["first", "second"]


def test_root_manifest_rejects_existing_benchmark_reports(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "out"
    output_root.mkdir()
    legacy_manifest = {
        "output_root": str(output_root),
        "reports": [{"dataset": "existing", "paths": {"manifest": "existing/manifest.json"}}],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(legacy_manifest),
        encoding="utf-8",
    )

    dataset_dir = output_root / "new"
    dataset_dir.mkdir()

    with pytest.raises(ValueError, match="unsupported schema"):
        convert_to_arrow._upsert_root_manifest(output_root, dataset_name="new", dataset_dir=dataset_dir)

    root_manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert root_manifest == legacy_manifest


def test_root_manifest_upsert_keeps_dataset_order_stable(tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    output_root.mkdir()
    for dataset_name in ("b", "a"):
        dataset_dir = output_root / dataset_name
        dataset_dir.mkdir()
        (dataset_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "dataset": dataset_name,
                    "conversion_kind": "table-runtime",
                    "source_dir": str(tmp_path / "source" / dataset_name),
                    "signature_count": 2,
                    "paper_count": 3,
                    "paper_embedding_count": 99,
                    "cluster_seeds_require_count": 99,
                    "cluster_seeds_disallow_count": 0,
                    "altered_cluster_signature_count": 0,
                    "paths": {
                        "signatures": "signatures.arrow",
                        "papers": "papers.arrow",
                        "paper_authors": "paper_authors.arrow",
                        "specter": "specter.arrow",
                        "cluster_seeds": "cluster_seeds.arrow",
                        "name_counts_index": "../name_counts_index",
                        "signatures_batch_index": "signatures.signatures_batch_index.bin",
                    },
                    "validation": {
                        "specter_count": 2,
                        "missing_specter_paper_count": 1,
                        "cluster_seed_count": 1,
                        "cluster_seed_disallow_count": 0,
                        "altered_cluster_signature_count": 0,
                    },
                }
            ),
            encoding="utf-8",
        )
        convert_to_arrow._upsert_root_manifest(output_root, dataset_name=dataset_name, dataset_dir=dataset_dir)

    dataset_dir = output_root / "a"
    convert_to_arrow._upsert_root_manifest(output_root, dataset_name="a", dataset_dir=dataset_dir)

    root_manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert root_manifest["datasets"] == ["a", "b"]
    assert [entry["dataset"] for entry in root_manifest["dataset_manifests"]] == ["a", "b"]
    assert root_manifest["generator"]["script"] == "scripts/convert_to_arrow.py"
    assert isinstance(root_manifest["generated_at_utc"], str)
    assert root_manifest["audit"] == {
        "dataset_count": 2,
        "datasets_with_missing_manifests": [],
        "total_signature_count": 4,
        "total_paper_count": 6,
        "total_embedding_row_count": 4,
        "total_missing_embedding_count": 2,
        "total_batch_index_count": 2,
    }
    assert root_manifest["validation_command_cwd"] == str(output_root)
    first_entry = root_manifest["dataset_manifests"][0]
    assert first_entry["manifest_exists"] is True
    assert first_entry["manifest_size_bytes"] > 0
    assert len(first_entry["manifest_sha256"]) == 64
    assert str(first_entry["audit"]["source_id"]).replace("\\", "/").endswith("source/a")
    assert first_entry["audit"]["embedding_row_count"] == 2
    assert first_entry["audit"]["cluster_seed_count"] == 1
    assert first_entry["audit"]["sidecar_keys"] == [
        "cluster_seeds",
        "name_counts_index",
        "signatures_batch_index",
    ]
    validation_command = (
        "uv run python scripts/convert_to_arrow.py validate --dataset-dir "
        "{dataset_dir} --require-embeddings --require-name-counts-index"
    )
    assert root_manifest["validation_commands"] == [
        validation_command.format(dataset_dir="a"),
        validation_command.format(dataset_dir="b"),
    ]


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


def test_linker_replay_main_run_full_discovers_datasets_only_when_explicit(
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

    monkeypatch.setattr(
        convert_to_arrow, "discover_linker_replay_datasets", lambda _raw_root, _embeddings_root: ["pubmed"]
    )
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
            "--run-full",
            "--skip-validation",
        ],
    )

    convert_to_arrow.main()

    assert [call["sources"].dataset for call in calls] == ["pubmed"]


def test_validate_manifest_require_embeddings_reports_missing_specter_rows(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")

    signatures_path = tmp_path / "signatures.arrow"
    papers_path = tmp_path / "papers.arrow"
    paper_authors_path = tmp_path / "paper_authors.arrow"
    specter_path = tmp_path / "specter.arrow"
    write_arrow_ipc_table(
        pa.table(
            {
                "signature_id": pa.array(["s1", "s2"], type=pa.string()),
                "paper_id": pa.array(["p1", "p2"], type=pa.string()),
            }
        ),
        signatures_path,
    )
    write_arrow_ipc_table(pa.table({"paper_id": pa.array(["p1", "p2"], type=pa.string())}), papers_path)
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1", "p2"], type=pa.string()),
                "position": pa.array([0, 0], type=pa.int64()),
            }
        ),
        paper_authors_path,
    )
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1"], type=pa.string()),
                "embedding": pa.FixedSizeListArray.from_arrays(pa.array([0.1, 0.2], type=pa.float32()), 2),
            }
        ),
        specter_path,
    )

    manifest = {
        "paths": {
            "signatures": str(signatures_path),
            "papers": str(papers_path),
            "paper_authors": str(paper_authors_path),
            "specter": str(specter_path),
        },
        "signature_count": 2,
        "paper_count": 2,
    }

    metrics = convert_to_arrow.validate_arrow_dataset_manifest(
        manifest,
        require_embeddings=True,
        require_name_counts_index=False,
    )

    assert metrics["specter_count"] == 1
    assert metrics["missing_specter_paper_count"] == 1
    assert metrics["missing_specter_paper_examples"] == ["p2"]


def test_validate_manifest_can_require_complete_specter_rows(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")

    signatures_path = tmp_path / "signatures.arrow"
    papers_path = tmp_path / "papers.arrow"
    paper_authors_path = tmp_path / "paper_authors.arrow"
    specter_path = tmp_path / "specter.arrow"
    write_arrow_ipc_table(
        pa.table(
            {
                "signature_id": pa.array(["s1", "s2"], type=pa.string()),
                "paper_id": pa.array(["p1", "p2"], type=pa.string()),
            }
        ),
        signatures_path,
    )
    write_arrow_ipc_table(pa.table({"paper_id": pa.array(["p1", "p2"], type=pa.string())}), papers_path)
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1", "p2"], type=pa.string()),
                "position": pa.array([0, 0], type=pa.int64()),
            }
        ),
        paper_authors_path,
    )
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1"], type=pa.string()),
                "embedding": pa.FixedSizeListArray.from_arrays(pa.array([0.1, 0.2], type=pa.float32()), 2),
            }
        ),
        specter_path,
    )

    manifest = {
        "paths": {
            "signatures": str(signatures_path),
            "papers": str(papers_path),
            "paper_authors": str(paper_authors_path),
            "specter": str(specter_path),
        },
        "signature_count": 2,
        "paper_count": 2,
    }

    with pytest.raises(ValueError, match="require_complete_embeddings=True.*p2"):
        convert_to_arrow.validate_arrow_dataset_manifest(
            manifest,
            require_embeddings=True,
            require_name_counts_index=False,
            require_complete_embeddings=True,
        )


def test_write_specter_arrow_reports_zero_size_vectors(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    pytest.importorskip("pyarrow")
    source_path = tmp_path / "specter.pkl"
    output_path = tmp_path / "specter.arrow"
    with source_path.open("wb") as outfile:
        pickle.dump(
            {
                "p1": np.array([0.1, 0.2], dtype=np.float32),
                "p2": np.array([], dtype=np.float32),
                "p3": np.array([0.3, 0.4], dtype=np.float32),
            },
            outfile,
        )

    with caplog.at_level("WARNING", logger="scripts.convert_to_arrow"):
        report = convert_to_arrow._write_specter_arrow(
            source_path=source_path,
            output_path=output_path,
            needed_paper_ids={"p1", "p2"},
            overwrite=True,
        )

    assert report["row_count"] == 1
    assert report["dropped_empty_embedding_count"] == 1
    assert "zero-size vectors" in caplog.text


def test_extra_specter_physical_layout_omits_nonportable_batch_index_path(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    specter_path = tmp_path / "specter2.arrow"
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1", "p2"], type=pa.string()),
                "embedding": pa.FixedSizeListArray.from_arrays(
                    pa.array([0.1, 0.2, 0.3, 0.4], type=pa.float32()),
                    2,
                ),
            }
        ),
        specter_path,
    )
    paths = {"specter2": str(specter_path)}
    raw_planner_index_metrics: dict[str, Any] = {}
    physical_layout: dict[str, Any] = {"tables": {}}

    convert_to_arrow._add_extra_specter_index_and_layout(
        paths=paths,
        raw_planner_index_metrics=raw_planner_index_metrics,
        physical_layout=physical_layout,
        table_key="specter2",
        output_dir=tmp_path,
        overwrite=True,
    )

    layout = physical_layout["tables"]["specter2"]
    assert layout["batch_index_path_key"] == "specter2_batch_index"
    assert "batch_index_path" not in layout
    assert Path(paths["specter2_batch_index"]).exists()


def test_validate_arrow_dataset_dir_resolves_relative_manifest_paths() -> None:
    pytest.importorskip("pyarrow")
    fixture = Path("tests/fixtures/arrow/pubmed_specter2/pubmed")

    metrics = convert_to_arrow.validate_arrow_dataset_dir(
        fixture,
        require_embeddings=True,
        require_name_counts_index=True,
    )

    assert metrics["signature_count"] == 2871
    assert metrics["name_counts_index_present"] is True


def test_validate_arrow_dataset_manifest_rejects_incomplete_name_counts_index(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    signatures_path = tmp_path / "signatures.arrow"
    papers_path = tmp_path / "papers.arrow"
    paper_authors_path = tmp_path / "paper_authors.arrow"
    name_counts_index = tmp_path / "name_counts_index"
    name_counts_index.mkdir()
    (name_counts_index / "manifest.json").write_text(
        json.dumps({"files": {"first": {"path": "missing-first.bin"}}}),
        encoding="utf-8",
    )
    write_arrow_ipc_table(
        pa.table({"signature_id": pa.array(["s1"], type=pa.string()), "paper_id": pa.array(["p1"], type=pa.string())}),
        signatures_path,
    )
    write_arrow_ipc_table(pa.table({"paper_id": pa.array(["p1"], type=pa.string())}), papers_path)
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1"], type=pa.string()),
                "position": pa.array([0], type=pa.int64()),
            }
        ),
        paper_authors_path,
    )

    with pytest.raises(ValueError, match="missing files.first.path target"):
        convert_to_arrow.validate_arrow_dataset_manifest(
            {
                "paths": {
                    "signatures": str(signatures_path),
                    "papers": str(papers_path),
                    "paper_authors": str(paper_authors_path),
                    "name_counts_index": str(name_counts_index),
                }
            },
            require_embeddings=False,
            require_name_counts_index=True,
        )


def test_validate_arrow_dataset_manifest_rejects_integer_id_columns(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    signatures_path = tmp_path / "signatures.arrow"
    papers_path = tmp_path / "papers.arrow"
    paper_authors_path = tmp_path / "paper_authors.arrow"
    write_arrow_ipc_table(
        pa.table(
            {
                "signature_id": pa.array([1], type=pa.int64()),
                "paper_id": pa.array(["p1"], type=pa.string()),
            }
        ),
        signatures_path,
    )
    write_arrow_ipc_table(pa.table({"paper_id": pa.array(["p1"], type=pa.string())}), papers_path)
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1"], type=pa.string()),
                "position": pa.array([0], type=pa.int64()),
            }
        ),
        paper_authors_path,
    )

    with pytest.raises(ValueError, match="expected string"):
        convert_to_arrow.validate_arrow_dataset_manifest(
            {
                "paths": {
                    "signatures": str(signatures_path),
                    "papers": str(papers_path),
                    "paper_authors": str(paper_authors_path),
                }
            },
            require_embeddings=False,
            require_name_counts_index=False,
        )


def test_validate_arrow_dataset_manifest_requires_batch_index_sidecar(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    signatures_path = tmp_path / "signatures.arrow"
    papers_path = tmp_path / "papers.arrow"
    paper_authors_path = tmp_path / "paper_authors.arrow"
    write_arrow_ipc_table(
        pa.table(
            {
                "signature_id": pa.array(["s1"], type=pa.string()),
                "paper_id": pa.array(["p1"], type=pa.string()),
            }
        ),
        signatures_path,
    )
    write_arrow_ipc_table(pa.table({"paper_id": pa.array(["p1"], type=pa.string())}), papers_path)
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1"], type=pa.string()),
                "position": pa.array([0], type=pa.int64()),
            }
        ),
        paper_authors_path,
    )
    manifest = {
        "paths": {
            "signatures": str(signatures_path),
            "papers": str(papers_path),
            "paper_authors": str(paper_authors_path),
        },
        "physical_layout": {
            "tables": {
                "signatures": {
                    "key": "signature_id",
                    "batch_index_path_key": "signatures_batch_index",
                    "batch_index_present": True,
                    "max_record_batch_rows": 16384,
                    "actual_max_batch_rows": 1,
                }
            }
        },
    }

    with pytest.raises(FileNotFoundError, match="signatures_batch_index"):
        convert_to_arrow.validate_arrow_dataset_manifest(
            manifest,
            require_embeddings=False,
            require_name_counts_index=False,
        )
