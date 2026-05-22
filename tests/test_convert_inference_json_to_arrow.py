from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pytest

from s2and.incremental_linking.feature_block import FEATURE_BLOCK_ARROW_MANIFEST_SCHEMA_VERSION
from scripts.convert_inference_json_to_arrow import _build_parser, convert_inference_json_to_arrow


def _read_table(path: str) -> pa.Table:
    with pa.memory_map(path, "r") as source:
        return pa.ipc.open_file(source).read_all()


def _minimal_service_payload(signature_id: str = "s1", paper_id: int = 1) -> dict[str, object]:
    return {
        "signatures": [
            {
                "signature_id": signature_id,
                "paper_id": paper_id,
                "author_info": {
                    "position": 0,
                    "block": "a smith",
                    "first": "Alice",
                    "middle": None,
                    "last": "Smith",
                    "suffix": None,
                    "email": None,
                    "affiliations": [],
                    "source_ids": [],
                },
            }
        ],
        "papers": [
            {
                "paper_id": paper_id,
                "title": "One",
                "abstract": "",
                "journal_name": "",
                "venue": "",
                "year": 2020,
                "authors": [{"position": 0, "author_name": "Alice Smith"}],
            }
        ],
        "cluster_seeds": {},
        "altered_cluster_signatures": [],
    }


def test_convert_inference_json_to_arrow_preserves_seed_and_altered_tables(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("S2AND_SKIP_FASTTEXT", "1")
    payload = {
        "signatures": [
            {
                "signature_id": "s1",
                "paper_id": 1,
                "author_info": {
                    "position": 0,
                    "block": "a smith",
                    "first": "Alice",
                    "middle": None,
                    "last": "Smith",
                    "suffix": None,
                    "email": None,
                    "affiliations": [],
                    "source_ids": [],
                },
            },
            {
                "signature_id": "s2",
                "paper_id": 2,
                "author_info": {
                    "position": 0,
                    "block": "a smith",
                    "first": "Alice",
                    "middle": None,
                    "last": "Smith",
                    "suffix": None,
                    "email": None,
                    "affiliations": [],
                    "source_ids": [],
                },
            },
            {
                "signature_id": "q",
                "paper_id": 3,
                "author_info": {
                    "position": 0,
                    "block": "a smith",
                    "first": "Alex",
                    "middle": None,
                    "last": "Smith",
                    "suffix": None,
                    "email": None,
                    "affiliations": [],
                    "source_ids": [],
                },
            },
        ],
        "papers": [
            {
                "paper_id": 1,
                "title": "One",
                "abstract": "Has Abstract",
                "journal_name": "",
                "venue": "",
                "year": 2020,
                "authors": [{"position": 0, "author_name": "Alice Smith"}],
            },
            {
                "paper_id": 2,
                "title": "Two",
                "abstract": "Has Abstract",
                "journal_name": "",
                "venue": "",
                "year": 2021,
                "authors": [{"position": 0, "author_name": "Alice Smith"}],
            },
            {
                "paper_id": 3,
                "title": "Three",
                "abstract": "",
                "journal_name": "",
                "venue": "",
                "year": 2022,
                "authors": [{"position": 0, "author_name": "Alex Smith"}],
            },
        ],
        "paper_embeddings": {"1": [0.1, 0.2], "2": [0.2, 0.3], "3": [0.3, 0.4]},
        "cluster_seeds": {"s1": {"s2": "require", "q": "disallow"}},
        "altered_cluster_signatures": ["s1"],
    }
    input_json = tmp_path / "service_payload.json"
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    manifest = convert_inference_json_to_arrow(
        input_json=input_json,
        output_root=tmp_path / "arrow",
        dataset_name="service_payload",
        name_counts_index_root=tmp_path,
        n_jobs=1,
        overwrite=True,
        skip_name_counts_index=True,
    )

    assert manifest["schema"] == FEATURE_BLOCK_ARROW_MANIFEST_SCHEMA_VERSION
    assert manifest["signature_count"] == 3
    assert manifest["paper_count"] == 3
    assert manifest["cluster_seeds_require_count"] == 2
    assert manifest["cluster_seeds_disallow_count"] == 1
    assert manifest["altered_cluster_signatures"] == ["s1"]

    cluster_seed_rows = _read_table(manifest["paths"]["cluster_seeds"]).to_pydict()
    assert cluster_seed_rows == {"signature_id": ["s1", "s2"], "cluster_id": ["0", "0"]}
    cluster_seed_disallow_rows = _read_table(manifest["paths"]["cluster_seed_disallows"]).to_pydict()
    assert cluster_seed_disallow_rows == {"signature_id_1": ["q"], "signature_id_2": ["s1"]}
    altered_path = Path(manifest["paths"]["altered_cluster_signatures"])
    assert altered_path.name == "altered_cluster_signatures.arrow"
    assert _read_table(str(altered_path)).to_pydict() == {"signature_id": ["s1"]}

    assert _read_table(manifest["paths"]["signatures"]).num_rows == 3
    assert _read_table(manifest["paths"]["papers"]).num_rows == 3
    assert _read_table(manifest["paths"]["paper_authors"]).num_rows == 3
    assert _read_table(manifest["paths"]["specter"]).num_rows == 3
    assert Path(manifest["paths"]["signatures_batch_index"]).name == "signatures.signatures_batch_index.bin"
    assert Path(manifest["paths"]["papers_batch_index"]).exists()
    assert "signatures_json" not in manifest["paths"]
    assert "papers_json" not in manifest["paths"]
    assert "cluster_seeds_json" not in manifest["paths"]
    assert not (Path(manifest["paths"]["signatures"]).parent / "signatures.json").exists()
    assert manifest["physical_layout"]["schema"] == "s2and_arrow_physical_v1"
    assert manifest["physical_layout"]["tables"]["signatures"]["batch_index_present"] is True
    assert manifest["raw_planner_batch_indexes"]["signatures_batch_index"]["record_count"] == 3


def test_convert_inference_json_to_arrow_source_json_is_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("S2AND_SKIP_FASTTEXT", "1")
    input_json = tmp_path / "service_payload.json"
    input_json.write_text(json.dumps(_minimal_service_payload()), encoding="utf-8")

    manifest = convert_inference_json_to_arrow(
        input_json=input_json,
        output_root=tmp_path / "arrow",
        dataset_name="service_payload",
        name_counts_index_root=tmp_path,
        n_jobs=1,
        overwrite=True,
        skip_name_counts_index=True,
        copy_source_json=True,
    )

    for key in ("signatures_json", "papers_json", "cluster_seeds_json"):
        assert Path(manifest["paths"][key]).exists()


def test_convert_inference_json_to_arrow_cli_defaults_keep_generated_index_output_local() -> None:
    args = _build_parser().parse_args(["--input-json", "service_payload.json"])

    assert args.name_counts_index_root is None
    assert args.copy_source_json is False


def test_convert_inference_json_to_arrow_rejects_duplicate_list_ids(tmp_path: Path) -> None:
    input_json = tmp_path / "service_payload.json"
    input_json.write_text(
        json.dumps(
            {
                "signatures": [
                    {"signature_id": "s1"},
                    {"signature_id": "s1"},
                ],
                "papers": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate signature_id"):
        convert_inference_json_to_arrow(
            input_json=input_json,
            output_root=tmp_path / "arrow",
            dataset_name="service_payload",
            name_counts_index_root=tmp_path,
            n_jobs=1,
            overwrite=False,
            skip_name_counts_index=True,
        )


def test_convert_inference_json_to_arrow_rejects_stale_output_without_overwrite(tmp_path: Path) -> None:
    input_json = tmp_path / "service_payload.json"
    input_json.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "arrow" / "service_payload"
    output_dir.mkdir(parents=True)
    (output_dir / "signatures.arrow").write_text("stale", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Use --overwrite"):
        convert_inference_json_to_arrow(
            input_json=input_json,
            output_root=tmp_path / "arrow",
            dataset_name="service_payload",
            name_counts_index_root=tmp_path,
            n_jobs=1,
            overwrite=False,
            skip_name_counts_index=True,
        )


def test_convert_inference_json_to_arrow_overwrite_preserves_other_root_manifest_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("S2AND_SKIP_FASTTEXT", "1")
    input_json = tmp_path / "service_payload.json"
    input_json.write_text(json.dumps(_minimal_service_payload()), encoding="utf-8")
    output_root = tmp_path / "arrow"
    output_root.mkdir()
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "inference_arrow_bundle_v1",
                "output_root": str(output_root),
                "datasets": ["existing_dataset"],
                "dataset_manifests": [
                    {
                        "dataset": "existing_dataset",
                        "dataset_dir": "existing_dataset",
                        "manifest_path": "existing_dataset/manifest.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    convert_inference_json_to_arrow(
        input_json=input_json,
        output_root=output_root,
        dataset_name="new_dataset",
        name_counts_index_root=tmp_path,
        n_jobs=1,
        overwrite=True,
        skip_name_counts_index=True,
    )

    root_manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert root_manifest["schema"] == "inference_arrow_bundle_v1"
    assert "source_path" not in root_manifest
    assert "reports" not in root_manifest
    assert root_manifest["datasets"] == ["existing_dataset", "new_dataset"]
    assert [report["dataset"] for report in root_manifest["dataset_manifests"]] == [
        "existing_dataset",
        "new_dataset",
    ]


def test_convert_inference_json_to_arrow_rejects_malformed_root_manifest_before_dataset_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("S2AND_SKIP_FASTTEXT", "1")
    input_json = tmp_path / "service_payload.json"
    input_json.write_text(json.dumps(_minimal_service_payload()), encoding="utf-8")
    output_root = tmp_path / "arrow"
    output_root.mkdir()
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "inference_arrow_bundle_v1",
                "output_root": str(output_root),
                "datasets": ["existing_dataset"],
                "dataset_manifests": [{"manifest_path": "existing_dataset/manifest.json"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"dataset_manifests\[0\].*dataset"):
        convert_inference_json_to_arrow(
            input_json=input_json,
            output_root=output_root,
            dataset_name="new_dataset",
            name_counts_index_root=tmp_path,
            n_jobs=1,
            overwrite=True,
            skip_name_counts_index=True,
        )

    assert not (output_root / "new_dataset" / "manifest.json").exists()


def test_convert_inference_json_to_arrow_rejects_legacy_root_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("S2AND_SKIP_FASTTEXT", "1")
    input_json = tmp_path / "service_payload.json"
    input_json.write_text(json.dumps(_minimal_service_payload()), encoding="utf-8")
    output_root = tmp_path / "arrow"
    output_root.mkdir()
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "source_path": "old.json",
                "output_root": str(output_root),
                "datasets": ["existing_dataset"],
                "reports": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="legacy source_path/reports"):
        convert_inference_json_to_arrow(
            input_json=input_json,
            output_root=output_root,
            dataset_name="new_dataset",
            name_counts_index_root=tmp_path,
            n_jobs=1,
            overwrite=True,
            skip_name_counts_index=True,
        )

    assert not (output_root / "new_dataset" / "manifest.json").exists()
