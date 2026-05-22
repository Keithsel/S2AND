from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa

from s2and.incremental_linking.feature_block import FEATURE_BLOCK_ARROW_MANIFEST_SCHEMA_VERSION
from scripts.convert_inference_json_to_arrow import convert_inference_json_to_arrow


def _read_table(path: str) -> pa.Table:
    with pa.memory_map(path, "r") as source:
        return pa.ipc.open_file(source).read_all()


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
    assert manifest["physical_layout"]["schema"] == "s2and_arrow_physical_v1"
    assert manifest["physical_layout"]["tables"]["signatures"]["batch_index_present"] is True
    assert manifest["raw_planner_batch_indexes"]["signatures_batch_index"]["record_count"] == 3
