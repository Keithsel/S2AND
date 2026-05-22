"""Convert service-shaped inference JSON into direct-Rust Arrow inputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as infile:
        return json.load(infile)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mapping_by_id(rows: Any, *, id_key: str, label: str) -> dict[str, Mapping[str, Any]]:
    if isinstance(rows, Mapping):
        return {str(key): value for key, value in rows.items()}
    if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
        raise TypeError(f"{label} must be a JSON object or list")
    mapped: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError(f"{label} rows must be objects")
        row_id = row.get(id_key)
        if row_id is None:
            raise ValueError(f"{label} row is missing {id_key!r}")
        mapped[str(row_id)] = row
    return mapped


def _altered_values(payload: Mapping[str, Any]) -> list[str]:
    values = payload.get("altered_cluster_signatures") or []
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise TypeError("altered_cluster_signatures must be a list when present")
    return [str(value) for value in values]


def convert_inference_json_to_arrow(
    *,
    input_json: Path,
    output_root: Path,
    dataset_name: str,
    name_counts_index_root: Path,
    n_jobs: int,
    overwrite: bool,
    skip_name_counts_index: bool,
) -> dict[str, Any]:
    """Write one Arrow inference dataset and return its manifest."""

    from s2and.data import ANDData
    from s2and.incremental_linking.feature_block import (
        FEATURE_BLOCK_ARROW_MANIFEST_SCHEMA_VERSION,
        raw_planner_arrow_physical_layout,
        write_arrow_ipc_table,
        write_feature_block_arrow_from_anddata,
        write_name_counts_index,
        write_raw_arrow_batch_lookup_indexes,
    )

    output_dir = output_root / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    payload = _load_json(input_json)
    if not isinstance(payload, Mapping):
        raise TypeError("input JSON must contain an object")
    load_seconds = time.perf_counter() - start

    signatures = _mapping_by_id(payload.get("signatures"), id_key="signature_id", label="signatures")
    papers = _mapping_by_id(payload.get("papers"), id_key="paper_id", label="papers")
    altered = _altered_values(payload)
    specter_embeddings = payload.get("paper_embeddings", payload.get("specter_embeddings"))

    start = time.perf_counter()
    dataset = ANDData(
        signatures=signatures,
        papers=papers,
        name=dataset_name,
        mode="inference",
        clusters=None,
        specter_embeddings=specter_embeddings,
        cluster_seeds=payload.get("cluster_seeds"),
        altered_cluster_signatures=altered,
        block_type="s2",
        train_pairs=None,
        val_pairs=None,
        test_pairs=None,
        train_pairs_size=1000,
        val_pairs_size=1000,
        test_pairs_size=1000,
        n_jobs=n_jobs,
        load_name_counts=not skip_name_counts_index,
        preprocess=True,
        random_seed=42,
        name_tuples="filtered",
        use_orcid_id=True,
        use_sinonym_overwrite=False,
        compute_reference_features=False,
    )
    anddata_seconds = time.perf_counter() - start

    start = time.perf_counter()
    paths = write_feature_block_arrow_from_anddata(
        dataset,
        output_dir,
        signature_ids=list(dataset.signatures),
        include_specter=specter_embeddings is not None,
        include_empty_cluster_seeds=True,
        overwrite=overwrite,
    )
    write_arrow_seconds = time.perf_counter() - start

    import pyarrow as pa

    altered_arrow_path = output_dir / "altered_cluster_signatures.arrow"
    if overwrite or not altered_arrow_path.exists():
        table = pa.table({"signature_id": pa.array(altered, type=pa.string())})
        write_arrow_ipc_table(table, altered_arrow_path)
    paths["altered_cluster_signatures"] = str(altered_arrow_path)

    source_paths = {
        "signatures_json": output_dir / "signatures.json",
        "papers_json": output_dir / "papers.json",
        "cluster_seeds_json": output_dir / "cluster_seeds.json",
    }
    if overwrite or not source_paths["signatures_json"].exists():
        _write_json(source_paths["signatures_json"], signatures)
    if overwrite or not source_paths["papers_json"].exists():
        _write_json(source_paths["papers_json"], papers)
    if overwrite or not source_paths["cluster_seeds_json"].exists():
        _write_json(source_paths["cluster_seeds_json"], payload.get("cluster_seeds") or {})
    paths.update({key: str(path) for key, path in source_paths.items()})

    start = time.perf_counter()
    paths, raw_planner_index_metrics = write_raw_arrow_batch_lookup_indexes(
        paths,
        output_dir,
        overwrite=overwrite,
    )
    write_raw_planner_indexes_seconds = time.perf_counter() - start
    physical_layout = raw_planner_arrow_physical_layout(paths)

    name_counts_index_metrics: dict[str, Any] = {"skipped": True}
    write_name_counts_index_seconds = 0.0
    if not skip_name_counts_index:
        start = time.perf_counter()
        name_counts_index_path, name_counts_index_metrics = write_name_counts_index(
            name_counts_index_root,
            overwrite=False,
        )
        write_name_counts_index_seconds = time.perf_counter() - start
        paths["name_counts_index"] = name_counts_index_path

    manifest = {
        "schema": FEATURE_BLOCK_ARROW_MANIFEST_SCHEMA_VERSION,
        "dataset": dataset_name,
        "source_path": str(input_json),
        "signature_count": len(dataset.signatures),
        "paper_count": len(dataset.papers),
        "paper_embedding_count": len(specter_embeddings or {}),
        "cluster_seeds_require_count": len(dataset.cluster_seeds_require),
        "cluster_seeds_disallow_count": len(dataset.cluster_seeds_disallow),
        "altered_cluster_signature_count": len(altered),
        "altered_cluster_signatures": altered,
        "paths": paths,
        "physical_layout": physical_layout,
        "raw_planner_batch_indexes": raw_planner_index_metrics,
        "name_counts_index": name_counts_index_metrics,
        "name_tuples": "default packaged filtered aliases",
        "timings_seconds": {
            "load_json_seconds": load_seconds,
            "anddata_seconds": anddata_seconds,
            "write_arrow_seconds": write_arrow_seconds,
            "write_raw_planner_indexes_seconds": write_raw_planner_indexes_seconds,
            "write_name_counts_index_seconds": write_name_counts_index_seconds,
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(
        output_root / "manifest.json",
        {
            "source_path": str(input_json),
            "output_root": str(output_root),
            "datasets": [dataset_name],
            "reports": [manifest],
        },
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("scratch/inference_arrow"))
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--name-counts-index-root", type=Path, default=Path("s2and/data"))
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-name-counts-index", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    dataset_name = str(args.dataset_name or args.input_json.stem)
    report = convert_inference_json_to_arrow(
        input_json=args.input_json,
        output_root=args.output_root,
        dataset_name=dataset_name,
        name_counts_index_root=args.name_counts_index_root,
        n_jobs=int(args.n_jobs),
        overwrite=bool(args.overwrite),
        skip_name_counts_index=bool(args.skip_name_counts_index),
    )
    print(
        json.dumps(
            {
                "dataset": report["dataset"],
                "signature_count": report["signature_count"],
                "paper_count": report["paper_count"],
                "cluster_seeds_require_count": report["cluster_seeds_require_count"],
                "cluster_seeds_disallow_count": report["cluster_seeds_disallow_count"],
                "altered_cluster_signature_count": report["altered_cluster_signature_count"],
                "paths": report["paths"],
                "timings_seconds": report["timings_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
