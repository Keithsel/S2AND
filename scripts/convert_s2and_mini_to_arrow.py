"""Convert bundled s2and_mini datasets into direct-Rust Arrow inputs."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


DATASETS = ("arnetminer", "inspire", "kisti", "pubmed", "qian", "zbmath")


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as infile:
        return json.load(infile)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_arrow_ipc_table(table: Any, path: Path) -> str:
    import pyarrow as pa

    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    return str(path)


def _specter_mapping(payload: Any) -> dict[str, np.ndarray]:
    if isinstance(payload, dict):
        return {str(key): np.asarray(value, dtype=np.float32) for key, value in payload.items()}
    if isinstance(payload, tuple) and len(payload) == 2:
        matrix, keys = payload
        matrix_array = np.asarray(matrix, dtype=np.float32)
        return {str(key): np.asarray(matrix_array[index], dtype=np.float32) for index, key in enumerate(keys)}
    raise TypeError(f"Unsupported SPECTER payload type: {type(payload).__name__}")


def _write_specter_arrow(
    *,
    source_path: Path,
    output_path: Path,
    needed_paper_ids: set[str],
    overwrite: bool,
) -> dict[str, Any]:
    import pyarrow as pa

    if output_path.exists() and not overwrite:
        return {"path": str(output_path), "reused": True}

    with source_path.open("rb") as infile:
        specter_by_paper_id = _specter_mapping(pickle.load(infile))
    selected_items = [
        (paper_id, vector)
        for paper_id, vector in specter_by_paper_id.items()
        if str(paper_id) in needed_paper_ids and vector.size > 0
    ]
    selected_items.sort(key=lambda item: item[0])
    if not selected_items:
        raise ValueError(f"No SPECTER embeddings from {source_path} matched the dataset papers")

    dimension = int(selected_items[0][1].shape[0])
    for paper_id, vector in selected_items:
        if int(vector.shape[0]) != dimension:
            raise ValueError(
                f"SPECTER dimension mismatch in {source_path}: paper_id={paper_id!r} "
                f"expected={dimension} got={vector.shape[0]}"
            )

    matrix = np.vstack([vector for _paper_id, vector in selected_items]).astype(np.float32, copy=False)
    flat = pa.array(np.ravel(matrix), type=pa.float32())
    table = pa.table(
        {
            "paper_id": pa.array([paper_id for paper_id, _vector in selected_items], type=pa.string()),
            "embedding": pa.FixedSizeListArray.from_arrays(flat, dimension),
        }
    )
    _write_arrow_ipc_table(table, output_path)
    return {
        "path": str(output_path),
        "reused": False,
        "row_count": int(table.num_rows),
        "dimension": dimension,
        "source_path": str(source_path),
    }


def _source_file(source_dir: Path, dataset: str, suffix: str) -> Path:
    path = source_dir / dataset / f"{dataset}{suffix}"
    if not path.exists():
        raise FileNotFoundError(f"Missing source file: {path}")
    return path


def convert_dataset(
    source_root: Path,
    output_root: Path,
    dataset: str,
    *,
    n_jobs: int,
    overwrite: bool,
) -> dict[str, Any]:
    from s2and.data import ANDData
    from s2and.incremental_linking.feature_block import (
        write_feature_block_arrow_from_anddata,
    )

    source_dir = source_root / dataset
    output_dir = output_root / dataset
    signatures_path = _source_file(source_root, dataset, "_signatures.json")
    papers_path = _source_file(source_root, dataset, "_papers.json")
    clusters_path = _source_file(source_root, dataset, "_clusters.json")

    start = time.perf_counter()
    dataset_obj = ANDData(
        signatures=str(signatures_path),
        papers=str(papers_path),
        name=dataset,
        mode="train",
        specter_embeddings=None,
        clusters=str(clusters_path),
        block_type="s2",
        train_pairs=None,
        val_pairs=None,
        test_pairs=None,
        train_pairs_size=100000,
        val_pairs_size=10000,
        test_pairs_size=10000,
        n_jobs=n_jobs,
        load_name_counts=True,
        preprocess=True,
        random_seed=42,
        name_tuples="filtered",
        use_orcid_id=True,
        use_sinonym_overwrite=False,
    )
    anddata_seconds = time.perf_counter() - start

    start = time.perf_counter()
    paths = write_feature_block_arrow_from_anddata(
        dataset_obj,
        output_dir,
        signature_ids=list(dataset_obj.signatures),
        include_specter=False,
        include_empty_cluster_seeds=False,
        drop_embedded_name_counts=False,
        overwrite=overwrite,
    )
    write_common_seconds = time.perf_counter() - start

    output_clusters_path = output_dir / f"{dataset}_clusters.json"
    if overwrite or not output_clusters_path.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(clusters_path, output_clusters_path)
    paths["clusters"] = str(output_clusters_path)

    needed_paper_ids = {str(signature.paper_id) for signature in dataset_obj.signatures.values()}
    specter_reports = {
        "_specter.pickle": _write_specter_arrow(
            source_path=source_dir / f"{dataset}_specter.pickle",
            output_path=output_dir / "specter.arrow",
            needed_paper_ids=needed_paper_ids,
            overwrite=overwrite,
        ),
        "_specter2.pkl": _write_specter_arrow(
            source_path=source_dir / f"{dataset}_specter2.pkl",
            output_path=output_dir / "specter2.arrow",
            needed_paper_ids=needed_paper_ids,
            overwrite=overwrite,
        ),
    }
    paths["specter"] = str(output_dir / "specter.arrow")
    paths["specter2"] = str(output_dir / "specter2.arrow")

    manifest = {
        "dataset": dataset,
        "source_dir": str(source_dir),
        "schema": "feature_block_arrow_v1",
        "signature_count": len(dataset_obj.signatures),
        "paper_count": len(dataset_obj.papers),
        "cluster_count": len(dataset_obj.clusters or {}),
        "paths": paths,
        "specter": specter_reports,
        "name_tuples": "default packaged filtered aliases",
        "timings_seconds": {
            "anddata_seconds": anddata_seconds,
            "write_common_seconds": write_common_seconds,
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("s2and/data/s2and_mini"))
    parser.add_argument("--output-root", type=Path, default=Path("s2and/data/s2and_mini_arrow"))
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--n-jobs", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for dataset in args.datasets:
        start = time.perf_counter()
        report = convert_dataset(
            args.source_root,
            output_root,
            str(dataset),
            n_jobs=int(args.n_jobs),
            overwrite=bool(args.overwrite),
        )
        report["total_seconds"] = time.perf_counter() - start
        reports.append(report)
        print(json.dumps({"dataset": dataset, "total_seconds": report["total_seconds"]}, sort_keys=True))

    root_manifest = {
        "source_root": str(args.source_root),
        "output_root": str(output_root),
        "datasets": [report["dataset"] for report in reports],
        "reports": reports,
    }
    _write_json(output_root / "manifest.json", root_manifest)
    print(json.dumps(root_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
