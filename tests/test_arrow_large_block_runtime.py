from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import s2and.model as model_module
from s2and.arrow_inputs import validate_arrow_prediction_artifacts
from s2and.consts import PROJECT_ROOT_PATH
from s2and.incremental_linking.feature_block import write_cluster_seeds_arrow
from s2and.runtime import build_runtime_context
from scripts._rust_suite.promoted_incremental_arrow_profile_cmd import (
    ArrowProfileDataset,
    _block_dict,
    _read_signature_rows,
    _select_workload,
    _signature_namespaces,
)

_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec"


def _is_lfs_pointer(path: Path) -> bool:
    return path.is_file() and path.read_bytes()[: len(_LFS_POINTER_PREFIX)] == _LFS_POINTER_PREFIX


def _skip_if_missing_or_lfs_pointer(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise pytest.skip.Exception(f"missing local Arrow/prod artifact(s): {missing}")
    pointers = [str(path) for path in paths if _is_lfs_pointer(path)]
    if pointers:
        raise pytest.skip.Exception(f"Git LFS artifact(s) not materialized: {pointers}")


def _resolve_manifest_path(dataset_root: Path, value: Any) -> str:
    raw_path = Path(str(value))
    candidates = [raw_path] if raw_path.is_absolute() else [dataset_root / raw_path, Path(PROJECT_ROOT_PATH) / raw_path]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str(candidates[0])


def _arrow_prediction_paths(dataset_root: Path) -> dict[str, str]:
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_paths = manifest.get("paths")
    if not isinstance(manifest_paths, Mapping):
        raise ValueError(f"Arrow manifest is missing object paths: {manifest_path}")
    paths: dict[str, str] = {}
    for key in (
        "signatures",
        "papers",
        "paper_authors",
        "specter",
        "specter2",
        "name_counts_index",
        "signatures_batch_index",
        "papers_batch_index",
        "paper_authors_batch_index",
        "specter_batch_index",
        "specter2_batch_index",
    ):
        value = manifest_paths.get(key)
        if value is not None:
            paths[key] = _resolve_manifest_path(dataset_root, value)
    if "specter" not in paths and "specter2" in paths:
        paths["specter"] = paths["specter2"]
    if "specter_batch_index" not in paths and "specter2_batch_index" in paths:
        paths["specter_batch_index"] = paths["specter2_batch_index"]
    return validate_arrow_prediction_artifacts(
        paths,
        require_specter=True,
        require_name_counts_index=True,
        require_batch_indexes=True,
        context="large-block Arrow runtime integration test",
        producer_hint="use the canonical s2and_and_big_blocks_linker_dataset_20260513_arrow bundle",
    )


@pytest.mark.requires_lfs
def test_canonical_pubmed_large_block_arrow_subblocking_and_incremental_no_anddata_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust_module = pytest.importorskip("s2and_rust")
    if not hasattr(rust_module, "make_subblocks_with_telemetry_arrow"):
        raise pytest.skip.Exception("s2and_rust.make_subblocks_with_telemetry_arrow is unavailable")
    if not hasattr(rust_module, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("s2and_rust.RawBlockQueryCandidatePlanner is unavailable")

    from s2and.production_model import load_production_model

    dataset_root = Path("s2and/data/s2and_and_big_blocks_linker_dataset_20260513_arrow/datasets/pubmed")
    model_root = Path("s2and/data/production_model_v1.21")
    _skip_if_missing_or_lfs_pointer(
        [
            dataset_root / "manifest.json",
            dataset_root / "signatures.arrow",
            dataset_root / "papers.arrow",
            dataset_root / "paper_authors.arrow",
            dataset_root / "specter2.arrow",
            dataset_root / "signatures.signatures_batch_index.bin",
            dataset_root / "papers.papers_batch_index.bin",
            dataset_root / "paper_authors.paper_authors_batch_index.bin",
            dataset_root / "specter2.specter_batch_index.bin",
            Path("s2and/data/name_counts_index/name_counts_index/manifest.json"),
            model_root / "manifest.json",
            model_root / "clusterer.json",
            model_root / "pairwise/main.lgb",
            model_root / "pairwise/nameless.lgb",
            model_root / "incremental_linker/booster.lgb",
            model_root / "incremental_linker/metadata.json",
        ]
    )

    arrow_paths = _arrow_prediction_paths(dataset_root)
    rows = _read_signature_rows(Path(arrow_paths["signatures"]))
    blocks = _block_dict(rows)
    target_block = "r agarwal"
    block_signature_ids = blocks[target_block]
    seed_signature_to_cluster = {
        signature_id: f"seed_component_{index}" for index, signature_id in enumerate(block_signature_ids[:20])
    }
    workload = _select_workload(
        blocks=blocks,
        signature_to_cluster_id=seed_signature_to_cluster,
        target_block=target_block,
        query_limit=2,
        max_seed_clusters=2,
    )
    cluster_seeds_path = tmp_path / "cluster_seeds.arrow"
    write_cluster_seeds_arrow(cluster_seeds_path, workload.seed_signature_to_cluster)
    dataset = ArrowProfileDataset(
        name="canonical_pubmed_large_block_arrow_runtime",
        arrow_paths=arrow_paths,
        signatures=_signature_namespaces(rows),
        cluster_seeds_path=cluster_seeds_path,
    )

    sync_calls: list[object] = []
    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *args, **kwargs: sync_calls.append(args))
    monkeypatch.setattr(
        model_module,
        "cluster_with_specter",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("legacy SPECTER subblocking fallback ran")),
    )

    clusterer = load_production_model(str(model_root))
    clusterer.use_cache = False
    clusterer.n_jobs = 1

    pred_clusters, dists = clusterer.predict(
        {workload.target_block: block_signature_ids},
        dataset,
        batching_threshold=64,
        backend="rust",
        total_ram_bytes=1_000_000_000_000,
        restore_rust_cluster_seeds_on_exit=False,
    )

    graph = clusterer._last_arrow_graph_subblocking_telemetry
    assert pred_clusters
    assert dists is None
    assert graph["enabled"] == 1
    assert graph["mode"] == "graph"
    assert graph["source"] == "arrow"
    assert graph["candidate_signature_count"] == len(block_signature_ids)
    assert graph["legacy_fallback_invocation_count"] == 0
    assert graph["graph_prepare_failed"] == 0
    assert graph["graph_prepare_error"] is None
    assert graph["graph_fallback_errors"] == []
    assert graph["fallback_invocation_count"] > 0

    result = clusterer.predict_incremental(
        workload.block_signatures,
        dataset,
        prevent_new_incompatibilities=False,
        batching_threshold=1,
        runtime_context=build_runtime_context("large_block_arrow_runtime_test", backend="rust"),
        total_ram_bytes=1_000_000_000_000,
    )

    telemetry = result["incremental_linker_telemetry"]
    assert result["incremental_linker_query_view"] == "raw_arrow"
    assert telemetry["arrow_promoted_incremental"] == 1
    assert telemetry["seed_setup_cluster_seeds_source"] == "arrow"
    assert telemetry["seed_setup_cluster_seeds_from_arrow"] == 1
    assert telemetry["seed_arrow_reused_source"] == 1
    assert telemetry["raw_arrow_window_plan_enabled"] == 1
    assert telemetry["raw_arrow_window_plan_count"] == 1
    assert telemetry["raw_arrow_window_featurizer_count"] == 1
    assert telemetry["raw_arrow_window_featurizer_reused_batch_count"] == 2
    assert telemetry["raw_arrow_featurizer_reused"] == 2
    assert telemetry["raw_arrow_seed_signature_count"] == 2
    assert telemetry["raw_arrow_seed_component_count"] == 2
    assert result["clusters"]
    assert sync_calls == []
