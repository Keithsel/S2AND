"""Production orchestration for the promoted incremental linker."""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import s2and.incremental_linking.artifact as artifact_module
import s2and.incremental_linking.query_adapter as query_adapter_module
import s2and.incremental_linking.runtime as runtime_module
from s2and import feature_port, memory_budget
from s2and.data import ANDData
from s2and.incremental_linking.feature_block import (
    TemporaryArrowPaths,
    arrow_paths_with_temporary_cluster_seeds,
    cluster_seed_disallows_from_arrow_paths,
    read_cluster_seeds_arrow,
)
from s2and.incremental_linking.policy import (
    clusterer_uses_name_count_features,
    request_cluster_seed_disallow_parts,
    require_arrow_name_counts_index_for_clusterer,
)
from s2and.runtime import RuntimeContext

logger = logging.getLogger("s2and")

_PROMOTED_INCREMENTAL_TELEMETRY_MERGE_POLICY = {
    "retrieval_top_k": "constant",
    "seed_signature_count": "constant",
    "seed_component_count": "constant",
    "raw_arrow_seed_signature_count": "constant",
    "raw_arrow_seed_component_count": "constant",
    "memory_total_ram_bytes": "first",
    "memory_available_bytes": "first",
    "memory_stage_budget_bytes": "first",
    "memory_predicted_peak_delta_bytes": "first",
    "memory_predicted_peak_rss_bytes": "first",
    "memory_rss_before_bytes": "first",
    "memory_rss_peak_bytes": "first",
    "memory_rss_after_bytes": "first",
    "memory_observed_peak_delta_bytes": "first",
    "memory_observed_end_delta_bytes": "first",
    "memory_prediction_error_ratio": "first",
    "memory_underpredicted": "first",
}

BuildIncrementalResultFn = Callable[..., dict[str, Any]]
BuildIncrementalConstraintBackendFn = Callable[..., Any]
GetRustFeaturizerFn = Callable[..., Any]
ResolveTotalRamBytesFn = Callable[[int | None], tuple[int, str]]


def _raw_window_plan_telemetry_fields(raw_candidate_plan: Mapping[str, Any]) -> dict[str, int | float | str]:
    """Return raw Arrow planner telemetry under the window-plan prefix."""

    telemetry = raw_candidate_plan.get("telemetry")
    if not isinstance(telemetry, Mapping):
        return {}
    fields: dict[str, int | float | str] = {}
    for key, value in telemetry.items():
        if key == "timings":
            continue
        if isinstance(value, bool):
            fields[f"raw_arrow_window_plan_{key}"] = int(value)
        elif isinstance(value, int | float | str):
            fields[f"raw_arrow_window_plan_{key}"] = value
    timings = telemetry.get("timings")
    if isinstance(timings, Mapping):
        for key, value in timings.items():
            if isinstance(value, int | float):
                fields[f"raw_arrow_window_plan_{key}"] = float(value)
    return fields


def _merge_raw_window_plan_telemetry(
    merged: dict[str, int | float | str],
    fields: Mapping[str, int | float | str],
) -> None:
    """Merge telemetry from one raw Arrow window into an aggregate payload."""

    for key, value in fields.items():
        if isinstance(value, int | float) and not isinstance(value, bool):
            merged[key] = float(merged.get(key, 0.0)) + float(value)
            continue
        existing = merged.get(key)
        if existing is None:
            merged[key] = value
        elif existing != value:
            merged[key] = "__mixed__"


def _request_cluster_seed_disallows(
    dataset: ANDData,
    arrow_paths: Mapping[str, Any],
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], set[tuple[str, str]]]:
    arrow_disallows = cluster_seed_disallows_from_arrow_paths(arrow_paths)
    return request_cluster_seed_disallow_parts(dataset, arrow_disallows)


def _cluster_seed_map_fingerprint(cluster_seeds_require: Mapping[Any, Any]) -> tuple[int, str]:
    digest = hashlib.blake2b(digest_size=16)
    items = sorted(
        (str(signature_id), str(component_id)) for signature_id, component_id in cluster_seeds_require.items()
    )
    for signature_id, component_id in items:
        for value in (signature_id, component_id):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little", signed=False))
            digest.update(encoded)
    return len(items), digest.hexdigest()


def _cluster_seeds_arrow_matches(path_value: Any, cluster_seeds_require: Mapping[Any, Any]) -> bool:
    if path_value is None:
        return False
    path = Path(str(path_value))
    if not path.exists():
        return False
    try:
        arrow_cluster_seeds = read_cluster_seeds_arrow(path)
    except (OSError, ValueError):
        return False
    except Exception as exc:
        if exc.__class__.__name__ == "ArrowInvalid":
            return False
        raise
    return _cluster_seed_map_fingerprint(arrow_cluster_seeds) == _cluster_seed_map_fingerprint(cluster_seeds_require)


def _unpack_incremental_seed_setup(
    seed_setup: Sequence[Any],
) -> tuple[
    Mapping[str, int | str],
    Mapping[int | str, int | str],
    Mapping[int | str, Sequence[str]],
    Mapping[int | str, Sequence[str]] | None,
]:
    if len(seed_setup) == 3:
        cluster_seeds_require, recluster_map, cluster_seeds_require_inverse = seed_setup
        return cluster_seeds_require, recluster_map, cluster_seeds_require_inverse, None
    if len(seed_setup) == 4:
        cluster_seeds_require, recluster_map, cluster_seeds_require_inverse, split_cluster_seeds_require_inverse = (
            seed_setup
        )
        return (
            cluster_seeds_require,
            recluster_map,
            cluster_seeds_require_inverse,
            split_cluster_seeds_require_inverse,
        )
    raise ValueError(f"incremental seed setup must have 3 or 4 entries, got {len(seed_setup)}")


def _finish_incremental_with_optional_split_inverse(
    clusterer: Any,
    unassigned_signature_ids: list[str],
    dataset: ANDData,
    linked_signature_clusters: Mapping[str, int | str],
    recluster_map: Mapping[int | str, int | str],
    cluster_seeds_require_inverse: Mapping[int | str, Sequence[str]],
    prevent_new_incompatibilities: bool,
    partial_supervision: Mapping[tuple[str, str], int | float],
    runtime_context: RuntimeContext,
    *,
    total_ram_bytes: int | None,
    arrow_paths: Mapping[str, Any] | None = None,
    split_cluster_seeds_require_inverse: Mapping[int | str, Sequence[str]] | None = None,
) -> dict[str, list[str]]:
    method = clusterer._finish_incremental_with_seed_links
    kwargs: dict[str, Any] = {"total_ram_bytes": total_ram_bytes}
    if arrow_paths is not None:
        kwargs["arrow_paths"] = arrow_paths
    if split_cluster_seeds_require_inverse is not None:
        kwargs["split_cluster_seeds_require_inverse"] = split_cluster_seeds_require_inverse
    return method(
        unassigned_signature_ids,
        dataset,
        linked_signature_clusters,
        recluster_map,
        cluster_seeds_require_inverse,
        prevent_new_incompatibilities,
        partial_supervision,
        runtime_context,
        **kwargs,
    )


def promoted_incremental_component_sizes(cluster_seeds_require: Mapping[str, int | str]) -> dict[str, int]:
    component_sizes: dict[str, int] = {}
    for cluster_id in cluster_seeds_require.values():
        component_key = str(cluster_id)
        component_sizes[component_key] = component_sizes.get(component_key, 0) + 1
    return component_sizes


def _signature_orcid(dataset: ANDData, signature_id: str) -> str | None:
    value = getattr(dataset.signatures[str(signature_id)], "author_info_orcid", None)
    return query_adapter_module.normalize_orcid(value)


def promoted_incremental_orcid_fanout_by_query(
    dataset: ANDData,
    query_signature_ids: Sequence[str],
    cluster_seeds_require: Mapping[str, int | str],
    *,
    orcid_enabled: bool,
) -> dict[str, tuple[int, int]]:
    """Return known ORCID return-all row/pair floors by query signature id."""

    if not orcid_enabled or not query_signature_ids or not cluster_seeds_require:
        return {}

    query_orcid_by_signature_id = {
        str(query_signature_id): query_orcid
        for query_signature_id in query_signature_ids
        if (query_orcid := _signature_orcid(dataset, str(query_signature_id))) is not None
    }
    if not query_orcid_by_signature_id:
        return {}

    query_orcids = set(query_orcid_by_signature_id.values())
    component_orcids: dict[str, set[str]] = {}
    component_sizes: dict[str, int] = {}
    for seed_signature_id, component in cluster_seeds_require.items():
        component_key = str(component)
        component_sizes[component_key] = component_sizes.get(component_key, 0) + 1
        seed_orcid = _signature_orcid(dataset, str(seed_signature_id))
        if seed_orcid in query_orcids:
            component_orcids.setdefault(component_key, set()).add(str(seed_orcid))

    if not component_orcids:
        return {}

    fanout_by_query: dict[str, tuple[int, int]] = {}
    for query_signature_id, query_orcid in query_orcid_by_signature_id.items():
        matching_components = [
            component_key for component_key, orcids in component_orcids.items() if query_orcid in orcids
        ]
        if matching_components:
            fanout_by_query[str(query_signature_id)] = (
                len(matching_components),
                sum(int(component_sizes[component_key]) for component_key in matching_components),
            )
    return fanout_by_query


def _top_k_candidate_floors(component_sizes: Mapping[str, int], retrieval_top_k: int) -> tuple[int, int]:
    candidate_rows = min(max(0, int(retrieval_top_k)), len(component_sizes))
    pairs = int(sum(sorted((max(0, int(size)) for size in component_sizes.values()), reverse=True)[:candidate_rows]))
    return candidate_rows, pairs


def _orcid_fanout_floor_estimates(
    fanout_by_query: Mapping[str, tuple[int, int]],
    query_signature_ids: Sequence[str],
) -> tuple[int | None, int | None]:
    rows = 0
    pairs = 0
    for signature_id in query_signature_ids:
        query_rows, query_pairs = fanout_by_query.get(str(signature_id), (0, 0))
        rows = max(rows, int(query_rows))
        pairs = max(pairs, int(query_pairs))
    return (rows if rows > 0 else None, pairs if pairs > 0 else None)


def _orcid_fanout_floor_totals(
    fanout_by_query: Mapping[str, tuple[int, int]],
    query_signature_ids: Sequence[str],
    *,
    base_candidate_rows_per_query: int,
    base_pairs_per_query: int,
) -> tuple[int | None, int | None]:
    if not query_signature_ids:
        return None, None
    row_total = 0
    pair_total = 0
    base_rows = max(0, int(base_candidate_rows_per_query))
    base_pairs = max(0, int(base_pairs_per_query))
    for signature_id in query_signature_ids:
        query_rows, query_pairs = fanout_by_query.get(str(signature_id), (0, 0))
        row_total += max(base_rows, int(query_rows))
        pair_total += max(base_pairs, int(query_pairs))
    base_row_total = base_rows * len(query_signature_ids)
    base_pair_total = base_pairs * len(query_signature_ids)
    return (
        row_total if row_total > base_row_total else None,
        pair_total if pair_total > base_pair_total else None,
    )


def compute_promoted_incremental_limits(
    *,
    query_count: int,
    component_sizes: Mapping[str, int],
    retrieval_top_k: int,
    total_ram_bytes: int | None,
    max_query_batch_size: int | None,
    observed_query_count: int = 0,
    observed_candidate_rows_per_query: int | None = None,
    observed_pairs_per_query: int | None = None,
    candidate_rows_per_query_floor: int | None = None,
    pairs_per_query_floor: int | None = None,
    candidate_rows_total_floor: int | None = None,
    pairs_total_floor: int | None = None,
) -> memory_budget.PromotedPhaseALimits:
    return memory_budget.compute_promoted_phase_a_limits(
        query_count=query_count,
        component_sizes=component_sizes,
        retrieval_top_k=retrieval_top_k,
        total_ram_bytes=total_ram_bytes,
        max_query_batch_size=max_query_batch_size,
        observed_query_count=observed_query_count,
        observed_candidate_rows_per_query=observed_candidate_rows_per_query,
        observed_pairs_per_query=observed_pairs_per_query,
        candidate_rows_per_query_floor=candidate_rows_per_query_floor,
        pairs_per_query_floor=pairs_per_query_floor,
        candidate_rows_total_floor=candidate_rows_total_floor,
        pairs_total_floor=pairs_total_floor,
        detect_cgroup_fn=memory_budget.detect_cgroup_total_ram_bytes_best_effort,
        detect_total_fn=memory_budget.detect_total_ram_bytes_best_effort,
        current_rss_fn=memory_budget.current_rss_bytes_best_effort,
    )


def raise_if_promoted_incremental_batch_over_budget(limits: memory_budget.PromotedPhaseALimits) -> None:
    if not bool(limits.single_query_exceeds_budget):
        return
    raise MemoryError(
        "Promoted incremental linker cannot fit a single query under the memory budget: "
        f"single_query_predicted_persistent_bytes={int(limits.single_query_predicted_persistent_bytes)} "
        f"stage_budget_bytes={int(limits.stage_budget_bytes)} "
        f"total_ram_bytes={int(limits.total_ram_bytes)} "
        f"current_rss_bytes={int(limits.current_rss_bytes)} "
        f"safety_margin_bytes={int(limits.safety_margin_bytes)}"
    )


def promoted_incremental_observed_probe(
    telemetry: Mapping[str, int | float | str],
    fallback_query_count: int,
) -> tuple[int, int, int] | None:
    try:
        query_count = int(telemetry.get("query_count", fallback_query_count))
        candidate_row_count = int(telemetry.get("candidate_row_count", 0))
        pair_count = int(telemetry.get("pair_count", 0))
    except (TypeError, ValueError):
        return None
    if query_count <= 0 or (candidate_row_count <= 0 and pair_count <= 0):
        return None
    rows_per_query = int(math.ceil(float(candidate_row_count) / float(query_count)))
    pairs_per_query = int(math.ceil(float(pair_count) / float(query_count)))
    return query_count, rows_per_query, pairs_per_query


def promoted_incremental_memory_telemetry_fields(
    limits: memory_budget.PromotedPhaseALimits,
    memory_summary: memory_budget.PredictionAccuracySummary,
) -> dict[str, int | float | str]:
    return {
        "memory_total_ram_bytes": int(limits.total_ram_bytes),
        "memory_available_bytes": int(limits.available_bytes),
        "memory_stage_budget_bytes": int(limits.stage_budget_bytes),
        "memory_predicted_peak_delta_bytes": int(memory_summary.predicted_peak_delta_bytes),
        "memory_predicted_peak_rss_bytes": int(memory_summary.predicted_peak_rss_bytes),
        "memory_rss_before_bytes": int(memory_summary.rss_before_bytes),
        "memory_rss_peak_bytes": int(memory_summary.rss_peak_bytes),
        "memory_rss_after_bytes": int(memory_summary.rss_after_bytes),
        "memory_observed_peak_delta_bytes": int(memory_summary.observed_peak_delta_bytes),
        "memory_observed_end_delta_bytes": int(memory_summary.observed_end_delta_bytes),
        "memory_prediction_error_ratio": float(memory_summary.prediction_error_ratio),
        "memory_underpredicted": int(bool(memory_summary.underpredicted)),
        "memory_prediction_contract_version": str(memory_summary.prediction_contract_version),
    }


def merge_promoted_incremental_batch_telemetry(
    batch_telemetries: list[Mapping[str, int | float | str]],
    *,
    batch_sizes: list[int],
    configured_batch_size: int | None,
    memory_telemetries: list[Mapping[str, int | float | str]] | None = None,
    initial_limits: memory_budget.PromotedPhaseALimits | None = None,
    final_limits: memory_budget.PromotedPhaseALimits | None = None,
    calibration_applied: bool = False,
) -> dict[str, int | float | str]:
    merged: dict[str, int | float | str] = {}
    conflict_counts: dict[str, int] = {}
    for telemetry in batch_telemetries:
        for key, value in telemetry.items():
            merge_policy = _PROMOTED_INCREMENTAL_TELEMETRY_MERGE_POLICY.get(key, "sum_numeric")
            if merge_policy == "first":
                if key not in merged:
                    merged[key] = value
                elif merged[key] != value:
                    conflict_counts[key] = conflict_counts.get(key, 0) + 1
                continue
            if merge_policy == "sum_numeric" and isinstance(value, int | float) and not isinstance(value, bool):
                previous = merged.get(key, 0)
                if isinstance(previous, int | float) and not isinstance(previous, bool):
                    merged[key] = previous + value
                else:
                    conflict_counts[key] = conflict_counts.get(key, 0) + 1
                continue
            if key not in merged:
                merged[key] = value
            elif merged[key] != value:
                conflict_counts[key] = conflict_counts.get(key, 0) + 1

    merged["query_batch_count"] = len(batch_sizes)
    merged["query_batch_size_configured"] = int(configured_batch_size or 0)
    merged["query_batch_size_max"] = max(batch_sizes, default=0)
    merged["query_batch_size_min"] = min(batch_sizes, default=0)
    merged.setdefault("query_count", sum(batch_sizes))
    if initial_limits is not None:
        merged["memory_initial_query_batch_size"] = int(initial_limits.query_batch_size)
        merged["memory_initial_predicted_peak_delta_bytes"] = int(initial_limits.predicted_peak_delta_bytes)
        merged["memory_initial_predicted_peak_rss_bytes"] = int(initial_limits.predicted_peak_rss_bytes)
        merged["memory_initial_operational_estimate_source"] = str(initial_limits.operational_estimate_source)
    if final_limits is not None:
        merged["memory_final_query_batch_size"] = int(final_limits.query_batch_size)
        merged["memory_final_predicted_peak_delta_bytes"] = int(final_limits.predicted_peak_delta_bytes)
        merged["memory_final_predicted_peak_rss_bytes"] = int(final_limits.predicted_peak_rss_bytes)
        merged["memory_final_operational_estimate_source"] = str(final_limits.operational_estimate_source)
    merged["memory_observed_calibration_applied"] = int(bool(calibration_applied))
    if memory_telemetries:
        int_max_fields = (
            "memory_predicted_peak_delta_bytes",
            "memory_predicted_peak_rss_bytes",
            "memory_rss_peak_bytes",
            "memory_observed_peak_delta_bytes",
            "memory_observed_end_delta_bytes",
            "memory_total_ram_bytes",
            "memory_available_bytes",
            "memory_stage_budget_bytes",
        )
        for field_name in int_max_fields:
            merged[f"{field_name}_max"] = max(int(item.get(field_name, 0)) for item in memory_telemetries)
        merged["memory_prediction_error_ratio_max"] = max(
            float(item.get("memory_prediction_error_ratio", 0.0)) for item in memory_telemetries
        )
        merged["memory_underpredicted_batch_count"] = sum(
            1 for item in memory_telemetries if bool(item.get("memory_underpredicted", 0))
        )
    for key, count in conflict_counts.items():
        merged[f"{key}_batch_conflict_count"] = int(count)
    return merged


def _summarize_query_views(query_views: tuple[str, ...]) -> str:
    if not query_views:
        return "none"
    unique_views = set(query_views)
    if len(unique_views) == 1:
        return query_views[0]
    return "mixed"


def predict_incremental_promoted_linker(
    clusterer: Any,
    block_signatures: list[str],
    dataset: ANDData,
    *,
    artifact_dir: Path,
    prevent_new_incompatibilities: bool,
    partial_supervision: dict[tuple[str, str], int | float],
    runtime_context: RuntimeContext,
    total_ram_bytes: int | None,
    batching_threshold: int | None,
    resolve_total_ram_bytes: ResolveTotalRamBytesFn,
    build_incremental_result: BuildIncrementalResultFn,
    get_rust_featurizer: GetRustFeaturizerFn,
    build_incremental_constraint_backend: BuildIncrementalConstraintBackendFn,
) -> dict[str, Any]:
    """Run the promoted linker as the incremental seed-link provider."""

    artifact = artifact_module.load_incremental_linking_artifact(artifact_dir)
    resolved_total_ram_bytes, _ = resolve_total_ram_bytes(total_ram_bytes)
    (
        cluster_seeds_require,
        recluster_map,
        cluster_seeds_require_inverse,
        split_cluster_seeds_require_inverse,
    ) = _unpack_incremental_seed_setup(
        clusterer._build_incremental_seed_setup(
            dataset,
            partial_supervision,
            runtime_context,
            total_ram_bytes=resolved_total_ram_bytes,
        )
    )
    seed_setup_telemetry = dict(getattr(clusterer, "_last_incremental_seed_setup_telemetry", {}) or {})
    if len(cluster_seeds_require) == 0:
        raise ValueError("Promoted incremental linker mode requires at least one seed cluster")

    unassigned_signature_ids = [
        str(signature_id) for signature_id in block_signatures if str(signature_id) not in cluster_seeds_require
    ]
    component_sizes = promoted_incremental_component_sizes(cluster_seeds_require)
    retrieval_top_k = int(artifact.metadata.retrieval_top_k)
    orcid_enabled = not bool(getattr(clusterer, "suppress_orcid", False))
    orcid_fanout_by_query = promoted_incremental_orcid_fanout_by_query(
        dataset,
        unassigned_signature_ids,
        cluster_seeds_require,
        orcid_enabled=orcid_enabled,
    )
    base_candidate_rows_per_query, base_pairs_per_query = _top_k_candidate_floors(component_sizes, retrieval_top_k)
    initial_row_floor, initial_pair_floor = _orcid_fanout_floor_estimates(
        orcid_fanout_by_query,
        unassigned_signature_ids,
    )
    initial_row_total_floor, initial_pair_total_floor = _orcid_fanout_floor_totals(
        orcid_fanout_by_query,
        unassigned_signature_ids,
        base_candidate_rows_per_query=base_candidate_rows_per_query,
        base_pairs_per_query=base_pairs_per_query,
    )
    initial_limits = compute_promoted_incremental_limits(
        query_count=len(unassigned_signature_ids),
        component_sizes=component_sizes,
        retrieval_top_k=retrieval_top_k,
        total_ram_bytes=resolved_total_ram_bytes,
        max_query_batch_size=batching_threshold,
        candidate_rows_per_query_floor=initial_row_floor,
        pairs_per_query_floor=initial_pair_floor,
        candidate_rows_total_floor=initial_row_total_floor,
        pairs_total_floor=initial_pair_total_floor,
    )
    resolved_total_ram_bytes = int(initial_limits.total_ram_bytes)
    raise_if_promoted_incremental_batch_over_budget(initial_limits)
    featurizer = get_rust_featurizer(
        dataset,
        runtime_context=runtime_context,
    )
    constraint_backend = build_incremental_constraint_backend(
        dataset,
        use_default_constraints_as_supervision=clusterer.use_default_constraints_as_supervision,
        runtime_context=runtime_context,
        suppress_orcid=getattr(clusterer, "suppress_orcid", False),
    )
    linked_signature_clusters: dict[str, int | str] = {}
    batch_telemetries: list[Mapping[str, int | float | str]] = []
    memory_telemetries: list[Mapping[str, int | float | str]] = []
    batch_sizes: list[int] = []
    final_limits = initial_limits
    calibration_applied = False
    observed_probe: tuple[int, int, int] | None = None
    resolved_query_views: tuple[str, ...] = ()
    if unassigned_signature_ids:
        linker_inputs = query_adapter_module.build_incremental_linker_inputs(
            dataset=dataset,
            query_signature_ids=unassigned_signature_ids,
            cluster_seeds_require=cluster_seeds_require,
            query_view=None,
            orcid_enabled=orcid_enabled,
        )
        resolved_query_views = tuple(str(view) for view in linker_inputs.query_views)

        def _extra_row_signal_builder(retrieval_batch: Any, query_signature_id_by_index: Any) -> Any:
            return query_adapter_module.build_name_count_rarity_row_signals(
                retrieval_batch,
                query_signature_id_by_index=query_signature_id_by_index,
                query_by_signature_id=linker_inputs.query_by_signature_id,
                summary_by_component=linker_inputs.summary_by_component,
            )

        next_query_index = 0
        current_limits = initial_limits
        current_query_batch_size = max(1, int(current_limits.query_batch_size))
        while next_query_index < len(unassigned_signature_ids):
            remaining_query_count = len(unassigned_signature_ids) - next_query_index
            query_batch_size = min(current_query_batch_size, remaining_query_count)
            query_batch = unassigned_signature_ids[next_query_index : next_query_index + query_batch_size]
            batch_limit_kwargs: dict[str, Any] = {}
            if observed_probe is not None:
                observed_query_count, observed_rows_per_query, observed_pairs_per_query = observed_probe
                batch_limit_kwargs = {
                    "observed_query_count": observed_query_count,
                    "observed_candidate_rows_per_query": observed_rows_per_query,
                    "observed_pairs_per_query": observed_pairs_per_query,
                }
            batch_row_floor, batch_pair_floor = _orcid_fanout_floor_estimates(orcid_fanout_by_query, query_batch)
            batch_row_total_floor, batch_pair_total_floor = _orcid_fanout_floor_totals(
                orcid_fanout_by_query,
                query_batch,
                base_candidate_rows_per_query=base_candidate_rows_per_query,
                base_pairs_per_query=base_pairs_per_query,
            )
            batch_limits = compute_promoted_incremental_limits(
                query_count=len(query_batch),
                component_sizes=component_sizes,
                retrieval_top_k=retrieval_top_k,
                total_ram_bytes=resolved_total_ram_bytes,
                max_query_batch_size=len(query_batch),
                candidate_rows_per_query_floor=batch_row_floor,
                pairs_per_query_floor=batch_pair_floor,
                candidate_rows_total_floor=batch_row_total_floor,
                pairs_total_floor=batch_pair_total_floor,
                **batch_limit_kwargs,
            )
            if 0 < int(batch_limits.query_batch_size) < len(query_batch):
                current_limits = batch_limits
                final_limits = batch_limits
                current_query_batch_size = int(batch_limits.query_batch_size)
                continue
            raise_if_promoted_incremental_batch_over_budget(batch_limits)
            batch_queries = tuple(
                linker_inputs.query_by_signature_id[str(signature_id)] for signature_id in query_batch
            )
            batch_query_views = tuple(
                str(linker_inputs.query_view_by_signature_id[str(signature_id)]) for signature_id in query_batch
            )
            batch_rss_before_bytes = int(batch_limits.current_rss_bytes)
            private_result = runtime_module._predict_incremental_link_or_abstain_production_private(
                clusterer,
                artifact,
                dataset=dataset,
                featurizer=featurizer,
                retriever=linker_inputs.retriever,
                queries=batch_queries,
                query_signature_ids=query_batch,
                query_view=batch_query_views,
                partial_supervision=partial_supervision,
                constraint_backend=constraint_backend,
                extra_row_signal_builder=_extra_row_signal_builder,
                seed_setup=(cluster_seeds_require, recluster_map, cluster_seeds_require_inverse),
                runtime_context=runtime_context,
                n_jobs=clusterer.n_jobs,
                total_ram_bytes=resolved_total_ram_bytes,
            )
            linked_signature_clusters.update(dict(private_result.linked_signature_clusters))
            batch_telemetry = dict(private_result.telemetry)
            batch_telemetries.append(batch_telemetry)
            next_query_index += len(query_batch)
            batch_sizes.append(len(query_batch))
            batch_rss_after_bytes, batch_rss_source = memory_budget.current_rss_bytes_best_effort(
                int(batch_limits.total_ram_bytes)
            )
            batch_rss_peak_bytes = max(batch_rss_before_bytes, int(batch_rss_after_bytes))
            memory_summary = memory_budget.summarize_prediction_accuracy(
                stage_name="incremental_promoted_query_batch",
                predicted_peak_delta_bytes=int(batch_limits.predicted_peak_delta_bytes),
                rss_before_bytes=batch_rss_before_bytes,
                rss_peak_bytes=batch_rss_peak_bytes,
                rss_after_bytes=int(batch_rss_after_bytes),
            )
            batch_memory_telemetry = promoted_incremental_memory_telemetry_fields(
                batch_limits,
                memory_summary,
            )
            memory_telemetries.append(batch_memory_telemetry)
            logger.info(
                "Telemetry: incremental_promoted_query_batch index=%d query_count=%d "
                "candidate_row_count=%d pair_count=%d link_count=%d abstain_count=%d "
                "query_batch_size=%d query_batch_size_configured=%d "
                "operational_estimate_source=%s predicted_pairs_per_batch=%d "
                "predicted_candidate_rows_per_batch=%d pair_chunk_pairs=%d pair_chunk_count=%d "
                "prediction_contract_version=%s predicted_peak_delta_bytes=%d "
                "predicted_peak_rss_bytes=%d rss_before_bytes=%d rss_peak_bytes=%d "
                "rss_after_bytes=%d observed_peak_delta_bytes=%d prediction_error_ratio=%.3f "
                "underpredicted=%s rss_source=%s run_id=%s",
                len(batch_sizes),
                len(query_batch),
                int(batch_telemetry.get("candidate_row_count", 0)),
                int(batch_telemetry.get("pair_count", 0)),
                int(batch_telemetry.get("link_count", 0)),
                int(batch_telemetry.get("abstain_count", 0)),
                int(batch_limits.query_batch_size),
                int(batching_threshold or 0),
                str(batch_limits.operational_estimate_source),
                int(batch_limits.predicted_pairs_per_batch),
                int(batch_limits.predicted_candidate_rows_per_batch),
                int(batch_limits.pair_chunk_pairs),
                int(batch_limits.pair_chunk_count),
                str(memory_summary.prediction_contract_version),
                int(memory_summary.predicted_peak_delta_bytes),
                int(memory_summary.predicted_peak_rss_bytes),
                int(memory_summary.rss_before_bytes),
                int(memory_summary.rss_peak_bytes),
                int(memory_summary.rss_after_bytes),
                int(memory_summary.observed_peak_delta_bytes),
                float(memory_summary.prediction_error_ratio),
                bool(memory_summary.underpredicted),
                str(batch_rss_source),
                runtime_context.run_id,
            )
            memory_budget.emit_memory_telemetry(
                {
                    "stage": memory_summary.stage_name,
                    "index": len(batch_sizes),
                    "query_count": len(query_batch),
                    "candidate_row_count": int(batch_telemetry.get("candidate_row_count", 0)),
                    "pair_count": int(batch_telemetry.get("pair_count", 0)),
                    "query_batch_size": batch_limits.query_batch_size,
                    "query_batch_size_configured": int(batching_threshold or 0),
                    "operational_estimate_source": batch_limits.operational_estimate_source,
                    "predicted_pairs_per_batch": batch_limits.predicted_pairs_per_batch,
                    "predicted_candidate_rows_per_batch": batch_limits.predicted_candidate_rows_per_batch,
                    "pair_chunk_pairs": batch_limits.pair_chunk_pairs,
                    "pair_chunk_count": batch_limits.pair_chunk_count,
                    "prediction_contract_version": memory_summary.prediction_contract_version,
                    "predicted_peak_delta_bytes": memory_summary.predicted_peak_delta_bytes,
                    "predicted_peak_rss_bytes": memory_summary.predicted_peak_rss_bytes,
                    "rss_before_bytes": memory_summary.rss_before_bytes,
                    "rss_peak_bytes": memory_summary.rss_peak_bytes,
                    "rss_after_bytes": memory_summary.rss_after_bytes,
                    "observed_peak_delta_bytes": memory_summary.observed_peak_delta_bytes,
                    "prediction_error_ratio": memory_summary.prediction_error_ratio,
                    "underpredicted": memory_summary.underpredicted,
                    "rss_source": batch_rss_source,
                    "run_id": runtime_context.run_id,
                }
            )
            if not calibration_applied and next_query_index < len(unassigned_signature_ids):
                observed_probe = promoted_incremental_observed_probe(batch_telemetry, len(query_batch))
                if observed_probe is not None:
                    remaining_after_probe = len(unassigned_signature_ids) - next_query_index
                    observed_query_count, observed_rows_per_query, observed_pairs_per_query = observed_probe
                    remaining_row_floor, remaining_pair_floor = _orcid_fanout_floor_estimates(
                        orcid_fanout_by_query,
                        unassigned_signature_ids[next_query_index:],
                    )
                    remaining_row_total_floor, remaining_pair_total_floor = _orcid_fanout_floor_totals(
                        orcid_fanout_by_query,
                        unassigned_signature_ids[next_query_index:],
                        base_candidate_rows_per_query=base_candidate_rows_per_query,
                        base_pairs_per_query=base_pairs_per_query,
                    )
                    calibrated_limits = compute_promoted_incremental_limits(
                        query_count=remaining_after_probe,
                        component_sizes=component_sizes,
                        retrieval_top_k=retrieval_top_k,
                        total_ram_bytes=resolved_total_ram_bytes,
                        max_query_batch_size=batching_threshold,
                        observed_query_count=observed_query_count,
                        observed_candidate_rows_per_query=observed_rows_per_query,
                        observed_pairs_per_query=observed_pairs_per_query,
                        candidate_rows_per_query_floor=remaining_row_floor,
                        pairs_per_query_floor=remaining_pair_floor,
                        candidate_rows_total_floor=remaining_row_total_floor,
                        pairs_total_floor=remaining_pair_total_floor,
                    )
                    old_query_batch_size = int(current_limits.query_batch_size)
                    raise_if_promoted_incremental_batch_over_budget(calibrated_limits)
                    current_limits = calibrated_limits
                    current_query_batch_size = max(1, int(calibrated_limits.query_batch_size))
                    final_limits = calibrated_limits
                    calibration_applied = True
                    logger.info(
                        "Telemetry: incremental_promoted_query_batch_calibration "
                        "observed_query_count=%d observed_candidate_rows_per_query=%d "
                        "observed_pairs_per_query=%d old_query_batch_size=%d "
                        "new_query_batch_size=%d operational_estimate_source=%s "
                        "predicted_peak_delta_bytes=%d predicted_peak_rss_bytes=%d run_id=%s",
                        observed_query_count,
                        observed_rows_per_query,
                        observed_pairs_per_query,
                        old_query_batch_size,
                        int(calibrated_limits.query_batch_size),
                        str(calibrated_limits.operational_estimate_source),
                        int(calibrated_limits.predicted_peak_delta_bytes),
                        int(calibrated_limits.predicted_peak_rss_bytes),
                        runtime_context.run_id,
                    )
    merged_telemetry = merge_promoted_incremental_batch_telemetry(
        batch_telemetries,
        batch_sizes=batch_sizes,
        configured_batch_size=batching_threshold,
        memory_telemetries=memory_telemetries,
        initial_limits=initial_limits,
        final_limits=final_limits,
        calibration_applied=calibration_applied,
    )
    query_view_counts = Counter(resolved_query_views)
    for query_view, count in query_view_counts.items():
        merged_telemetry[f"query_view_{query_view}_count"] = int(count)
    logger.info(
        "Telemetry: incremental_promoted_query_batches query_count=%d batch_count=%d "
        "batch_size_min=%d batch_size_max=%d query_batch_size_configured=%d "
        "initial_query_batch_size=%d final_query_batch_size=%d calibration_applied=%s "
        "predicted_peak_delta_bytes_max=%d observed_peak_delta_bytes_max=%d "
        "underpredicted_batch_count=%d run_id=%s",
        int(merged_telemetry.get("query_count", len(unassigned_signature_ids))),
        int(merged_telemetry.get("query_batch_count", 0)),
        int(merged_telemetry.get("query_batch_size_min", 0)),
        int(merged_telemetry.get("query_batch_size_max", 0)),
        int(merged_telemetry.get("query_batch_size_configured", 0)),
        int(merged_telemetry.get("memory_initial_query_batch_size", 0)),
        int(merged_telemetry.get("memory_final_query_batch_size", 0)),
        bool(merged_telemetry.get("memory_observed_calibration_applied", 0)),
        int(merged_telemetry.get("memory_predicted_peak_delta_bytes_max", 0)),
        int(merged_telemetry.get("memory_observed_peak_delta_bytes_max", 0)),
        int(merged_telemetry.get("memory_underpredicted_batch_count", 0)),
        runtime_context.run_id,
    )
    finish_start = time.perf_counter()
    predicted_clusters = _finish_incremental_with_optional_split_inverse(
        clusterer,
        unassigned_signature_ids,
        dataset,
        linked_signature_clusters,
        recluster_map,
        cluster_seeds_require_inverse,
        prevent_new_incompatibilities,
        partial_supervision,
        runtime_context,
        total_ram_bytes=resolved_total_ram_bytes,
        split_cluster_seeds_require_inverse=split_cluster_seeds_require_inverse,
    )
    finish_seconds = time.perf_counter() - finish_start
    merged_telemetry = {
        **seed_setup_telemetry,
        **merged_telemetry,
        "incremental_finish_seconds": float(finish_seconds),
    }
    residual_count = sum(
        1 for signature_id in unassigned_signature_ids if signature_id not in linked_signature_clusters
    )
    phase_b_required_bytes = residual_count * (residual_count - 1) // 2 * 8
    payload = build_incremental_result(
        predicted_clusters,
        phase_b_mode="exact",
        phase_b_budget_bytes=phase_b_required_bytes,
        phase_b_required_bytes=phase_b_required_bytes,
        phase_b_residual_count=residual_count,
    )
    payload["incremental_linker_artifact_path"] = str(artifact_dir)
    payload["incremental_linker_query_view"] = _summarize_query_views(resolved_query_views)
    payload["incremental_linker_telemetry"] = merged_telemetry
    return payload


def predict_incremental_promoted_linker_from_arrow_paths(
    clusterer: Any,
    block_signatures: list[str],
    dataset: ANDData,
    *,
    arrow_paths: Mapping[str, Any],
    artifact_dir: Path,
    prevent_new_incompatibilities: bool,
    partial_supervision: dict[tuple[str, str], int | float],
    runtime_context: RuntimeContext,
    total_ram_bytes: int | None,
    batching_threshold: int | None,
    resolve_total_ram_bytes: ResolveTotalRamBytesFn,
    build_incremental_result: BuildIncrementalResultFn,
) -> dict[str, Any]:
    """Run the promoted linker from Arrow artifacts, then finish residuals through the normal path."""

    artifact = artifact_module.load_incremental_linking_artifact(artifact_dir)
    resolved_total_ram_bytes, _ = resolve_total_ram_bytes(total_ram_bytes)
    base_arrow_path_payload = feature_port.normalize_arrow_paths(arrow_paths)
    require_arrow_name_counts_index_for_clusterer(clusterer, base_arrow_path_payload, context="Raw Arrow scoring")
    (
        cluster_seeds_require,
        recluster_map,
        cluster_seeds_require_inverse,
        split_cluster_seeds_require_inverse,
    ) = _unpack_incremental_seed_setup(
        clusterer._build_incremental_seed_setup(
            dataset,
            partial_supervision,
            runtime_context,
            total_ram_bytes=resolved_total_ram_bytes,
            arrow_paths=base_arrow_path_payload,
        )
    )
    seed_setup_telemetry = dict(getattr(clusterer, "_last_incremental_seed_setup_telemetry", {}) or {})
    if len(cluster_seeds_require) == 0:
        raise ValueError("Promoted incremental linker mode requires at least one seed cluster")

    unassigned_signature_ids = [
        str(signature_id) for signature_id in block_signatures if str(signature_id) not in cluster_seeds_require
    ]
    component_sizes = promoted_incremental_component_sizes(cluster_seeds_require)
    retrieval_top_k = int(artifact.metadata.retrieval_top_k)
    orcid_enabled = not bool(getattr(clusterer, "suppress_orcid", False))
    orcid_fanout_by_query = (
        promoted_incremental_orcid_fanout_by_query(
            dataset,
            unassigned_signature_ids,
            cluster_seeds_require,
            orcid_enabled=orcid_enabled,
        )
        if hasattr(dataset, "signatures")
        else {}
    )
    base_candidate_rows_per_query, base_pairs_per_query = _top_k_candidate_floors(component_sizes, retrieval_top_k)
    initial_row_floor, initial_pair_floor = _orcid_fanout_floor_estimates(
        orcid_fanout_by_query,
        unassigned_signature_ids,
    )
    initial_row_total_floor, initial_pair_total_floor = _orcid_fanout_floor_totals(
        orcid_fanout_by_query,
        unassigned_signature_ids,
        base_candidate_rows_per_query=base_candidate_rows_per_query,
        base_pairs_per_query=base_pairs_per_query,
    )
    initial_limits = compute_promoted_incremental_limits(
        query_count=len(unassigned_signature_ids),
        component_sizes=component_sizes,
        retrieval_top_k=retrieval_top_k,
        total_ram_bytes=resolved_total_ram_bytes,
        max_query_batch_size=batching_threshold,
        candidate_rows_per_query_floor=initial_row_floor,
        pairs_per_query_floor=initial_pair_floor,
        candidate_rows_total_floor=initial_row_total_floor,
        pairs_total_floor=initial_pair_total_floor,
    )
    resolved_total_ram_bytes = int(initial_limits.total_ram_bytes)
    raise_if_promoted_incremental_batch_over_budget(initial_limits)
    linked_signature_clusters: dict[str, int | str] = {}
    batch_telemetries: list[Mapping[str, int | float | str]] = []
    batch_sizes: list[int] = []
    query_batch_size = max(1, int(initial_limits.query_batch_size or 1))
    name_tuples = getattr(dataset, "name_tuples", "filtered")
    request_disallows, dataset_disallows, arrow_disallows = _request_cluster_seed_disallows(
        dataset,
        base_arrow_path_payload,
    )
    seed_arrow_start = time.perf_counter()
    seed_arrow_matches_cluster_seeds_require = _cluster_seeds_arrow_matches(
        base_arrow_path_payload.get("cluster_seeds"),
        cluster_seeds_require,
    )
    seed_arrow_reused_source = (
        str(seed_setup_telemetry.get("seed_setup_cluster_seeds_source", "")) == "arrow"
        and len(recluster_map) == 0
        and seed_arrow_matches_cluster_seeds_require
        and request_disallows == arrow_disallows
    )
    if seed_arrow_reused_source:
        arrow_path_payload = dict(base_arrow_path_payload)
        arrow_path_bundle: TemporaryArrowPaths | None = None
    else:
        arrow_path_bundle = arrow_paths_with_temporary_cluster_seeds(
            base_arrow_path_payload,
            cluster_seeds_require,
            prefix="s2and_arrow_incremental_cluster_seeds_",
            cluster_seeds_disallow=request_disallows,
        )
        arrow_path_payload = arrow_path_bundle.paths
    seed_arrow_assignment_seconds = time.perf_counter() - seed_arrow_start
    try:
        plan_window_multiplier = 8
        plan_window_size = query_batch_size
        if query_batch_size < len(unassigned_signature_ids):
            plan_window_size = min(len(unassigned_signature_ids), query_batch_size * plan_window_multiplier)
        use_windowed_raw_plan = plan_window_size > query_batch_size
        raw_window_plan_count = 0
        raw_window_plan_seconds = 0.0
        raw_window_plan_query_count = 0
        raw_window_featurizer_count = 0
        raw_window_featurizer_seconds = 0.0
        raw_window_featurizer_signature_count = 0
        raw_window_subset_seconds = 0.0
        raw_window_plan_telemetry: dict[str, int | float | str] = {}

        rust_module = feature_port._require_rust_runtime() if use_windowed_raw_plan else None
        for plan_start_index in range(0, len(unassigned_signature_ids), plan_window_size):
            query_plan_window = unassigned_signature_ids[plan_start_index : plan_start_index + plan_window_size]
            raw_candidate_plan: Mapping[str, Any] | None = None
            raw_window_featurizer: Any | None = None
            if use_windowed_raw_plan:
                raw_window_start = time.perf_counter()
                assert rust_module is not None
                raw_candidate_plan = rust_module.raw_block_query_candidate_plan_arrow(
                    arrow_path_payload,
                    list(query_plan_window),
                    top_k=retrieval_top_k,
                    query_view="auto",
                    orcid_enabled=bool(orcid_enabled),
                    num_threads=clusterer.n_jobs,
                    max_exemplars=4,
                )
                raw_window_plan_seconds += time.perf_counter() - raw_window_start
                raw_window_plan_count += 1
                raw_window_plan_query_count += len(query_plan_window)
                _merge_raw_window_plan_telemetry(
                    raw_window_plan_telemetry,
                    _raw_window_plan_telemetry_fields(raw_candidate_plan),
                )
                raw_window_featurizer_start = time.perf_counter()
                signature_order = runtime_module.feature_block_signature_order_from_raw_candidate_plan(
                    raw_candidate_plan
                )
                raw_window_featurizer = feature_port.build_rust_featurizer_from_arrow_paths(
                    arrow_path_payload,
                    signature_ids=signature_order.signature_ids,
                    name_tuples=name_tuples,
                    load_name_counts=clusterer_uses_name_count_features(clusterer),
                    preprocess=True,
                    compute_reference_features=False,
                    num_threads=clusterer.n_jobs,
                )
                raw_window_featurizer_seconds += time.perf_counter() - raw_window_featurizer_start
                raw_window_featurizer_count += 1
                raw_window_featurizer_signature_count += len(signature_order.signature_ids)

            for local_start_index in range(0, len(query_plan_window), query_batch_size):
                query_batch = query_plan_window[local_start_index : local_start_index + query_batch_size]
                batch_raw_candidate_plan = None
                batch_raw_window_featurizer = None
                if raw_candidate_plan is not None:
                    raw_window_subset_start = time.perf_counter()
                    batch_raw_candidate_plan = runtime_module.subset_raw_candidate_plan_for_query_ids(
                        raw_candidate_plan,
                        query_batch,
                        zero_plan_timings=True,
                    )
                    raw_window_subset_seconds += time.perf_counter() - raw_window_subset_start
                    batch_raw_window_featurizer = raw_window_featurizer
                result = runtime_module.predict_incremental_link_or_abstain_from_raw_arrow_paths(
                    clusterer,
                    artifact,
                    arrow_paths=arrow_path_payload,
                    query_signature_ids=query_batch,
                    top_k=retrieval_top_k,
                    partial_supervision=partial_supervision,
                    runtime_context=runtime_context,
                    n_jobs=clusterer.n_jobs,
                    total_ram_bytes=resolved_total_ram_bytes,
                    load_name_counts=clusterer_uses_name_count_features(clusterer),
                    name_tuples=name_tuples,
                    orcid_enabled=orcid_enabled,
                    raw_candidate_plan=batch_raw_candidate_plan,
                    rust_featurizer=batch_raw_window_featurizer,
                    partial_supervision_seed_signature_to_component=cluster_seeds_require,
                )
                linked_signature_clusters.update(dict(result.linked_signature_clusters))
                batch_telemetries.append(dict(result.telemetry))
                batch_sizes.append(len(query_batch))

        merged_telemetry = merge_promoted_incremental_batch_telemetry(
            batch_telemetries,
            batch_sizes=batch_sizes,
            configured_batch_size=batching_threshold,
            initial_limits=initial_limits,
            final_limits=initial_limits,
        )
        merged_telemetry.setdefault("seed_signature_count", int(len(cluster_seeds_require)))
        merged_telemetry.setdefault("seed_component_count", int(len(cluster_seeds_require_inverse)))
        merged_telemetry.setdefault("raw_arrow_seed_signature_count", int(len(cluster_seeds_require)))
        merged_telemetry.setdefault("raw_arrow_seed_component_count", int(len(cluster_seeds_require_inverse)))
        finish_start = time.perf_counter()
        predicted_clusters = _finish_incremental_with_optional_split_inverse(
            clusterer,
            unassigned_signature_ids,
            dataset,
            linked_signature_clusters,
            recluster_map,
            cluster_seeds_require_inverse,
            prevent_new_incompatibilities,
            partial_supervision,
            runtime_context,
            total_ram_bytes=resolved_total_ram_bytes,
            arrow_paths=arrow_path_payload,
            split_cluster_seeds_require_inverse=split_cluster_seeds_require_inverse,
        )
        finish_seconds = time.perf_counter() - finish_start
        residual_phase_b_telemetry = dict(getattr(clusterer, "_last_incremental_residual_phase_b_telemetry", {}) or {})
        residual_count = sum(
            1 for signature_id in unassigned_signature_ids if signature_id not in linked_signature_clusters
        )
        phase_b_required_bytes = residual_count * (residual_count - 1) // 2 * 8
        payload = build_incremental_result(
            predicted_clusters,
            phase_b_mode="exact",
            phase_b_budget_bytes=phase_b_required_bytes,
            phase_b_required_bytes=phase_b_required_bytes,
            phase_b_residual_count=residual_count,
        )
        payload["incremental_linker_artifact_path"] = str(artifact_dir)
        payload["incremental_linker_query_view"] = "raw_arrow"
        payload["incremental_linker_telemetry"] = {
            **seed_setup_telemetry,
            **merged_telemetry,
            **residual_phase_b_telemetry,
            **raw_window_plan_telemetry,
            "incremental_finish_seconds": float(finish_seconds),
            "seed_arrow_assignment_seconds": float(seed_arrow_assignment_seconds),
            "seed_arrow_reused_source": int(bool(seed_arrow_reused_source)),
            "seed_arrow_dataset_disallow_count": int(len(dataset_disallows)),
            "seed_arrow_disallow_count": int(len(request_disallows)),
            "arrow_promoted_incremental": 1,
            "arrow_path_count": len(arrow_path_payload),
            "raw_arrow_window_plan_count": int(raw_window_plan_count),
            "raw_arrow_window_plan_query_count": int(raw_window_plan_query_count),
            "raw_arrow_window_plan_seconds": float(raw_window_plan_seconds),
            "raw_arrow_window_featurizer_count": int(raw_window_featurizer_count),
            "raw_arrow_window_featurizer_signature_count": int(raw_window_featurizer_signature_count),
            "raw_arrow_window_featurizer_seconds": float(raw_window_featurizer_seconds),
            "raw_arrow_window_subset_seconds": float(raw_window_subset_seconds),
        }
        return payload
    finally:
        if arrow_path_bundle is not None:
            arrow_path_bundle.close()
