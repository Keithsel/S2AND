"""Production orchestration for the promoted incremental linker."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import s2and.incremental_linking.artifact as artifact_module
import s2and.incremental_linking.query_adapter as query_adapter_module
import s2and.incremental_linking.runtime as runtime_module
from s2and import memory_budget
from s2and.data import ANDData
from s2and.runtime import RuntimeContext

logger = logging.getLogger("s2and")

_PROMOTED_INCREMENTAL_TELEMETRY_CONSTANT_KEYS = frozenset(
    {
        "retrieval_top_k",
        "seed_signature_count",
        "seed_component_count",
    }
)
_PROMOTED_INCREMENTAL_TELEMETRY_FIRST_VALUE_KEYS = frozenset(
    {
        "memory_total_ram_bytes",
        "memory_available_bytes",
        "memory_stage_budget_bytes",
        "memory_predicted_peak_delta_bytes",
        "memory_predicted_peak_rss_bytes",
        "memory_rss_before_bytes",
        "memory_rss_peak_bytes",
        "memory_rss_after_bytes",
        "memory_observed_peak_delta_bytes",
        "memory_observed_end_delta_bytes",
        "memory_prediction_error_ratio",
        "memory_underpredicted",
    }
)

BuildIncrementalResultFn = Callable[..., dict[str, Any]]
BuildIncrementalConstraintBackendFn = Callable[..., Any]
GetRustFeaturizerFn = Callable[..., Any]
ResolveTotalRamBytesFn = Callable[[int | None], tuple[int, str]]


def promoted_incremental_component_sizes(cluster_seeds_require: Mapping[str, int | str]) -> dict[str, int]:
    component_sizes: dict[str, int] = {}
    for cluster_id in cluster_seeds_require.values():
        component_key = str(cluster_id)
        component_sizes[component_key] = component_sizes.get(component_key, 0) + 1
    return component_sizes


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
            if key in _PROMOTED_INCREMENTAL_TELEMETRY_FIRST_VALUE_KEYS:
                if key not in merged:
                    merged[key] = value
                elif merged[key] != value:
                    conflict_counts[key] = conflict_counts.get(key, 0) + 1
                continue
            if (
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and key not in _PROMOTED_INCREMENTAL_TELEMETRY_CONSTANT_KEYS
            ):
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

    query_view = "initial_only"
    artifact = artifact_module.load_incremental_linking_artifact(artifact_dir)
    resolved_total_ram_bytes, _ = resolve_total_ram_bytes(total_ram_bytes)
    cluster_seeds_require, recluster_map, cluster_seeds_require_inverse = clusterer._build_incremental_seed_setup(
        dataset,
        partial_supervision,
        runtime_context,
        total_ram_bytes=resolved_total_ram_bytes,
    )
    if len(cluster_seeds_require) == 0:
        raise ValueError("Promoted incremental linker mode requires at least one seed cluster")

    unassigned_signature_ids = [
        str(signature_id) for signature_id in block_signatures if str(signature_id) not in cluster_seeds_require
    ]
    component_sizes = promoted_incremental_component_sizes(cluster_seeds_require)
    retrieval_top_k = int(artifact.metadata.retrieval_top_k)
    initial_limits = compute_promoted_incremental_limits(
        query_count=len(unassigned_signature_ids),
        component_sizes=component_sizes,
        retrieval_top_k=retrieval_top_k,
        total_ram_bytes=resolved_total_ram_bytes,
        max_query_batch_size=batching_threshold,
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
    if unassigned_signature_ids:
        linker_inputs = query_adapter_module.build_incremental_linker_inputs(
            dataset=dataset,
            query_signature_ids=unassigned_signature_ids,
            cluster_seeds_require=cluster_seeds_require,
            query_view=query_view,
        )

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
            batch_limits = compute_promoted_incremental_limits(
                query_count=len(query_batch),
                component_sizes=component_sizes,
                retrieval_top_k=retrieval_top_k,
                total_ram_bytes=resolved_total_ram_bytes,
                max_query_batch_size=len(query_batch),
                **batch_limit_kwargs,
            )
            raise_if_promoted_incremental_batch_over_budget(batch_limits)
            batch_queries = tuple(
                linker_inputs.query_by_signature_id[str(signature_id)] for signature_id in query_batch
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
                query_view=query_view,
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
                    calibrated_limits = compute_promoted_incremental_limits(
                        query_count=remaining_after_probe,
                        component_sizes=component_sizes,
                        retrieval_top_k=retrieval_top_k,
                        total_ram_bytes=resolved_total_ram_bytes,
                        max_query_batch_size=batching_threshold,
                        observed_query_count=observed_query_count,
                        observed_candidate_rows_per_query=observed_rows_per_query,
                        observed_pairs_per_query=observed_pairs_per_query,
                    )
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
                        int(initial_limits.query_batch_size),
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
    predicted_clusters = clusterer._finish_incremental_with_seed_links(
        unassigned_signature_ids,
        dataset,
        linked_signature_clusters,
        recluster_map,
        cluster_seeds_require_inverse,
        prevent_new_incompatibilities,
        partial_supervision,
        runtime_context,
        total_ram_bytes=resolved_total_ram_bytes,
    )
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
    payload["incremental_linker_query_view"] = str(query_view)
    payload["incremental_linker_telemetry"] = merged_telemetry
    return payload
