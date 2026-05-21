"""Private link-or-abstain orchestration helpers for incremental linking."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np

import s2and.incremental_linking.query_adapter as query_adapter_module
from s2and import feature_port
from s2and.consts import LARGE_DISTANCE, LARGE_INTEGER
from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.artifact import IncrementalLinkingArtifact
from s2and.incremental_linking.feature_block import (
    FeatureBlock,
    feature_block_for_signature_order,
    feature_block_from_raw_payloads,
    feature_block_signature_order_from_raw_candidate_plan,
)
from s2and.incremental_linking.features import LinkerFeatureMatrix, assemble_linker_feature_matrix
from s2and.incremental_linking.linker_pairwise import (
    PROMOTED_PAIRWISE_AGG_BASE_FEATURE_NAMES,
    PROMOTED_PAIRWISE_AGG_FEATURE_COLUMNS,
    PROMOTED_PAIRWISE_AGG_FEATURE_INDICES,
    LinkerCandidateBatch,
    PairwiseAggregateStats,
    compute_candidate_batch_pairwise_aggregate_stats_rust,
    compute_linker_pair_chunk_plan,
    iter_candidate_batch_pair_feature_chunks_rust,
)
from s2and.incremental_linking.logistic_gate import (
    NumpyLogisticGate,
    build_runtime_logistic_gate_matrix,
    load_logistic_gate_config,
)
from s2and.incremental_linking.retrieval import (
    LinkerRetrievalBatch,
    build_linker_retrieval_batch_from_raw_candidate_plan,
    build_linker_retrieval_batch_rust,
)
from s2and.incremental_linking.row_features import build_promoted_non_pairwise_row_features_with_telemetry
from s2and.runtime import build_runtime_context
from s2and.thread_config import resolve_n_jobs

LinkAction = Literal["link", "abstain"]

# Production 1.2 dense output semantics. The pairwise distance model preserves
# NaNs internally; only the exported pw_* aggregate features are zero-filled.
_PAIRWISE_MODEL_NAN_VALUE: float = float("nan")
_PAIRWISE_AGGREGATE_NAN_VALUE: float = 0.0


@dataclass(frozen=True)
class CandidateBatchPairwiseModelResult:
    """Fused candidate-batch pairwise model outputs and promoted aggregates."""

    row_signals: dict[str, np.ndarray]
    pairwise_stats: PairwiseAggregateStats
    telemetry: dict[str, int | float]


def signature_id_to_index_map(featurizer: Any) -> dict[str, int]:
    """Return the Rust signature index map for candidate-batch construction."""

    return {str(signature_id): index for index, signature_id in enumerate(featurizer.signature_ids())}


@dataclass(frozen=True)
class LinkOrAbstainDecision:
    """One private compact decision for a query signature."""

    query_signature_index: int
    action: LinkAction
    row_index: int | None
    component_key: str | None
    score: float | None
    runner_up_score: float | None
    score_margin: float | None


@dataclass(frozen=True)
class LinkOrAbstainCompactResult:
    """Private compact result for artifact-scored candidate rows."""

    probabilities: np.ndarray
    decisions: tuple[LinkOrAbstainDecision, ...]


@dataclass(frozen=True)
class LinkOrAbstainRetrievedCandidatesResult:
    """Private result for artifact-scored retrieved candidates."""

    feature_matrix: LinkerFeatureMatrix
    compact_result: LinkOrAbstainCompactResult
    telemetry: dict[str, int | float | str]


@dataclass(frozen=True)
class LinkOrAbstainProductionResult(LinkOrAbstainRetrievedCandidatesResult):
    """Private end-to-end result for the M3a production slice."""

    retrieval_batch: LinkerRetrievalBatch
    pairwise_model_result: CandidateBatchPairwiseModelResult
    linked_signature_clusters: dict[str, Any]


def _ordered_group_indices(query_indices: np.ndarray) -> tuple[np.ndarray, ...]:
    groups: list[np.ndarray] = []
    for query_index in tuple(dict.fromkeys(int(value) for value in query_indices)):
        groups.append(np.flatnonzero(query_indices == np.uint32(query_index)))
    return tuple(groups)


def _best_row_for_group(
    group: np.ndarray,
    *,
    probabilities: np.ndarray,
    retrieval_ranks: np.ndarray | None,
    component_keys: tuple[object, ...] | None,
) -> int:
    def sort_key(row_index: int) -> tuple[float, int, str]:
        rank = 0 if retrieval_ranks is None else int(retrieval_ranks[row_index])
        component_key = "" if component_keys is None else str(component_keys[row_index])
        return (-float(probabilities[row_index]), rank, component_key)

    return min((int(row_index) for row_index in group), key=sort_key)


def _artifact_logistic_gate(artifact: IncrementalLinkingArtifact) -> NumpyLogisticGate:
    gate_model = getattr(artifact, "gate_model", None)
    if gate_model is not None:
        return gate_model
    return load_logistic_gate_config(artifact.metadata.gate_config)


def _orcid_match_signal(row_signals: Mapping[str, Any] | None, row_count: int) -> np.ndarray | None:
    if row_signals is None or "orcid_match" not in row_signals:
        return None
    values = np.asarray(row_signals["orcid_match"])
    if values.ndim != 1 or len(values) != int(row_count):
        raise ValueError(f"orcid_match row signal must be 1D with row_count={row_count}, got {values.shape}")
    return values.astype(bool, copy=False)


def _optional_float_row_signal(
    row_signals: Mapping[str, Any] | None,
    name: str,
    row_count: int,
) -> np.ndarray | None:
    if row_signals is None or name not in row_signals:
        return None
    values = np.asarray(row_signals[name], dtype=np.float64)
    if values.ndim != 1 or len(values) != int(row_count):
        raise ValueError(f"{name} row signal must be 1D with row_count={row_count}, got {values.shape}")
    return values


def _constraint_require_signal(row_signals: Mapping[str, Any] | None, row_count: int) -> np.ndarray | None:
    require_count = _optional_float_row_signal(row_signals, "constraint_require_count", row_count)
    if require_count is None:
        return None
    return require_count > 0


def _constraint_disallow_veto_signal(row_signals: Mapping[str, Any] | None, row_count: int) -> np.ndarray | None:
    disallow_count = _optional_float_row_signal(row_signals, "constraint_disallow_count", row_count)
    if disallow_count is None:
        return None
    pair_count = _optional_float_row_signal(row_signals, "constraint_pair_count", row_count)
    disallow_fraction = _optional_float_row_signal(row_signals, "constraint_disallow_fraction", row_count)
    if pair_count is None or disallow_fraction is None:
        missing = [
            name
            for name, value in (
                ("constraint_pair_count", pair_count),
                ("constraint_disallow_fraction", disallow_fraction),
            )
            if value is None
        ]
        raise ValueError(f"constraint disallow veto requires row signals: {missing}")

    has_disallow = disallow_count > 0
    single_pair_disallow = (pair_count <= 1) & has_disallow
    all_pairs_disallow = (pair_count > 0) & (disallow_count >= pair_count)
    mostly_disallow = (pair_count >= 3) & (disallow_fraction >= 0.8)
    veto = has_disallow & (single_pair_disallow | all_pairs_disallow | mostly_disallow)

    # Positive hard evidence is allowed to choose an identity-bearing row even if
    # the candidate component also has incompatible historical members.
    require_rows = _constraint_require_signal(row_signals, row_count)
    if require_rows is not None:
        veto &= ~require_rows
    orcid_rows = _orcid_match_signal(row_signals, row_count)
    if orcid_rows is not None:
        veto &= ~orcid_rows
    return veto


def _validate_single_constraint_require_target(
    *,
    forced_constraint_rows: np.ndarray,
    component_keys: tuple[object, ...] | None,
    query_signature_index: int,
) -> None:
    if len(forced_constraint_rows) <= 1:
        return
    if component_keys is None:
        raise ValueError(
            "constraint_require_conflicting_candidate_components: "
            f"query_signature_index={query_signature_index} require_row_count={len(forced_constraint_rows)} "
            "component_keys_missing=True"
        )
    required_components = tuple(sorted({str(component_keys[int(row_index)]) for row_index in forced_constraint_rows}))
    if len(required_components) > 1:
        raise ValueError(
            "constraint_require_conflicting_candidate_components: "
            f"query_signature_index={query_signature_index} component_keys={required_components}"
        )


def _constraint_row_signals(candidate_batch: LinkerCandidateBatch, pair_labels: np.ndarray) -> dict[str, np.ndarray]:
    row_count = int(candidate_batch.row_count)
    labels = np.asarray(pair_labels, dtype=np.float64)
    pair_count = int(candidate_batch.pair_count)
    if labels.shape != (pair_count,):
        raise ValueError(f"pair_labels must have shape ({pair_count},), got {labels.shape}")
    row_indices = np.asarray(candidate_batch.pair_row_indices, dtype=np.int64)
    if row_indices.shape != (pair_count,):
        raise ValueError(f"pair_row_indices must have shape ({pair_count},), got {row_indices.shape}")

    pair_counts = np.bincount(row_indices, minlength=row_count).astype(np.float32, copy=False)
    require_counts = np.zeros(row_count, dtype=np.float32)
    disallow_counts = np.zeros(row_count, dtype=np.float32)
    hit_counts = np.zeros(row_count, dtype=np.float32)
    finite = np.isfinite(labels)
    if np.any(finite):
        finite_rows = row_indices[finite]
        distances = labels[finite] + float(LARGE_INTEGER)
        hit_counts += np.bincount(finite_rows, minlength=row_count).astype(np.float32, copy=False)
        require_rows = finite_rows[np.isclose(distances, 0.0)]
        if len(require_rows):
            require_counts += np.bincount(require_rows, minlength=row_count).astype(np.float32, copy=False)
        disallow_rows = finite_rows[distances >= float(LARGE_DISTANCE)]
        if len(disallow_rows):
            disallow_counts += np.bincount(disallow_rows, minlength=row_count).astype(np.float32, copy=False)

    disallow_fraction = np.zeros(row_count, dtype=np.float32)
    np.divide(disallow_counts, pair_counts, out=disallow_fraction, where=pair_counts > 0)
    return {
        "constraint_pair_count": pair_counts,
        "constraint_hit_count": hit_counts,
        "constraint_require_count": require_counts,
        "constraint_disallow_count": disallow_counts,
        "constraint_disallow_fraction": disallow_fraction,
    }


def _subset_row_signals(
    row_signals: Mapping[str, Any] | None,
    row_indices: np.ndarray,
    row_count: int,
) -> dict[str, Any] | None:
    if row_signals is None:
        return None
    subset: dict[str, Any] = {}
    for name, values in row_signals.items():
        array = np.asarray(values)
        if array.ndim != 1 or len(array) != row_count:
            raise ValueError(f"row signal {name!r} must be 1D with row_count={row_count}, got {array.shape}")
        subset[name] = array[row_indices]
    return subset


def _subset_feature_matrix_for_rows(
    feature_matrix: LinkerFeatureMatrix,
    row_indices: np.ndarray,
) -> LinkerFeatureMatrix:
    candidate_batch = feature_matrix.candidate_batch
    row_indices = np.asarray(row_indices, dtype=np.int64)
    row_component_keys = (
        None
        if candidate_batch.row_component_keys is None
        else tuple(candidate_batch.row_component_keys[int(row_index)] for row_index in row_indices)
    )
    retrieval_scores = (
        None if candidate_batch.retrieval_scores is None else np.asarray(candidate_batch.retrieval_scores)[row_indices]
    )
    retrieval_ranks = (
        None if candidate_batch.retrieval_ranks is None else np.asarray(candidate_batch.retrieval_ranks)[row_indices]
    )
    row_query_signature_indices = np.asarray(candidate_batch.row_query_signature_indices, dtype=np.uint32)[row_indices]
    subset_batch = LinkerCandidateBatch(
        row_count=len(row_indices),
        left_signature_indices=np.zeros(0, dtype=np.uint32),
        right_signature_indices=np.zeros(0, dtype=np.uint32),
        pair_row_indices=np.zeros(0, dtype=np.uint32),
        row_query_signature_indices=row_query_signature_indices,
        row_component_keys=row_component_keys,
        retrieval_scores=retrieval_scores,
        retrieval_ranks=retrieval_ranks,
    )
    return LinkerFeatureMatrix(
        matrix=np.asarray(feature_matrix.matrix)[row_indices],
        feature_columns=feature_matrix.feature_columns,
        candidate_batch=subset_batch,
        # Pairwise aggregate columns are already materialized in matrix; keeping
        # pairwise_stats would only overlay the same feature values during gating.
        pairwise_stats=None,
    )


def _predict_incremental_link_or_abstain_compact(
    artifact: IncrementalLinkingArtifact,
    feature_matrix: LinkerFeatureMatrix,
    *,
    row_signals: Mapping[str, Any] | None = None,
) -> LinkOrAbstainCompactResult:
    """Score artifact-ordered rows and apply the artifact's logistic gate.

    This is intentionally not a public API. It exists to keep the first vertical
    slice concrete while retrieval policy, constraint handling, and telemetry are
    still private implementation details.
    """

    candidate_batch = feature_matrix.candidate_batch
    if candidate_batch.row_query_signature_indices is None:
        raise ValueError("candidate_batch.row_query_signature_indices is required for compact decisions")
    probabilities = artifact.predict_probabilities(feature_matrix.matrix)
    if len(probabilities) != candidate_batch.row_count:
        raise ValueError("artifact probability count must match candidate row_count")
    query_indices = np.asarray(candidate_batch.row_query_signature_indices, dtype=np.uint32)
    component_keys = candidate_batch.row_component_keys
    gate = _artifact_logistic_gate(artifact)
    gate_matrix, gate_query_rows = build_runtime_logistic_gate_matrix(
        gate,
        feature_matrix,
        np.asarray(probabilities, dtype=np.float64),
        row_signals=row_signals,
    )
    gate_links = gate.predict_link(gate_matrix)
    orcid_matches = _orcid_match_signal(row_signals, candidate_batch.row_count)
    constraint_requires = _constraint_require_signal(row_signals, candidate_batch.row_count)
    constraint_vetoes = _constraint_disallow_veto_signal(row_signals, candidate_batch.row_count)
    decisions: list[LinkOrAbstainDecision] = []
    for query_pos, group in enumerate(gate_query_rows.groups):
        best_row = int(gate_query_rows.best_rows[query_pos])
        forced_orcid_rows = np.asarray([], dtype=np.int64)
        if orcid_matches is not None:
            forced_orcid_rows = group[orcid_matches[group]]
        forced_constraint_rows = np.asarray([], dtype=np.int64)
        if constraint_requires is not None:
            forced_constraint_rows = group[constraint_requires[group]]
        if len(forced_orcid_rows):
            best_row = _best_row_for_group(
                forced_orcid_rows,
                probabilities=probabilities,
                retrieval_ranks=candidate_batch.retrieval_ranks,
                component_keys=component_keys,
            )
            ranked_nonbest = [int(row) for row in gate_query_rows.ranked_groups[query_pos] if int(row) != best_row]
            runner_up_score = float(probabilities[ranked_nonbest[0]]) if ranked_nonbest else float("nan")
            margin = None if np.isnan(runner_up_score) else float(probabilities[best_row] - runner_up_score)
            action: LinkAction = "link"
        elif len(forced_constraint_rows):
            _validate_single_constraint_require_target(
                forced_constraint_rows=forced_constraint_rows,
                component_keys=component_keys,
                query_signature_index=int(query_indices[best_row]),
            )
            best_row = _best_row_for_group(
                forced_constraint_rows,
                probabilities=probabilities,
                retrieval_ranks=candidate_batch.retrieval_ranks,
                component_keys=component_keys,
            )
            ranked_nonbest = [int(row) for row in gate_query_rows.ranked_groups[query_pos] if int(row) != best_row]
            runner_up_score = float(probabilities[ranked_nonbest[0]]) if ranked_nonbest else float("nan")
            margin = None if np.isnan(runner_up_score) else float(probabilities[best_row] - runner_up_score)
            action = "link"
        elif constraint_vetoes is not None and np.any(constraint_vetoes[group]):
            eligible_original_rows = group[~constraint_vetoes[group]]
            if len(eligible_original_rows) == 0:
                runner_up_score = float(gate_query_rows.runner_up_scores[query_pos])
                margin = None if np.isnan(runner_up_score) else float(gate_query_rows.score_margins[query_pos])
                action = "abstain"
            else:
                eligible_matrix = _subset_feature_matrix_for_rows(feature_matrix, eligible_original_rows)
                eligible_row_signals = _subset_row_signals(
                    row_signals,
                    eligible_original_rows,
                    candidate_batch.row_count,
                )
                eligible_gate_matrix, eligible_gate_rows = build_runtime_logistic_gate_matrix(
                    gate,
                    eligible_matrix,
                    np.asarray(probabilities[eligible_original_rows], dtype=np.float64),
                    row_signals=eligible_row_signals,
                )
                eligible_gate_links = gate.predict_link(eligible_gate_matrix)
                eligible_ranked = eligible_gate_rows.ranked_groups[0]
                best_row = int(eligible_original_rows[int(eligible_gate_rows.best_rows[0])])
                runner_up_score = (
                    float(probabilities[int(eligible_original_rows[int(eligible_ranked[1])])])
                    if len(eligible_ranked) > 1
                    else float("nan")
                )
                margin = None if np.isnan(runner_up_score) else float(probabilities[best_row] - runner_up_score)
                action = "link" if bool(eligible_gate_links[0]) else "abstain"
        else:
            runner_up_score = float(gate_query_rows.runner_up_scores[query_pos])
            margin = None if np.isnan(runner_up_score) else float(gate_query_rows.score_margins[query_pos])
            action = "link" if bool(gate_links[query_pos]) else "abstain"
        component_key = None
        if action == "link" and component_keys is not None:
            component_key = str(component_keys[best_row])
        decisions.append(
            LinkOrAbstainDecision(
                query_signature_index=int(query_indices[best_row]),
                action=action,
                row_index=best_row if action == "link" else None,
                component_key=component_key,
                score=float(probabilities[best_row]),
                runner_up_score=None if np.isnan(runner_up_score) else float(runner_up_score),
                score_margin=margin,
            )
        )
    return LinkOrAbstainCompactResult(
        probabilities=np.asarray(probabilities, dtype=np.float64),
        decisions=tuple(decisions),
    )


def _pairwise_model_feature_indices(featurizer_info: FeaturizationInfo) -> tuple[int, ...]:
    selected: set[int] = set()
    for feature_group in featurizer_info.features_to_use:
        selected.update(featurizer_info.feature_group_to_index[str(feature_group)])
    return tuple(sorted(selected))


def _matrix_positions(matrix_indices: Sequence[int], selected_indices: Sequence[int]) -> tuple[int, ...]:
    position_by_index = {int(index): position for position, index in enumerate(matrix_indices)}
    missing = [int(index) for index in selected_indices if int(index) not in position_by_index]
    if missing:
        raise ValueError(f"selected pairwise model feature indices are missing from matrix_indices: {missing[:5]}")
    return tuple(position_by_index[int(index)] for index in selected_indices)


def _predict_pairwise_class0(classifier: Any, features: np.ndarray, *, num_threads: int) -> np.ndarray:
    # Estimator threading is configured through propagated n_jobs; predict_proba(num_threads=...)
    # is LightGBM-specific and breaks sklearn-compatible wrappers.
    del num_threads

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
        probabilities = classifier.predict_proba(features)
    return np.asarray(probabilities, dtype=np.float64)[:, 0]


def _predict_pairwise_model_distances(
    *,
    classifier: Any,
    features: np.ndarray,
    labels: np.ndarray,
    num_threads: int,
    nameless_classifier: Any | None = None,
    nameless_features: np.ndarray | None = None,
) -> np.ndarray:
    predictions = np.zeros(len(labels), dtype=np.float64)
    predict = np.isnan(labels)
    if np.any(predict):
        predicted = _predict_pairwise_class0(classifier, features[predict], num_threads=num_threads)
        if nameless_classifier is not None and nameless_features is not None:
            nameless_predicted = _predict_pairwise_class0(
                nameless_classifier,
                nameless_features[predict],
                num_threads=num_threads,
            )
            predicted = (predicted + nameless_predicted) / 2.0
        predictions[predict] = predicted
    return predictions


def _update_top_distances(top_distances: np.ndarray, row_index: int, distance: float) -> None:
    row = top_distances[row_index]
    if distance >= row[-1]:
        return
    row[-1] = distance
    row.sort()


def _distance_row_signals(
    *,
    counts: np.ndarray,
    sums: np.ndarray,
    mins: np.ndarray,
    top_distances: np.ndarray,
    empty_distance_value: float = 1.0,
) -> dict[str, np.ndarray]:
    row_count = len(counts)
    observed = counts > 0
    min_distance = np.full(row_count, float(empty_distance_value), dtype=np.float32)
    mean_distance = np.full(row_count, float(empty_distance_value), dtype=np.float32)
    top3_mean_distance = np.full(row_count, float(empty_distance_value), dtype=np.float32)
    top5_mean_distance = np.full(row_count, float(empty_distance_value), dtype=np.float32)
    pair_count = counts.astype(np.float32, copy=False)
    if np.any(observed):
        min_distance[observed] = mins[observed].astype(np.float32, copy=False)
        mean_distance[observed] = (sums[observed] / counts[observed]).astype(np.float32, copy=False)
        for row_index in np.flatnonzero(observed):
            finite = top_distances[row_index][np.isfinite(top_distances[row_index])]
            if len(finite) == 0:
                continue
            top3_mean_distance[row_index] = float(np.mean(finite[:3]))
            top5_mean_distance[row_index] = float(np.mean(finite[:5]))
    return {
        "min_distance": min_distance,
        "mean_distance": mean_distance,
        "top3_mean_distance": top3_mean_distance,
        "top5_mean_distance": top5_mean_distance,
        "pair_count": pair_count,
    }


def _accumulate_pairwise_distance_chunk(
    *,
    dataset: ANDData | None,
    row_indices: np.ndarray,
    row_count: int,
    model_distances: np.ndarray,
    labels: np.ndarray,
    n_jobs: int,
    runtime_context: Any | None,
    featurizer: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    if featurizer is not None and not callable(getattr(featurizer, "linker_pair_distance_accumulators", None)):
        raise RuntimeError(
            "RustFeaturizer.linker_pair_distance_accumulators is required for promoted linker distance "
            "aggregation; rebuild/install the current s2and-rust extension."
        )
    return feature_port.build_linker_pair_distance_accumulators_rust(
        dataset,
        row_indices,
        int(row_count),
        model_distances,
        pair_labels=labels,
        num_threads=resolve_n_jobs(n_jobs),
        runtime_context=runtime_context,
        featurizer=featurizer,
    )


def compute_candidate_batch_pairwise_model_and_aggregate_stats(
    dataset: ANDData | None,
    candidate_batch: LinkerCandidateBatch,
    *,
    classifier: Any,
    featurizer_info: FeaturizationInfo,
    nameless_classifier: Any | None = None,
    nameless_featurizer_info: FeaturizationInfo | None = None,
    pair_labels: np.ndarray | None = None,
    n_jobs: int = 1,
    total_ram_bytes: int | None = None,
    pairwise_model_nan_value: float = _PAIRWISE_MODEL_NAN_VALUE,
    pairwise_aggregate_nan_value: float = _PAIRWISE_AGGREGATE_NAN_VALUE,
    runtime_context: Any | None = None,
    featurizer: Any | None = None,
) -> CandidateBatchPairwiseModelResult:
    """Score candidate pairs and compute promoted pairwise aggregates in one Rust feature pass.

    Production defaults reproduce the dense production 1.2 matrix by preserving
    NaNs for pairwise distance model inputs and zero-filling the exported
    promoted pairwise aggregate values.
    """

    start_seconds = time.perf_counter()
    row_count = int(candidate_batch.row_count)
    pair_count = int(candidate_batch.pair_count)
    labels = (
        np.full(pair_count, np.nan, dtype=np.float64)
        if pair_labels is None
        else np.asarray(pair_labels, dtype=np.float64)
    )
    if labels.shape != (pair_count,):
        raise ValueError(f"pair_labels must have shape ({pair_count},), got {labels.shape}")

    main_indices = _pairwise_model_feature_indices(featurizer_info)
    if not main_indices:
        raise ValueError("featurizer_info selects no pairwise model features")
    nameless_indices = (
        ()
        if nameless_classifier is None or nameless_featurizer_info is None
        else _pairwise_model_feature_indices(nameless_featurizer_info)
    )
    aggregate_indices = tuple(int(index) for index in PROMOTED_PAIRWISE_AGG_FEATURE_INDICES)
    matrix_indices = tuple(dict.fromkeys((*main_indices, *nameless_indices, *aggregate_indices)))
    main_positions = _matrix_positions(matrix_indices, main_indices)
    nameless_positions = _matrix_positions(matrix_indices, nameless_indices) if nameless_indices else ()
    aggregate_feature_names = tuple(PROMOTED_PAIRWISE_AGG_BASE_FEATURE_NAMES)
    aggregate_columns = tuple(PROMOTED_PAIRWISE_AGG_FEATURE_COLUMNS)
    plan = compute_linker_pair_chunk_plan(
        total_pairs=pair_count,
        row_count=row_count,
        matrix_feature_count=len(matrix_indices),
        aggregate_feature_count=len(aggregate_indices),
        total_ram_bytes=total_ram_bytes,
    )
    aggregate_counts = np.zeros(row_count, dtype=np.uint64)
    aggregate_valid_counts = np.zeros((row_count, len(aggregate_indices)), dtype=np.uint64)
    aggregate_sums = np.zeros((row_count, len(aggregate_indices)), dtype=np.float64)
    aggregate_mins = np.full((row_count, len(aggregate_indices)), np.inf, dtype=np.float64)
    aggregate_maxs = np.full((row_count, len(aggregate_indices)), -np.inf, dtype=np.float64)
    distance_counts = np.zeros(row_count, dtype=np.uint64)
    distance_sums = np.zeros(row_count, dtype=np.float64)
    distance_mins = np.full(row_count, np.inf, dtype=np.float64)
    top_distances = np.full((row_count, 5), np.inf, dtype=np.float64)
    hard_disallow_distance_pair_count = 0
    if featurizer is None:
        if dataset is None:
            raise ValueError("dataset is required when featurizer is not provided")
        featurizer = feature_port._get_rust_featurizer(  # noqa: SLF001
            dataset,
            runtime_context=runtime_context,
        )

    chunk_count = 0
    feature_seconds = 0.0
    predict_seconds = 0.0
    for chunk in iter_candidate_batch_pair_feature_chunks_rust(
        dataset,
        candidate_batch,
        matrix_indices=matrix_indices,
        aggregate_indices=aggregate_indices,
        n_jobs=n_jobs,
        total_ram_bytes=total_ram_bytes,
        nan_value=float(pairwise_model_nan_value),
        aggregate_nan_value=float(pairwise_aggregate_nan_value),
        runtime_context=runtime_context,
        featurizer=featurizer,
        chunk_plan=plan,
    ):
        pair_features = chunk.pair_features
        feature_seconds += float(chunk.feature_seconds)
        chunk_count += 1

        observed = chunk.counts > 0
        if np.any(observed):
            rows = chunk.global_row_indices[observed]
            aggregate_counts[rows] += chunk.counts[observed].astype(np.uint64, copy=False)
            if chunk.valid_counts is None:
                raise RuntimeError("nan-aware pairwise aggregate chunks must include valid_counts")
            aggregate_valid_counts[rows] += chunk.valid_counts[observed]
            aggregate_sums[rows] += chunk.sums[observed]
            aggregate_mins[rows] = np.minimum(aggregate_mins[rows], chunk.mins[observed])
            aggregate_maxs[rows] = np.maximum(aggregate_maxs[rows], chunk.maxs[observed])

        predict_start = time.perf_counter()
        labels_chunk = labels[chunk.start : chunk.stop]
        model_pair_features = pair_features
        if not np.isnan(float(pairwise_model_nan_value)):
            model_pair_features = pair_features.copy()
            model_pair_features[np.isnan(model_pair_features)] = float(pairwise_model_nan_value)
        model_distances = _predict_pairwise_model_distances(
            classifier=classifier,
            features=model_pair_features[:, main_positions],
            labels=labels_chunk,
            num_threads=resolve_n_jobs(n_jobs),
            nameless_classifier=nameless_classifier,
            nameless_features=model_pair_features[:, nameless_positions] if nameless_positions else None,
        )
        predict_seconds += time.perf_counter() - predict_start
        distance_accumulators = _accumulate_pairwise_distance_chunk(
            dataset=dataset,
            row_indices=chunk.local_row_indices,
            row_count=len(chunk.global_row_indices),
            model_distances=model_distances,
            labels=labels_chunk,
            n_jobs=n_jobs,
            runtime_context=runtime_context,
            featurizer=featurizer,
        )
        chunk_counts, chunk_sums, chunk_mins, chunk_top_distances, chunk_hard_disallow_count = distance_accumulators
        observed_distance_rows = chunk_counts > 0
        if np.any(observed_distance_rows):
            rows = chunk.global_row_indices[observed_distance_rows]
            distance_counts[rows] += chunk_counts[observed_distance_rows].astype(np.uint64, copy=False)
            distance_sums[rows] += chunk_sums[observed_distance_rows]
            distance_mins[rows] = np.minimum(distance_mins[rows], chunk_mins[observed_distance_rows])
            for local_row_index, global_row_index in zip(
                np.flatnonzero(observed_distance_rows),
                rows,
                strict=True,
            ):
                row_top_distances = chunk_top_distances[int(local_row_index)]
                finite = row_top_distances[np.isfinite(row_top_distances)]
                for value in finite:
                    _update_top_distances(top_distances, int(global_row_index), float(value))
        hard_disallow_distance_pair_count += int(chunk_hard_disallow_count)

    pairwise_stats = PairwiseAggregateStats(
        counts=aggregate_counts,
        sums=aggregate_sums,
        mins=aggregate_mins,
        maxs=aggregate_maxs,
        base_feature_names=aggregate_feature_names,
        aggregate_feature_columns=aggregate_columns,
        chunk_plan=plan,
        chunk_count=int(chunk_count),
        matrix_indices=matrix_indices,
        aggregate_indices=aggregate_indices,
        valid_counts=aggregate_valid_counts,
    )
    telemetry: dict[str, int | float] = {
        "candidate_row_count": row_count,
        "pair_count": pair_count,
        "chunk_count": int(chunk_count),
        "matrix_feature_count": int(len(matrix_indices)),
        "aggregate_feature_count": int(len(aggregate_indices)),
        "feature_seconds": float(feature_seconds),
        "predict_seconds": float(predict_seconds),
        "total_seconds": float(time.perf_counter() - start_seconds),
        "hard_disallow_distance_pair_count": int(hard_disallow_distance_pair_count),
    }
    return CandidateBatchPairwiseModelResult(
        row_signals=_distance_row_signals(
            counts=distance_counts,
            sums=distance_sums,
            mins=distance_mins,
            top_distances=top_distances,
        ),
        pairwise_stats=pairwise_stats,
        telemetry=telemetry,
    )


def _merge_extra_row_signals(
    base_row_signals: Mapping[str, Any],
    extra_row_signals: Mapping[str, Any] | None,
) -> dict[str, Any]:
    row_signals = dict(base_row_signals)
    if extra_row_signals is None:
        return row_signals
    overlap = sorted(set(row_signals) & set(extra_row_signals))
    if overlap:
        raise ValueError(f"extra_row_signals may not override existing row signals: {overlap}")
    row_signals.update(extra_row_signals)
    return row_signals


def _merge_row_signal_sources(*sources: Mapping[str, Any] | None) -> dict[str, Any]:
    row_signals: dict[str, Any] = {}
    for source in sources:
        row_signals = _merge_extra_row_signals(row_signals, source)
    return row_signals


def _query_author_for_gate(query: Any) -> str:
    value = getattr(query, "query_author", None)
    if value is not None and str(value).strip():
        return str(value)

    def first_present(*names: str) -> Any:
        for name in names:
            attr_value = getattr(query, name, None)
            if attr_value is not None and str(attr_value).strip():
                return attr_value
        return None

    parts = [
        first_present("first", "author_info_first"),
        first_present("middle", "author_info_middle"),
        first_present("last", "author_info_last"),
        first_present("suffix", "author_info_suffix"),
    ]
    return " ".join(str(part).strip() for part in parts if part is not None and str(part).strip())


def _production_query_author_row_signals(
    retrieval_batch: LinkerRetrievalBatch,
    *,
    query_signature_id_by_index: Mapping[int, str],
    query_by_signature_id: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    candidate_batch = retrieval_batch.candidate_batch
    row_count = int(candidate_batch.row_count)
    existing_query_author = retrieval_batch.row_signals.get("query_author")
    if existing_query_author is not None:
        values = np.asarray(existing_query_author, dtype=object)
        if values.shape != (row_count,):
            raise ValueError(f"query_author row signal must have shape ({row_count},), got {values.shape}")
        return {}
    row_query_indices = candidate_batch.row_query_signature_indices
    if row_query_indices is None:
        raise ValueError("candidate_batch.row_query_signature_indices is required for query_author row signals")
    query_author = np.empty(row_count, dtype=object)
    for row_index, query_index in enumerate(row_query_indices):
        query_signature_id = query_signature_id_by_index.get(int(query_index))
        if query_signature_id is None:
            raise KeyError(f"Missing query signature id for index {int(query_index)}")
        query_author[row_index] = _query_author_for_gate(query_by_signature_id[str(query_signature_id)])
    return {"query_author": query_author}


def _featureize_linker_candidates_with_telemetry(
    *,
    dataset: ANDData | None,
    candidate_batch: LinkerCandidateBatch,
    row_signals: Mapping[str, Any],
    feature_columns: Sequence[str],
    pairwise_stats: PairwiseAggregateStats | None = None,
    n_jobs: int = 1,
    total_ram_bytes: int | None = None,
    nan_value: float = _PAIRWISE_AGGREGATE_NAN_VALUE,
    runtime_context: Any | None = None,
    featurizer: Any | None = None,
) -> tuple[LinkerFeatureMatrix, dict[str, int]]:
    """Private featureizer from compact production-shaped candidate inputs."""

    resolved_feature_columns = tuple(str(column) for column in feature_columns)
    if candidate_batch.row_count == 0:
        return (
            LinkerFeatureMatrix(
                matrix=np.empty((0, len(resolved_feature_columns)), dtype=np.float32),
                feature_columns=resolved_feature_columns,
                candidate_batch=candidate_batch,
                pairwise_stats=pairwise_stats,
            ),
            {"generated_family_id_count": 0, "generic_family_override_count": 0},
        )
    if pairwise_stats is None:
        if dataset is None:
            raise ValueError("dataset is required when pairwise_stats is not provided")
        pairwise_stats = compute_candidate_batch_pairwise_aggregate_stats_rust(
            dataset,
            candidate_batch,
            n_jobs=n_jobs,
            total_ram_bytes=total_ram_bytes,
            nan_value=nan_value,
            runtime_context=runtime_context,
            featurizer=featurizer,
        )
    row_features, row_feature_telemetry = build_promoted_non_pairwise_row_features_with_telemetry(
        candidate_batch,
        row_signals,
    )
    return (
        assemble_linker_feature_matrix(
            candidate_batch,
            row_features,
            pairwise_stats=pairwise_stats,
            feature_columns=resolved_feature_columns,
        ),
        row_feature_telemetry,
    )


def _featureize_linker_candidates(
    *,
    dataset: ANDData | None,
    candidate_batch: LinkerCandidateBatch,
    row_signals: Mapping[str, Any],
    feature_columns: Sequence[str],
    pairwise_stats: PairwiseAggregateStats | None = None,
    n_jobs: int = 1,
    total_ram_bytes: int | None = None,
    nan_value: float = _PAIRWISE_AGGREGATE_NAN_VALUE,
    runtime_context: Any | None = None,
    featurizer: Any | None = None,
) -> LinkerFeatureMatrix:
    feature_matrix, _row_feature_telemetry = _featureize_linker_candidates_with_telemetry(
        dataset=dataset,
        candidate_batch=candidate_batch,
        row_signals=row_signals,
        feature_columns=feature_columns,
        pairwise_stats=pairwise_stats,
        n_jobs=n_jobs,
        total_ram_bytes=total_ram_bytes,
        nan_value=nan_value,
        runtime_context=runtime_context,
        featurizer=featurizer,
    )
    return feature_matrix


def _no_candidate_abstain_decisions(
    query_signature_indices: Sequence[int] | np.ndarray,
) -> tuple[LinkOrAbstainDecision, ...]:
    return tuple(
        LinkOrAbstainDecision(
            query_signature_index=int(query_index),
            action="abstain",
            row_index=None,
            component_key=None,
            score=None,
            runner_up_score=None,
            score_margin=None,
        )
        for query_index in query_signature_indices
    )


def _signature_id_to_index(signature_id_to_index: Mapping[str, int], signature_id: Any) -> int:
    key = str(signature_id)
    if key not in signature_id_to_index:
        raise KeyError(f"signature_id not present in linker runtime signature_ids: {key!r}")
    return int(signature_id_to_index[key])


def _build_component_member_indices_by_key(
    cluster_seeds_require: Mapping[Any, Any],
    signature_id_to_index: Mapping[str, int],
) -> dict[str, np.ndarray]:
    component_member_indices: dict[str, list[int]] = {}
    for signature_id, component_key in cluster_seeds_require.items():
        component_member_indices.setdefault(str(component_key), []).append(
            _signature_id_to_index(signature_id_to_index, signature_id)
        )
    return {
        component_key: np.asarray(member_indices, dtype=np.uint32)
        for component_key, member_indices in component_member_indices.items()
        if member_indices
    }


def _empty_retrieval_batch() -> LinkerRetrievalBatch:
    candidate_batch = LinkerCandidateBatch(
        row_count=0,
        left_signature_indices=np.zeros(0, dtype=np.uint32),
        right_signature_indices=np.zeros(0, dtype=np.uint32),
        pair_row_indices=np.zeros(0, dtype=np.uint32),
        row_query_signature_indices=np.zeros(0, dtype=np.uint32),
        row_component_keys=(),
        retrieval_scores=np.zeros(0, dtype=np.float32),
        retrieval_ranks=np.zeros(0, dtype=np.uint16),
    )
    return LinkerRetrievalBatch(candidate_batch=candidate_batch, row_signals={})


def _candidate_pair_ids(
    signature_ids_by_index: Sequence[Any],
    candidate_batch: LinkerCandidateBatch,
) -> list[tuple[str, str]]:
    signature_count = len(signature_ids_by_index)
    pair_ids: list[tuple[str, str]] = []
    for left_index, right_index in zip(
        candidate_batch.left_signature_indices,
        candidate_batch.right_signature_indices,
        strict=True,
    ):
        left = int(left_index)
        right = int(right_index)
        if left >= signature_count or right >= signature_count:
            raise IndexError(
                "candidate batch pair index out of range for linker runtime signature_ids: "
                f"left={left} right={right} signature_count={signature_count}"
            )
        pair_ids.append((str(signature_ids_by_index[left]), str(signature_ids_by_index[right])))
    return pair_ids


def _resolve_candidate_batch_pair_labels_rust(
    *,
    dataset: ANDData | None,
    candidate_batch: LinkerCandidateBatch,
    signature_ids_by_index: Sequence[Any],
    partial_supervision: Mapping[tuple[str, str], int | float],
    use_default_constraints_as_supervision: bool,
    dont_merge_cluster_seeds: bool,
    suppress_orcid: bool,
    n_jobs: int,
    runtime_context: Any | None,
    featurizer: Any | None,
) -> tuple[np.ndarray, Any]:
    pair_count = int(candidate_batch.pair_count)
    start_seconds = time.perf_counter()
    labels = np.full(pair_count, np.nan, dtype=np.float64)
    if use_default_constraints_as_supervision:
        method = None if featurizer is None else getattr(featurizer, "linker_pair_index_arrays_constraint_labels", None)
        if featurizer is not None and not callable(method):
            raise RuntimeError(
                "RustFeaturizer.linker_pair_index_arrays_constraint_labels is required for promoted linker "
                "constraint resolution; rebuild/install the current s2and-rust extension."
            )
        labels = feature_port.get_constraint_labels_index_arrays_rust(
            dataset,
            candidate_batch.left_signature_indices,
            candidate_batch.right_signature_indices,
            dont_merge_cluster_seeds=dont_merge_cluster_seeds,
            incremental_dont_use_cluster_seeds=False,
            num_threads=resolve_n_jobs(n_jobs),
            runtime_context=runtime_context,
            featurizer=featurizer,
            suppress_orcid=suppress_orcid,
        )

    partial_hits = 0
    if partial_supervision:
        signature_count = len(signature_ids_by_index)
        for pair_offset, (left_index, right_index) in enumerate(
            zip(candidate_batch.left_signature_indices, candidate_batch.right_signature_indices, strict=True)
        ):
            left = int(left_index)
            right = int(right_index)
            if left >= signature_count or right >= signature_count:
                raise IndexError(
                    "candidate batch pair index out of range for linker runtime signature_ids: "
                    f"left={left} right={right} signature_count={signature_count}"
                )
            left_id = str(signature_ids_by_index[left])
            right_id = str(signature_ids_by_index[right])
            if (left_id, right_id) in partial_supervision:
                labels[pair_offset] = float(partial_supervision[(left_id, right_id)] - LARGE_INTEGER)
                partial_hits += 1
            elif (right_id, left_id) in partial_supervision:
                labels[pair_offset] = float(partial_supervision[(right_id, left_id)] - LARGE_INTEGER)
                partial_hits += 1

    api_mode = "rust_index_arrays" if use_default_constraints_as_supervision else "partial_only"
    telemetry = SimpleNamespace(
        total_pairs=pair_count,
        partial_supervision_hits=int(partial_hits),
        unresolved_pairs=int(pair_count - partial_hits),
        rust_batch_call_count=int(use_default_constraints_as_supervision),
        api_mode=api_mode,
        elapsed_seconds=float(time.perf_counter() - start_seconds),
    )
    return labels, telemetry


def _partial_supervision_kind(value: int | float) -> str:
    value_float = float(value)
    if value_float == 0.0:
        return "require"
    if value_float == float(LARGE_DISTANCE):
        return "disallow"
    return "other"


def _validate_partial_supervision_window(
    *,
    partial_supervision: Mapping[tuple[str, str], int | float],
    query_signature_ids: set[str],
    seed_signature_to_component: Mapping[str, Any],
    candidate_pair_ids: Sequence[tuple[str, str]],
) -> dict[str, int]:
    telemetry = {
        "partial_supervision_pair_count": int(len(partial_supervision)),
        "partial_supervision_disallow_outside_retrieval_window": 0,
        "partial_supervision_disallow_between_residual_queries": 0,
        "partial_supervision_ignored_outside_window": 0,
    }
    inside_window_pairs: set[tuple[str, str]] = set()
    require_components_by_query: dict[str, set[str]] = {}
    for left, right in candidate_pair_ids:
        inside_window_pairs.add((left, right))
        inside_window_pairs.add((right, left))

    for (left_raw, right_raw), value in partial_supervision.items():
        left = str(left_raw)
        right = str(right_raw)
        kind = _partial_supervision_kind(value)
        left_is_query = left in query_signature_ids
        right_is_query = right in query_signature_ids
        if left_is_query and right_is_query:
            if kind == "require":
                raise ValueError(
                    "partial_supervision_require_between_residual_queries: "
                    f"query_signature_id_1={left!r} query_signature_id_2={right!r}"
                )
            if kind == "disallow":
                telemetry["partial_supervision_disallow_between_residual_queries"] += 1
            else:
                telemetry["partial_supervision_ignored_outside_window"] += 1
            continue

        query_signature_id: str | None = None
        seed_signature_id: str | None = None
        if left_is_query and right in seed_signature_to_component:
            query_signature_id = left
            seed_signature_id = right
        elif right_is_query and left in seed_signature_to_component:
            query_signature_id = right
            seed_signature_id = left

        if query_signature_id is None or seed_signature_id is None:
            telemetry["partial_supervision_ignored_outside_window"] += 1
            continue
        seed_component = seed_signature_to_component[seed_signature_id]
        if kind == "require":
            require_components = require_components_by_query.setdefault(query_signature_id, set())
            require_components.add(str(seed_component))
            if len(require_components) > 1:
                raise ValueError(
                    "partial_supervision_require_conflicting_seed_components: "
                    f"query_signature_id={query_signature_id!r} components={tuple(sorted(require_components))}"
                )
        if (query_signature_id, seed_signature_id) in inside_window_pairs:
            continue
        if kind == "require":
            raise ValueError(
                "partial_supervision_require_outside_retrieval_window: "
                f"query_signature_id={query_signature_id!r} seed_signature_id={seed_signature_id!r} "
                f"seed_component={seed_component!r}"
            )
        if kind == "disallow":
            telemetry["partial_supervision_disallow_outside_retrieval_window"] += 1
        else:
            telemetry["partial_supervision_ignored_outside_window"] += 1
    return telemetry


def _constraint_telemetry_dict(telemetry: Any) -> dict[str, int | float | str]:
    out: dict[str, int | float | str] = {}
    for name in (
        "total_pairs",
        "partial_supervision_hits",
        "unresolved_pairs",
        "rust_batch_call_count",
        "api_mode",
        "elapsed_seconds",
    ):
        value = getattr(telemetry, name, None)
        if value is not None:
            out[f"constraint_{name}"] = value
    return out


def _predict_incremental_link_or_abstain_retrieved_candidates(
    artifact: IncrementalLinkingArtifact,
    retrieval_batch: LinkerRetrievalBatch,
    *,
    dataset: ANDData | None = None,
    extra_row_signals: Mapping[str, Any] | None = None,
    pairwise_stats: PairwiseAggregateStats | None = None,
    no_candidate_query_signature_indices: Sequence[int] | np.ndarray = (),
    partial_supervision: Mapping[Any, Any] | None = None,
    n_jobs: int = 1,
    total_ram_bytes: int | None = None,
    nan_value: float = _PAIRWISE_AGGREGATE_NAN_VALUE,
    runtime_context: Any | None = None,
    featurizer: Any | None = None,
) -> LinkOrAbstainRetrievedCandidatesResult:
    """Private vertical slice over retrieved candidates.

    This intentionally remains private while retrieval parity, partial
    supervision, constraints, and telemetry are still under M2/M3 validation.
    """

    if partial_supervision:
        raise NotImplementedError("partial supervision is not yet wired into the compact linker runtime")
    candidate_batch = retrieval_batch.candidate_batch
    row_signals = _merge_extra_row_signals(retrieval_batch.row_signals, extra_row_signals)
    feature_matrix, row_feature_telemetry = _featureize_linker_candidates_with_telemetry(
        dataset=dataset,
        candidate_batch=candidate_batch,
        row_signals=row_signals,
        feature_columns=artifact.metadata.feature_columns,
        pairwise_stats=pairwise_stats,
        n_jobs=n_jobs,
        total_ram_bytes=total_ram_bytes,
        nan_value=nan_value,
        runtime_context=runtime_context,
        featurizer=featurizer,
    )
    compact_result = _predict_incremental_link_or_abstain_compact(
        artifact,
        feature_matrix,
        row_signals=row_signals,
    )
    no_candidate_decisions = _no_candidate_abstain_decisions(no_candidate_query_signature_indices)
    if no_candidate_decisions:
        compact_result = LinkOrAbstainCompactResult(
            probabilities=compact_result.probabilities,
            decisions=(*compact_result.decisions, *no_candidate_decisions),
        )
    link_count = sum(1 for decision in compact_result.decisions if decision.action == "link")
    abstain_count = sum(1 for decision in compact_result.decisions if decision.action == "abstain")
    return LinkOrAbstainRetrievedCandidatesResult(
        feature_matrix=feature_matrix,
        compact_result=compact_result,
        telemetry={
            "candidate_row_count": int(candidate_batch.row_count),
            "pair_count": int(candidate_batch.pair_count),
            "no_candidate_query_count": int(len(no_candidate_decisions)),
            "decision_count": int(len(compact_result.decisions)),
            "link_count": int(link_count),
            "abstain_count": int(abstain_count),
            **{f"row_feature_{key}": int(value) for key, value in row_feature_telemetry.items()},
        },
    )


def _predict_incremental_link_or_abstain_production_private(
    clusterer: Any,
    artifact: IncrementalLinkingArtifact,
    *,
    dataset: ANDData,
    featurizer: Any,
    retriever: Any,
    queries: Sequence[Any],
    query_signature_ids: Sequence[Any],
    query_view: str | Sequence[str] = "initial_only",
    top_k: int | None = None,
    partial_supervision: Mapping[tuple[Any, Any], int | float] | None = None,
    constraint_backend: Any | None = None,
    extra_row_signals: Mapping[str, Any] | None = None,
    extra_row_signal_builder: Callable[[LinkerRetrievalBatch, Mapping[int, str]], Mapping[str, Any]] | None = None,
    seed_setup: tuple[
        Mapping[str, int | str],
        Mapping[int | str, int | str],
        Mapping[int | str, Sequence[str]],
    ]
    | None = None,
    runtime_context: Any | None = None,
    n_jobs: int | None = None,
    total_ram_bytes: int | None = None,
    retrieval_top_k: int | None = None,
) -> LinkOrAbstainProductionResult:
    """Run the private M3a production-shaped link-or-abstain slice.

    The caller still owns production summary/query construction and the
    constraint backend so this runtime package stays free of `scripts.*` and
    `s2and.model` imports. This helper wires the pieces that are already runtime
    surfaces: seed setup, Rust retrieval into `LinkerCandidateBatch`, existing
    constraint-label resolution, fused pairwise scoring/aggregation, gate
    application, no-candidate abstains, and altered-cluster naturalization.
    """

    if len(queries) != len(query_signature_ids):
        raise ValueError(
            "queries and query_signature_ids must have equal length: " f"{len(queries)} != {len(query_signature_ids)}"
        )
    resolved_runtime_context = runtime_context or build_runtime_context("incremental_link_or_abstain_private")
    partial_supervision_dict = {
        (str(left), str(right)): value for (left, right), value in (partial_supervision or {}).items()
    }
    n_jobs_resolved = resolve_n_jobs(getattr(clusterer, "n_jobs", 1) if n_jobs is None else n_jobs)
    retrieval_top_k = int(artifact.metadata.retrieval_top_k if top_k is None else top_k)

    if seed_setup is None:
        build_seed_setup = getattr(clusterer, "_build_incremental_seed_setup", None)
        if not callable(build_seed_setup):
            raise TypeError("clusterer must expose _build_incremental_seed_setup for the private M3a slice")
        cluster_seeds_require, recluster_map, _cluster_seeds_require_inverse = build_seed_setup(
            dataset,
            partial_supervision_dict,
            resolved_runtime_context,
        )
    else:
        cluster_seeds_require, recluster_map, _cluster_seeds_require_inverse = seed_setup
    cluster_seeds_require = dict(cluster_seeds_require)
    recluster_map = dict(recluster_map)

    signature_id_to_index = signature_id_to_index_map(featurizer)
    query_signature_id_strings = tuple(str(signature_id) for signature_id in query_signature_ids)
    query_signature_indices = np.asarray(
        [_signature_id_to_index(signature_id_to_index, signature_id) for signature_id in query_signature_id_strings],
        dtype=np.uint32,
    )
    component_member_indices_by_key = _build_component_member_indices_by_key(
        cluster_seeds_require,
        signature_id_to_index,
    )
    if len(queries) == 0 or len(component_member_indices_by_key) == 0:
        retrieval_batch = _empty_retrieval_batch()
    else:
        retrieval_batch = build_linker_retrieval_batch_rust(
            retriever=retriever,
            queries=queries,
            query_signature_indices=query_signature_indices,
            component_member_indices_by_key=component_member_indices_by_key,
            top_k=retrieval_top_k,
            query_view=query_view,
            n_jobs=n_jobs_resolved,
        )
    return _predict_incremental_link_or_abstain_production_from_retrieval_private(
        clusterer,
        artifact,
        dataset=dataset,
        featurizer=featurizer,
        retrieval_batch=retrieval_batch,
        queries=queries,
        query_signature_ids=query_signature_ids,
        partial_supervision=partial_supervision_dict,
        constraint_backend=constraint_backend,
        extra_row_signals=extra_row_signals,
        extra_row_signal_builder=extra_row_signal_builder,
        seed_setup=(cluster_seeds_require, recluster_map, _cluster_seeds_require_inverse),
        runtime_context=resolved_runtime_context,
        n_jobs=n_jobs_resolved,
        total_ram_bytes=total_ram_bytes,
        retrieval_top_k=retrieval_top_k,
    )


def _predict_incremental_link_or_abstain_production_from_retrieval_private(
    clusterer: Any,
    artifact: IncrementalLinkingArtifact,
    *,
    dataset: ANDData | None,
    featurizer: Any,
    retrieval_batch: LinkerRetrievalBatch,
    queries: Sequence[Any],
    query_signature_ids: Sequence[Any],
    partial_supervision: Mapping[tuple[Any, Any], int | float] | None = None,
    constraint_backend: Any | None = None,
    extra_row_signals: Mapping[str, Any] | None = None,
    extra_row_signal_builder: Callable[[LinkerRetrievalBatch, Mapping[int, str]], Mapping[str, Any]] | None = None,
    seed_setup: tuple[
        Mapping[str, int | str],
        Mapping[int | str, int | str],
        Mapping[int | str, Sequence[str]],
    ]
    | None = None,
    runtime_context: Any | None = None,
    n_jobs: int | None = None,
    total_ram_bytes: int | None = None,
    retrieval_top_k: int | None = None,
) -> LinkOrAbstainProductionResult:
    """Run production scoring/gating from an already retrieved candidate batch."""

    if len(queries) != len(query_signature_ids):
        raise ValueError(
            "queries and query_signature_ids must have equal length: " f"{len(queries)} != {len(query_signature_ids)}"
        )
    resolved_runtime_context = runtime_context or build_runtime_context(
        "incremental_link_or_abstain_from_retrieval_private"
    )
    partial_supervision_dict = {
        (str(left), str(right)): value for (left, right), value in (partial_supervision or {}).items()
    }
    n_jobs_resolved = resolve_n_jobs(getattr(clusterer, "n_jobs", 1) if n_jobs is None else n_jobs)
    if seed_setup is None:
        build_seed_setup = getattr(clusterer, "_build_incremental_seed_setup", None)
        if not callable(build_seed_setup):
            raise TypeError("clusterer must expose _build_incremental_seed_setup for the private M3a slice")
        cluster_seeds_require, recluster_map, _cluster_seeds_require_inverse = build_seed_setup(
            dataset,
            partial_supervision_dict,
            resolved_runtime_context,
        )
    else:
        cluster_seeds_require, recluster_map, _cluster_seeds_require_inverse = seed_setup
    cluster_seeds_require = dict(cluster_seeds_require)
    recluster_map = dict(recluster_map)

    signature_id_to_index = signature_id_to_index_map(featurizer)
    signature_ids_by_index = tuple(str(signature_id) for signature_id in featurizer.signature_ids())
    query_signature_id_strings = tuple(str(signature_id) for signature_id in query_signature_ids)
    query_signature_indices = np.asarray(
        [_signature_id_to_index(signature_id_to_index, signature_id) for signature_id in query_signature_id_strings],
        dtype=np.uint32,
    )
    query_signature_id_by_index = {
        int(query_index): query_signature_id
        for query_index, query_signature_id in zip(query_signature_indices, query_signature_id_strings, strict=True)
    }
    query_by_signature_id = {
        query_signature_id: query for query_signature_id, query in zip(query_signature_id_strings, queries, strict=True)
    }

    candidate_batch = retrieval_batch.candidate_batch
    retrieved_query_indices = (
        set()
        if candidate_batch.row_query_signature_indices is None
        else {int(value) for value in np.asarray(candidate_batch.row_query_signature_indices, dtype=np.uint32)}
    )
    no_candidate_query_signature_indices = np.asarray(
        [
            int(query_index)
            for query_index in query_signature_indices
            if int(query_index) not in retrieved_query_indices
        ],
        dtype=np.uint32,
    )
    if partial_supervision_dict:
        pair_ids = _candidate_pair_ids(signature_ids_by_index, candidate_batch)
        partial_telemetry = _validate_partial_supervision_window(
            partial_supervision=partial_supervision_dict,
            query_signature_ids=set(query_signature_id_strings),
            seed_signature_to_component={
                str(signature_id): component for signature_id, component in cluster_seeds_require.items()
            },
            candidate_pair_ids=pair_ids,
        )
    else:
        partial_telemetry = {
            "partial_supervision_pair_count": 0,
            "partial_supervision_disallow_outside_retrieval_window": 0,
            "partial_supervision_disallow_between_residual_queries": 0,
            "partial_supervision_ignored_outside_window": 0,
        }

    constraint_featurizer = getattr(constraint_backend, "rust_featurizer", None) or featurizer
    pair_labels, constraint_telemetry = _resolve_candidate_batch_pair_labels_rust(
        dataset=dataset,
        candidate_batch=candidate_batch,
        signature_ids_by_index=signature_ids_by_index,
        partial_supervision=partial_supervision_dict,
        use_default_constraints_as_supervision=bool(getattr(clusterer, "use_default_constraints_as_supervision", True)),
        dont_merge_cluster_seeds=bool(getattr(clusterer, "dont_merge_cluster_seeds", True)),
        suppress_orcid=bool(getattr(clusterer, "suppress_orcid", False)),
        n_jobs=n_jobs_resolved,
        runtime_context=resolved_runtime_context,
        featurizer=constraint_featurizer,
    )
    if pair_labels.shape != (candidate_batch.pair_count,):
        raise ValueError(
            "constraint label count must match pair_count: " f"{pair_labels.shape} != ({candidate_batch.pair_count},)"
        )
    constraint_row_signals = _constraint_row_signals(candidate_batch, pair_labels)

    pairwise_model_result = compute_candidate_batch_pairwise_model_and_aggregate_stats(
        dataset,
        candidate_batch,
        classifier=clusterer.classifier,
        featurizer_info=clusterer.featurizer_info,
        nameless_classifier=getattr(clusterer, "nameless_classifier", None),
        nameless_featurizer_info=getattr(clusterer, "nameless_featurizer_info", None),
        pair_labels=pair_labels,
        n_jobs=n_jobs_resolved,
        total_ram_bytes=total_ram_bytes,
        runtime_context=resolved_runtime_context,
        featurizer=featurizer,
    )
    built_runtime_row_signals = _production_query_author_row_signals(
        retrieval_batch,
        query_signature_id_by_index=query_signature_id_by_index,
        query_by_signature_id=query_by_signature_id,
    )
    built_extra_row_signals = (
        {}
        if extra_row_signal_builder is None
        else dict(extra_row_signal_builder(retrieval_batch, query_signature_id_by_index))
    )
    merged_extra_row_signals = _merge_row_signal_sources(
        built_runtime_row_signals,
        built_extra_row_signals,
        extra_row_signals,
    )
    decision_row_signals = _merge_row_signal_sources(
        pairwise_model_result.row_signals,
        constraint_row_signals,
        merged_extra_row_signals,
    )
    private_result = _predict_incremental_link_or_abstain_retrieved_candidates(
        artifact,
        retrieval_batch,
        dataset=dataset,
        extra_row_signals=decision_row_signals,
        pairwise_stats=pairwise_model_result.pairwise_stats,
        no_candidate_query_signature_indices=no_candidate_query_signature_indices,
        n_jobs=n_jobs_resolved,
        total_ram_bytes=total_ram_bytes,
        nan_value=_PAIRWISE_AGGREGATE_NAN_VALUE,
        runtime_context=resolved_runtime_context,
        featurizer=featurizer,
    )

    raw_linked_clusters = {
        query_signature_id_by_index[decision.query_signature_index]: decision.component_key
        for decision in private_result.compact_result.decisions
        if decision.action == "link"
        and decision.component_key is not None
        and decision.query_signature_index in query_signature_id_by_index
    }
    linked_signature_clusters = naturalize_incremental_clusters(raw_linked_clusters, recluster_map)
    component_keys = candidate_batch.row_component_keys or ()
    telemetry: dict[str, int | float | str] = {
        **private_result.telemetry,
        **{f"pairwise_{key}": value for key, value in pairwise_model_result.telemetry.items()},
        **_constraint_telemetry_dict(constraint_telemetry),
        **partial_telemetry,
        "query_count": int(len(query_signature_id_strings)),
        "seed_signature_count": int(len(cluster_seeds_require)),
        "seed_component_count": int(len({str(value) for value in cluster_seeds_require.values()})),
        "retrieval_top_k": int(
            retrieval_top_k if retrieval_top_k is not None else _max_retrieval_rank(candidate_batch)
        ),
        "retrieved_component_count": int(len(component_keys)),
    }
    return LinkOrAbstainProductionResult(
        feature_matrix=private_result.feature_matrix,
        compact_result=private_result.compact_result,
        telemetry=telemetry,
        retrieval_batch=retrieval_batch,
        pairwise_model_result=pairwise_model_result,
        linked_signature_clusters=linked_signature_clusters,
    )


def _raw_plan_query_views(raw_candidate_plan: Mapping[str, Any], query_count: int) -> tuple[str, ...]:
    raw_views = raw_candidate_plan.get("query_views", raw_candidate_plan.get("query_views_resolved"))
    if raw_views is None:
        raise KeyError("raw candidate plan is missing query_views")
    views = tuple(str(value) for value in raw_views)
    if len(views) != int(query_count):
        raise ValueError(f"raw candidate plan query_views length must match query count: {len(views)} != {query_count}")
    return views


def _max_retrieval_rank(candidate_batch: LinkerCandidateBatch) -> int:
    retrieval_ranks = candidate_batch.retrieval_ranks
    if retrieval_ranks is None:
        return 0
    return max((int(rank) for rank in retrieval_ranks), default=0)


def _identity_seed_setup(
    cluster_seeds_require: Mapping[str, int | str],
) -> tuple[dict[str, int | str], dict[int | str, int | str], dict[int | str, list[str]]]:
    recluster_map: dict[int | str, int | str] = {
        component_id: component_id for component_id in cluster_seeds_require.values()
    }
    inverse: dict[int | str, list[str]] = {}
    for signature_id, component_id in cluster_seeds_require.items():
        inverse.setdefault(component_id, []).append(str(signature_id))
    return (
        {str(signature_id): component_id for signature_id, component_id in cluster_seeds_require.items()},
        recluster_map,
        inverse,
    )


def _raw_candidate_plan_telemetry_fields(raw_candidate_plan: Mapping[str, Any]) -> dict[str, int | float | str]:
    telemetry = raw_candidate_plan.get("telemetry")
    if not isinstance(telemetry, Mapping):
        return {}
    fields: dict[str, int | float | str] = {}
    window_plan_reused = bool(telemetry.get("window_plan_reused", 0))
    for key, value in telemetry.items():
        if key == "timings":
            continue
        if window_plan_reused and key in _RAW_CANDIDATE_PLAN_WINDOW_REUSE_ZERO_TELEMETRY_KEYS:
            fields[f"raw_arrow_plan_{key}"] = 0
            continue
        if isinstance(value, bool):
            fields[f"raw_arrow_plan_{key}"] = int(value)
        elif isinstance(value, int | float | str):
            fields[f"raw_arrow_plan_{key}"] = value
    timings = telemetry.get("timings")
    if isinstance(timings, Mapping):
        for key, value in timings.items():
            if isinstance(value, int | float):
                fields[f"raw_arrow_plan_{key}"] = float(value)
    return fields


_RAW_CANDIDATE_PLAN_ROW_KEYS: tuple[str, ...] = (
    "row_query_signature_indices",
    "row_component_keys",
    "retrieval_scores",
    "retrieval_ranks",
    "row_component_sizes",
    "row_named_signature_counts",
    "row_dominant_first_names",
    "row_candidate_year_min",
    "row_candidate_year_max",
    "row_candidate_year_range_missing",
    "row_query_first_tokens",
    "row_query_years",
    "row_query_year_missing",
    "row_query_has_affiliations",
    "row_query_has_coauthors",
    "row_orcid_match",
    "middle_initial_compatibility",
    "affiliation_overlap",
    "coauthor_overlap",
    "venue_overlap",
    "year_compatibility",
    "title_overlap",
    "specter_centroid_similarity",
    "specter_exemplar_similarity",
    "row_last_name_count_min_rarity",
    "row_candidate_last_name_count_min_rarity",
    "row_candidate_last_first_name_count_min_rarity",
    "row_last_first_name_count_min_rarity",
    "row_first_prefix_x_last_first_name_count_min_rarity",
    "row_candidate_cluster_max_paper_author_count",
    "row_paper_author_list_max_jaccard",
    "row_paper_author_list_max_containment",
    "row_paper_author_list_max_overlap_count",
    "row_local_author_window10_jaccard_max",
    "row_local_author_window10_overlap_count_max",
    "row_best_author_count_log_absdiff",
)
_RAW_CANDIDATE_PLAN_PAIR_KEYS: tuple[str, ...] = (
    "left_signature_indices",
    "right_signature_indices",
    "left_signature_ids",
    "right_signature_ids",
    "pair_row_indices",
)
_RAW_CANDIDATE_PLAN_WINDOW_REUSE_ZERO_TELEMETRY_KEYS: tuple[str, ...] = (
    "signature_count",
    "paper_count",
    "paper_author_paper_count",
    "specter_count",
    "seed_signature_count",
    "cluster_count",
    "unidecode_char_count",
    "excluded_query_seed_count",
    "indexed_arrow_candidate_plan",
    "signature_batches_read",
    "signature_rows_scanned",
    "paper_batches_read",
    "paper_rows_scanned",
    "paper_author_batches_read",
    "paper_author_rows_scanned",
    "specter_batches_read",
    "specter_rows_scanned",
)


def _subset_sequence_or_array(values: Any, mask: np.ndarray) -> Any:
    if isinstance(values, np.ndarray):
        return values[mask]
    return [value for value, keep in zip(values, mask, strict=True) if bool(keep)]


def _slice_sequence_or_array(values: Any, start: int, stop: int) -> Any:
    if isinstance(values, np.ndarray):
        return values[start:stop]
    return list(values[start:stop])


def _zero_raw_plan_timings(telemetry: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(telemetry)
    timings = out.get("timings")
    if isinstance(timings, Mapping):
        out["timings"] = {str(key): 0.0 for key in timings}
    for key in _RAW_CANDIDATE_PLAN_WINDOW_REUSE_ZERO_TELEMETRY_KEYS:
        if key in {"seed_signature_count", "cluster_count"}:
            continue
        value = out.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            out[key] = 0
    return out


def subset_raw_candidate_plan_for_query_ids(
    raw_candidate_plan: Mapping[str, Any],
    query_signature_ids: Sequence[Any],
    *,
    zero_plan_timings: bool = False,
) -> dict[str, Any]:
    """Return a raw candidate plan restricted to a query-id subset.

    The raw Arrow planner is query-separable: candidate rows and pair rows for
    each query depend on the shared seed table, not on other queries in the same
    planner call. This helper preserves the exact per-query row payload while
    remapping query offsets so downstream scoring sees the normal batch-local
    raw-plan contract.
    """

    requested_query_ids = tuple(str(signature_id) for signature_id in query_signature_ids)
    plan_query_ids = tuple(str(signature_id) for signature_id in raw_candidate_plan["query_signature_ids"])
    query_offset_by_id = {signature_id: offset for offset, signature_id in enumerate(plan_query_ids)}
    missing = [signature_id for signature_id in requested_query_ids if signature_id not in query_offset_by_id]
    if missing:
        raise ValueError(f"raw candidate plan is missing requested query_signature_ids: {missing[:10]}")

    old_query_offsets = np.asarray([query_offset_by_id[signature_id] for signature_id in requested_query_ids])
    old_to_new_query_offset = {
        int(old_query_offset): int(new_query_offset)
        for new_query_offset, old_query_offset in enumerate(old_query_offsets)
    }
    row_query_offsets = np.asarray(raw_candidate_plan["row_query_signature_indices"], dtype=np.uint32)
    pair_row_indices = np.asarray(raw_candidate_plan["pair_row_indices"], dtype=np.uint32)
    contiguous_query_offsets = (
        len(old_query_offsets) > 0
        and np.array_equal(
            old_query_offsets,
            np.arange(
                int(old_query_offsets[0]),
                int(old_query_offsets[0]) + len(old_query_offsets),
                dtype=old_query_offsets.dtype,
            ),
        )
    )
    sorted_row_offsets = len(row_query_offsets) < 2 or bool(np.all(row_query_offsets[:-1] <= row_query_offsets[1:]))
    sorted_pair_rows = len(pair_row_indices) < 2 or bool(np.all(pair_row_indices[:-1] <= pair_row_indices[1:]))
    if contiguous_query_offsets and sorted_row_offsets and sorted_pair_rows:
        old_query_start = int(old_query_offsets[0])
        old_query_stop = old_query_start + len(old_query_offsets)
        row_start = int(np.searchsorted(row_query_offsets, old_query_start, side="left"))
        row_stop = int(np.searchsorted(row_query_offsets, old_query_stop, side="left"))
        pair_start = int(np.searchsorted(pair_row_indices, row_start, side="left"))
        pair_stop = int(np.searchsorted(pair_row_indices, row_stop, side="left"))

        out = dict(raw_candidate_plan)
        out["query_signature_ids"] = list(requested_query_ids)
        out["query_views"] = [
            raw_candidate_plan["query_views"][query_offset_by_id[signature_id]] for signature_id in requested_query_ids
        ]
        out["query_authors"] = [
            raw_candidate_plan["query_authors"][query_offset_by_id[signature_id]]
            for signature_id in requested_query_ids
        ]
        out["row_count"] = int(row_stop - row_start)
        out["pair_count"] = int(pair_stop - pair_start)

        for key in _RAW_CANDIDATE_PLAN_ROW_KEYS:
            if key not in raw_candidate_plan:
                continue
            if key == "row_query_signature_indices":
                out[key] = (row_query_offsets[row_start:row_stop] - old_query_start).astype(np.uint32, copy=False)
            else:
                out[key] = _slice_sequence_or_array(raw_candidate_plan[key], row_start, row_stop)

        old_query_count = len(plan_query_ids)
        new_query_count = len(requested_query_ids)

        def remap_contiguous_signature_indices(values: Any) -> np.ndarray:
            raw_values = np.asarray(values, dtype=np.uint32)[pair_start:pair_stop]
            remapped = np.empty(len(raw_values), dtype=np.uint32)
            query_mask = raw_values < old_query_count
            if np.any(query_mask):
                remapped[query_mask] = raw_values[query_mask] - old_query_start
            if np.any(~query_mask):
                remapped[~query_mask] = new_query_count + (raw_values[~query_mask] - old_query_count)
            return remapped

        for key in _RAW_CANDIDATE_PLAN_PAIR_KEYS:
            if key not in raw_candidate_plan:
                continue
            if key in {"left_signature_indices", "right_signature_indices"}:
                out[key] = remap_contiguous_signature_indices(raw_candidate_plan[key])
            elif key == "pair_row_indices":
                out[key] = (pair_row_indices[pair_start:pair_stop] - row_start).astype(np.uint32, copy=False)
            else:
                out[key] = _slice_sequence_or_array(raw_candidate_plan[key], pair_start, pair_stop)

        retrieved_component_keys = set(out.get("row_component_keys", ()))
        component_members = raw_candidate_plan.get("component_members")
        if isinstance(component_members, Mapping):
            out["component_members"] = {
                str(component_key): list(members)
                for component_key, members in component_members.items()
                if str(component_key) in retrieved_component_keys
            }

        telemetry = raw_candidate_plan.get("telemetry")
        if isinstance(telemetry, Mapping):
            out["telemetry"] = _zero_raw_plan_timings(telemetry) if zero_plan_timings else dict(telemetry)
            out["telemetry"]["query_signature_count"] = int(len(requested_query_ids))
            if len(plan_query_ids) != len(requested_query_ids):
                out["telemetry"]["window_plan_reused"] = 1
        return out

    row_mask = np.isin(row_query_offsets, old_query_offsets)
    old_row_indices = np.flatnonzero(row_mask)
    old_row_to_new = np.full(int(raw_candidate_plan["row_count"]), -1, dtype=np.int64)
    old_row_to_new[old_row_indices] = np.arange(len(old_row_indices), dtype=np.int64)

    pair_mask = old_row_to_new[pair_row_indices] >= 0

    out = dict(raw_candidate_plan)
    out["query_signature_ids"] = list(requested_query_ids)
    out["query_views"] = [
        raw_candidate_plan["query_views"][query_offset_by_id[signature_id]] for signature_id in requested_query_ids
    ]
    out["query_authors"] = [
        raw_candidate_plan["query_authors"][query_offset_by_id[signature_id]] for signature_id in requested_query_ids
    ]
    out["row_count"] = int(len(old_row_indices))
    out["pair_count"] = int(np.count_nonzero(pair_mask))

    for key in _RAW_CANDIDATE_PLAN_ROW_KEYS:
        if key not in raw_candidate_plan:
            continue
        if key == "row_query_signature_indices":
            out[key] = np.asarray(
                [old_to_new_query_offset[int(value)] for value in row_query_offsets[row_mask]],
                dtype=np.uint32,
            )
        else:
            out[key] = _subset_sequence_or_array(raw_candidate_plan[key], row_mask)

    old_query_count = len(plan_query_ids)
    new_query_count = len(requested_query_ids)

    def remap_signature_indices(values: Any) -> np.ndarray:
        raw_values = np.asarray(values, dtype=np.uint32)[pair_mask]
        remapped = np.empty(len(raw_values), dtype=np.uint32)
        for offset, raw_value in enumerate(raw_values):
            index = int(raw_value)
            if index < old_query_count:
                remapped[offset] = old_to_new_query_offset[index]
            else:
                remapped[offset] = new_query_count + (index - old_query_count)
        return remapped

    for key in _RAW_CANDIDATE_PLAN_PAIR_KEYS:
        if key not in raw_candidate_plan:
            continue
        if key in {"left_signature_indices", "right_signature_indices"}:
            out[key] = remap_signature_indices(raw_candidate_plan[key])
        elif key == "pair_row_indices":
            out[key] = old_row_to_new[pair_row_indices[pair_mask]].astype(np.uint32, copy=False)
        else:
            out[key] = _subset_sequence_or_array(raw_candidate_plan[key], pair_mask)

    retrieved_component_keys = set(out.get("row_component_keys", ()))
    component_members = raw_candidate_plan.get("component_members")
    if isinstance(component_members, Mapping):
        out["component_members"] = {
            str(component_key): list(members)
            for component_key, members in component_members.items()
            if str(component_key) in retrieved_component_keys
        }

    telemetry = raw_candidate_plan.get("telemetry")
    if isinstance(telemetry, Mapping):
        out["telemetry"] = _zero_raw_plan_timings(telemetry) if zero_plan_timings else dict(telemetry)
        out["telemetry"]["query_signature_count"] = int(len(requested_query_ids))
        if len(plan_query_ids) != len(requested_query_ids):
            out["telemetry"]["window_plan_reused"] = 1
    return out


def _raw_candidate_plan_seed_setup(
    raw_candidate_plan: Mapping[str, Any],
) -> tuple[dict[str, int | str], dict[int | str, int | str], dict[int | str, list[str]]]:
    seed_signature_ids = raw_candidate_plan.get("seed_signature_ids")
    seed_component_keys = raw_candidate_plan.get("seed_component_keys")
    if isinstance(seed_signature_ids, Sequence) and not isinstance(seed_signature_ids, str | bytes):
        if not isinstance(seed_component_keys, Sequence) or isinstance(seed_component_keys, str | bytes):
            raise ValueError("raw candidate plan seed_component_keys must accompany seed_signature_ids")
        if len(seed_signature_ids) != len(seed_component_keys):
            raise ValueError(
                "raw candidate plan seed_signature_ids and seed_component_keys must have equal length: "
                f"{len(seed_signature_ids)} != {len(seed_component_keys)}"
            )
        return _identity_seed_setup(
            {
                str(signature_id): str(component_key)
                for signature_id, component_key in zip(seed_signature_ids, seed_component_keys, strict=True)
            }
        )

    component_members = raw_candidate_plan.get("component_members")
    if not isinstance(component_members, Mapping):
        raise ValueError("raw candidate plan must include component_members for scoring")
    cluster_seeds_require = {
        str(signature_id): str(component_key)
        for component_key, members in component_members.items()
        for signature_id in members
    }
    return _identity_seed_setup(cluster_seeds_require)


def _raw_candidate_plan_query_placeholders(
    raw_candidate_plan: Mapping[str, Any],
    query_signature_ids: Sequence[str],
) -> tuple[SimpleNamespace, ...]:
    query_authors = raw_candidate_plan.get("query_authors")
    if not isinstance(query_authors, Sequence) or isinstance(query_authors, str | bytes):
        raise ValueError("raw Arrow candidate plan must include query_authors")
    if len(query_authors) != len(query_signature_ids):
        raise ValueError(
            "raw Arrow candidate plan query_authors length must match query_signature_ids: "
            f"{len(query_authors)} != {len(query_signature_ids)}"
        )
    return tuple(SimpleNamespace(query_author=str(value or "")) for value in query_authors)


def predict_incremental_link_or_abstain_from_raw_feature_block(
    clusterer: Any,
    artifact: IncrementalLinkingArtifact,
    *,
    feature_block: FeatureBlock,
    raw_candidate_plan: Mapping[str, Any],
    partial_supervision: Mapping[tuple[Any, Any], int | float] | None = None,
    runtime_context: Any | None = None,
    n_jobs: int | None = None,
    total_ram_bytes: int | None = None,
    max_exemplars: int = 4,
    load_name_counts: bool | dict[str, Any] = False,
    name_tuples: set[tuple[str, str]] | str | None = "filtered",
) -> LinkOrAbstainProductionResult:
    """Score a raw candidate plan through `FeatureBlock` without full-block `ANDData`.

    This is the production bridge for raw retrieval output. It keeps the hot
    request bounded to the query plus returned candidate members, builds the
    Rust featurizer directly from `FeatureBlock`, and does not rerun retrieval.
    """

    if isinstance(load_name_counts, dict):
        raise ValueError("raw FeatureBlock scoring accepts load_name_counts as a bool, not a dict")
    resolved_runtime_context = runtime_context or build_runtime_context("incremental_link_or_abstain_raw_feature_block")
    n_jobs_resolved = resolve_n_jobs(getattr(clusterer, "n_jobs", 1) if n_jobs is None else n_jobs)
    stage_start = time.perf_counter()
    signature_order = feature_block_signature_order_from_raw_candidate_plan(raw_candidate_plan)
    mini_feature_block = feature_block_for_signature_order(feature_block, signature_order)
    order_seconds = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    featurizer = feature_port.build_rust_featurizer_from_feature_block(
        mini_feature_block,
        name_tuples=name_tuples,
        load_name_counts=bool(load_name_counts),
        preprocess=True,
        compute_reference_features=False,
        num_threads=n_jobs_resolved,
    )
    feature_block_featurizer_seconds = time.perf_counter() - stage_start
    featurizer_signature_ids = tuple(str(signature_id) for signature_id in featurizer.signature_ids())
    missing_plan_signature_ids = sorted(set(signature_order.signature_ids).difference(featurizer_signature_ids))
    if missing_plan_signature_ids:
        raise ValueError(
            "mini FeatureBlock featurizer is missing raw candidate plan signatures: "
            f"{missing_plan_signature_ids[:10]}"
        )
    featurizer_signature_id_to_index = signature_id_to_index_map(featurizer)

    retrieval_batch = build_linker_retrieval_batch_from_raw_candidate_plan(
        raw_candidate_plan,
        signature_id_to_index=featurizer_signature_id_to_index,
    )
    query_signature_ids = tuple(signature_order.query_signature_ids)
    query_views = _raw_plan_query_views(raw_candidate_plan, len(query_signature_ids))

    stage_start = time.perf_counter()
    feature_cache: dict[str, query_adapter_module.QueryFeatures] = {}
    base_query_by_signature_id = {
        query_signature_id: query_adapter_module.extract_query_features_from_feature_block(
            mini_feature_block,
            query_signature_id,
            feature_cache=feature_cache,
            orcid_enabled=True,
        )
        for query_signature_id in query_signature_ids
    }
    query_by_signature_id = {
        query_signature_id: query_adapter_module.mask_query_features(
            base_query_by_signature_id[query_signature_id],
            query_view,
            orcid_enabled=True,
        )
        for query_signature_id, query_view in zip(query_signature_ids, query_views, strict=True)
    }

    component_members = mini_feature_block.seed_component_members
    summary_by_component = {
        component_key: query_adapter_module.build_cluster_summary_from_feature_block(
            mini_feature_block,
            cluster_id=component_key,
            component_key=component_key,
            signature_ids=member_signature_ids,
            max_exemplars=max_exemplars,
            feature_cache=feature_cache,
            orcid_enabled=True,
        )
        for component_key, member_signature_ids in component_members.items()
    }
    feature_block_signal_seconds = time.perf_counter() - stage_start

    def extra_row_signal_builder(
        current_retrieval_batch: LinkerRetrievalBatch,
        query_signature_id_by_index: Mapping[int, str],
    ) -> Mapping[str, Any]:
        return query_adapter_module.build_name_count_rarity_row_signals(
            current_retrieval_batch,
            query_signature_id_by_index=query_signature_id_by_index,
            query_by_signature_id=query_by_signature_id,
            summary_by_component=summary_by_component,
        )

    result = _predict_incremental_link_or_abstain_production_from_retrieval_private(
        clusterer,
        artifact,
        dataset=None,
        featurizer=featurizer,
        retrieval_batch=retrieval_batch,
        queries=tuple(query_by_signature_id[query_signature_id] for query_signature_id in query_signature_ids),
        query_signature_ids=query_signature_ids,
        partial_supervision=partial_supervision,
        constraint_backend=None,
        extra_row_signal_builder=extra_row_signal_builder,
        seed_setup=_identity_seed_setup(dict(mini_feature_block.cluster_seeds_require)),
        runtime_context=resolved_runtime_context,
        n_jobs=n_jobs_resolved,
        total_ram_bytes=total_ram_bytes,
        retrieval_top_k=int(artifact.metadata.retrieval_top_k),
    )
    telemetry = {
        **result.telemetry,
        "feature_block_order_seconds": float(order_seconds),
        "feature_block_rust_featurizer_seconds": float(feature_block_featurizer_seconds),
        "feature_block_signal_seconds": float(feature_block_signal_seconds),
        "feature_block_signature_count": int(len(mini_feature_block.signatures)),
        "feature_block_paper_count": int(len(mini_feature_block.papers)),
        "feature_block_paper_author_count": int(len(mini_feature_block.paper_authors)),
        "feature_block_seed_signature_count": int(len(mini_feature_block.cluster_seeds_require)),
        "feature_block_seed_component_count": int(len(component_members)),
    }
    return LinkOrAbstainProductionResult(
        feature_matrix=result.feature_matrix,
        compact_result=result.compact_result,
        telemetry=telemetry,
        retrieval_batch=result.retrieval_batch,
        pairwise_model_result=result.pairwise_model_result,
        linked_signature_clusters=result.linked_signature_clusters,
    )


def predict_incremental_link_or_abstain_from_raw_arrow_paths(
    clusterer: Any,
    artifact: IncrementalLinkingArtifact,
    *,
    arrow_paths: Mapping[str, Any],
    query_signature_ids: Sequence[Any],
    query_view: str = "auto",
    top_k: int | None = None,
    partial_supervision: Mapping[tuple[Any, Any], int | float] | None = None,
    runtime_context: Any | None = None,
    n_jobs: int | None = None,
    total_ram_bytes: int | None = None,
    max_exemplars: int = 4,
    load_name_counts: bool | dict[str, Any] = False,
    name_tuples: set[tuple[str, str]] | str | None = "filtered",
    orcid_enabled: bool = True,
    raw_candidate_plan: Mapping[str, Any] | None = None,
    rust_featurizer: Any | None = None,
) -> LinkOrAbstainProductionResult:
    """Retrieve and score raw Arrow IPC inputs through Rust without `ANDData`."""

    if isinstance(load_name_counts, dict):
        raise ValueError("raw Arrow scoring accepts load_name_counts as a bool, not a dict")
    resolved_runtime_context = runtime_context or build_runtime_context("incremental_link_or_abstain_raw_arrow")
    n_jobs_resolved = resolve_n_jobs(getattr(clusterer, "n_jobs", 1) if n_jobs is None else n_jobs)
    top_k_resolved = int(artifact.metadata.retrieval_top_k if top_k is None else top_k)
    arrow_path_payload = {str(key): str(value) for key, value in arrow_paths.items()}
    query_signature_id_strings = tuple(str(signature_id) for signature_id in query_signature_ids)

    if raw_candidate_plan is None:
        stage_start = time.perf_counter()
        rust_module = feature_port._require_rust_runtime()  # noqa: SLF001
        raw_candidate_plan = rust_module.raw_block_query_candidate_plan_arrow(
            arrow_path_payload,
            list(query_signature_id_strings),
            top_k=top_k_resolved,
            query_view=str(query_view),
            orcid_enabled=bool(orcid_enabled),
            num_threads=n_jobs_resolved,
            max_exemplars=int(max_exemplars),
            include_seed_assignments=bool(partial_supervision),
        )
        raw_arrow_retrieval_seconds = time.perf_counter() - stage_start
    else:
        raw_candidate_plan = dict(raw_candidate_plan)
        raw_arrow_retrieval_seconds = 0.0

    stage_start = time.perf_counter()
    signature_order = feature_block_signature_order_from_raw_candidate_plan(raw_candidate_plan)
    if rust_featurizer is None:
        featurizer = feature_port.build_rust_featurizer_from_arrow_paths(
            arrow_path_payload,
            signature_ids=signature_order.signature_ids,
            name_tuples=name_tuples,
            load_name_counts=bool(load_name_counts),
            preprocess=True,
            compute_reference_features=False,
            num_threads=n_jobs_resolved,
        )
        raw_arrow_featurizer_reused = 0
    else:
        featurizer = rust_featurizer
        raw_arrow_featurizer_reused = 1
    raw_arrow_featurizer_seconds = time.perf_counter() - stage_start
    featurizer_signature_id_to_index = signature_id_to_index_map(featurizer)

    retrieval_batch = build_linker_retrieval_batch_from_raw_candidate_plan(
        raw_candidate_plan,
        signature_id_to_index=featurizer_signature_id_to_index,
    )
    _raw_plan_query_views(raw_candidate_plan, len(query_signature_id_strings))
    stage_start = time.perf_counter()
    query_placeholders = _raw_candidate_plan_query_placeholders(raw_candidate_plan, query_signature_id_strings)
    seed_setup = _raw_candidate_plan_seed_setup(raw_candidate_plan)
    seed_signature_ids = raw_candidate_plan.get("seed_signature_ids", ())
    seed_signature_count = len(seed_signature_ids) if isinstance(seed_signature_ids, Sequence) else 0
    plan_telemetry = raw_candidate_plan.get("telemetry")
    if seed_signature_count == 0 and isinstance(plan_telemetry, Mapping):
        seed_signature_count = int(plan_telemetry.get("seed_signature_count", 0) or 0)
    seed_component_count = len(seed_setup[1])
    if isinstance(plan_telemetry, Mapping):
        seed_component_count = int(plan_telemetry.get("cluster_count", seed_component_count) or seed_component_count)
    raw_arrow_signal_seconds = time.perf_counter() - stage_start

    result = _predict_incremental_link_or_abstain_production_from_retrieval_private(
        clusterer,
        artifact,
        dataset=None,
        featurizer=featurizer,
        retrieval_batch=retrieval_batch,
        queries=query_placeholders,
        query_signature_ids=query_signature_id_strings,
        partial_supervision=partial_supervision,
        constraint_backend=None,
        extra_row_signal_builder=None,
        seed_setup=seed_setup,
        runtime_context=resolved_runtime_context,
        n_jobs=n_jobs_resolved,
        total_ram_bytes=total_ram_bytes,
        retrieval_top_k=top_k_resolved,
    )
    raw_plan_telemetry_fields = _raw_candidate_plan_telemetry_fields(raw_candidate_plan)
    telemetry = {
        **result.telemetry,
        **raw_plan_telemetry_fields,
        "seed_signature_count": int(seed_signature_count),
        "seed_component_count": int(seed_component_count),
        "raw_arrow_retrieval_seconds": float(raw_arrow_retrieval_seconds),
        "raw_arrow_featurizer_seconds": float(raw_arrow_featurizer_seconds),
        "raw_arrow_featurizer_reused": int(raw_arrow_featurizer_reused),
        "raw_arrow_signal_seconds": float(raw_arrow_signal_seconds),
        "raw_arrow_signature_count": int(len(signature_order.signature_ids)),
        "raw_arrow_seed_signature_count": int(seed_signature_count),
        "raw_arrow_seed_component_count": int(seed_component_count),
    }
    return LinkOrAbstainProductionResult(
        feature_matrix=result.feature_matrix,
        compact_result=result.compact_result,
        telemetry=telemetry,
        retrieval_batch=result.retrieval_batch,
        pairwise_model_result=result.pairwise_model_result,
        linked_signature_clusters=result.linked_signature_clusters,
    )


def predict_incremental_link_or_abstain_from_raw_payloads(
    clusterer: Any,
    artifact: IncrementalLinkingArtifact,
    *,
    signatures: Mapping[str, Mapping[str, Any]],
    papers: Mapping[str, Mapping[str, Any]],
    raw_candidate_plan: Mapping[str, Any],
    cluster_seeds_require: Mapping[Any, Any],
    cluster_seeds_disallow: Iterable[tuple[Any, Any]] | None = None,
    specter_embeddings: Mapping[Any, Any] | tuple[Any, Any] | None = None,
    partial_supervision: Mapping[tuple[Any, Any], int | float] | None = None,
    runtime_context: Any | None = None,
    n_jobs: int | None = None,
    total_ram_bytes: int | None = None,
    max_exemplars: int = 4,
    load_name_counts: bool | dict[str, Any] = False,
    name_tuples: set[tuple[str, str]] | str | None = "filtered",
) -> LinkOrAbstainProductionResult:
    """Build a raw `FeatureBlock` from payloads and score a raw candidate plan."""

    stage_start = time.perf_counter()
    feature_block = feature_block_from_raw_payloads(
        signatures=signatures,
        papers=papers,
        raw_candidate_plan=raw_candidate_plan,
        cluster_seeds_require=cluster_seeds_require,
        cluster_seeds_disallow=cluster_seeds_disallow,
        specter_embeddings=specter_embeddings,
    )
    raw_payload_build_seconds = time.perf_counter() - stage_start
    result = predict_incremental_link_or_abstain_from_raw_feature_block(
        clusterer,
        artifact,
        feature_block=feature_block,
        raw_candidate_plan=raw_candidate_plan,
        partial_supervision=partial_supervision,
        runtime_context=runtime_context,
        n_jobs=n_jobs,
        total_ram_bytes=total_ram_bytes,
        max_exemplars=max_exemplars,
        load_name_counts=load_name_counts,
        name_tuples=name_tuples,
    )
    return LinkOrAbstainProductionResult(
        feature_matrix=result.feature_matrix,
        compact_result=result.compact_result,
        telemetry={
            **result.telemetry,
            "feature_block_raw_payload_build_seconds": float(raw_payload_build_seconds),
        },
        retrieval_batch=result.retrieval_batch,
        pairwise_model_result=result.pairwise_model_result,
        linked_signature_clusters=result.linked_signature_clusters,
    )


_predict_incremental_link_or_abstain_raw_feature_block_private = (
    predict_incremental_link_or_abstain_from_raw_feature_block
)


def naturalize_incremental_clusters(
    predicted_clusters: Mapping[str, Any],
    split_cluster_to_natural_cluster: Mapping[Any, Any],
) -> dict[str, Any]:
    """Naturalize split incremental cluster IDs back to caller-visible IDs."""

    return {
        str(signature_id): split_cluster_to_natural_cluster.get(cluster_id, cluster_id)
        for signature_id, cluster_id in predicted_clusters.items()
    }
