"""Thin Rust operation wrappers around the per-dataset Rust featurizer."""

from __future__ import annotations

from typing import Any

import numpy as np

from s2and.consts import LARGE_DISTANCE, LARGE_INTEGER
from s2and.data import ANDData
from s2and.thread_config import resolve_n_jobs


def _get_rust_featurizer(*args: Any, **kwargs: Any) -> Any:
    from s2and import feature_port

    return feature_port._get_rust_featurizer(*args, **kwargs)  # noqa: SLF001


def update_rust_cluster_seeds(
    dataset: ANDData,
    runtime_context: Any | None = None,
) -> None:
    featurizer = _get_rust_featurizer(dataset, runtime_context=runtime_context)
    featurizer.update_cluster_seeds(dataset.cluster_seeds_require, dataset.cluster_seeds_disallow)


def get_constraint_rust(
    dataset: ANDData,
    sig_id_1: str,
    sig_id_2: str,
    low_value: float = 0.0,
    high_value: float = LARGE_DISTANCE,
    dont_merge_cluster_seeds: bool = True,
    incremental_dont_use_cluster_seeds: bool = False,
    featurizer: Any | None = None,
    runtime_context: Any | None = None,
    suppress_orcid: bool = False,
):
    if featurizer is None:
        featurizer = _get_rust_featurizer(dataset, runtime_context=runtime_context)
    return featurizer.get_constraint(
        sig_id_1,
        sig_id_2,
        low_value,
        high_value,
        dont_merge_cluster_seeds,
        incremental_dont_use_cluster_seeds,
        suppress_orcid=suppress_orcid,
    )


def get_constraints_matrix_rust(
    dataset: ANDData,
    pairs: list[tuple[str, str]],
    low_value: float = 0.0,
    high_value: float = LARGE_DISTANCE,
    dont_merge_cluster_seeds: bool = True,
    incremental_dont_use_cluster_seeds: bool = False,
    num_threads: int | None = None,
    featurizer: Any | None = None,
    runtime_context: Any | None = None,
    suppress_orcid: bool = False,
) -> list[float | None]:
    if featurizer is None:
        featurizer = _get_rust_featurizer(dataset, runtime_context=runtime_context)
    resolved_num_threads = None if num_threads is None else resolve_n_jobs(num_threads)

    get_constraints_matrix = getattr(featurizer, "get_constraints_matrix", None)
    if not callable(get_constraints_matrix):
        raise RuntimeError("RustFeaturizer.get_constraints_matrix is unavailable; rebuild/install s2and-rust>=0.50.0.")
    return list(
        get_constraints_matrix(
            pairs,
            low_value,
            high_value,
            dont_merge_cluster_seeds,
            incremental_dont_use_cluster_seeds,
            resolved_num_threads,
            suppress_orcid=suppress_orcid,
        )
    )


def get_constraints_matrix_indexed_rust(
    dataset: ANDData,
    pairs: list[tuple[int, int]],
    low_value: float = 0.0,
    high_value: float = LARGE_DISTANCE,
    dont_merge_cluster_seeds: bool = True,
    incremental_dont_use_cluster_seeds: bool = False,
    num_threads: int | None = None,
    featurizer: Any | None = None,
    runtime_context: Any | None = None,
    suppress_orcid: bool = False,
) -> list[float | None]:
    if featurizer is None:
        featurizer = _get_rust_featurizer(dataset, runtime_context=runtime_context)
    resolved_num_threads = None if num_threads is None else resolve_n_jobs(num_threads)

    return list(
        featurizer.get_constraints_matrix_indexed(
            pairs,
            low_value,
            high_value,
            dont_merge_cluster_seeds,
            incremental_dont_use_cluster_seeds,
            resolved_num_threads,
            suppress_orcid=suppress_orcid,
        )
    )


def get_constraint_labels_index_arrays_rust(
    dataset: ANDData,
    left_signature_indices: np.ndarray,
    right_signature_indices: np.ndarray,
    low_value: float = 0.0,
    high_value: float = LARGE_DISTANCE,
    dont_merge_cluster_seeds: bool = True,
    incremental_dont_use_cluster_seeds: bool = False,
    num_threads: int | None = None,
    featurizer: Any | None = None,
    runtime_context: Any | None = None,
    suppress_orcid: bool = False,
    large_integer: float = LARGE_INTEGER,
) -> np.ndarray:
    """Resolve constraint labels for numeric pair-index arrays in Rust.

    Returned values use the existing pairwise-label convention:
    ``NaN`` means unconstrained, otherwise ``constraint_distance - LARGE_INTEGER``.
    """

    if featurizer is None:
        featurizer = _get_rust_featurizer(dataset, runtime_context=runtime_context)
    resolved_num_threads = None if num_threads is None else resolve_n_jobs(num_threads)

    method = getattr(featurizer, "linker_pair_index_arrays_constraint_labels", None)
    if not callable(method):
        raise RuntimeError(
            "RustFeaturizer.linker_pair_index_arrays_constraint_labels is unavailable; "
            "rebuild/install a newer s2and-rust extension."
        )
    return np.asarray(
        method(
            np.ascontiguousarray(left_signature_indices, dtype=np.uint32),
            np.ascontiguousarray(right_signature_indices, dtype=np.uint32),
            float(low_value),
            float(high_value),
            bool(dont_merge_cluster_seeds),
            bool(incremental_dont_use_cluster_seeds),
            resolved_num_threads,
            bool(suppress_orcid),
            float(large_integer),
        ),
        dtype=np.float64,
    )


def build_linker_pair_distance_accumulators_rust(
    dataset: ANDData,
    row_indices: np.ndarray,
    row_count: int,
    pair_distances: np.ndarray,
    pair_labels: np.ndarray | None = None,
    num_threads: int | None = None,
    featurizer: Any | None = None,
    runtime_context: Any | None = None,
    large_integer: float = LARGE_INTEGER,
    hard_disallow_distance: float = LARGE_DISTANCE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Aggregate candidate pair distances into row-level accumulators in Rust."""

    if featurizer is None:
        featurizer = _get_rust_featurizer(dataset, runtime_context=runtime_context)

    method = getattr(featurizer, "linker_pair_distance_accumulators", None)
    if not callable(method):
        raise RuntimeError(
            "RustFeaturizer.linker_pair_distance_accumulators is unavailable; "
            "rebuild/install a newer s2and-rust extension."
        )
    resolved_num_threads = None if num_threads is None else resolve_n_jobs(num_threads)
    labels_arg = None if pair_labels is None else np.ascontiguousarray(pair_labels, dtype=np.float64)
    counts, sums, mins, top_distances, hard_disallow_pair_count = method(
        np.ascontiguousarray(row_indices, dtype=np.uint32),
        int(row_count),
        np.ascontiguousarray(pair_distances, dtype=np.float64),
        labels_arg,
        resolved_num_threads,
        float(large_integer),
        float(hard_disallow_distance),
    )
    return (
        np.asarray(counts, dtype=np.uint32),
        np.asarray(sums, dtype=np.float64),
        np.asarray(mins, dtype=np.float64),
        np.asarray(top_distances, dtype=np.float64),
        int(hard_disallow_pair_count),
    )


def get_constraints_block_upper_triangle_indexed_rust(
    dataset: ANDData,
    block_signature_indices: list[int],
    start_offset: int = 0,
    max_pairs: int | None = None,
    low_value: float = 0.0,
    high_value: float = LARGE_DISTANCE,
    dont_merge_cluster_seeds: bool = True,
    incremental_dont_use_cluster_seeds: bool = False,
    num_threads: int | None = None,
    featurizer: Any | None = None,
    runtime_context: Any | None = None,
    suppress_orcid: bool = False,
) -> tuple[list[int], list[int], list[float | None]]:
    if featurizer is None:
        featurizer = _get_rust_featurizer(dataset, runtime_context=runtime_context)

    method = getattr(featurizer, "get_constraints_block_upper_triangle_indexed", None)
    if not callable(method):
        raise RuntimeError(
            "RustFeaturizer.get_constraints_block_upper_triangle_indexed is unavailable; "
            "rebuild/install a newer s2and-rust extension."
        )
    resolved_num_threads = None if num_threads is None else resolve_n_jobs(num_threads)

    left_indices, right_indices, values = method(
        block_signature_indices,
        start_offset,
        max_pairs,
        low_value,
        high_value,
        dont_merge_cluster_seeds,
        incremental_dont_use_cluster_seeds,
        resolved_num_threads,
        suppress_orcid=suppress_orcid,
    )
    return (
        [int(value) for value in left_indices],
        [int(value) for value in right_indices],
        list(values),
    )


def featurize_pair_rust(
    dataset: ANDData,
    sig_id_1: str,
    sig_id_2: str,
    runtime_context: Any | None = None,
):
    featurizer = _get_rust_featurizer(dataset, runtime_context=runtime_context)
    return featurizer.featurize_pair(sig_id_1, sig_id_2)


def build_pair_feature_matrix_rust(
    dataset: ANDData,
    pairs: list[tuple[str, str]],
    selected_indices: list[int] | None = None,
    num_threads: int | None = None,
    nan_value: float = np.nan,
    runtime_context: Any | None = None,
) -> np.ndarray:
    featurizer = _get_rust_featurizer(dataset, runtime_context=runtime_context)
    if not hasattr(featurizer, "featurize_pairs_matrix"):
        raise RuntimeError("RustFeaturizer.featurize_pairs_matrix is unavailable in the loaded extension")
    resolved_num_threads = None if num_threads is None else resolve_n_jobs(num_threads)
    matrix = featurizer.featurize_pairs_matrix(
        pairs,
        selected_indices,
        resolved_num_threads,
        nan_value,
    )
    return np.asarray(matrix, dtype=np.float64)


def build_linker_pair_features_and_aggregate_stats_indexed_rust(
    dataset: ANDData,
    pairs: list[tuple[int, int]],
    row_indices: list[int],
    row_count: int,
    matrix_indices: list[int] | None = None,
    aggregate_indices: list[int] | None = None,
    num_threads: int | None = None,
    nan_value: float = np.nan,
    runtime_context: Any | None = None,
    featurizer: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build one indexed pair-feature chunk and row-level aggregate stats in one Rust pass."""

    if featurizer is None:
        featurizer = _get_rust_featurizer(dataset, runtime_context=runtime_context)
    method = getattr(featurizer, "linker_pair_features_and_aggregate_stats_indexed", None)
    if not callable(method):
        raise RuntimeError(
            "RustFeaturizer.linker_pair_features_and_aggregate_stats_indexed is unavailable; "
            "rebuild/install a newer s2and-rust extension."
        )
    resolved_num_threads = None if num_threads is None else resolve_n_jobs(num_threads)
    matrix, counts, sums, mins, maxs = method(
        pairs,
        row_indices,
        int(row_count),
        matrix_indices,
        aggregate_indices,
        resolved_num_threads,
        nan_value,
    )
    return (
        np.asarray(matrix, dtype=np.float64),
        np.asarray(counts, dtype=np.uint32),
        np.asarray(sums, dtype=np.float64),
        np.asarray(mins, dtype=np.float64),
        np.asarray(maxs, dtype=np.float64),
    )


def build_linker_pair_features_and_aggregate_stats_arrays_rust(
    dataset: ANDData,
    left_signature_indices: np.ndarray,
    right_signature_indices: np.ndarray,
    row_indices: np.ndarray,
    row_count: int,
    matrix_indices: list[int] | None = None,
    aggregate_indices: list[int] | None = None,
    num_threads: int | None = None,
    nan_value: float = np.nan,
    aggregate_nan_value: float | None = None,
    runtime_context: Any | None = None,
    featurizer: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build pair features and row-level aggregate stats from numeric index arrays.

    ``nan_value`` controls the pair-feature matrix returned for model prediction.
    ``aggregate_nan_value`` can differ when callers need separate missing-value
    policies for the pairwise model matrix and promoted ``pw_*`` aggregates.
    """

    if featurizer is None:
        featurizer = _get_rust_featurizer(dataset, runtime_context=runtime_context)
    method = getattr(featurizer, "linker_pair_index_arrays_and_aggregate_stats", None)
    if not callable(method):
        raise RuntimeError(
            "RustFeaturizer.linker_pair_index_arrays_and_aggregate_stats is unavailable; "
            "rebuild/install a newer s2and-rust extension."
        )
    resolved_num_threads = None if num_threads is None else resolve_n_jobs(num_threads)
    resolved_aggregate_nan_value = nan_value if aggregate_nan_value is None else float(aggregate_nan_value)
    result = method(
        np.ascontiguousarray(left_signature_indices, dtype=np.uint32),
        np.ascontiguousarray(right_signature_indices, dtype=np.uint32),
        np.ascontiguousarray(row_indices, dtype=np.uint32),
        int(row_count),
        matrix_indices,
        aggregate_indices,
        resolved_num_threads,
        nan_value,
        resolved_aggregate_nan_value,
    )
    try:
        result_len = len(result)
    except TypeError as exc:
        raise RuntimeError(
            "RustFeaturizer.linker_pair_index_arrays_and_aggregate_stats returned an outdated "
            "aggregate contract; rebuild/install a newer s2and-rust extension."
        ) from exc
    if result_len != 6:
        raise RuntimeError(
            "RustFeaturizer.linker_pair_index_arrays_and_aggregate_stats returned an outdated "
            "aggregate contract; rebuild/install a newer s2and-rust extension."
        )
    matrix, counts, valid_counts, sums, mins, maxs = result
    return (
        np.asarray(matrix, dtype=np.float64),
        np.asarray(counts, dtype=np.uint32),
        np.asarray(valid_counts, dtype=np.uint64),
        np.asarray(sums, dtype=np.float64),
        np.asarray(mins, dtype=np.float64),
        np.asarray(maxs, dtype=np.float64),
    )


def build_linker_pair_aggregate_stats_arrays_rust(
    dataset: ANDData,
    left_signature_indices: np.ndarray,
    right_signature_indices: np.ndarray,
    row_indices: np.ndarray,
    row_count: int,
    aggregate_indices: list[int] | None = None,
    num_threads: int | None = None,
    nan_value: float = np.nan,
    runtime_context: Any | None = None,
    featurizer: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build row-level aggregate stats from numeric pair index arrays without returning pair features."""

    if featurizer is None:
        featurizer = _get_rust_featurizer(dataset, runtime_context=runtime_context)
    method = getattr(featurizer, "linker_pair_index_arrays_aggregate_stats", None)
    if not callable(method):
        raise RuntimeError(
            "RustFeaturizer.linker_pair_index_arrays_aggregate_stats is unavailable; "
            "rebuild/install a newer s2and-rust extension."
        )
    resolved_num_threads = None if num_threads is None else resolve_n_jobs(num_threads)
    counts, sums, mins, maxs = method(
        np.ascontiguousarray(left_signature_indices, dtype=np.uint32),
        np.ascontiguousarray(right_signature_indices, dtype=np.uint32),
        np.ascontiguousarray(row_indices, dtype=np.uint32),
        int(row_count),
        aggregate_indices,
        resolved_num_threads,
        nan_value,
    )
    return (
        np.asarray(counts, dtype=np.uint32),
        np.asarray(sums, dtype=np.float64),
        np.asarray(mins, dtype=np.float64),
        np.asarray(maxs, dtype=np.float64),
    )


def build_block_upper_triangle_feature_matrix_indexed_rust(
    dataset: ANDData,
    block_signature_indices: list[int],
    start_offset: int = 0,
    max_pairs: int | None = None,
    selected_indices: list[int] | None = None,
    num_threads: int | None = None,
    nan_value: float = np.nan,
    runtime_context: Any | None = None,
    featurizer: Any | None = None,
) -> np.ndarray:
    if featurizer is None:
        featurizer = _get_rust_featurizer(dataset, runtime_context=runtime_context)
    method = getattr(featurizer, "featurize_block_upper_triangle_matrix_indexed", None)
    if not callable(method):
        raise RuntimeError(
            "RustFeaturizer.featurize_block_upper_triangle_matrix_indexed is unavailable; "
            "rebuild/install a newer s2and-rust extension."
        )
    resolved_num_threads = None if num_threads is None else resolve_n_jobs(num_threads)
    matrix = method(
        block_signature_indices,
        start_offset,
        max_pairs,
        selected_indices,
        resolved_num_threads,
        nan_value,
    )
    return np.asarray(matrix, dtype=np.float64)
