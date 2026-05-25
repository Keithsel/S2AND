"""Rust retrieval-to-candidate-batch bridge for incremental linking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from s2and.incremental_linking.array_validation import as_uint16_1d, as_uint32_1d
from s2and.incremental_linking.feature_block import FeatureBlockSignatureOrder
from s2and.incremental_linking.gate_buckets import first_name_bucket_array, normalize_query_views
from s2and.incremental_linking.linker_pairwise import LinkerCandidateBatch

_REQUIRED_RUST_PAIR_PLAN_KEYS: tuple[str, ...] = ("row_orcid_match",)
RAW_CANDIDATE_PLAN_SCHEMA_VERSION = "raw_arrow_candidate_plan_v1"
RAW_CANDIDATE_PLAN_BATCH_ROW_KEYS: tuple[str, ...] = (
    "row_query_signature_indices",
    "row_component_keys",
    "retrieval_scores",
    "retrieval_ranks",
)
RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS: tuple[tuple[str, str, Any], ...] = (
    ("row_component_sizes", "cluster_size", np.float32),
    ("row_named_signature_counts", "named_signature_count", np.float32),
    ("row_dominant_first_names", "dominant_first_name", object),
    ("row_candidate_year_min", "candidate_year_min", np.int32),
    ("row_candidate_year_max", "candidate_year_max", np.int32),
    ("row_candidate_year_range_missing", "candidate_year_range_missing", np.uint8),
    ("row_query_first_tokens", "query_first_token", object),
    ("row_query_years", "query_year", np.int32),
    ("row_query_year_missing", "query_year_missing", np.uint8),
    ("row_query_has_affiliations", "query_has_affiliations", np.uint8),
    ("row_query_has_coauthors", "query_has_coauthors", np.uint8),
    ("row_orcid_match", "orcid_match", np.uint8),
    ("middle_initial_compatibility", "middle_initial_compatibility", np.float32),
    ("affiliation_overlap", "affiliation_overlap", np.float32),
    ("coauthor_overlap", "coauthor_overlap", np.float32),
    ("venue_overlap", "venue_overlap", np.float32),
    ("year_compatibility", "year_compatibility", np.float32),
    ("title_overlap", "title_overlap", np.float32),
    ("specter_centroid_similarity", "specter_centroid_similarity", np.float32),
    ("specter_exemplar_similarity", "specter_exemplar_similarity", np.float32),
    ("row_last_name_count_min_rarity", "last_name_count_min_rarity", np.float32),
    ("row_candidate_last_name_count_min_rarity", "candidate_last_name_count_min_rarity", np.float32),
    (
        "row_candidate_last_first_name_count_min_rarity",
        "candidate_last_first_name_count_min_rarity",
        np.float32,
    ),
    ("row_last_first_name_count_min_rarity", "last_first_name_count_min_rarity", np.float32),
    (
        "row_first_prefix_x_last_first_name_count_min_rarity",
        "first_prefix_x_last_first_name_count_min_rarity",
        np.float32,
    ),
    ("row_candidate_cluster_max_paper_author_count", "candidate_cluster_max_paper_author_count", np.float32),
    ("row_paper_author_list_max_jaccard", "paper_author_list_max_jaccard", np.float32),
    ("row_paper_author_list_max_containment", "paper_author_list_max_containment", np.float32),
    ("row_paper_author_list_max_overlap_count", "paper_author_list_max_overlap_count", np.float32),
    ("row_local_author_window10_jaccard_max", "local_author_window10_jaccard_max", np.float32),
    (
        "row_local_author_window10_overlap_count_max",
        "local_author_window10_overlap_count_max",
        np.float32,
    ),
    ("row_best_author_count_log_absdiff", "best_author_count_log_absdiff", np.float32),
)
RAW_CANDIDATE_PLAN_ROW_KEYS: tuple[str, ...] = (
    *RAW_CANDIDATE_PLAN_BATCH_ROW_KEYS,
    *(raw_key for raw_key, _signal_key, _dtype in RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS),
)
RAW_CANDIDATE_PLAN_PAIR_INDEX_KEYS: tuple[str, ...] = (
    "left_signature_indices",
    "right_signature_indices",
    "pair_row_indices",
)
RAW_CANDIDATE_PLAN_PAIR_ID_KEYS: tuple[str, ...] = (
    "left_signature_ids",
    "right_signature_ids",
)
RAW_CANDIDATE_PLAN_PAIR_KEYS: tuple[str, ...] = (
    *RAW_CANDIDATE_PLAN_PAIR_INDEX_KEYS,
    *RAW_CANDIDATE_PLAN_PAIR_ID_KEYS,
)


@dataclass(frozen=True)
class LinkerRetrievalBatch:
    """Retrieved candidate rows, flat pair plan, and compact row-level signals."""

    candidate_batch: LinkerCandidateBatch
    row_signals: dict[str, Any]


def _rust_retriever_object(retriever: Any) -> Any:
    return getattr(retriever, "retriever", retriever)


def _as_uint32_mapping(component_member_indices_by_key: Mapping[str, Sequence[int] | np.ndarray]) -> dict[str, Any]:
    return {
        str(component_key): as_uint32_1d(f"component_member_indices_by_key[{component_key!r}]", member_indices)
        for component_key, member_indices in component_member_indices_by_key.items()
    }


def _validate_rust_pair_plan_schema(plan: Mapping[str, Any]) -> None:
    missing = sorted(key for key in _REQUIRED_RUST_PAIR_PLAN_KEYS if key not in plan)
    if missing:
        raise RuntimeError(
            "RustHybridCentroidRetriever.top_k_hybrid_centroid_pair_plan returned a stale pair-plan schema; "
            f"missing keys={missing}. Rebuild/install the current s2and-rust extension."
        )


def _required_raw_plan_value(plan: Mapping[str, Any], key: str) -> Any:
    if key not in plan:
        raise KeyError(f"raw candidate plan is missing required key: {key}")
    return plan[key]


def _raw_plan_array(plan: Mapping[str, Any], key: str, dtype: Any, expected_length: int) -> np.ndarray:
    if dtype == np.uint32:
        values = as_uint32_1d(key, _required_raw_plan_value(plan, key))
    else:
        values = np.asarray(_required_raw_plan_value(plan, key), dtype=dtype)
    if values.ndim != 1 or len(values) != int(expected_length):
        raise ValueError(f"raw candidate plan key {key!r} must be 1D with length {expected_length}, got {values.shape}")
    return values


def _raw_plan_nonnegative_count(plan: Mapping[str, Any], key: str) -> int:
    value = int(_required_raw_plan_value(plan, key))
    if value < 0:
        raise ValueError(f"raw candidate plan {key} must be non-negative")
    return value


def _raw_plan_sequence_length(plan: Mapping[str, Any], key: str) -> int:
    try:
        return len(_required_raw_plan_value(plan, key))
    except TypeError as exc:
        raise ValueError(f"raw candidate plan key {key!r} must be a sized 1D sequence") from exc


def validate_raw_candidate_plan_schema(plan: Mapping[str, Any]) -> None:
    """Validate the raw Arrow candidate-plan payload before slicing or remapping it."""

    schema_version = _required_raw_plan_value(plan, "schema_version")
    if schema_version != RAW_CANDIDATE_PLAN_SCHEMA_VERSION:
        raise ValueError(
            "raw candidate plan schema_version must be "
            f"{RAW_CANDIDATE_PLAN_SCHEMA_VERSION!r}, got {schema_version!r}"
        )
    row_count = _raw_plan_nonnegative_count(plan, "row_count")
    pair_count = _raw_plan_nonnegative_count(plan, "pair_count")
    missing = sorted(
        key
        for key in (
            "query_signature_ids",
            "query_views",
            "query_authors",
            *RAW_CANDIDATE_PLAN_ROW_KEYS,
            *RAW_CANDIDATE_PLAN_PAIR_INDEX_KEYS,
        )
        if key not in plan
    )
    if missing:
        raise KeyError(f"raw candidate plan is missing required keys: {missing}")
    has_pair_signature_ids = all(key in plan for key in RAW_CANDIDATE_PLAN_PAIR_ID_KEYS)
    if any(key in plan for key in RAW_CANDIDATE_PLAN_PAIR_ID_KEYS) and not has_pair_signature_ids:
        raise KeyError(
            "raw candidate plan must include both left_signature_ids and right_signature_ids when either is present"
        )
    if not has_pair_signature_ids and "seed_signature_ids" not in plan:
        raise KeyError("raw candidate plan without pair signature ids must include seed_signature_ids")

    query_count = _raw_plan_sequence_length(plan, "query_signature_ids")
    if query_count == 0:
        raise ValueError("raw candidate plan query_signature_ids must be non-empty")
    for key in ("query_views", "query_authors"):
        length = _raw_plan_sequence_length(plan, key)
        if length != query_count:
            raise ValueError(
                f"raw candidate plan {key} length must match query_signature_ids: {length} != {query_count}"
            )
    for key in RAW_CANDIDATE_PLAN_ROW_KEYS:
        _raw_plan_array(plan, key, object, row_count)
    for key in RAW_CANDIDATE_PLAN_PAIR_INDEX_KEYS:
        _raw_plan_array(plan, key, np.uint32, pair_count)
    if has_pair_signature_ids:
        for key in RAW_CANDIDATE_PLAN_PAIR_ID_KEYS:
            length = _raw_plan_sequence_length(plan, key)
            if length != pair_count:
                raise ValueError(f"raw candidate plan {key} length must match pair_count: {length} != {pair_count}")


def _signature_indices_from_ids(
    signature_ids: Sequence[Any],
    signature_id_to_index: Mapping[str, int],
    *,
    field_name: str,
) -> np.ndarray:
    try:
        return as_uint32_1d(
            field_name,
            [int(signature_id_to_index[str(signature_id)]) for signature_id in signature_ids],
        )
    except KeyError as exc:
        raise KeyError(
            f"{field_name} contains signature_id not present in signature_id_to_index: {str(exc.args[0])!r}"
        ) from exc


def _signature_id_to_index_from_order(
    signature_order: FeatureBlockSignatureOrder | Sequence[Any],
) -> dict[str, int]:
    if isinstance(signature_order, FeatureBlockSignatureOrder):
        return signature_order.signature_id_to_index
    return {str(signature_id): index for index, signature_id in enumerate(signature_order)}


def build_linker_retrieval_batch_from_raw_candidate_plan(
    plan: Mapping[str, Any],
    *,
    signature_id_to_index: Mapping[str, int] | None = None,
    feature_block_signature_order: FeatureBlockSignatureOrder | Sequence[Any] | None = None,
) -> LinkerRetrievalBatch:
    """Convert a raw id-based candidate plan into the numeric linker batch contract.

    The raw Arrow API uses request-local query offsets and signature ids. The
    downstream linker runtime expects numeric indices in the current
    dataset/featurizer signature order. This bridge performs only that mapping;
    it does not rerun retrieval.
    """

    validate_raw_candidate_plan_schema(plan)
    if signature_id_to_index is None:
        if feature_block_signature_order is None:
            raise ValueError("signature_id_to_index or feature_block_signature_order is required")
        signature_id_to_index = _signature_id_to_index_from_order(feature_block_signature_order)
    elif feature_block_signature_order is not None:
        raise ValueError("Pass only one of signature_id_to_index or feature_block_signature_order")

    row_count = int(_required_raw_plan_value(plan, "row_count"))
    pair_count = int(_required_raw_plan_value(plan, "pair_count"))
    query_signature_ids = [str(value) for value in _required_raw_plan_value(plan, "query_signature_ids")]
    row_query_offsets = _raw_plan_array(plan, "row_query_signature_indices", np.uint32, row_count)
    if len(query_signature_ids) == 0:
        raise ValueError("raw candidate plan query_signature_ids must be non-empty")
    if np.any(row_query_offsets >= len(query_signature_ids)):
        raise ValueError("raw candidate plan row_query_signature_indices contains an out-of-range query offset")

    query_indices_by_offset = _signature_indices_from_ids(
        query_signature_ids,
        signature_id_to_index,
        field_name="query_signature_ids",
    )
    row_query_signature_indices = query_indices_by_offset[row_query_offsets]
    if "left_signature_ids" in plan and "right_signature_ids" in plan:
        left_signature_indices = _signature_indices_from_ids(
            [str(value) for value in _required_raw_plan_value(plan, "left_signature_ids")],
            signature_id_to_index,
            field_name="left_signature_ids",
        )
        right_signature_indices = _signature_indices_from_ids(
            [str(value) for value in _required_raw_plan_value(plan, "right_signature_ids")],
            signature_id_to_index,
            field_name="right_signature_ids",
        )
    else:
        seed_signature_ids = [str(value) for value in _required_raw_plan_value(plan, "seed_signature_ids")]
        signature_ids_by_plan_index = (*query_signature_ids, *seed_signature_ids)

        def _remap_plan_signature_indices(key: str) -> np.ndarray:
            raw_indices = _raw_plan_array(plan, key, np.uint32, pair_count)
            remapped: list[int] = []
            for raw_index in raw_indices:
                plan_index = int(raw_index)
                if plan_index >= len(signature_ids_by_plan_index):
                    raise ValueError(
                        f"raw candidate plan {key} contains signature index {plan_index} outside "
                        f"signature table length {len(signature_ids_by_plan_index)}"
                    )
                signature_id = signature_ids_by_plan_index[plan_index]
                try:
                    remapped.append(int(signature_id_to_index[signature_id]))
                except KeyError as exc:
                    raise KeyError(
                        f"{key} contains signature_id not present in signature_id_to_index: {signature_id!r}"
                    ) from exc
            return as_uint32_1d(key, remapped)

        left_signature_indices = _remap_plan_signature_indices("left_signature_indices")
        right_signature_indices = _remap_plan_signature_indices("right_signature_indices")
    if len(left_signature_indices) != pair_count or len(right_signature_indices) != pair_count:
        raise ValueError(
            "raw candidate plan pair_count does not match left/right signature id lengths: "
            f"{pair_count} != {len(left_signature_indices)} / {len(right_signature_indices)}"
        )
    pair_row_indices = _raw_plan_array(plan, "pair_row_indices", np.uint32, pair_count)
    row_component_keys = tuple(str(value) for value in _required_raw_plan_value(plan, "row_component_keys"))
    if len(row_component_keys) != row_count:
        raise ValueError(
            "raw candidate plan row_component_keys length must match row_count: "
            f"{len(row_component_keys)} != {row_count}"
        )

    retrieval_scores = _raw_plan_array(plan, "retrieval_scores", np.float32, row_count)
    retrieval_ranks = as_uint16_1d("retrieval_ranks", _required_raw_plan_value(plan, "retrieval_ranks"))
    if len(retrieval_ranks) != row_count:
        raise ValueError(
            "raw candidate plan key 'retrieval_ranks' must be 1D with length "
            f"{row_count}, got {retrieval_ranks.shape}"
        )
    candidate_batch = LinkerCandidateBatch(
        row_count=row_count,
        left_signature_indices=left_signature_indices,
        right_signature_indices=right_signature_indices,
        pair_row_indices=pair_row_indices,
        row_query_signature_indices=row_query_signature_indices,
        row_component_keys=row_component_keys,
        retrieval_scores=retrieval_scores,
        retrieval_ranks=retrieval_ranks,
    )

    raw_query_views = [str(value) for value in _required_raw_plan_value(plan, "query_views")]
    if len(raw_query_views) != len(query_signature_ids):
        raise ValueError(
            "raw candidate plan query_views length must match query_signature_ids: "
            f"{len(raw_query_views)} != {len(query_signature_ids)}"
        )
    raw_query_authors = [str(value or "") for value in _required_raw_plan_value(plan, "query_authors")]
    if len(raw_query_authors) != len(query_signature_ids):
        raise ValueError(
            "raw candidate plan query_authors length must match query_signature_ids: "
            f"{len(raw_query_authors)} != {len(query_signature_ids)}"
        )
    query_views = np.asarray(raw_query_views, dtype=object)[row_query_offsets]
    query_authors = np.asarray(raw_query_authors, dtype=object)[row_query_offsets]
    query_first_tokens = _raw_plan_array(plan, "row_query_first_tokens", object, row_count)
    row_signals: dict[str, Any] = {
        "retrieval_score": retrieval_scores,
        "retrieval_rank": retrieval_ranks,
        "candidate_component_key": np.asarray(row_component_keys, dtype=object),
        "query_view": query_views,
        "query_author": query_authors,
        "first_name_bucket": first_name_bucket_array(query_first_tokens, query_views),
    }
    for raw_key, signal_key, dtype in RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS:
        row_signals[signal_key] = _raw_plan_array(plan, raw_key, dtype, row_count)
    return LinkerRetrievalBatch(candidate_batch=candidate_batch, row_signals=row_signals)


def build_linker_retrieval_batch_rust(
    *,
    retriever: Any,
    queries: Sequence[Any],
    query_signature_indices: Sequence[int] | np.ndarray,
    query_signature_ids: Sequence[str] | None = None,
    component_member_indices_by_key: Mapping[str, Sequence[int] | np.ndarray],
    top_k: int,
    query_view: str | Sequence[str],
    n_jobs: int | None = None,
    retrieval_subblock_index: Mapping[str, Any] | None = None,
    query_candidate_component_keys_by_signature_id: Mapping[str, Sequence[str]] | None = None,
    full_first_global_backfill_count: int = 5,
) -> LinkerRetrievalBatch:
    """Retrieve candidates in Rust and return the shared numeric candidate-batch contract."""

    normalized_query_views = normalize_query_views(query_view, len(queries))
    rust_retriever = _rust_retriever_object(retriever)
    method = getattr(rust_retriever, "top_k_hybrid_centroid_pair_plan", None)
    if method is None:
        raise RuntimeError("RustHybridCentroidRetriever.top_k_hybrid_centroid_pair_plan is unavailable")
    if retrieval_subblock_index is not None or query_candidate_component_keys_by_signature_id is not None:
        if query_signature_ids is None:
            raise ValueError(
                "query_signature_ids are required when retrieval_subblock_index or query candidate keys are provided"
            )
        if len(query_signature_ids) != len(queries):
            raise ValueError(
                "queries and query_signature_ids must have equal length: "
                f"{len(queries)} != {len(query_signature_ids)}"
            )
        plan = method(
            list(queries),
            as_uint32_1d("query_signature_indices", query_signature_indices),
            _as_uint32_mapping(component_member_indices_by_key),
            int(top_k),
            None if n_jobs is None else int(n_jobs),
            [str(value) for value in query_signature_ids],
            None if retrieval_subblock_index is None else dict(retrieval_subblock_index),
            (
                None
                if query_candidate_component_keys_by_signature_id is None
                else {
                    str(query_signature_id): [str(component_key) for component_key in component_keys]
                    for query_signature_id, component_keys in query_candidate_component_keys_by_signature_id.items()
                }
            ),
            int(full_first_global_backfill_count),
        )
    else:
        plan = method(
            list(queries),
            as_uint32_1d("query_signature_indices", query_signature_indices),
            _as_uint32_mapping(component_member_indices_by_key),
            int(top_k),
            None if n_jobs is None else int(n_jobs),
        )
    _validate_rust_pair_plan_schema(plan)
    row_count = int(plan["row_count"])
    candidate_batch = LinkerCandidateBatch(
        row_count=row_count,
        left_signature_indices=as_uint32_1d("left_signature_indices", plan["left_signature_indices"]),
        right_signature_indices=as_uint32_1d("right_signature_indices", plan["right_signature_indices"]),
        pair_row_indices=as_uint32_1d("pair_row_indices", plan["pair_row_indices"]),
        row_query_signature_indices=as_uint32_1d(
            "row_query_signature_indices",
            plan["row_query_signature_indices"],
        ),
        row_component_keys=tuple(str(value) for value in plan["row_component_keys"]),
        retrieval_scores=np.asarray(plan["retrieval_scores"], dtype=np.float32),
        retrieval_ranks=np.asarray(plan["retrieval_ranks"], dtype=np.uint16),
    )
    if isinstance(normalized_query_views, str):
        query_views: Any = np.full(row_count, normalized_query_views, dtype=object)
    else:
        row_query_signature_indices = candidate_batch.row_query_signature_indices
        if row_query_signature_indices is None:
            raise RuntimeError("Rust retrieval plan did not provide row_query_signature_indices")
        query_view_by_query_index = {
            int(query_index): str(current_query_view)
            for query_index, current_query_view in zip(query_signature_indices, normalized_query_views, strict=True)
        }
        query_views = np.asarray(
            [query_view_by_query_index[int(query_index)] for query_index in row_query_signature_indices],
            dtype=object,
        )
    query_first_tokens = np.asarray(plan["row_query_first_tokens"], dtype=object)
    row_signals: dict[str, Any] = {
        "retrieval_score": candidate_batch.retrieval_scores,
        "retrieval_rank": candidate_batch.retrieval_ranks,
        "candidate_component_key": np.asarray(candidate_batch.row_component_keys, dtype=object),
        "query_view": query_views,
        "cluster_size": np.asarray(plan["row_component_sizes"], dtype=np.float32),
        "named_signature_count": np.asarray(plan["row_named_signature_counts"], dtype=np.float32),
        "dominant_first_name": np.asarray(plan["row_dominant_first_names"], dtype=object),
        "candidate_year_min": np.asarray(plan["row_candidate_year_min"], dtype=np.int32),
        "candidate_year_max": np.asarray(plan["row_candidate_year_max"], dtype=np.int32),
        "candidate_year_range_missing": np.asarray(plan["row_candidate_year_range_missing"], dtype=np.uint8),
        "query_first_token": query_first_tokens,
        "first_name_bucket": first_name_bucket_array(query_first_tokens, query_views),
        "query_year": np.asarray(plan["row_query_years"], dtype=np.int32),
        "query_year_missing": np.asarray(plan["row_query_year_missing"], dtype=np.uint8),
        "query_has_affiliations": np.asarray(plan["row_query_has_affiliations"], dtype=np.uint8),
        "query_has_coauthors": np.asarray(plan["row_query_has_coauthors"], dtype=np.uint8),
        "orcid_match": np.asarray(plan["row_orcid_match"], dtype=np.uint8),
        "middle_initial_compatibility": np.asarray(plan["middle_initial_compatibility"], dtype=np.float32),
        "affiliation_overlap": np.asarray(plan["affiliation_overlap"], dtype=np.float32),
        "coauthor_overlap": np.asarray(plan["coauthor_overlap"], dtype=np.float32),
        "venue_overlap": np.asarray(plan["venue_overlap"], dtype=np.float32),
        "year_compatibility": np.asarray(plan["year_compatibility"], dtype=np.float32),
        "title_overlap": np.asarray(plan["title_overlap"], dtype=np.float32),
        "specter_centroid_similarity": np.asarray(plan["specter_centroid_similarity"], dtype=np.float32),
        "specter_exemplar_similarity": np.asarray(plan["specter_exemplar_similarity"], dtype=np.float32),
    }
    return LinkerRetrievalBatch(candidate_batch=candidate_batch, row_signals=row_signals)
