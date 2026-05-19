"""Rust retrieval-to-candidate-batch bridge for incremental linking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from s2and.incremental_linking.gate_buckets import first_name_bucket_array, normalize_query_views
from s2and.incremental_linking.linker_pairwise import LinkerCandidateBatch

_REQUIRED_RUST_PAIR_PLAN_KEYS: tuple[str, ...] = ("row_orcid_match",)


@dataclass(frozen=True)
class LinkerRetrievalBatch:
    """Retrieved candidate rows, flat pair plan, and compact row-level signals."""

    candidate_batch: LinkerCandidateBatch
    row_signals: dict[str, Any]


def _rust_retriever_object(retriever: Any) -> Any:
    return getattr(retriever, "retriever", retriever)


def _as_uint32_mapping(component_member_indices_by_key: Mapping[str, Sequence[int] | np.ndarray]) -> dict[str, Any]:
    return {
        str(component_key): np.ascontiguousarray(member_indices, dtype=np.uint32)
        for component_key, member_indices in component_member_indices_by_key.items()
    }


def _validate_rust_pair_plan_schema(plan: Mapping[str, Any]) -> None:
    missing = sorted(key for key in _REQUIRED_RUST_PAIR_PLAN_KEYS if key not in plan)
    if missing:
        raise RuntimeError(
            "RustHybridCentroidRetriever.top_k_hybrid_centroid_pair_plan returned a stale pair-plan schema; "
            f"missing keys={missing}. Rebuild/install the current s2and-rust extension."
        )


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
            np.ascontiguousarray(query_signature_indices, dtype=np.uint32),
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
            np.ascontiguousarray(query_signature_indices, dtype=np.uint32),
            _as_uint32_mapping(component_member_indices_by_key),
            int(top_k),
            None if n_jobs is None else int(n_jobs),
        )
    _validate_rust_pair_plan_schema(plan)
    row_count = int(plan["row_count"])
    candidate_batch = LinkerCandidateBatch(
        row_count=row_count,
        left_signature_indices=np.asarray(plan["left_signature_indices"], dtype=np.uint32),
        right_signature_indices=np.asarray(plan["right_signature_indices"], dtype=np.uint32),
        pair_row_indices=np.asarray(plan["pair_row_indices"], dtype=np.uint32),
        row_query_signature_indices=np.asarray(plan["row_query_signature_indices"], dtype=np.uint32),
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
        "query_has_affiliations": np.asarray(plan["row_query_has_affiliations"], dtype=np.float32),
        "query_has_coauthors": np.asarray(plan["row_query_has_coauthors"], dtype=np.float32),
        "orcid_match": np.asarray(plan["row_orcid_match"], dtype=np.float32),
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
