"""Classic train/calibrate/eval helpers for promoted incremental linker training."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from s2and.incremental_linking.artifact import save_incremental_linking_artifact
from s2and.incremental_linking.contracts import INCREMENTAL_LINKING_RUST_CAPABILITIES
from s2and.incremental_linking.features import PROMOTED_NON_PAIRWISE_FEATURE_COLUMNS
from s2and.incremental_linking.gate_buckets import first_name_bucket_from_token_view, normalize_bucket_letters
from s2and.incremental_linking.linker_pairwise import promoted_pairwise_aggregate_columns
from s2and.incremental_linking_training.query_support import (
    FROZEN_BEST_RUST_HYBRID_CENTROID_POLICY,
    FROZEN_BEST_RUST_HYBRID_CENTROID_POLICY_NAME,
)
from s2and.thread_config import resolve_n_jobs

DEFAULT_CLASSIC_N_JOBS = 20

# Shared training, calibration, and evaluation helpers for the official replay
# target. CLI entrypoints should import this module instead of owning copies.
_ANCHOR_EVIDENCE_FEATURE_COLUMNS = (
    "anchor_evidence_count",
    "strong_positive_anchor_score",
    "weak_residual_anchor_score",
    "sparse_relative_winner_score",
)
_DERIVED_PROMOTED_FEATURE_COLUMNS = (
    "retrieval_reciprocal_rank",
    "cluster_size_log",
    "candidate_year_span",
    "year_gap_to_candidate_range",
    "year_gap_signed_to_candidate_range",
    "candidate_dominant_first_name_length",
    "query_first_prefix_match_any_length",
    "same_dominant_first_as_best_top5",
    "same_family_as_heuristic_choice",
)
_ANCHOR_EVIDENCE_PREREQUISITES = (
    "min_distance",
    "retrieval_score_gap_vs_best_competitor",
    "same_family_as_top1",
    "retrieval_rank",
)
_CLASSIC_DERIVABLE_FEATURE_PREREQUISITES: dict[str, tuple[str, ...]] = {
    "retrieval_reciprocal_rank": ("retrieval_rank",),
    "cluster_size_log": ("cluster_size",),
    "candidate_year_span": ("candidate_year_min", "candidate_year_max", "candidate_year_range_missing"),
    "year_gap_to_candidate_range": (
        "candidate_year_min",
        "candidate_year_max",
        "candidate_year_range_missing",
        "query_year",
        "query_year_missing",
    ),
    "year_gap_signed_to_candidate_range": (
        "candidate_year_min",
        "candidate_year_max",
        "candidate_year_range_missing",
        "query_year",
        "query_year_missing",
    ),
    "candidate_dominant_first_name_length": ("dominant_first_name",),
    "query_first_prefix_match_any_length": ("dominant_first_name",),
    "same_dominant_first_as_best_top5": (
        "query_group_id",
        "dominant_first_name",
        "retrieval_rank",
        "top5_mean_distance",
        "candidate_component_key",
    ),
    "same_family_as_heuristic_choice": (
        "query_group_id",
        "dominant_first_name",
        "retrieval_rank",
        "top5_mean_distance",
        "candidate_component_key",
        "retrieval_score",
    ),
    **{feature: _ANCHOR_EVIDENCE_PREREQUISITES for feature in _ANCHOR_EVIDENCE_FEATURE_COLUMNS},
}


@dataclass(frozen=True)
class OfficialBundle:
    """Single-file metadata and paths for the official stack."""

    root: Path
    bundle_name: str
    assets: dict[str, Any]
    models: dict[str, Any]
    expected_metrics: dict[str, Any]


@dataclass(frozen=True)
class TotalErrorGateSpec:
    """Bucketed score/margin gate selected by total query errors."""

    name: str
    score_thresholds: dict[str, float]
    margin_thresholds: dict[str, float]


_TOTAL_ERROR_SCORE_BUCKETS = (
    "multi_candidate|multi_letter_first",
    "multi_candidate|single_letter_first",
    "single_candidate|multi_letter_first",
    "single_candidate|single_letter_first",
)
_TOTAL_ERROR_MARGIN_BUCKETS = (
    "multi_candidate|multi_letter_first",
    "multi_candidate|single_letter_first",
)
_DEFAULT_PROMOTED_GATE_FIXED_GRID_STEP = 0.01
_DEFAULT_PROMOTED_GATE_ERROR_WEIGHTS = {
    "false_abstain": 0.25,
    "false_link": 1.0,
    "wrong_candidate_link": 1.5,
}
CALIBRATION_DATASET_SOURCE_KEY_BY_DATASET = {
    "a_khan": "a_khan_eval",
    "a_silva": "a_silva_eval",
    "h_wang": "hwang_eval",
    "j_smith": "j_smith_eval",
    "s_gupta": "s_gupta_eval",
    "s_lee": "s_lee_eval",
    "s_park": "s_park_eval",
}


def load_bundle(root: Path) -> OfficialBundle:
    """Load bundle metadata from an explicit bundle root."""

    root = root.resolve()
    payload = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    return OfficialBundle(
        root=root,
        bundle_name=str(payload["bundle_name"]),
        assets=dict(payload["assets"]),
        models=dict(payload["models"]),
        expected_metrics=dict(payload["expected_metrics"]),
    )


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if path.suffix == ".parquet":
        parquet_kwargs: dict[str, Any] = {}
        if "usecols" in kwargs:
            parquet_kwargs["columns"] = kwargs["usecols"]
        return pd.read_parquet(path, **parquet_kwargs)
    defaults = {"low_memory": False}
    defaults.update(kwargs)
    read_csv = cast(Any, pd.read_csv)
    return read_csv(path, **defaults)


def _resolve_path(bundle: OfficialBundle, path_like: str | Path) -> Path:
    """Resolve a stored bundle path relative to the bundle root."""

    path = Path(path_like)
    if path.is_absolute():
        resolved = path.resolve()
        try:
            resolved.relative_to(bundle.root.resolve())
        except ValueError as exc:
            raise ValueError(f"Bundle asset path escapes bundle root: {path_like}") from exc
    else:
        resolved = (bundle.root / path).resolve()
        try:
            resolved.relative_to(bundle.root.resolve())
        except ValueError as exc:
            raise ValueError(f"Bundle asset path escapes bundle root: {path_like}") from exc
    if not resolved.exists():
        raise FileNotFoundError(f"Bundle asset does not exist: {resolved}")
    return resolved


def _normalize_dataset_slug(value: Any) -> str:
    """Normalize a dataset-like identifier into a stable lowercase slug."""

    normalized = re.sub(r"[^0-9a-z_]+", "_", str(value).strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        raise ValueError(f"Cannot derive dataset slug from {value!r}")
    return normalized


def _summary_key_for_eval_dataset(dataset_name: Any) -> str:
    """Return the runtime summary key for one eval dataset."""

    return f"overall_{_normalize_dataset_slug(dataset_name)}_eval"


def _iter_extra_eval_paths(spec: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Return all optional eval datasets configured for the classic bundle."""

    eval_paths: list[tuple[str, str]] = []
    seen_datasets: set[str] = set()
    for path_key, dataset_name in (
        ("s_park_eval_path", "s_park"),
        ("s_lee_eval_path", "s_lee"),
    ):
        if path_key not in spec:
            continue
        normalized_dataset = _normalize_dataset_slug(dataset_name)
        eval_paths.append((normalized_dataset, str(spec[path_key])))
        seen_datasets.add(normalized_dataset)

    extra_eval_paths = spec.get("extra_eval_paths", {})
    if extra_eval_paths is None:
        return tuple(eval_paths)
    if not isinstance(extra_eval_paths, dict):
        raise ValueError("classic.extra_eval_paths must be a mapping of dataset slug to relative path")
    for dataset_name, path_like in extra_eval_paths.items():
        normalized_dataset = _normalize_dataset_slug(dataset_name)
        if normalized_dataset in seen_datasets:
            raise ValueError(f"Duplicate extra eval dataset configured: {normalized_dataset}")
        eval_paths.append((normalized_dataset, str(path_like)))
        seen_datasets.add(normalized_dataset)
    return tuple(eval_paths)


def _iter_classic_train_holdout_paths(spec: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Return calibration/eval row files whose identities must stay out of classic training."""

    holdout_paths: list[tuple[str, str]] = []
    seen_paths: set[str] = set()
    for path_key, source_name in (
        ("classic_gate_source_path", "classic_gate_source"),
        ("s2and_eval_path", "s2and"),
        ("hwang_eval_path", "hwang"),
    ):
        path_like = spec.get(path_key)
        if path_like is None:
            continue
        normalized_path = str(path_like)
        if normalized_path in seen_paths:
            continue
        holdout_paths.append((source_name, normalized_path))
        seen_paths.add(normalized_path)

    for source_name, path_like in _iter_extra_eval_paths(spec):
        normalized_path = str(path_like)
        if normalized_path in seen_paths:
            continue
        holdout_paths.append((source_name, normalized_path))
        seen_paths.add(normalized_path)
    return tuple(holdout_paths)


def _nonempty_string_values(series: pd.Series) -> set[str]:
    """Return nonempty string values from one identity column."""

    values: set[str] = set()
    for value in series.dropna():
        text = str(value).strip()
        if text:
            values.add(text)
    return values


def _read_classic_holdout_identity_sets(
    bundle: OfficialBundle,
    spec: dict[str, Any],
) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    """Load query/base identity sets from calibration and eval row files."""

    query_group_ids: set[str] = set()
    base_group_ids: set[str] = set()
    source_summaries: list[dict[str, Any]] = []
    for source_name, path_like in _iter_classic_train_holdout_paths(spec):
        path = _resolve_path(bundle, path_like)
        header = _read_csv(path, nrows=0).columns
        identity_columns = [column for column in ("query_group_id", "base_group_id") if column in header]
        if not identity_columns:
            source_summaries.append(
                {
                    "source": source_name,
                    "path": str(path.relative_to(bundle.root)),
                    "query_groups": 0,
                    "base_groups": 0,
                }
            )
            continue
        rows = _read_csv(path, usecols=identity_columns)
        source_query_group_ids = (
            _nonempty_string_values(rows["query_group_id"]) if "query_group_id" in rows.columns else set()
        )
        source_base_group_ids = (
            _nonempty_string_values(rows["base_group_id"]) if "base_group_id" in rows.columns else set()
        )
        query_group_ids.update(source_query_group_ids)
        base_group_ids.update(source_base_group_ids)
        source_summaries.append(
            {
                "source": source_name,
                "path": str(path.relative_to(bundle.root)),
                "query_groups": int(len(source_query_group_ids)),
                "base_groups": int(len(source_base_group_ids)),
            }
        )
    return query_group_ids, base_group_ids, source_summaries


def _apply_classic_train_holdout_filter(
    train_df: pd.DataFrame,
    *,
    holdout_query_group_ids: set[str],
    holdout_base_group_ids: set[str],
    holdout_sources: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop training rows whose query/base identities appear in calibration or eval."""

    rows_before = int(len(train_df))
    queries_before = int(train_df["query_group_id"].astype(str).nunique()) if "query_group_id" in train_df else 0
    labels = pd.to_numeric(train_df["label"], errors="coerce").fillna(0) if "label" in train_df else pd.Series()
    positive_rows_before = int(labels.sum()) if not labels.empty else 0
    positive_queries_before = (
        int(train_df.loc[labels == 1, "query_group_id"].astype(str).nunique())
        if "query_group_id" in train_df and not labels.empty
        else 0
    )
    query_overlap_mask = (
        train_df["query_group_id"].astype(str).isin(holdout_query_group_ids)
        if "query_group_id" in train_df and holdout_query_group_ids
        else pd.Series(False, index=train_df.index)
    )
    base_overlap_mask = (
        train_df["base_group_id"].astype(str).isin(holdout_base_group_ids)
        if "base_group_id" in train_df and holdout_base_group_ids
        else pd.Series(False, index=train_df.index)
    )
    remove_mask = query_overlap_mask | base_overlap_mask
    removed = train_df[remove_mask].copy()
    filtered = train_df[~remove_mask].copy()

    removed_labels = pd.to_numeric(removed["label"], errors="coerce").fillna(0) if "label" in removed else pd.Series()
    filtered_labels = (
        pd.to_numeric(filtered["label"], errors="coerce").fillna(0) if "label" in filtered else pd.Series()
    )
    summary = {
        "rows_before": rows_before,
        "rows_after": int(len(filtered)),
        "rows_removed": int(len(removed)),
        "queries_before": queries_before,
        "queries_after": int(filtered["query_group_id"].astype(str).nunique()) if "query_group_id" in filtered else 0,
        "queries_removed": int(removed["query_group_id"].astype(str).nunique()) if "query_group_id" in removed else 0,
        "positive_rows_before": positive_rows_before,
        "positive_rows_after": int(filtered_labels.sum()) if not filtered_labels.empty else 0,
        "positive_rows_removed": int(removed_labels.sum()) if not removed_labels.empty else 0,
        "positive_queries_before": positive_queries_before,
        "positive_queries_after": int(filtered.loc[filtered_labels == 1, "query_group_id"].astype(str).nunique())
        if "query_group_id" in filtered and not filtered_labels.empty
        else 0,
        "positive_queries_removed": int(removed.loc[removed_labels == 1, "query_group_id"].astype(str).nunique())
        if "query_group_id" in removed and not removed_labels.empty
        else 0,
        "overlapping_query_groups": int(train_df.loc[query_overlap_mask, "query_group_id"].astype(str).nunique())
        if "query_group_id" in train_df
        else 0,
        "overlapping_base_groups": int(train_df.loc[base_overlap_mask, "base_group_id"].astype(str).nunique())
        if "base_group_id" in train_df
        else 0,
        "holdout_query_groups": int(len(holdout_query_group_ids)),
        "holdout_base_groups": int(len(holdout_base_group_ids)),
        "holdout_sources": list(holdout_sources or []),
    }
    return filtered, summary


def _bounded_threshold_grid(values: np.ndarray, grid_size: int) -> np.ndarray:
    """Build a bounded quantile grid with inclusive edge thresholds."""

    cleaned = np.asarray(values, dtype=np.float64)
    if cleaned.size == 0:
        return np.array([0.0], dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, num=max(int(grid_size), 2), dtype=np.float64)
    thresholds = np.unique(np.quantile(cleaned, quantiles))
    epsilon = 1e-12
    return np.unique(
        np.concatenate(
            (
                np.array([float(cleaned.min()) - epsilon], dtype=np.float64),
                thresholds.astype(np.float64, copy=False),
                np.array([float(cleaned.max()) + epsilon], dtype=np.float64),
            )
        )
    )


def _normalize_letters(value: Any) -> str:
    """Normalize a name-like token down to lowercase letters."""

    return re.sub(r"[^a-z]", "", str(value).lower())


def _is_missing_scalar(value: Any) -> bool:
    """Return whether a scalar-like value is pandas-missing."""

    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, bool | np.bool_):
        return bool(missing)
    return False


def _normalize_optional_letters(value: Any) -> str:
    """Normalize a name token, treating missing values as absent."""

    if _is_missing_scalar(value):
        return ""
    return _normalize_letters(value)


def _cluster_size_log(cluster_size: Any) -> float:
    """Return an uncapped log-size primitive."""

    return float(math.log1p(max(0.0, float(cluster_size or 0.0))))


def _numeric_feature_series(df: pd.DataFrame, column: str, *, default: float = 0.0) -> pd.Series:
    """Return one numeric feature column with a deterministic default for formula derivation."""

    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=np.float32)
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(np.float32)


def _query_first_series_for_prefix(out: pd.DataFrame) -> list[str]:
    if "query_author" in out.columns:
        query_first_from_author = out["query_author"].map(_query_first_token)
    else:
        query_first_from_author = pd.Series([""] * len(out), index=out.index, dtype="string")
    if "query_first_token" in out.columns:
        query_first_from_token = out["query_first_token"].map(_normalize_optional_letters)
    else:
        query_first_from_token = pd.Series([""] * len(out), index=out.index, dtype="string")
    return [
        author_token if author_token else token
        for author_token, token in zip(query_first_from_author, query_first_from_token, strict=True)
    ]


def _derive_promoted_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive promoted row features from portable candidate-row primitives."""

    out = df.copy()
    if "retrieval_rank" in out.columns:
        retrieval_rank = pd.to_numeric(out["retrieval_rank"], errors="coerce").fillna(0.0).astype(np.float32)
        out["retrieval_reciprocal_rank"] = 1.0 / np.maximum(retrieval_rank.to_numpy(dtype=np.float32), 1.0)
    if "cluster_size" in out.columns:
        cluster_size = pd.to_numeric(out["cluster_size"], errors="coerce").fillna(0.0).astype(np.float32)
        out["cluster_size_log"] = cluster_size.map(_cluster_size_log)
    if {"candidate_year_min", "candidate_year_max", "candidate_year_range_missing"}.issubset(out.columns):
        candidate_year_min = pd.to_numeric(out["candidate_year_min"], errors="coerce").fillna(0.0).astype(np.float32)
        candidate_year_max = pd.to_numeric(out["candidate_year_max"], errors="coerce").fillna(0.0).astype(np.float32)
        candidate_missing = (
            pd.to_numeric(out["candidate_year_range_missing"], errors="coerce").fillna(1.0).astype(np.float32) > 0.0
        )
        out["candidate_year_span"] = np.where(
            candidate_missing,
            0.0,
            np.maximum(
                candidate_year_max.to_numpy(dtype=np.float32) - candidate_year_min.to_numpy(dtype=np.float32),
                0.0,
            ),
        )
        if {"query_year", "query_year_missing"}.issubset(out.columns):
            query_year = pd.to_numeric(out["query_year"], errors="coerce").fillna(0.0).astype(np.float32)
            query_missing = (
                pd.to_numeric(out["query_year_missing"], errors="coerce").fillna(1.0).astype(np.float32) > 0.0
            )
            observed = ~(candidate_missing | query_missing)
            lower = observed & (query_year < candidate_year_min)
            upper = observed & (query_year > candidate_year_max)
            gap = np.zeros(len(out), dtype=np.float32)
            signed_gap = np.zeros(len(out), dtype=np.float32)
            lower_gap = (candidate_year_min[lower] - query_year[lower]).to_numpy(dtype=np.float32)
            upper_gap = (query_year[upper] - candidate_year_max[upper]).to_numpy(dtype=np.float32)
            gap[lower.to_numpy(dtype=bool)] = lower_gap
            gap[upper.to_numpy(dtype=bool)] = upper_gap
            signed_gap[lower.to_numpy(dtype=bool)] = -lower_gap
            signed_gap[upper.to_numpy(dtype=bool)] = upper_gap
            out["year_gap_to_candidate_range"] = gap
            out["year_gap_signed_to_candidate_range"] = signed_gap
    if "dominant_first_name" in out.columns:
        query_first = _query_first_series_for_prefix(out)
        dominant_first = out["dominant_first_name"].map(_normalize_optional_letters)
        out["candidate_dominant_first_name_length"] = [float(len(value)) for value in dominant_first]
        out["query_first_prefix_match_any_length"] = [
            1.0 if query and dominant and (query.startswith(dominant) or dominant.startswith(query)) else 0.0
            for query, dominant in zip(query_first, dominant_first, strict=True)
        ]
        if {
            "query_group_id",
            "retrieval_rank",
            "top5_mean_distance",
            "candidate_component_key",
        }.issubset(out.columns):
            group_key = out["query_group_id"].astype(str)
            retrieval_rank_numeric = pd.to_numeric(out["retrieval_rank"], errors="coerce")
            retrieval_score_sort = (
                -pd.to_numeric(out["retrieval_score"], errors="coerce")
                if "retrieval_score" in out.columns
                else retrieval_rank_numeric
            )
            grouping_frame = out.assign(
                _query_group_key=group_key,
                _dominant_first_alpha=dominant_first,
                _retrieval_rank_numeric=retrieval_rank_numeric,
                _retrieval_score_sort=retrieval_score_sort,
                _top5_mean_distance_numeric=pd.to_numeric(out["top5_mean_distance"], errors="coerce"),
                _row_order=np.arange(len(out)),
            )
            top1_rows = grouping_frame.sort_values(
                [
                    "_query_group_key",
                    "_retrieval_score_sort",
                    "_retrieval_rank_numeric",
                    "candidate_component_key",
                    "_row_order",
                ],
                kind="stable",
            )
            top1_by_group = top1_rows.drop_duplicates("_query_group_key").set_index("_query_group_key")
            top1_dominant = group_key.map(top1_by_group["_dominant_first_alpha"])
            best_top5_rows = grouping_frame.sort_values(
                [
                    "_query_group_key",
                    "_top5_mean_distance_numeric",
                    "_retrieval_score_sort",
                    "_retrieval_rank_numeric",
                    "candidate_component_key",
                    "_row_order",
                ],
                kind="stable",
            )
            best_top5_by_group = best_top5_rows.drop_duplicates("_query_group_key").set_index("_query_group_key")
            best_top5_dominant = group_key.map(best_top5_by_group["_dominant_first_alpha"])
            dominant_first_top1_match = np.asarray(
                [
                    1.0 if dominant and top1 and dominant == top1 else 0.0
                    for dominant, top1 in zip(dominant_first, top1_dominant, strict=True)
                ],
                dtype=np.float32,
            )
            same_dominant_first_as_best_top5 = np.asarray(
                [
                    1.0 if dominant and best and dominant == best else 0.0
                    for dominant, best in zip(dominant_first, best_top5_dominant, strict=True)
                ],
                dtype=np.float32,
            )
            out["same_dominant_first_as_best_top5"] = same_dominant_first_as_best_top5
            out["same_family_as_heuristic_choice"] = (
                dominant_first_top1_match
                * pd.to_numeric(out["retrieval_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                + same_dominant_first_as_best_top5
                * (
                    1.0
                    - pd.to_numeric(out["top5_mean_distance"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                )
            ).astype(np.float32)
    return out


def _derive_anchor_evidence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive anchor-evidence formulas from existing candidate-row evidence columns."""

    out = df.copy()
    if {
        "query_group_id",
        "retrieval_score",
        "retrieval_rank",
        "candidate_component_key",
    }.issubset(out.columns):
        retrieval_score_values = pd.to_numeric(out["retrieval_score"], errors="coerce").astype(np.float32)
        stored_rank_values = pd.to_numeric(out["retrieval_rank"], errors="coerce").fillna(99.0).astype(np.float32)
        current_rank = np.zeros(len(out), dtype=np.float32)
        current_gap = np.zeros(len(out), dtype=np.float32)
        best_gap = np.zeros(len(out), dtype=np.float32)
        ordering_frame = pd.DataFrame(
            {
                "_query_group_key": out["query_group_id"].astype(str),
                "_score": retrieval_score_values,
                "_stored_rank": stored_rank_values,
                "_component_key": out["candidate_component_key"].astype(str),
                "_row_index": np.arange(len(out)),
            }
        )
        for _group_key, group in ordering_frame.groupby("_query_group_key", sort=False):
            ordered = group.sort_values(
                ["_score", "_stored_rank", "_component_key", "_row_index"],
                ascending=[False, True, True, True],
                kind="stable",
            )
            indices = ordered["_row_index"].to_numpy(dtype=np.int64)
            if len(indices) == 0:
                continue
            scores = retrieval_score_values.iloc[indices].to_numpy(dtype=np.float32)
            top1 = int(indices[0])
            runner_up = int(indices[1]) if len(indices) > 1 else top1
            best_score = float(np.max(scores))
            for rank, row_index in enumerate(indices, start=1):
                competitor = runner_up if int(row_index) == top1 else top1
                current_rank[int(row_index)] = float(rank)
                current_gap[int(row_index)] = float(
                    retrieval_score_values.iloc[int(row_index)] - retrieval_score_values.iloc[int(competitor)]
                )
                best_gap[int(row_index)] = float(best_score - retrieval_score_values.iloc[int(row_index)])
        out["retrieval_rank"] = current_rank
        out["retrieval_score_gap_vs_best_competitor"] = np.round(current_gap, 6).astype(np.float32)
        out["retrieval_score_best_gap"] = np.round(best_gap, 6).astype(np.float32)

    min_distance = _numeric_feature_series(out, "min_distance", default=10000.0)
    retrieval_gap = _numeric_feature_series(out, "retrieval_score_gap_vs_best_competitor")
    same_top1 = _numeric_feature_series(out, "same_family_as_top1")
    retrieval_rank = _numeric_feature_series(out, "retrieval_rank", default=99.0)

    min_distance_values = min_distance.to_numpy(dtype=np.float32, copy=False)
    retrieval_gap_values = retrieval_gap.to_numpy(dtype=np.float32, copy=False)
    same_top1_values = same_top1.to_numpy(dtype=np.float32, copy=False)
    retrieval_rank_values = retrieval_rank.to_numpy(dtype=np.float32, copy=False)

    min_distance_clip = np.clip(min_distance_values, 0.0, 1.0)
    same_top1_clip = np.clip(same_top1_values, 0.0, 1.0)
    retrieval_gap_positive = np.clip(retrieval_gap_values, 0.0, 0.3) / 0.3
    retrieval_gap_normalized = np.clip((np.clip(retrieval_gap_values, -0.2, 0.3) + 0.2) / 0.5, 0.0, 1.0)

    out["anchor_evidence_count"] = (
        (min_distance_values <= 0.15).astype(np.float32) + (retrieval_gap_values >= 0.02).astype(np.float32)
    ).astype(np.float32)

    distance_signal = 1.0 - min_distance_clip
    support_strength = 0.20 * distance_signal
    out["strong_positive_anchor_score"] = (np.clip(support_strength, 0.0, 1.0) * (0.5 + 0.5 * same_top1_clip)).astype(
        np.float32
    )

    residual_support = 0.28 * distance_signal + 0.08 * retrieval_gap_normalized
    out["weak_residual_anchor_score"] = (same_top1_clip * np.clip(residual_support, 0.0, 1.0)).astype(np.float32)

    out["sparse_relative_winner_score"] = (
        (retrieval_rank_values <= 1.0).astype(np.float32)
        * same_top1_clip
        * np.clip(retrieval_gap_positive, 0.0, 1.0)
        * np.clip(residual_support, 0.0, 1.0)
    ).astype(np.float32)
    return out


def _query_first_token(author: Any) -> str:
    """Return the first alphabetic token from a query author string."""

    if _is_missing_scalar(author):
        return ""
    tokens = re.findall(r"[A-Za-z]+", str(author))
    return tokens[0].lower() if tokens else ""


def _classic_gate_first_name_bucket(row: pd.Series | dict[str, Any]) -> str:
    """Classify a query into the gate's first-name-length bucket."""

    token = normalize_bucket_letters(row.get("query_first_token", ""))
    query_author = row.get("query_author")
    if not token and query_author is not None and pd.notna(query_author) and str(query_author).strip():
        token = _query_first_token(row.get("query_author"))
    return first_name_bucket_from_token_view(token, row.get("query_view", ""))


def _promoted_stratified_gate_spec(spec: dict[str, Any]) -> dict[str, Any] | None:
    """Return the configured promoted stratified gate calibration, if active."""

    configured = spec.get("promoted_stratified_gate")
    if configured is None:
        return None
    if not isinstance(configured, dict):
        raise ValueError("classic.promoted_stratified_gate must be a mapping when provided")
    if str(configured.get("mode")) != "full_calibration_fixed_grid_4score_2margin":
        raise ValueError("classic.promoted_stratified_gate.mode must be full_calibration_fixed_grid_4score_2margin")
    out = dict(configured)
    split_spec = spec.get("stratified_eval_test_split")
    split_spec = split_spec if isinstance(split_spec, dict) else {}
    calibration_splits = out.get("calibration_splits")
    if calibration_splits is None:
        calibration_splits = [
            split_spec.get("calibration_fit_split", "calibration_fit"),
            split_spec.get("calibration_check_split", "calibration_check"),
        ]
    if isinstance(calibration_splits, str) or not isinstance(calibration_splits, Sequence):
        raise ValueError("classic.promoted_stratified_gate.calibration_splits must be a sequence of split names")
    out["calibration_splits"] = [str(split) for split in calibration_splits]
    if not out["calibration_splits"]:
        raise ValueError("classic.promoted_stratified_gate.calibration_splits must be non-empty")
    out["test_split"] = str(out.get("test_split", split_spec.get("test_split", "test")))
    out["fixed_grid_step"] = float(out.get("fixed_grid_step", _DEFAULT_PROMOTED_GATE_FIXED_GRID_STEP))
    out["selection_metric"] = str(out.get("selection_metric", "weighted_average_error"))
    if out["selection_metric"] != "weighted_average_error":
        raise ValueError("classic.promoted_stratified_gate.selection_metric must be weighted_average_error")
    out["error_weights"] = dict(_DEFAULT_PROMOTED_GATE_ERROR_WEIGHTS)
    return out


def _weighted_error_metrics(
    *,
    n_queries: int,
    false_abstain: int,
    false_link: int,
    wrong_candidate_link: int,
    error_weights: Mapping[str, float] = _DEFAULT_PROMOTED_GATE_ERROR_WEIGHTS,
) -> dict[str, Any]:
    false_abstain_error_rate = float(false_abstain) / float(n_queries) if n_queries else 0.0
    false_link_error_rate = float(false_link) / float(n_queries) if n_queries else 0.0
    wrong_link_error_rate = float(wrong_candidate_link) / float(n_queries) if n_queries else 0.0
    weight_total = float(sum(float(value) for value in error_weights.values()))
    weighted_average_error = (
        (
            float(error_weights["false_abstain"]) * false_abstain_error_rate
            + float(error_weights["false_link"]) * false_link_error_rate
            + float(error_weights["wrong_candidate_link"]) * wrong_link_error_rate
        )
        / weight_total
        if weight_total
        else 0.0
    )
    return {
        "false_abstain_error_rate": false_abstain_error_rate,
        "false_link_error_rate": false_link_error_rate,
        "wrong_link_error_rate": wrong_link_error_rate,
        "weighted_average_error": weighted_average_error,
        "weighted_average_error_weights": {
            "false_abstain_error_rate": float(error_weights["false_abstain"]),
            "false_link_error_rate": float(error_weights["false_link"]),
            "wrong_link_error_rate": float(error_weights["wrong_candidate_link"]),
        },
    }


def _summarize_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    positives = predictions[predictions["query_safe_target"] == 1]
    negatives = predictions[predictions["query_safe_target"] == 0]
    accepted = predictions["predicted_action"] == "link_candidate"
    correct_link = accepted & (predictions["query_safe_target"] == 1) & (predictions["chosen_candidate_target"] == 1)
    false_abstain = (~accepted) & (predictions["query_safe_target"] == 1)
    false_link = accepted & (predictions["query_safe_target"] == 0)
    wrong_candidate_link = (
        accepted & (predictions["query_safe_target"] == 1) & (predictions["chosen_candidate_target"] == 0)
    )
    tp = int(correct_link.sum())
    # A wrong candidate link is both an accepted incorrect link (precision error)
    # and a missed correct positive link (recall error), so these diagnostic
    # counts are intentionally not a mutually exclusive confusion matrix.
    fp = int((accepted & ~correct_link).sum())
    tn = int(((predictions["predicted_action"] == "abstain") & (predictions["query_safe_target"] == 0)).sum())
    fn = int(
        (
            ((predictions["predicted_action"] == "abstain") & (predictions["query_safe_target"] == 1))
            | (accepted & (predictions["query_safe_target"] == 1) & (predictions["chosen_candidate_target"] == 0))
        ).sum()
    )
    positive_recall = float(positives["correct"].mean()) if len(positives) else 0.0
    negative_recall = float(negatives["correct"].mean()) if len(negatives) else 0.0
    link_precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    link_recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    errors = int((predictions["correct"] == 0).sum())
    summary = {
        "target_semantics": "query_safe_target_with_explicit_source",
        "n_queries": int(len(predictions)),
        "n_positive_queries": int(len(positives)),
        "n_negative_queries": int(len(negatives)),
        "accuracy": float(predictions["correct"].mean()) if len(predictions) else 0.0,
        "errors": errors,
        "error_rate": float(errors / len(predictions)) if len(predictions) else 0.0,
        "balanced_accuracy": (positive_recall + negative_recall) / 2.0,
        "positive_recall": positive_recall,
        "negative_recall": negative_recall,
        "link_precision": link_precision,
        "link_recall": link_recall,
        "abstain_rate": float((predictions["predicted_action"] == "abstain").mean()) if len(predictions) else 0.0,
        "positive_forced_choice_accuracy": (
            float(positives["chosen_candidate_target"].mean()) if len(positives) else 0.0
        ),
        "false_abstain": int(false_abstain.sum()),
        "false_link": int(false_link.sum()),
        "wrong_candidate_link": int(wrong_candidate_link.sum()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    summary.update(
        _weighted_error_metrics(
            n_queries=int(len(predictions)),
            false_abstain=int(false_abstain.sum()),
            false_link=int(false_link.sum()),
            wrong_candidate_link=int(wrong_candidate_link.sum()),
        )
    )
    return summary


def _normalize_augmented_feature_frame(df: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    requested_anchor_features = set(feature_columns) & set(_ANCHOR_EVIDENCE_FEATURE_COLUMNS)
    if requested_anchor_features:
        out = _derive_anchor_evidence_features(out)
    requested_derived_features = set(feature_columns) & set(_DERIVED_PROMOTED_FEATURE_COLUMNS)
    if requested_derived_features:
        out = _derive_promoted_features(out)
    for column in feature_columns:
        if column not in out.columns:
            out[column] = np.nan
    for column in feature_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _augmented_feature_matrix(df: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    prepared = _normalize_augmented_feature_frame(df, feature_columns)
    return prepared.loc[:, list(feature_columns)].copy().astype(np.float32)


def _validate_classic_feature_inputs(df: pd.DataFrame, feature_columns: tuple[str, ...]) -> None:
    """Require active features to be present or explicitly derivable."""

    missing_required: list[str] = []
    missing_prerequisites: dict[str, list[str]] = {}
    for column in feature_columns:
        if column in df.columns:
            continue
        prerequisites = _CLASSIC_DERIVABLE_FEATURE_PREREQUISITES.get(str(column))
        if prerequisites is None:
            missing_required.append(str(column))
            continue
        missing_for_column = [required for required in prerequisites if required not in df.columns]
        if missing_for_column:
            missing_prerequisites[str(column)] = missing_for_column
    if missing_required or missing_prerequisites:
        raise ValueError(
            "Classic feature matrix is missing required feature inputs: "
            f"missing_features={missing_required}, missing_prerequisites={missing_prerequisites}"
        )


def _coerce_classic_feature_matrix(features: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    """Coerce the final classic feature matrix while preserving numeric NaNs."""

    out = features.loc[:, list(feature_columns)].copy()
    non_numeric_cells: dict[str, int] = {}
    for column in feature_columns:
        raw_values = out[column]
        coerced = pd.to_numeric(raw_values, errors="coerce")
        non_numeric = coerced.isna() & raw_values.notna()
        if non_numeric.any():
            non_numeric_cells[str(column)] = int(non_numeric.sum())
        out[column] = coerced
    if non_numeric_cells:
        raise ValueError(f"Classic feature matrix contains non-numeric feature values: {non_numeric_cells}")
    infinite_cells = {
        str(column): int(np.isinf(out[column].to_numpy(dtype=np.float64, copy=False)).sum())
        for column in feature_columns
        if np.isinf(out[column].to_numpy(dtype=np.float64, copy=False)).any()
    }
    if infinite_cells:
        raise ValueError(f"Classic feature matrix contains infinite feature values: {infinite_cells}")
    return out.astype(np.float32)


def _resolve_classic_monotone_constraints(
    spec: dict[str, Any],
    feature_columns: tuple[str, ...],
) -> list[int] | None:
    """Resolve the active classic monotone constraints from explicit spec config."""

    configured = spec.get("monotone_constraints")
    if configured is None:
        return None
    if not isinstance(configured, list):
        raise ValueError("classic.monotone_constraints must be a list when provided")
    if len(configured) != len(feature_columns):
        raise ValueError(
            "classic.monotone_constraints length must match classic.feature_columns "
            f"({len(configured)} != {len(feature_columns)})"
        )
    constraints = [int(value) for value in configured]
    invalid = [value for value in constraints if value not in {-1, 0, 1}]
    if invalid:
        raise ValueError(f"classic.monotone_constraints values must be in {{-1, 0, 1}}: {invalid}")
    return constraints if any(value != 0 for value in constraints) else None


def _build_classic_classifier(
    params: dict[str, Any],
    *,
    monotone_constraints: list[int] | None = None,
    n_jobs: int = DEFAULT_CLASSIC_N_JOBS,
) -> LGBMClassifier:
    classifier_params = {key: value for key, value in params.items()}
    if monotone_constraints is not None:
        classifier_params["monotone_constraints"] = list(monotone_constraints)
    return LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        random_state=13,
        data_random_seed=13,
        feature_fraction_seed=13,
        verbosity=-1,
        n_jobs=resolve_n_jobs(n_jobs),
        class_weight=None,
        **classifier_params,
    )


def _classic_feature_matrix(df: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    """Build a classic feature frame, allowing union-style augmented features when requested."""

    _validate_classic_feature_inputs(df, feature_columns)
    missing_feature_columns = [column for column in feature_columns if column not in df.columns]
    if missing_feature_columns:
        features = _augmented_feature_matrix(df, feature_columns)
        return _coerce_classic_feature_matrix(features, feature_columns)
    out = df.copy()
    return _coerce_classic_feature_matrix(out, feature_columns)


def _apply_classic_train_row_cap(
    train_df: pd.DataFrame,
    *,
    rule_name: str | None,
    min_train_limit: int | None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Apply an optional per-query classic training row cap."""

    if rule_name is None:
        return train_df.copy(), None
    if rule_name != "max_of_min_limit_and_first_positive_rank":
        raise ValueError(f"Unsupported classic train row cap rule: {rule_name}")
    if min_train_limit is None:
        raise ValueError("classic train row cap rule requires train_row_cap_min_limit")

    train_df = train_df.copy()
    train_df["_row_cap_query_group_key"] = train_df["query_group_id"].astype(str)
    positive_ranks = (
        train_df.loc[train_df["label"] == 1]
        .groupby("_row_cap_query_group_key", sort=False)["retrieval_rank"]
        .min()
        .rename("first_positive_rank")
    )
    query_caps = train_df[["query_group_id", "_row_cap_query_group_key"]].drop_duplicates().copy()
    query_caps["first_positive_rank"] = (
        query_caps["_row_cap_query_group_key"].map(positive_ranks.to_dict()).astype("float64")
    )
    query_caps["row_cap"] = (
        query_caps["first_positive_rank"].fillna(float(min_train_limit)).clip(lower=float(min_train_limit))
    )
    cap_map = query_caps.set_index("_row_cap_query_group_key")["row_cap"].to_dict()
    selected = train_df[
        train_df["retrieval_rank"] <= train_df["_row_cap_query_group_key"].map(cap_map).astype(float)
    ].copy()
    selected = selected.drop(columns=["_row_cap_query_group_key"])

    positive_queries_before = set(train_df.loc[train_df["label"] == 1, "query_group_id"].astype(str))
    positive_queries_after = set(selected.loc[selected["label"] == 1, "query_group_id"].astype(str))
    retained_beyond_min = int((query_caps["row_cap"] > float(min_train_limit)).sum())
    positive_rows_before = int(train_df["label"].sum())
    positive_rows_after = int(selected["label"].sum())
    return selected, {
        "rule_name": rule_name,
        "min_train_limit": int(min_train_limit),
        "train_rows_before": int(len(train_df)),
        "train_rows_after": int(len(selected)),
        "positive_rows_before": positive_rows_before,
        "positive_rows_after": positive_rows_after,
        "positive_rows_retained_pct": (
            float(positive_rows_after / positive_rows_before) if positive_rows_before else None
        ),
        "queries_before": int(train_df["query_group_id"].astype(str).nunique()),
        "queries_after": int(selected["query_group_id"].astype(str).nunique()),
        "positive_queries_before": int(len(positive_queries_before)),
        "positive_queries_after": int(len(positive_queries_after)),
        "lost_positive_queries": int(len(positive_queries_before - positive_queries_after)),
        "queries_with_row_cap_above_min": retained_beyond_min,
        "queries_with_row_cap_equal_min": int(len(query_caps) - retained_beyond_min),
    }


def _score_query_choices(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    query_id_column: str,
    include_margin: bool,
    bucket_column: str | None = None,
) -> pd.DataFrame:
    keep_columns = [query_id_column, "dataset", "query_view", "candidate_component_key", "retrieval_rank", "label"]
    for optional_column in ("query_author", "query_first_token"):
        if optional_column in df.columns:
            keep_columns.append(optional_column)
    if "supervision_type" in df.columns:
        keep_columns.append("supervision_type")
    if "base_group_id" in df.columns:
        keep_columns.append("base_group_id")
    if bucket_column:
        keep_columns.append(bucket_column)
    scored = df[keep_columns].copy()
    scored["candidate_probability"] = probabilities.astype(np.float32)
    rows: list[dict[str, Any]] = []
    for query_id, group in scored.groupby(query_id_column, sort=False):
        ranked = group.sort_values(
            by=["candidate_probability", "retrieval_rank", "candidate_component_key"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        chosen = ranked.iloc[0]
        second_probability = float(ranked.iloc[1]["candidate_probability"]) if len(ranked) > 1 else np.nan
        retrieved_window_safe_target = int(group["label"].max())
        row = {
            "query_case_id": str(query_id),
            "dataset": str(chosen["dataset"]),
            "query_view": str(chosen["query_view"]),
            "query_safe_target": retrieved_window_safe_target,
            "retrieved_window_safe_target": retrieved_window_safe_target,
            "query_safe_target_source": "retrieved_window",
            "chosen_candidate_target": int(chosen["label"]),
            "chosen_probability": float(chosen["candidate_probability"]),
            "chosen_candidate_component_key": str(chosen["candidate_component_key"]),
            "predicted_action": "abstain",
            "correct": 0,
            "first_name_bucket": _classic_gate_first_name_bucket(chosen),
        }
        if "supervision_type" in group.columns:
            row["supervision_type"] = str(chosen["supervision_type"])
        if "base_group_id" in group.columns:
            row["base_group_id"] = str(chosen["base_group_id"])
        if bucket_column:
            row["review_bucket"] = str(chosen[bucket_column])
        if include_margin:
            row["second_probability"] = second_probability if pd.notna(second_probability) else None
            row["score_margin"] = (
                float(chosen["candidate_probability"]) - float(second_probability)
                if pd.notna(second_probability)
                else None
            )
            row["has_runner_up"] = int(len(ranked) > 1)
            row["candidate_kind"] = "multi_candidate" if len(ranked) > 1 else "single_candidate"
            row["top1_correct"] = int(chosen["label"])
        rows.append(row)
    return pd.DataFrame(rows)


def _total_error_gate_bucket(rows: pd.DataFrame) -> pd.Series:
    """Return candidate-kind/first-name gate buckets for scored query choices."""

    if "candidate_kind" in rows.columns:
        candidate_kind = rows["candidate_kind"].astype(str)
    else:
        has_runner_up = pd.to_numeric(rows["has_runner_up"], errors="coerce").fillna(0).astype(int) == 1
        score_margin = pd.to_numeric(rows["score_margin"], errors="coerce")
        candidate_kind = pd.Series(
            np.where(has_runner_up & score_margin.notna(), "multi_candidate", "single_candidate"),
            index=rows.index,
        )
    if "first_name_bucket" in rows.columns:
        first_name_bucket = rows["first_name_bucket"].astype(str)
    else:
        first_name_bucket = rows.apply(_classic_gate_first_name_bucket, axis=1)
    return candidate_kind + "|" + first_name_bucket


def _summarize_training_gate_buckets(train_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Count post-filter training rows and queries by promoted gate bucket."""

    query_ids = train_df["query_group_id"].astype(str)
    row_counts = query_ids.value_counts(sort=False)
    query_representatives = train_df.assign(_query_group_id=query_ids).groupby("_query_group_id", sort=False).head(1)
    candidate_kind = pd.Series(
        np.where(
            query_representatives["_query_group_id"].map(row_counts).astype(int).gt(1),
            "multi_candidate",
            "single_candidate",
        ),
        index=query_representatives.index,
    )
    bucket_frame = pd.DataFrame(
        {
            "bucket": candidate_kind + "|" + query_representatives.apply(_classic_gate_first_name_bucket, axis=1),
            "row_count": query_representatives["_query_group_id"].map(row_counts).astype(int),
        }
    )
    query_counts = bucket_frame["bucket"].value_counts(sort=False).to_dict()
    training_row_counts = bucket_frame.groupby("bucket", sort=False)["row_count"].sum().to_dict()
    return {
        "query_counts": {bucket: int(query_counts.get(bucket, 0)) for bucket in _TOTAL_ERROR_SCORE_BUCKETS},
        "row_counts": {bucket: int(training_row_counts.get(bucket, 0)) for bucket in _TOTAL_ERROR_SCORE_BUCKETS},
    }


def _gate_bucket_split_counts(predictions: pd.DataFrame, split_order: Sequence[str]) -> dict[str, dict[str, int]]:
    """Count scored query choices by promoted gate bucket and split."""

    if predictions.empty:
        return {bucket: {str(split): 0 for split in split_order} for bucket in _TOTAL_ERROR_SCORE_BUCKETS}
    counts = predictions.groupby(["gate_bucket", "split"], dropna=False, sort=False).size()
    return {
        bucket: {str(split): int(counts.get((bucket, str(split)), 0)) for split in split_order}
        for bucket in _TOTAL_ERROR_SCORE_BUCKETS
    }


def _fixed_probability_threshold_grid(step: float) -> np.ndarray:
    """Return an inclusive fixed probability grid from 0.0 to 1.0."""

    step = float(step)
    if not 0.0 < step <= 1.0:
        raise ValueError("classic.promoted_stratified_gate.fixed_grid_step must be in (0, 1]")
    interval_count = int(round(1.0 / step))
    if not np.isclose(float(interval_count) * step, 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("classic.promoted_stratified_gate.fixed_grid_step must evenly divide 1.0")
    return np.round(np.linspace(0.0, 1.0, interval_count + 1, dtype=np.float64), 6)


def _total_error_components(rows: pd.DataFrame, link: np.ndarray) -> dict[str, np.ndarray]:
    """Count error components for one or more link/abstain decisions."""

    query_target = rows["query_safe_target"].to_numpy(dtype=np.int8, copy=False)
    chosen_target = rows["chosen_candidate_target"].to_numpy(dtype=np.int8, copy=False)
    matrix = np.asarray(link, dtype=bool)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    false_abstain = ((~matrix) & (query_target[:, None] == 1)).sum(axis=0).astype(np.int64)
    false_link = (matrix & (query_target[:, None] == 0)).sum(axis=0).astype(np.int64)
    wrong_candidate_link = (
        (matrix & (query_target[:, None] == 1) & (chosen_target[:, None] == 0)).sum(axis=0).astype(np.int64)
    )
    return {
        "false_abstain": false_abstain,
        "false_link": false_link,
        "wrong_candidate_link": wrong_candidate_link,
        "errors": false_abstain + false_link + wrong_candidate_link,
    }


def _weighted_error_counts_from_components(
    components: Mapping[str, np.ndarray],
    *,
    error_weights: Mapping[str, float] = _DEFAULT_PROMOTED_GATE_ERROR_WEIGHTS,
) -> np.ndarray:
    """Return weighted error counts from vectorized error components."""

    return (
        float(error_weights["false_abstain"]) * np.asarray(components["false_abstain"], dtype=np.float64)
        + float(error_weights["false_link"]) * np.asarray(components["false_link"], dtype=np.float64)
        + float(error_weights["wrong_candidate_link"])
        * np.asarray(components["wrong_candidate_link"], dtype=np.float64)
    )


def _best_total_error_threshold_index(
    components: Mapping[str, np.ndarray],
    *,
    score_thresholds: np.ndarray,
    error_weights: Mapping[str, float],
    margin_thresholds: np.ndarray | None = None,
) -> int:
    """Select the best fixed-grid threshold by weighted error with deterministic ties."""

    weighted_errors = _weighted_error_counts_from_components(components, error_weights=error_weights)
    tie_keys: list[np.ndarray] = [np.asarray(score_thresholds, dtype=np.float64)]
    if margin_thresholds is not None:
        tie_keys.append(np.asarray(margin_thresholds, dtype=np.float64))
    ranking = np.lexsort(
        tuple(
            [
                *tie_keys,
                np.asarray(components["false_abstain"], dtype=np.int64),
                np.asarray(components["false_link"], dtype=np.int64),
                np.asarray(components["wrong_candidate_link"], dtype=np.int64),
                weighted_errors,
            ]
        )
    )
    return int(ranking[0])


def _total_error_fit_metrics(
    rows: pd.DataFrame,
    components: Mapping[str, np.ndarray],
    best_index: int,
    *,
    error_weights: Mapping[str, float],
) -> dict[str, Any]:
    """Return scalar fit metrics for the selected threshold candidate."""

    false_abstain = int(np.asarray(components["false_abstain"])[best_index])
    false_link = int(np.asarray(components["false_link"])[best_index])
    wrong_candidate_link = int(np.asarray(components["wrong_candidate_link"])[best_index])
    weighted_error_count = float(
        _weighted_error_counts_from_components(components, error_weights=error_weights)[best_index]
    )
    return {
        "n_queries": int(len(rows)),
        "errors": int(np.asarray(components["errors"])[best_index]),
        "false_abstain": false_abstain,
        "false_link": false_link,
        "wrong_candidate_link": wrong_candidate_link,
        "weighted_error_count": weighted_error_count,
        **_weighted_error_metrics(
            n_queries=int(len(rows)),
            false_abstain=false_abstain,
            false_link=false_link,
            wrong_candidate_link=wrong_candidate_link,
            error_weights=error_weights,
        ),
    }


def _fit_total_error_single_score(
    rows: pd.DataFrame,
    *,
    threshold_grid: np.ndarray,
    error_weights: Mapping[str, float],
) -> tuple[float, dict[str, Any]]:
    """Fit a score-only threshold on a fixed probability grid."""

    if rows.empty:
        return float(threshold_grid[-1]), {
            "n_queries": 0,
            "errors": 0,
            "false_abstain": 0,
            "false_link": 0,
            "wrong_candidate_link": 0,
            "weighted_error_count": 0.0,
            **_weighted_error_metrics(
                n_queries=0,
                false_abstain=0,
                false_link=0,
                wrong_candidate_link=0,
                error_weights=error_weights,
            ),
        }
    score = pd.to_numeric(rows["chosen_probability"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    links = score[:, None] >= threshold_grid[None, :]
    components = _total_error_components(rows, links)
    best_index = _best_total_error_threshold_index(
        components,
        score_thresholds=threshold_grid,
        error_weights=error_weights,
    )
    return float(threshold_grid[best_index]), _total_error_fit_metrics(
        rows,
        components,
        best_index,
        error_weights=error_weights,
    )


def _fit_total_error_score_margin(
    rows: pd.DataFrame,
    *,
    threshold_grid: np.ndarray,
    error_weights: Mapping[str, float],
) -> tuple[float, float, dict[str, Any]]:
    """Fit a score-or-margin threshold pair on a fixed probability grid."""

    if rows.empty:
        empty_metrics = {
            "n_queries": 0,
            "errors": 0,
            "false_abstain": 0,
            "false_link": 0,
            "wrong_candidate_link": 0,
            "weighted_error_count": 0.0,
            **_weighted_error_metrics(
                n_queries=0,
                false_abstain=0,
                false_link=0,
                wrong_candidate_link=0,
                error_weights=error_weights,
            ),
        }
        return float(threshold_grid[-1]), float(threshold_grid[-1]), empty_metrics
    score = pd.to_numeric(rows["chosen_probability"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    margin = pd.to_numeric(rows["score_margin"], errors="coerce").fillna(-np.inf).to_numpy(dtype=np.float64)
    best_key: tuple[float, int, int, int, float, float] | None = None
    best_score_threshold = float(threshold_grid[-1])
    best_margin_threshold = float(threshold_grid[-1])
    best_metrics: dict[str, Any] | None = None
    for score_threshold in threshold_grid:
        links = (score[:, None] >= float(score_threshold)) | (margin[:, None] >= threshold_grid[None, :])
        components = _total_error_components(rows, links)
        score_thresholds = np.repeat(float(score_threshold), len(threshold_grid))
        best_index = _best_total_error_threshold_index(
            components,
            score_thresholds=score_thresholds,
            margin_thresholds=threshold_grid,
            error_weights=error_weights,
        )
        metrics = _total_error_fit_metrics(rows, components, best_index, error_weights=error_weights)
        key = (
            float(metrics["weighted_error_count"]),
            int(metrics["wrong_candidate_link"]),
            int(metrics["false_link"]),
            int(metrics["false_abstain"]),
            float(score_threshold),
            float(threshold_grid[best_index]),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_score_threshold = float(score_threshold)
            best_margin_threshold = float(threshold_grid[best_index])
            best_metrics = metrics
    if best_metrics is None:
        raise ValueError("Unable to fit promoted score/margin gate on non-empty calibration rows")
    return best_score_threshold, best_margin_threshold, best_metrics


def _fit_total_error_gate(
    calibration_rows: pd.DataFrame,
    *,
    fixed_grid_step: float,
    error_weights: Mapping[str, float],
) -> dict[str, Any]:
    """Fit the promoted 4-score/2-margin gate on all calibration rows."""

    threshold_grid = _fixed_probability_threshold_grid(fixed_grid_step)
    labels = _total_error_gate_bucket(calibration_rows)
    score_thresholds: dict[str, float] = {}
    margin_thresholds: dict[str, float] = {}
    bucket_metrics: dict[str, dict[str, Any]] = {}
    for bucket in _TOTAL_ERROR_SCORE_BUCKETS:
        rows = calibration_rows[labels == bucket].copy()
        if bucket in _TOTAL_ERROR_MARGIN_BUCKETS:
            score_threshold, margin_threshold, metrics = _fit_total_error_score_margin(
                rows,
                threshold_grid=threshold_grid,
                error_weights=error_weights,
            )
            score_thresholds[bucket] = score_threshold
            margin_thresholds[bucket] = margin_threshold
        else:
            score_threshold, metrics = _fit_total_error_single_score(
                rows,
                threshold_grid=threshold_grid,
                error_weights=error_weights,
            )
            score_thresholds[bucket] = score_threshold
        bucket_metrics[bucket] = {
            "score_threshold": float(score_thresholds[bucket]),
            "margin_threshold": float(margin_thresholds[bucket]) if bucket in margin_thresholds else None,
            **metrics,
        }
    return {
        "gate": TotalErrorGateSpec(
            name=f"full_calibration_fixed_grid_{float(fixed_grid_step):g}",
            score_thresholds=score_thresholds,
            margin_thresholds=margin_thresholds,
        ),
        "bucket_metrics": bucket_metrics,
        "threshold_grid_points": int(len(threshold_grid)),
    }


def _apply_total_error_gate(rows: pd.DataFrame, gate: TotalErrorGateSpec) -> pd.DataFrame:
    """Apply a total-error 4-score/2-margin gate to scored query choices."""

    predictions = _apply_classic_gate(
        rows,
        score_threshold=float(gate.score_thresholds["multi_candidate|multi_letter_first"]),
        margin_threshold=float(gate.margin_thresholds["multi_candidate|multi_letter_first"]),
        single_candidate_score_threshold=float(gate.score_thresholds["single_candidate|single_letter_first"]),
        bucketed_score_thresholds=gate.score_thresholds,
        bucketed_margin_thresholds=gate.margin_thresholds,
    )
    predictions["gate_bucket"] = _total_error_gate_bucket(predictions)
    return predictions


def _fit_promoted_stratified_total_error_gate(
    choices: pd.DataFrame,
    gate_config: dict[str, Any],
) -> dict[str, Any]:
    """Fit the promoted bucketed gate on all configured calibration splits."""

    calibration_splits = tuple(str(split) for split in gate_config["calibration_splits"])
    calibration_rows = choices[choices["split"].astype(str).isin(calibration_splits)].copy()
    if calibration_rows.empty:
        raise ValueError(
            "Promoted stratified gate requires non-empty calibration splits: " f"splits={list(calibration_splits)}"
        )
    error_weights = dict(gate_config.get("error_weights", _DEFAULT_PROMOTED_GATE_ERROR_WEIGHTS))
    fit_result = _fit_total_error_gate(
        calibration_rows,
        fixed_grid_step=float(gate_config["fixed_grid_step"]),
        error_weights=error_weights,
    )
    selected_gate = fit_result["gate"]
    calibration_predictions = _apply_total_error_gate(calibration_rows, selected_gate)
    calibration_metrics = _summarize_predictions(calibration_predictions)
    split_series = choices["split"].astype(str)
    split_metrics = {
        split: _summarize_predictions(_apply_total_error_gate(choices[split_series == split].copy(), selected_gate))
        for split in calibration_splits
    }
    return {
        "gate": selected_gate,
        "calibration_metrics": calibration_metrics,
        "calibration_split_metrics": split_metrics,
        "bucket_metrics": dict(fit_result["bucket_metrics"]),
        "threshold_grid_points": int(fit_result["threshold_grid_points"]),
        "selection_key": {
            "calibration_weighted_average_error": float(calibration_metrics["weighted_average_error"]),
            "calibration_false_abstain_error_rate": float(calibration_metrics["false_abstain_error_rate"]),
            "calibration_false_link_error_rate": float(calibration_metrics["false_link_error_rate"]),
            "calibration_wrong_link_error_rate": float(calibration_metrics["wrong_link_error_rate"]),
            "error_weights": error_weights,
            "calibration_errors": int(calibration_metrics["errors"]),
            "calibration_wrong_candidate_link": int(calibration_metrics["wrong_candidate_link"]),
            "calibration_false_link": int(calibration_metrics["false_link"]),
            "calibration_false_abstain": int(calibration_metrics["false_abstain"]),
        },
    }


def _apply_clean_hwang_overrides(predictions: pd.DataFrame, override_df: pd.DataFrame) -> pd.DataFrame:
    merged = predictions.merge(
        override_df[["query_group_id", "manual_safe_target"]].rename(columns={"query_group_id": "query_case_id"}),
        on="query_case_id",
        how="left",
        validate="one_to_one",
    )
    manual_override = merged["manual_safe_target"].notna()
    if "query_safe_target_source" not in merged.columns:
        merged["query_safe_target_source"] = "retrieved_window"
    merged.loc[manual_override, "query_safe_target_source"] = "manual_safe_target"
    merged["query_safe_target"] = merged["manual_safe_target"].fillna(merged["query_safe_target"]).astype(int)
    merged = merged.drop(columns=["manual_safe_target"])
    merged["correct"] = (
        ((merged["predicted_action"] == "abstain") & (merged["query_safe_target"] == 0))
        | (
            (merged["predicted_action"] == "link_candidate")
            & (merged["query_safe_target"] == 1)
            & (merged["chosen_candidate_target"] == 1)
        )
    ).astype(int)
    return merged


def _score_abstain_rule(
    rows: pd.DataFrame,
    score_threshold: float,
    margin_threshold: float,
    *,
    single_candidate_score_threshold: float | None = None,
    bucketed_score_thresholds: dict[str, float] | None = None,
    bucketed_margin_threshold: float | None = None,
    bucketed_margin_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    scored_rows = _apply_classic_gate(
        rows,
        score_threshold=float(score_threshold),
        margin_threshold=float(margin_threshold),
        single_candidate_score_threshold=single_candidate_score_threshold,
        bucketed_score_thresholds=bucketed_score_thresholds,
        bucketed_margin_threshold=bucketed_margin_threshold,
        bucketed_margin_thresholds=bucketed_margin_thresholds,
    )
    scored_rows["accepted"] = scored_rows["predicted_action"] == "link_candidate"
    positives = scored_rows[scored_rows["query_safe_target"] == 1]
    negatives = scored_rows[scored_rows["query_safe_target"] == 0]
    positive_correct = int(
        (
            (scored_rows["accepted"])
            & (scored_rows["chosen_candidate_target"] == 1)
            & (scored_rows["query_safe_target"] == 1)
        ).sum()
    )
    positive_accuracy = float(positive_correct / len(positives)) if len(positives) else None
    negative_reject = int((~scored_rows["accepted"] & (scored_rows["query_safe_target"] == 0)).sum())
    negative_reject_accuracy = float(negative_reject / len(negatives)) if len(negatives) else None
    balanced_accuracy = (
        float(positive_accuracy if positive_accuracy is not None else 0.0)
        + float(negative_reject_accuracy if negative_reject_accuracy is not None else 0.0)
    ) / 2.0
    return {
        "score_threshold": float(score_threshold),
        "margin_threshold": float(margin_threshold),
        "single_candidate_score_threshold": (
            float(single_candidate_score_threshold) if single_candidate_score_threshold is not None else None
        ),
        "queries": int(len(rows)),
        "eligible_queries": int(len(scored_rows)),
        "runner_up_queries": int(
            (
                (pd.to_numeric(scored_rows["has_runner_up"], errors="coerce").fillna(0).astype(int) == 1)
                & scored_rows["score_margin"].notna()
            ).sum()
        ),
        "single_candidate_queries": int(
            (
                (pd.to_numeric(scored_rows["has_runner_up"], errors="coerce").fillna(0).astype(int) == 0)
                | scored_rows["score_margin"].isna()
            ).sum()
        ),
        "positive_queries": int(len(positives)),
        "negative_queries": int(len(negatives)),
        "balanced_accuracy": float(balanced_accuracy),
        "positive_accuracy": positive_accuracy,
        "negative_reject_accuracy": negative_reject_accuracy,
        "positive_accept_rate": (
            float(scored_rows.loc[scored_rows["query_safe_target"] == 1, "accepted"].mean()) if len(positives) else None
        ),
        "rejection_rate": float((~scored_rows["accepted"]).mean()) if len(scored_rows) else 0.0,
        "false_positive_links": int(
            (
                scored_rows["accepted"]
                & ~((scored_rows["query_safe_target"] == 1) & (scored_rows["chosen_candidate_target"] == 1))
            ).sum()
        ),
    }


def _apply_classic_gate(
    query_choices: pd.DataFrame,
    score_threshold: float,
    margin_threshold: float,
    *,
    single_candidate_score_threshold: float | None = None,
    bucketed_score_thresholds: dict[str, float] | None = None,
    bucketed_margin_threshold: float | None = None,
    bucketed_margin_thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    pred = query_choices.copy()
    score_margin = pd.to_numeric(pred["score_margin"], errors="coerce")
    if "has_runner_up" in pred.columns:
        no_runner_up = pd.to_numeric(pred["has_runner_up"], errors="coerce").fillna(0).astype(int) == 0
    else:
        no_runner_up = pd.Series(False, index=pred.index)
    single_candidate = no_runner_up | score_margin.isna()
    if bucketed_score_thresholds is not None:
        normalized_thresholds = {str(key): float(value) for key, value in bucketed_score_thresholds.items()}
        if "first_name_bucket" in pred.columns:
            first_name_bucket = pred["first_name_bucket"].astype(str)
        else:
            first_name_bucket = pred.apply(_classic_gate_first_name_bucket, axis=1)
        candidate_kind = pd.Series(
            np.where(single_candidate, "single_candidate", "multi_candidate"),
            index=pred.index,
        )
        bucket_keys = candidate_kind + "|" + first_name_bucket
        single_threshold = (
            float(single_candidate_score_threshold)
            if single_candidate_score_threshold is not None
            else float(score_threshold)
        )
        fallback_thresholds = pd.Series(
            np.where(
                single_candidate,
                single_threshold,
                float(score_threshold),
            ),
            index=pred.index,
        )
        threshold_values = bucket_keys.map(normalized_thresholds).fillna(fallback_thresholds).astype(float)
        score_link = pd.to_numeric(pred["chosen_probability"], errors="coerce") >= threshold_values
        if bucketed_margin_thresholds is not None:
            normalized_margin_thresholds = {str(key): float(value) for key, value in bucketed_margin_thresholds.items()}
            margin_threshold_values = bucket_keys.map(normalized_margin_thresholds)
            margin_link = (
                (~single_candidate)
                & score_margin.notna()
                & margin_threshold_values.notna()
                & (score_margin >= margin_threshold_values.astype(float))
            )
        elif bucketed_margin_threshold is None:
            margin_link = pd.Series(False, index=pred.index)
        else:
            margin_link = (
                (~single_candidate) & score_margin.notna() & (score_margin >= float(bucketed_margin_threshold))
            )
        abstain = ~(score_link | margin_link)
    else:
        runner_up_abstain = (
            ~single_candidate
            & (pred["chosen_probability"] < float(score_threshold))
            & (score_margin < float(margin_threshold))
        )
        single_candidate_threshold = (
            float(score_threshold)
            if single_candidate_score_threshold is None
            else float(single_candidate_score_threshold)
        )
        single_candidate_abstain = single_candidate & (pred["chosen_probability"] < single_candidate_threshold)
        abstain = runner_up_abstain | single_candidate_abstain
    pred["predicted_action"] = np.where(abstain, "abstain", "link_candidate")
    pred["correct"] = (
        ((pred["predicted_action"] == "abstain") & (pred["query_safe_target"] == 0))
        | (
            (pred["predicted_action"] == "link_candidate")
            & (pred["query_safe_target"] == 1)
            & (pred["chosen_candidate_target"] == 1)
        )
    ).astype(int)
    return pred


def _evaluate_classic_manual_holdout(
    manual_holdout: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    score_threshold: float,
    margin_threshold: float,
    single_candidate_score_threshold: float | None = None,
    bucketed_score_thresholds: dict[str, float] | None = None,
    bucketed_margin_threshold: float | None = None,
    bucketed_margin_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Score and summarize the manual holdout for the freshly fit classic model."""

    query_choices = _score_query_choices(
        manual_holdout.rename(columns={"binary_safe_link_target": "label"}),
        probabilities,
        query_id_column="query_case_id",
        include_margin=True,
        bucket_column="review_bucket",
    )
    predictions = _apply_classic_gate(
        query_choices,
        score_threshold=float(score_threshold),
        margin_threshold=float(margin_threshold),
        single_candidate_score_threshold=single_candidate_score_threshold,
        bucketed_score_thresholds=bucketed_score_thresholds,
        bucketed_margin_threshold=bucketed_margin_threshold,
        bucketed_margin_thresholds=bucketed_margin_thresholds,
    )
    return {
        "overall": _summarize_predictions(predictions),
        "by_bucket": {
            str(bucket): _summarize_predictions(group.copy())
            for bucket, group in predictions.groupby("review_bucket", sort=False)
        },
    }


def _evaluate_scored_windows(
    query_choices: pd.DataFrame,
    *,
    score_threshold: float,
    margin_threshold: float,
    single_candidate_score_threshold: float | None = None,
    override_df: pd.DataFrame | None = None,
    bucketed_score_thresholds: dict[str, float] | None = None,
    bucketed_margin_threshold: float | None = None,
    bucketed_margin_thresholds: dict[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for limit in sorted(query_choices["retrieval_rank_limit"].dropna().astype(int).unique()):
        limited = query_choices[query_choices["retrieval_rank_limit"] == limit].copy()
        predictions = _apply_classic_gate(
            limited,
            float(score_threshold),
            float(margin_threshold),
            single_candidate_score_threshold=single_candidate_score_threshold,
            bucketed_score_thresholds=bucketed_score_thresholds,
            bucketed_margin_threshold=bucketed_margin_threshold,
            bucketed_margin_thresholds=bucketed_margin_thresholds,
        )
        if override_df is not None:
            predictions = _apply_clean_hwang_overrides(predictions, override_df)
        results[str(limit)] = {"overall": _summarize_predictions(predictions)}
    return results


def _classic_stratified_eval_source_specs(spec: dict[str, Any]) -> tuple[dict[str, str], ...]:
    """Return source files used by the promoted eval/test stratified split."""

    sources = [
        {
            "source_key": "calibration_source",
            "path": str(spec["classic_gate_source_path"]),
            "source_kind": "calibration_source",
        },
        {"source_key": "s2and_eval", "path": str(spec["s2and_eval_path"]), "source_kind": "public_test"},
        {"source_key": "hwang_eval", "path": str(spec["hwang_eval_path"]), "source_kind": "public_test"},
    ]
    for path_key, source_key in (
        ("s_park_eval_path", "s_park_eval"),
        ("s_lee_eval_path", "s_lee_eval"),
    ):
        if path_key in spec:
            sources.append(
                {
                    "source_key": source_key,
                    "path": str(spec[path_key]),
                    "source_kind": "public_test",
                }
            )
    for dataset_name, path_like in sorted(dict(spec.get("extra_eval_paths") or {}).items()):
        sources.append(
            {
                "source_key": f"{_normalize_dataset_slug(dataset_name)}_eval",
                "path": str(path_like),
                "source_kind": "public_test",
            }
        )
    return tuple(sources)


def _drop_shadowed_calibration_source_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Drop calibration rows when the active public source has the same query."""

    query_source_key = ["query_group_id", "source_key"]
    public_query_sources = rows.loc[
        ~rows["source_kind"].astype(str).eq("calibration_source"),
        query_source_key,
    ].drop_duplicates()
    if public_query_sources.empty:
        return rows

    marked = rows.merge(
        public_query_sources.assign(_has_public_source_rows=True),
        on=query_source_key,
        how="left",
    )
    shadowed = marked["source_kind"].astype(str).eq("calibration_source") & marked["_has_public_source_rows"].fillna(
        False
    )
    return marked.loc[~shadowed].drop(columns=["_has_public_source_rows"]).copy()


def _validate_unique_stratified_candidate_rows(rows: pd.DataFrame) -> None:
    """Fail if one selected query/source/candidate has multiple active rows."""

    key_columns = ["query_group_id", "source_key", "candidate_component_key"]
    duplicate_mask = rows.duplicated(key_columns, keep=False)
    if not duplicate_mask.any():
        return

    duplicate_rows = rows.loc[duplicate_mask, key_columns + ["label"]].copy()
    duplicate_summary = (
        duplicate_rows.groupby(key_columns, dropna=False)
        .agg(row_count=("label", "size"), labels=("label", lambda values: sorted({str(value) for value in values})))
        .reset_index()
    )
    conflict_count = int(duplicate_summary["labels"].map(len).gt(1).sum())
    sample = duplicate_summary.head(5).to_dict(orient="records")
    raise ValueError(
        "Promoted stratified eval rows contain duplicate query/source/candidate rows: "
        f"duplicate_pairs={len(duplicate_summary)}, conflicting_pairs={conflict_count}, sample={sample}"
    )


def _active_stratified_label_metadata(rows: pd.DataFrame) -> pd.DataFrame:
    """Return query/source metadata recomputed from active candidate labels."""

    metadata_input = rows[["query_group_id", "source_key", "candidate_component_key", "retrieval_rank", "label"]].copy()
    metadata_input["_label"] = pd.to_numeric(metadata_input["label"], errors="coerce").fillna(0).astype(int)
    metadata_input["_retrieval_rank"] = pd.to_numeric(metadata_input["retrieval_rank"], errors="coerce").fillna(
        np.iinfo(np.int32).max
    )
    grouped = metadata_input.groupby(["query_group_id", "source_key"], sort=False)
    metadata = grouped.agg(
        candidate_count=("candidate_component_key", "nunique"),
        min_retrieval_rank=("_retrieval_rank", "min"),
        max_retrieval_rank=("_retrieval_rank", "max"),
        positive_candidate_rows=("_label", "sum"),
    ).reset_index()
    min_positive_rank = (
        metadata_input.loc[metadata_input["_label"].eq(1)]
        .groupby(["query_group_id", "source_key"], sort=False)["_retrieval_rank"]
        .min()
    )
    positive_rank_frame = min_positive_rank.rename("min_positive_rank").reset_index()
    metadata = metadata.merge(positive_rank_frame, on=["query_group_id", "source_key"], how="left")
    has_positive = metadata["positive_candidate_rows"].astype(int).gt(0)
    positive_first = has_positive & metadata["min_positive_rank"].eq(metadata["min_retrieval_rank"])
    metadata["has_positive_candidate"] = has_positive
    metadata["positive_first"] = positive_first
    metadata["positive_rank_bucket"] = np.select(
        [~has_positive, positive_first],
        ["no_positive", "positive_first"],
        default="positive_not_first",
    )
    metadata["raw_has_positive_candidate"] = metadata["has_positive_candidate"]
    metadata["raw_positive_first"] = metadata["positive_first"]
    metadata["manual_safe_target"] = has_positive.astype(int)
    metadata["multiple_candidates"] = metadata["candidate_count"].astype(int).gt(1)
    metadata["min_positive_rank"] = metadata["min_positive_rank"].astype(object)
    metadata.loc[~has_positive, "min_positive_rank"] = ""
    for column in ("candidate_count", "min_retrieval_rank", "max_retrieval_rank", "positive_candidate_rows"):
        metadata[column] = metadata[column].astype(int)
    return metadata


def _refresh_stratified_metadata_from_active_labels(
    rows: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refresh split metadata from the selected active candidate rows."""

    metadata = _active_stratified_label_metadata(rows)
    metadata_columns = [
        "has_positive_candidate",
        "positive_first",
        "positive_rank_bucket",
        "raw_has_positive_candidate",
        "raw_positive_first",
        "manual_safe_target",
        "multiple_candidates",
        "candidate_count",
        "min_positive_rank",
        "min_retrieval_rank",
        "max_retrieval_rank",
        "positive_candidate_rows",
    ]
    refreshed_rows = rows.drop(columns=[column for column in metadata_columns if column in rows.columns]).merge(
        metadata,
        on=["query_group_id", "source_key"],
        how="left",
    )
    refreshed_assignments = assignments.drop(
        columns=[column for column in metadata_columns if column in assignments.columns]
    ).merge(
        metadata,
        on=["query_group_id", "source_key"],
        how="left",
    )
    if "source_stratum" in refreshed_assignments.columns and "first_name_bucket" in refreshed_assignments.columns:
        refreshed_assignments["stratum_key"] = refreshed_assignments.apply(
            lambda row: (
                f"{row['source_stratum']}|has_pos={int(bool(row['has_positive_candidate']))}|"
                f"{row['positive_rank_bucket']}|{row['first_name_bucket']}|"
                f"multi_cand={int(bool(row['multiple_candidates']))}"
            ),
            axis=1,
        )
    if "stratum_key" in refreshed_assignments.columns:
        refreshed_rows = refreshed_rows.drop(columns=["stratum_key"], errors="ignore").merge(
            refreshed_assignments[["query_group_id", "source_key", "stratum_key"]],
            on=["query_group_id", "source_key"],
            how="left",
        )
    return refreshed_rows, refreshed_assignments


def _load_classic_stratified_eval_rows(
    bundle: OfficialBundle,
    spec: dict[str, Any],
    split_spec: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load source rows selected by the promoted query-level stratified split."""

    assignments = _read_csv(_resolve_path(bundle, str(split_spec["assignments_path"])))
    required_assignment_columns = {"query_group_id", "source_key", "split"}
    missing_assignment_columns = sorted(required_assignment_columns - set(assignments.columns))
    if missing_assignment_columns:
        raise ValueError(f"Stratified split assignments missing required columns: {missing_assignment_columns}")
    source_frames: list[pd.DataFrame] = []
    for source_spec in _classic_stratified_eval_source_specs(spec):
        rows = _read_csv(_resolve_path(bundle, source_spec["path"]), compression="gzip")
        if str(source_spec["source_key"]) == "calibration_source":
            rows["source_key"] = (
                rows["dataset"].astype(str).map(CALIBRATION_DATASET_SOURCE_KEY_BY_DATASET).fillna("s2and_eval")
            )
        else:
            rows["source_key"] = str(source_spec["source_key"])
        rows["source_kind"] = str(source_spec["source_kind"])
        source_frames.append(rows)
    all_rows = _drop_shadowed_calibration_source_rows(pd.concat(source_frames, ignore_index=True))
    assignment_columns = [
        "query_group_id",
        "source_key",
        "split",
        "stratum_key",
        "source_stratum",
        "has_positive_candidate",
        "positive_rank_bucket",
        "first_name_bucket",
        "multiple_candidates",
        "manual_safe_target",
        "correction_type",
    ]
    selected_assignment_columns = [column for column in assignment_columns if column in assignments.columns]
    rows = all_rows.merge(
        assignments[selected_assignment_columns],
        on=["query_group_id", "source_key"],
        how="inner",
        suffixes=("", "_split"),
    )
    for column in selected_assignment_columns:
        if column in {"query_group_id", "source_key"}:
            continue
        assignment_column = f"{column}_split"
        if assignment_column in rows.columns:
            rows[column] = rows[assignment_column]
            rows = rows.drop(columns=[assignment_column])
    matched_assignments = rows[["query_group_id", "source_key"]].drop_duplicates()
    if len(matched_assignments) != len(assignments):
        raise ValueError(
            "Stratified split assignments did not all match source rows: "
            f"matched={len(matched_assignments)}, expected={len(assignments)}"
        )
    _validate_unique_stratified_candidate_rows(rows)
    rows, assignments = _refresh_stratified_metadata_from_active_labels(rows, assignments)
    return rows, assignments


def _breakdown_predictions(predictions: pd.DataFrame, column: str) -> dict[str, dict[str, Any]]:
    """Summarize predictions by one column for JSON reports."""

    return {
        str(value): _summarize_predictions(group.copy())
        for value, group in predictions.groupby(column, dropna=False, sort=True)
    }


def _score_classic_stratified_eval_test_choices(
    bundle: OfficialBundle,
    spec: dict[str, Any],
    split_spec: dict[str, Any],
    model: LGBMClassifier,
    feature_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score query choices for the promoted stratified calibration/test split."""

    split_rows, assignments = _load_classic_stratified_eval_rows(bundle, spec, split_spec)
    probabilities = model.predict_proba(_classic_feature_matrix(split_rows, feature_columns))[:, 1]
    choices = _score_query_choices(
        split_rows,
        probabilities,
        query_id_column="query_group_id",
        include_margin=True,
    )
    metadata_columns = [
        "query_group_id",
        "source_key",
        "source_kind",
        "split",
        "stratum_key",
        "source_stratum",
        "has_positive_candidate",
        "positive_rank_bucket",
        "first_name_bucket",
        "multiple_candidates",
        "manual_safe_target",
        "correction_type",
    ]
    metadata = split_rows[[column for column in metadata_columns if column in split_rows.columns]].drop_duplicates(
        "query_group_id"
    )
    metadata = metadata.rename(columns={"query_group_id": "query_case_id"})
    choices = choices.merge(metadata, on="query_case_id", how="left", suffixes=("", "_split"))
    if "first_name_bucket_split" in choices.columns:
        choices["first_name_bucket"] = choices["first_name_bucket_split"].fillna(choices["first_name_bucket"])
        choices = choices.drop(columns=["first_name_bucket_split"])
    if "query_safe_target_source" not in choices.columns:
        choices["query_safe_target_source"] = "retrieved_window"
    if "manual_safe_target" in choices.columns:
        manual_target = pd.to_numeric(choices["manual_safe_target"], errors="coerce")
        choices["manual_safe_target_matches_active_label"] = manual_target.isna() | manual_target.astype("Int64").eq(
            choices["query_safe_target"].astype("Int64")
        )
    return choices, assignments


def _summarize_classic_stratified_predictions(
    predictions: pd.DataFrame,
    assignments: pd.DataFrame,
    split_spec: dict[str, Any],
) -> dict[str, Any]:
    """Build promoted stratified split summary and test breakdowns from predictions."""

    predictions = predictions.copy()
    predictions["gate_bucket"] = (
        _total_error_gate_bucket(predictions) if not predictions.empty else pd.Series(dtype="string")
    )
    split_order = tuple(
        str(value)
        for value in split_spec.get(
            "split_order",
            ("calibration_fit", "calibration_check", "test"),
        )
    )
    overall_by_split = {
        split: _summarize_predictions(predictions[predictions["split"] == split].copy()) for split in split_order
    }
    test_predictions = predictions[predictions["split"] == str(split_spec.get("test_split", "test"))].copy()
    factor_columns = [
        "gate_bucket",
        "source_key",
        "source_stratum",
        "has_positive_candidate",
        "positive_rank_bucket",
        "first_name_bucket",
        "multiple_candidates",
    ]
    return {
        "assignment_query_counts": {
            str(split): int(count) for split, count in assignments["split"].value_counts().sort_index().items()
        },
        "scored_query_counts": {
            str(split): int(count) for split, count in predictions["split"].value_counts().sort_index().items()
        },
        "overall": overall_by_split,
        "gate_bucket_split_counts": _gate_bucket_split_counts(predictions, split_order),
        "test_breakdowns": {
            column: _breakdown_predictions(test_predictions, column)
            for column in factor_columns
            if column in test_predictions.columns
        },
    }


def _format_metric_float(value: Any) -> str:
    """Format a metric value compactly for markdown tables."""

    return f"{float(value):.4f}"


def _metric_balanced_accuracy_cell(metrics: dict[str, Any]) -> str:
    """Return a balanced-accuracy table cell, suppressing single-class slices."""

    positive_queries = metrics.get("n_positive_queries")
    negative_queries = metrics.get("n_negative_queries")
    if positive_queries is not None and negative_queries is not None:
        if int(positive_queries) == 0 or int(negative_queries) == 0:
            return "n/a"
    return _format_metric_float(metrics["balanced_accuracy"])


def _metric_count_cell(metrics: dict[str, Any], key: str) -> str:
    """Return a query-count cell for optional metric count fields."""

    value = metrics.get(key)
    if value is None:
        return "n/a"
    return str(int(value))


def _optional_metric_float_cell(value: Any) -> str:
    """Return a formatted float cell, preserving missing values as n/a."""

    if value is None:
        return "n/a"
    return _format_metric_float(value)


def _count_from_mapping(values: Mapping[str, Any], key: str) -> int:
    """Return an integer count from a JSON-style mapping."""

    value = values.get(key, 0)
    if value is None:
        return 0
    return int(value)


def _metric_breakdown_row(label: str, metrics: dict[str, Any]) -> list[str]:
    """Return one selected-gate breakdown table row."""

    return [
        str(label),
        str(int(metrics.get("n_queries", metrics.get("queries", 0)))),
        _metric_count_cell(metrics, "n_positive_queries"),
        _metric_count_cell(metrics, "n_negative_queries"),
        _metric_balanced_accuracy_cell(metrics),
        _format_metric_float(metrics["error_rate"]),
        str(int(metrics.get("false_abstain", 0))),
        str(int(metrics.get("false_link", 0))),
        str(int(metrics.get("wrong_candidate_link", 0))),
    ]


def _metric_factor_row(factor: str, group: str, metrics: dict[str, Any]) -> list[str]:
    """Return one requested factor breakdown table row."""

    return [
        str(factor),
        str(group),
        str(int(metrics.get("n_queries", metrics.get("queries", 0)))),
        _metric_count_cell(metrics, "n_positive_queries"),
        _metric_count_cell(metrics, "n_negative_queries"),
        _format_metric_float(metrics["error_rate"]),
        str(int(metrics.get("false_abstain", 0))),
        str(int(metrics.get("false_link", 0))),
        str(int(metrics.get("wrong_candidate_link", 0))),
    ]


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Render a markdown table from string cells."""

    def cell(value: str) -> str:
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return lines


def _classic_gate_bucket_table_rows(summary: dict[str, Any], breakdowns: dict[str, Any]) -> list[list[str]]:
    """Return selected-gate calibration bucket rows for the promoted bucketed gate."""

    abstain_rule = summary.get("abstain_rule")
    split_summary = summary.get("stratified_eval_test_split")
    if not isinstance(abstain_rule, dict) or not isinstance(split_summary, dict):
        return []
    score_thresholds = abstain_rule.get("bucketed_score_thresholds")
    if not isinstance(score_thresholds, dict):
        return []
    margin_thresholds = abstain_rule.get("bucketed_margin_thresholds")
    if not isinstance(margin_thresholds, dict):
        margin_thresholds = {}

    promoted_gate = abstain_rule.get("promoted_stratified_gate")
    if isinstance(promoted_gate, dict):
        calibration_splits_value = promoted_gate.get("calibration_splits", ["calibration_fit", "calibration_check"])
        if isinstance(calibration_splits_value, str) or not isinstance(calibration_splits_value, Sequence):
            calibration_splits = ["calibration_fit", "calibration_check"]
        else:
            calibration_splits = [str(split) for split in calibration_splits_value]
        fit_split = calibration_splits[0] if calibration_splits else "calibration_fit"
        check_split = calibration_splits[1] if len(calibration_splits) > 1 else ""
        test_split = str(promoted_gate.get("test_split", "test"))
    else:
        fit_split = "calibration_fit"
        check_split = "calibration_check"
        test_split = "test"

    training_summary = summary.get("training_summary")
    if not isinstance(training_summary, dict):
        training_summary = {}
    train_query_counts = training_summary.get("gate_bucket_query_counts")
    if not isinstance(train_query_counts, dict):
        train_query_counts = {}
    train_row_counts = training_summary.get("gate_bucket_row_counts")
    if not isinstance(train_row_counts, dict):
        train_row_counts = {}

    split_counts = split_summary.get("gate_bucket_split_counts")
    if not isinstance(split_counts, dict):
        split_counts = {}
    test_breakdowns = breakdowns.get("gate_bucket")
    if not isinstance(test_breakdowns, dict):
        test_breakdowns = {}

    rows: list[list[str]] = []
    for bucket in _TOTAL_ERROR_SCORE_BUCKETS:
        bucket_split_counts = split_counts.get(bucket)
        if not isinstance(bucket_split_counts, dict):
            bucket_split_counts = {}
        fit_count = _count_from_mapping(bucket_split_counts, fit_split)
        check_count = _count_from_mapping(bucket_split_counts, check_split)
        test_metrics = test_breakdowns.get(bucket)
        if not isinstance(test_metrics, dict):
            test_metrics = {}
        test_count = _count_from_mapping(test_metrics, "n_queries") or _count_from_mapping(
            bucket_split_counts,
            test_split,
        )
        calibration_count = fit_count + check_count
        margin_threshold = margin_thresholds.get(bucket) if bucket in _TOTAL_ERROR_MARGIN_BUCKETS else None
        rows.append(
            [
                bucket,
                _optional_metric_float_cell(score_thresholds.get(bucket)),
                _optional_metric_float_cell(margin_threshold),
                str(_count_from_mapping(train_query_counts, bucket)),
                str(_count_from_mapping(train_row_counts, bucket)),
                str(fit_count),
                str(check_count),
                str(calibration_count),
                str(test_count),
                str(_count_from_mapping(test_metrics, "n_positive_queries")),
                str(_count_from_mapping(test_metrics, "n_negative_queries")),
                str(_count_from_mapping(test_metrics, "errors")),
                "n/a" if test_count == 0 else _format_metric_float(test_metrics.get("error_rate", 0.0)),
                str(_count_from_mapping(test_metrics, "false_abstain")),
                str(_count_from_mapping(test_metrics, "false_link")),
                str(_count_from_mapping(test_metrics, "wrong_candidate_link")),
            ]
        )
    return rows


def format_classic_selected_gate_tables(summary: dict[str, Any]) -> str:
    """Return the selected-gate stratified-test breakdown tables."""

    split_summary = summary.get("stratified_eval_test_split")
    if not isinstance(split_summary, dict):
        return ""
    breakdowns = split_summary.get("test_breakdowns")
    if not isinstance(breakdowns, dict):
        return ""

    lines: list[str] = []
    bucket_rows = _classic_gate_bucket_table_rows(summary, breakdowns)
    if bucket_rows:
        lines.extend(["## By Calibration Bucket, Selected Gate", ""])
        lines.extend(
            _markdown_table(
                [
                    "bucket",
                    "score threshold",
                    "margin threshold",
                    "train queries",
                    "train rows",
                    "calibration fit",
                    "calibration check",
                    "calibration total",
                    "test queries",
                    "test positive queries",
                    "test negative queries",
                    "test errors",
                    "test error rate",
                    "false abstain",
                    "false link",
                    "wrong link",
                ],
                bucket_rows,
            )
        )
        lines.append("")

    lines.extend(["## By Dataset Slice, Selected Gate", ""])
    source_breakdown = dict(breakdowns.get("source_key", {}))
    source_rows = [
        _metric_breakdown_row(str(slice_name), dict(metrics))
        for slice_name, metrics in sorted(source_breakdown.items(), key=lambda item: str(item[0]))
    ]
    lines.extend(
        _markdown_table(
            [
                "slice",
                "queries",
                "positive queries",
                "negative queries",
                "BA",
                "error rate",
                "false abstain",
                "false link",
                "wrong link",
            ],
            source_rows,
        )
    )
    lines.extend(["", "BA is n/a for single-class slices.", ""])

    lines.extend(["## Requested Factor Breakdowns", ""])
    factor_rows: list[list[str]] = []
    for factor in (
        "has_positive_candidate",
        "positive_rank_bucket",
        "first_name_bucket",
        "multiple_candidates",
        "source_stratum",
    ):
        factor_breakdown = breakdowns.get(factor)
        if not isinstance(factor_breakdown, dict):
            continue
        for group_name, metrics in sorted(factor_breakdown.items(), key=lambda item: str(item[0])):
            factor_rows.append(_metric_factor_row(factor, str(group_name), dict(metrics)))
    lines.extend(
        _markdown_table(
            [
                "factor",
                "group",
                "queries",
                "positive queries",
                "negative queries",
                "error rate",
                "false abstain",
                "false link",
                "wrong link",
            ],
            factor_rows,
        )
    )
    return "\n".join(lines) + "\n"


def _evaluate_classic_stratified_eval_test_split(
    bundle: OfficialBundle,
    spec: dict[str, Any],
    split_spec: dict[str, Any],
    model: LGBMClassifier,
    feature_columns: tuple[str, ...],
    *,
    score_threshold: float,
    margin_threshold: float,
    single_candidate_score_threshold: float | None = None,
    bucketed_score_thresholds: dict[str, float] | None = None,
    bucketed_margin_threshold: float | None = None,
    bucketed_margin_thresholds: dict[str, float] | None = None,
    scored_choices: tuple[pd.DataFrame, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Score the promoted stratified calibration/test split with the active gate."""

    if scored_choices is None:
        choices, assignments = _score_classic_stratified_eval_test_choices(
            bundle,
            spec,
            split_spec,
            model,
            feature_columns,
        )
    else:
        choices, assignments = scored_choices

    predictions = _apply_classic_gate(
        choices,
        score_threshold=score_threshold,
        margin_threshold=margin_threshold,
        single_candidate_score_threshold=single_candidate_score_threshold,
        bucketed_score_thresholds=bucketed_score_thresholds,
        bucketed_margin_threshold=bucketed_margin_threshold,
        bucketed_margin_thresholds=bucketed_margin_thresholds,
    )
    return _summarize_classic_stratified_predictions(predictions, assignments, split_spec)


def _score_eval_candidate_rows(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    include_margin: bool,
    limits: tuple[int, ...] = (5, 25),
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if len(probabilities) != len(df):
        raise ValueError(f"probabilities length must match df rows: {len(probabilities)} != {len(df)}")
    for limit in sorted(set(int(limit) for limit in limits)):
        mask = (df["retrieval_rank"] <= limit).to_numpy(dtype=bool)
        limited = df.loc[mask].copy()
        limited_probabilities = probabilities[np.flatnonzero(mask)]
        choices = _score_query_choices(
            limited.rename(columns={"query_group_id": "query_case_id"}),
            limited_probabilities,
            query_id_column="query_case_id",
            include_margin=include_margin,
        )
        choices["retrieval_rank_limit"] = limit
        frames.append(choices)
    return pd.concat(frames, ignore_index=True)


def run_classic(
    bundle: OfficialBundle,
    output_dir: Path,
    *,
    save_artifact_to: Path | None = None,
    artifact_audit_metadata: Mapping[str, Any] | None = None,
    required_rust_capabilities: Sequence[str] = INCREMENTAL_LINKING_RUST_CAPABILITIES,
    n_jobs: int = DEFAULT_CLASSIC_N_JOBS,
) -> dict[str, Any]:
    """Fit, calibrate, and evaluate the official classic pipeline."""

    output_dir.mkdir(parents=True, exist_ok=True)
    spec = bundle.models["classic"]
    feature_columns = tuple(spec["feature_columns"])
    monotone_constraints = _resolve_classic_monotone_constraints(spec, feature_columns)
    train_df = _read_csv(_resolve_path(bundle, spec["train_path"]), compression="gzip")
    train_df["retrieval_rank"] = pd.to_numeric(train_df["retrieval_rank"], errors="coerce")
    train_df = train_df[train_df["retrieval_rank"] <= 25].copy()
    train_df["label"] = pd.to_numeric(train_df["label"], errors="coerce").fillna(0).astype(np.int8)
    holdout_query_group_ids, holdout_base_group_ids, holdout_sources = _read_classic_holdout_identity_sets(
        bundle,
        spec,
    )
    train_df, train_holdout_filter_summary = _apply_classic_train_holdout_filter(
        train_df,
        holdout_query_group_ids=holdout_query_group_ids,
        holdout_base_group_ids=holdout_base_group_ids,
        holdout_sources=holdout_sources,
    )
    train_df, train_filter_summary = _apply_classic_train_row_cap(
        train_df,
        rule_name=spec.get("train_row_cap_rule"),
        min_train_limit=(int(spec["train_row_cap_min_limit"]) if "train_row_cap_min_limit" in spec else None),
    )
    training_gate_bucket_summary = _summarize_training_gate_buckets(train_df)
    train_matrix = _classic_feature_matrix(train_df, feature_columns).to_numpy(dtype=np.float32)
    train_labels = train_df["label"].to_numpy(dtype=np.int8, copy=False)
    group_sizes = train_df["query_group_id"].astype(str).value_counts(sort=False)
    sample_weight = (1.0 / train_df["query_group_id"].astype(str).map(group_sizes).astype(float)).to_numpy(
        dtype=np.float32
    )
    model = _build_classic_classifier(
        spec["best_params"],
        monotone_constraints=monotone_constraints,
        n_jobs=n_jobs,
    )
    started = perf_counter()
    model.fit(train_matrix, train_labels, sample_weight=sample_weight)
    train_seconds = float(perf_counter() - started)

    gate_source_path = _resolve_path(bundle, spec.get("classic_gate_source_path", spec["hwang_eval_path"]))
    gate_source_eval = _read_csv(gate_source_path, compression="gzip")
    gate_source_probabilities = model.predict_proba(_classic_feature_matrix(gate_source_eval, feature_columns))[:, 1]
    calibration_limit = int(spec.get("classic_gate_calibration_retrieval_limit", 50))
    gate_source_query_choices = _score_eval_candidate_rows(
        gate_source_eval,
        gate_source_probabilities,
        include_margin=True,
        limits=(calibration_limit,),
    )

    internal_eval_groups = set(
        _read_csv(_resolve_path(bundle, spec["classic_gate_internal_eval_base_groups_path"]))["base_group_id"].astype(
            str
        )
    )
    internal_eval_rows = gate_source_query_choices[
        (gate_source_query_choices["retrieval_rank_limit"] == calibration_limit)
        & (gate_source_query_choices["base_group_id"].isin(internal_eval_groups))
    ].copy()
    promoted_gate_config = _promoted_stratified_gate_spec(spec)
    stratified_scored_choices: tuple[pd.DataFrame, pd.DataFrame] | None = None
    promoted_gate_summary: dict[str, Any] | None = None
    if promoted_gate_config is None:
        raise ValueError("classic.promoted_stratified_gate is required")
    if spec.get("stratified_eval_test_split") is None:
        raise ValueError("classic.promoted_stratified_gate requires classic.stratified_eval_test_split")

    split_spec = dict(spec["stratified_eval_test_split"])
    stratified_scored_choices = _score_classic_stratified_eval_test_choices(
        bundle,
        spec,
        split_spec,
        model,
        feature_columns,
    )
    selected_gate_result = _fit_promoted_stratified_total_error_gate(
        stratified_scored_choices[0],
        promoted_gate_config,
    )
    selected_gate = selected_gate_result["gate"]
    bucketed_score_thresholds = dict(selected_gate.score_thresholds)
    bucketed_margin_thresholds = dict(selected_gate.margin_thresholds)
    bucketed_margin_threshold = None
    score_threshold = float(bucketed_score_thresholds["multi_candidate|multi_letter_first"])
    margin_threshold = float(bucketed_margin_thresholds["multi_candidate|multi_letter_first"])
    single_candidate_score_threshold = float(bucketed_score_thresholds["single_candidate|single_letter_first"])
    calibration_splits = tuple(str(split) for split in promoted_gate_config["calibration_splits"])
    calibration_split_label = "+".join(calibration_splits)
    calibration_metrics = {
        "split": calibration_split_label,
        "score_threshold": score_threshold,
        "margin_threshold": margin_threshold,
        **dict(selected_gate_result["calibration_metrics"]),
    }
    calibration_predictions = _apply_total_error_gate(
        stratified_scored_choices[0][stratified_scored_choices[0]["split"].astype(str).isin(calibration_splits)].copy(),
        selected_gate,
    )
    single_candidate_predictions = calibration_predictions[
        (pd.to_numeric(calibration_predictions["has_runner_up"], errors="coerce").fillna(0).astype(int) == 0)
        | calibration_predictions["score_margin"].isna()
    ].copy()
    single_candidate_calibration_metrics = {
        "single_candidate_score_threshold": single_candidate_score_threshold,
        **_summarize_predictions(single_candidate_predictions),
    }
    promoted_gate_summary = {
        "mode": str(promoted_gate_config["mode"]),
        "calibration_splits": list(calibration_splits),
        "test_split": str(promoted_gate_config["test_split"]),
        "fixed_grid_step": float(promoted_gate_config["fixed_grid_step"]),
        "threshold_grid_points": int(selected_gate_result["threshold_grid_points"]),
        "selection_metric": str(promoted_gate_config["selection_metric"]),
        "error_weights": dict(promoted_gate_config["error_weights"]),
        "selected_gate": {
            "name": selected_gate.name,
            "score_thresholds": dict(selected_gate.score_thresholds),
            "margin_thresholds": dict(selected_gate.margin_thresholds),
        },
        "selection_key": dict(selected_gate_result["selection_key"]),
        "calibration_metrics": dict(selected_gate_result["calibration_metrics"]),
        "calibration_split_metrics": dict(selected_gate_result["calibration_split_metrics"]),
        "bucket_metrics": dict(selected_gate_result["bucket_metrics"]),
    }

    s2and_eval = _read_csv(_resolve_path(bundle, spec["s2and_eval_path"]), compression="gzip")
    s2and_probabilities = model.predict_proba(_classic_feature_matrix(s2and_eval, feature_columns))[:, 1]
    s2and_query_choices = _score_eval_candidate_rows(s2and_eval, s2and_probabilities, include_margin=True)
    s2and_eval_summary = _evaluate_scored_windows(
        s2and_query_choices,
        score_threshold=score_threshold,
        margin_threshold=margin_threshold,
        single_candidate_score_threshold=single_candidate_score_threshold,
        bucketed_score_thresholds=bucketed_score_thresholds,
        bucketed_margin_threshold=bucketed_margin_threshold,
        bucketed_margin_thresholds=bucketed_margin_thresholds,
    )
    hwang_eval = _read_csv(_resolve_path(bundle, spec["hwang_eval_path"]), compression="gzip")
    hwang_probabilities = model.predict_proba(_classic_feature_matrix(hwang_eval, feature_columns))[:, 1]
    hwang_query_choices = _score_eval_candidate_rows(hwang_eval, hwang_probabilities, include_margin=True)
    optional_eval_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset_name, path_like in _iter_extra_eval_paths(spec):
        eval_df = _read_csv(_resolve_path(bundle, path_like), compression="gzip")
        eval_probabilities = model.predict_proba(_classic_feature_matrix(eval_df, feature_columns))[:, 1]
        eval_query_choices = _score_eval_candidate_rows(eval_df, eval_probabilities, include_margin=True)
        optional_eval_summaries[_summary_key_for_eval_dataset(dataset_name)] = _evaluate_scored_windows(
            eval_query_choices,
            score_threshold=score_threshold,
            margin_threshold=margin_threshold,
            single_candidate_score_threshold=single_candidate_score_threshold,
            bucketed_score_thresholds=bucketed_score_thresholds,
            bucketed_margin_threshold=bucketed_margin_threshold,
            bucketed_margin_thresholds=bucketed_margin_thresholds,
        )

    override_path = spec.get("hwang_clean_override_path")
    override_df = _read_csv(_resolve_path(bundle, str(override_path))) if override_path else None
    hwang_eval_summary = _evaluate_scored_windows(
        hwang_query_choices,
        score_threshold=score_threshold,
        margin_threshold=margin_threshold,
        single_candidate_score_threshold=single_candidate_score_threshold,
        override_df=override_df,
        bucketed_score_thresholds=bucketed_score_thresholds,
        bucketed_margin_threshold=bucketed_margin_threshold,
        bucketed_margin_thresholds=bucketed_margin_thresholds,
    )
    internal_eval_summary = _score_abstain_rule(
        internal_eval_rows,
        score_threshold=score_threshold,
        margin_threshold=margin_threshold,
        single_candidate_score_threshold=single_candidate_score_threshold,
        bucketed_score_thresholds=bucketed_score_thresholds,
        bucketed_margin_threshold=bucketed_margin_threshold,
        bucketed_margin_thresholds=bucketed_margin_thresholds,
    )
    stratified_eval_test_summary = None
    if spec.get("stratified_eval_test_split") is not None:
        stratified_eval_test_summary = _evaluate_classic_stratified_eval_test_split(
            bundle,
            spec,
            dict(spec["stratified_eval_test_split"]),
            model,
            feature_columns,
            score_threshold=score_threshold,
            margin_threshold=margin_threshold,
            single_candidate_score_threshold=single_candidate_score_threshold,
            bucketed_score_thresholds=bucketed_score_thresholds,
            bucketed_margin_threshold=bucketed_margin_threshold,
            bucketed_margin_thresholds=bucketed_margin_thresholds,
            scored_choices=stratified_scored_choices,
        )

    hwang_cleaned_eval = {
        f"w{limit}": {
            "cleaned_balanced_accuracy": window_summary["overall"]["balanced_accuracy"],
            "cleaned_positive_recall": window_summary["overall"]["positive_recall"],
            "cleaned_negative_recall": window_summary["overall"]["negative_recall"],
        }
        for limit, window_summary in hwang_eval_summary.items()
    }

    summary = {
        "model": "classic",
        "training_summary": {
            "rows": int(len(train_df)),
            "queries": int(train_df["query_group_id"].astype(str).nunique()),
            "positive_rows": int(train_df["label"].sum()),
            "gate_bucket_query_counts": training_gate_bucket_summary["query_counts"],
            "gate_bucket_row_counts": training_gate_bucket_summary["row_counts"],
            "elapsed_seconds": train_seconds,
            "train_holdout_filter_summary": train_holdout_filter_summary,
            "train_filter_summary": train_filter_summary,
        },
        "abstain_rule": {
            "score_threshold": score_threshold,
            "margin_threshold": margin_threshold,
            "single_candidate_score_threshold": single_candidate_score_threshold,
            "calibration_mode": "promoted_stratified_full_calibration_fixed_grid_4score_2margin",
            "promoted_stratified_gate": promoted_gate_summary,
            "bucketed_score_thresholds": bucketed_score_thresholds,
            "bucketed_margin_threshold": bucketed_margin_threshold,
            "bucketed_margin_thresholds": bucketed_margin_thresholds,
            "calibration_retrieval_limit": calibration_limit,
            "calibration_metrics": calibration_metrics,
            "single_candidate_calibration_metrics": single_candidate_calibration_metrics,
            "internal_eval_metrics": internal_eval_summary,
        },
        "overall_s2and_eval": s2and_eval_summary,
        "hwang_cleaned_eval": hwang_cleaned_eval,
    }
    if stratified_eval_test_summary is not None:
        summary["stratified_eval_test_split"] = stratified_eval_test_summary
    manual_holdout_path = spec.get("manual_holdout_candidates_path")
    if manual_holdout_path:
        manual_holdout = _read_csv(_resolve_path(bundle, manual_holdout_path), low_memory=False)
        manual_probabilities = model.predict_proba(_classic_feature_matrix(manual_holdout, feature_columns))[:, 1]
        summary["manual_holdout"] = _evaluate_classic_manual_holdout(
            manual_holdout,
            manual_probabilities,
            score_threshold=score_threshold,
            margin_threshold=margin_threshold,
            single_candidate_score_threshold=single_candidate_score_threshold,
            bucketed_score_thresholds=bucketed_score_thresholds,
            bucketed_margin_threshold=bucketed_margin_threshold,
            bucketed_margin_thresholds=bucketed_margin_thresholds,
        )
    for summary_key, eval_summary in optional_eval_summaries.items():
        summary[summary_key] = eval_summary
    selected_gate_tables = format_classic_selected_gate_tables(summary)
    if selected_gate_tables:
        selected_gate_tables_path = output_dir / "selected_gate_tables.md"
        selected_gate_tables_path.write_text(selected_gate_tables, encoding="utf-8")
        summary["selected_gate_tables_path"] = str(selected_gate_tables_path.relative_to(output_dir))
    if save_artifact_to is not None:
        fixture_source = gate_source_eval.head(5)
        if len(fixture_source) == 0:
            fixture_matrix = train_matrix[:5]
        else:
            fixture_matrix = _classic_feature_matrix(fixture_source, feature_columns).to_numpy(dtype=np.float32)
        artifact_metadata = save_incremental_linking_artifact(
            model,
            Path(save_artifact_to),
            feature_columns=feature_columns,
            retrieval_top_k=25,
            gate_config={
                "bucketed_score_thresholds": bucketed_score_thresholds,
                "bucketed_margin_thresholds": bucketed_margin_thresholds,
                "calibration_mode": summary["abstain_rule"]["calibration_mode"],
            },
            prediction_fixture_matrix=fixture_matrix,
            required_rust_capabilities=required_rust_capabilities,
            audit_metadata=artifact_audit_metadata,
        )
        summary["artifact"] = {
            "path": str(Path(save_artifact_to)),
            "schema_version": artifact_metadata.schema_version,
            "feature_schema_digest": artifact_metadata.feature_schema_digest,
            "production_contract_digest": artifact_metadata.production_contract_digest,
            "retrieval_stack_digest": artifact_metadata.retrieval_stack_digest,
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


PROMOTED_PAIRWISE_COLUMNS = promoted_pairwise_aggregate_columns()
PROMOTED_NON_PAIRWISE_COLUMNS = tuple(PROMOTED_NON_PAIRWISE_FEATURE_COLUMNS)
SUPPORTED_PROMOTED_FEATURE_COLUMNS = frozenset(PROMOTED_NON_PAIRWISE_COLUMNS) | frozenset(PROMOTED_PAIRWISE_COLUMNS)
FROZEN_RETRIEVAL_POLICY = FROZEN_BEST_RUST_HYBRID_CENTROID_POLICY
FROZEN_RETRIEVAL_POLICY_NAME = FROZEN_BEST_RUST_HYBRID_CENTROID_POLICY_NAME
WEIGHTED_ERROR_WEIGHTS = {
    "false_abstain_error_rate": float(_DEFAULT_PROMOTED_GATE_ERROR_WEIGHTS["false_abstain"]),
    "false_link_error_rate": float(_DEFAULT_PROMOTED_GATE_ERROR_WEIGHTS["false_link"]),
    "wrong_link_error_rate": float(_DEFAULT_PROMOTED_GATE_ERROR_WEIGHTS["wrong_candidate_link"]),
}
NAN_POLICY_CHOICES = ("preserve", "zero")
ROW_NAN_POLICY_CHOICES = ("finite", "semantic")
