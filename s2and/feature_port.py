import logging
import math
import os
import threading
import time
import weakref
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from s2and.consts import CLUSTER_SEEDS_LOOKUP
from s2and.data import ANDData
from s2and.runtime import (
    detect_rust_runtime_capabilities,
    load_s2and_rust_extension,
    min_supported_rust_extension_version_string,
)
from s2and.rust_calls import (
    build_block_upper_triangle_feature_matrix_indexed_rust,
    build_linker_pair_aggregate_stats_arrays_rust,
    build_linker_pair_distance_accumulators_rust,
    build_linker_pair_features_and_aggregate_stats_arrays_rust,
    build_linker_pair_features_and_aggregate_stats_indexed_rust,
    build_pair_feature_matrix_rust,
    featurize_pair_rust,
    get_constraint_labels_index_arrays_rust,
    get_constraint_rust,
    get_constraints_block_upper_triangle_indexed_rust,
    get_constraints_matrix_indexed_rust,
    get_constraints_matrix_rust,
    update_rust_cluster_seeds,
)
from s2and.rust_lifecycle import (
    RustBuildPath,
    RustJsonIngestContract,
    RustLifecyclePolicy,
    build_rust_json_ingest_contract,
)
from s2and.thread_config import resolve_n_jobs

# Treat extension as Any for typing; it is optional.
_s2and_rust: Any | None


_s2and_rust = load_s2and_rust_extension()
s2and_rust: Any | None = _s2and_rust

logger = logging.getLogger("s2and")
_S2AND_RUST_LOAD_LOCK = threading.Lock()


class _CacheEntry:
    """Composite cache entry for one Rust featurizer build option."""

    __slots__ = ("featurizer", "build_count")

    def __init__(
        self,
        featurizer: Any,
        build_count: int = 0,
    ):
        self.featurizer = featurizer
        self.build_count = build_count


class _InFlightFeaturizerBuild:
    """Tracks a single in-flight Rust featurizer build for a dataset."""

    __slots__ = ("event", "error", "build_path")

    def __init__(self, build_path: RustBuildPath) -> None:
        self.event = threading.Event()
        self.error: Exception | None = None
        self.build_path = build_path


@dataclass(frozen=True, slots=True)
class _RustNameCountsOverlay:
    first: float | None
    first_last: float | None
    last: float | None
    last_first_initial: float | None


@dataclass(frozen=True, slots=True)
class _RustSignatureNameCountsOverlay:
    author_info_name_counts: _RustNameCountsOverlay


_RustFeaturizerCacheKey = tuple[RustBuildPath, bool, int]

# Single WeakKeyDictionary eliminates desync risk between separate weak dicts.
_RUST_FEATURIZER_CACHE: "weakref.WeakKeyDictionary[ANDData, dict[_RustFeaturizerCacheKey, _CacheEntry]]" = (
    weakref.WeakKeyDictionary()
)
_RUST_FEATURIZER_CACHE_LOCK = threading.Lock()
_RUST_FEATURIZER_INFLIGHT_BUILDS: weakref.WeakKeyDictionary[
    ANDData, dict[_RustFeaturizerCacheKey, _InFlightFeaturizerBuild]
] = weakref.WeakKeyDictionary()
_RUST_BUILD_ERROR = "s2and_rust extension not built. Build with: maturin develop -m s2and_rust/Cargo.toml"
# Default remains "legacy_compat" until canonical artifacts (name counts, name tuples,
# ORCID prefix counts) are regenerated per docs/normalization_migration.md.
DEFAULT_NORMALIZATION_VERSION = "legacy_compat"
NORMALIZATION_VERSION_ENV = "S2AND_NORMALIZATION_VERSION"
RUST_FEATURIZER_EMPTY_WAIT_MAX_RETRIES = 3
RUST_FEATURIZER_EMPTY_WAIT_BACKOFF_SECONDS = 0.01


def _require_rust_runtime() -> Any:
    rust_module = _ensure_s2and_rust_loaded()
    if rust_module is None:
        raise RuntimeError(_RUST_BUILD_ERROR)
    capabilities = detect_rust_runtime_capabilities(extension_module=rust_module)
    if not capabilities.core_runtime_available:
        raise RuntimeError(f"Rust runtime unavailable: {capabilities.reason}")
    return rust_module


def _ensure_s2and_rust_loaded() -> Any | None:
    global s2and_rust
    if s2and_rust is not None:
        return s2and_rust
    with _S2AND_RUST_LOAD_LOCK:
        if s2and_rust is None:
            s2and_rust = load_s2and_rust_extension()
    return s2and_rust


def _dataset_name_for_logs(dataset: Any) -> str:
    name = getattr(dataset, "name", None)
    return str(name) if name is not None else f"<unnamed:{id(dataset)}>"


def _dataset_mode_for_logs(dataset: Any) -> str:
    mode = str(getattr(dataset, "mode", "")).strip()
    return mode if mode else "unknown"


def _runtime_callsite_for_logs(dataset: Any, runtime_context: Any | None = None) -> tuple[str, str]:
    context = runtime_context if runtime_context is not None else getattr(dataset, "runtime_context", None)
    operation = str(getattr(context, "operation", "unknown"))
    run_id_raw = getattr(context, "run_id", None)
    run_id = str(run_id_raw) if run_id_raw is not None else f"dataset-{id(dataset)}"
    return operation, run_id


def _rust_featurizer_empty_wait_max_retries() -> int:
    return max(1, int(RUST_FEATURIZER_EMPTY_WAIT_MAX_RETRIES))


def _rust_featurizer_empty_wait_backoff_seconds() -> float:
    return max(0.0, float(RUST_FEATURIZER_EMPTY_WAIT_BACKOFF_SECONDS))


def _rust_featurizer_cache_key(
    requested_build_path: RustBuildPath,
    allow_normalization_version_mismatch: bool,
    cluster_seeds_version: int = 0,
) -> _RustFeaturizerCacheKey:
    return requested_build_path, bool(allow_normalization_version_mismatch), int(cluster_seeds_version)


def _cluster_seeds_version_for_cache(dataset: Any) -> int:
    missing = object()
    raw_version = getattr(dataset, "_cluster_seeds_version", missing)
    if raw_version is missing:
        return 0
    return int(raw_version)


def _rust_featurizer_cache_entry_count_locked() -> int:
    return sum(len(entries) for entries in _RUST_FEATURIZER_CACHE.values())


def _increment_rust_featurizer_build_count(
    dataset: ANDData,
    cache_key: _RustFeaturizerCacheKey | None = None,
) -> int:
    resolved_cache_key = cache_key or _rust_featurizer_cache_key("from_dataset", False)
    with _RUST_FEATURIZER_CACHE_LOCK:
        entries = _RUST_FEATURIZER_CACHE.get(dataset)
        entry = None if entries is None else entries.get(resolved_cache_key)
        if entry is not None:
            entry.build_count += 1
            return int(entry.build_count)
        # Entry not yet inserted — caller will set build_count on insertion.
        return 1


def _rust_featurizer_build_count(
    dataset: ANDData,
    cache_key: _RustFeaturizerCacheKey | None = None,
) -> int:
    resolved_cache_key = cache_key or _rust_featurizer_cache_key("from_dataset", False)
    with _RUST_FEATURIZER_CACHE_LOCK:
        entries = _RUST_FEATURIZER_CACHE.get(dataset)
        entry = None if entries is None else entries.get(resolved_cache_key)
        return int(entry.build_count) if entry is not None else 0


def _dataset_rust_lifecycle_policy(dataset: Any) -> RustLifecyclePolicy | None:
    policy = getattr(dataset, "rust_lifecycle_policy", None)
    if isinstance(policy, RustLifecyclePolicy):
        return policy
    return None


def _rust_name_counts_artifact_path() -> str | None:
    configured = os.environ.get("S2AND_RUST_NAME_COUNTS_JSON", "").strip()
    return configured or None


def _name_count_value_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float):
        return not math.isnan(value)
    if isinstance(value, np.floating):
        return not math.isnan(float(value))
    return True


def _extract_overlay_name_counts_values(
    counts: Any,
) -> tuple[float | None, float | None, float | None, float | None] | None:
    if counts is None:
        return None

    if all(hasattr(counts, field) for field in ("first", "last", "first_last", "last_first_initial")):
        return (
            counts.first,
            counts.last,
            counts.first_last,
            counts.last_first_initial,
        )

    if isinstance(counts, tuple | list) and len(counts) >= 4:
        return (
            counts[0],
            counts[1],
            counts[2],
            counts[3],
        )

    return None


def _build_signature_name_counts_overlay_entry(signature: Any) -> _RustSignatureNameCountsOverlay | None:
    counts = getattr(signature, "author_info_name_counts", None)
    values = _extract_overlay_name_counts_values(counts)
    if values is None:
        return None
    first, last, first_last, last_first_initial = values
    if not any(_name_count_value_present(v) for v in values):
        return None
    return _RustSignatureNameCountsOverlay(
        author_info_name_counts=_RustNameCountsOverlay(
            first=first,
            first_last=first_last,
            last=last,
            last_first_initial=last_first_initial,
        )
    )


def _signature_has_materialized_name_counts(signature: Any) -> bool:
    return _build_signature_name_counts_overlay_entry(signature) is not None


def _signature_name_counts_overlay_payload_from_dataset(dataset: Any) -> tuple[int, int, dict[str, Any]]:
    signatures = getattr(dataset, "signatures", None)
    if signatures is None:
        return 0, 0, {}
    dataset_name_for_logs = _dataset_name_for_logs(dataset)
    try:
        signatures_total = int(len(signatures))
    except (TypeError, AttributeError) as length_exc:
        logger.warning(
            "Rust JSON ingest: failed to compute signatures length for name-count overlay dataset=%s: %s",
            dataset_name_for_logs,
            length_exc,
        )
        signatures_total = 0

    payload: dict[str, Any] = {}
    try:
        signature_items = signatures.items()
    except (TypeError, AttributeError) as items_exc:
        logger.exception(
            "Rust JSON ingest: failed to iterate signatures for name-count overlay dataset=%s",
            dataset_name_for_logs,
        )
        raise RuntimeError(
            "Rust JSON ingest failed while iterating signatures for name-count overlay payload "
            f"(dataset={dataset_name_for_logs})"
        ) from items_exc

    try:
        for signature_id, signature in signature_items:
            overlay_entry = _build_signature_name_counts_overlay_entry(signature)
            if overlay_entry is not None:
                payload[str(signature_id)] = overlay_entry
    except (TypeError, AttributeError) as overlay_exc:
        logger.exception(
            "Rust JSON ingest: failed to materialize signature name-count overlay payload dataset=%s",
            dataset_name_for_logs,
        )
        raise RuntimeError(
            "Rust JSON ingest failed while materializing signature name-count overlay payload "
            f"(dataset={dataset_name_for_logs})"
        ) from overlay_exc
    return signatures_total, len(payload), payload


def _resolve_json_ingest_name_counts_plan(
    dataset: Any,
    *,
    rust_featurizer_cls: Any,
    signature_name_counts_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the name-count source plan for Rust JSON ingest."""

    rust_can_overlay_signature_counts = hasattr(rust_featurizer_cls, "update_signature_name_counts")
    if signature_name_counts_payload is not None:
        dataset_signature_counts_payload = {}
        for signature_id, signature in signature_name_counts_payload.items():
            overlay_entry = _build_signature_name_counts_overlay_entry(signature)
            if overlay_entry is not None:
                dataset_signature_counts_payload[str(signature_id)] = overlay_entry
        signatures_with_counts = len(dataset_signature_counts_payload)
        signatures_total = signatures_with_counts
        try:
            signatures_total = int(len(getattr(dataset, "signatures", {})))
        except Exception:
            pass
    else:
        signatures_total, signatures_with_counts, dataset_signature_counts_payload = (
            _signature_name_counts_overlay_payload_from_dataset(dataset)
        )
    dataset_has_signature_counts = signatures_with_counts > 0
    configured_name_counts_path = _rust_name_counts_artifact_path()
    artifact_configured = configured_name_counts_path is not None
    use_dataset_signature_counts = dataset_has_signature_counts and rust_can_overlay_signature_counts
    name_counts_source = "none"
    name_counts_path = None
    if use_dataset_signature_counts:
        name_counts_source = "dataset"
    elif not dataset_has_signature_counts and artifact_configured:
        name_counts_source = "artifact"
        name_counts_path = configured_name_counts_path
    return {
        "dataset_signature_counts_payload": dataset_signature_counts_payload,
        "signatures_total": int(signatures_total),
        "signatures_with_counts": int(signatures_with_counts),
        "dataset_has_signature_counts": bool(dataset_has_signature_counts),
        "artifact_configured": bool(artifact_configured),
        "rust_can_overlay_signature_counts": bool(rust_can_overlay_signature_counts),
        "name_counts_source": str(name_counts_source),
        "name_counts_path": name_counts_path,
    }


def inspect_json_ingest_name_counts_source(
    dataset: Any,
    *,
    signature_name_counts_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Inspect the name-count source Rust JSON ingest would use for ``dataset``."""

    rust_module = _require_rust_runtime()
    rust_featurizer_cls = getattr(rust_module, "RustFeaturizer", None)
    if rust_featurizer_cls is None:
        raise RuntimeError("s2and_rust extension does not expose RustFeaturizer")
    return _resolve_json_ingest_name_counts_plan(
        dataset,
        rust_featurizer_cls=rust_featurizer_cls,
        signature_name_counts_payload=signature_name_counts_payload,
    )


def _expected_normalization_version() -> str:
    configured = os.environ.get(NORMALIZATION_VERSION_ENV, DEFAULT_NORMALIZATION_VERSION).strip()
    return configured or DEFAULT_NORMALIZATION_VERSION


def _from_json_paths_with_contract(rust_featurizer_cls: Any, contract: RustJsonIngestContract) -> Any:
    return rust_featurizer_cls.from_json_paths(
        contract.signatures_path,
        contract.papers_path,
        contract.cluster_seeds_path,
        contract.specter_embeddings,
        contract.name_tuples_path,
        contract.name_counts_path,
        contract.preprocess,
        contract.compute_reference_features,
        contract.cluster_seed_require_value,
        contract.cluster_seed_disallow_value,
        contract.num_threads,
        contract.expected_normalization_version,
        contract.allow_normalization_version_mismatch,
    )


def _build_rust_featurizer_from_json_paths(
    dataset: Any,
    num_threads: int,
    *,
    signature_name_counts_payload: dict[str, Any] | None = None,
    allow_normalization_version_mismatch: bool = False,
) -> tuple[Any, dict[str, float]]:
    pre_build_start = time.perf_counter()
    rust_module = _require_rust_runtime()
    rust_featurizer_cls = getattr(rust_module, "RustFeaturizer", None)
    if rust_featurizer_cls is None or not hasattr(rust_featurizer_cls, "from_json_paths"):
        raise RuntimeError("s2and_rust extension does not expose RustFeaturizer.from_json_paths")

    signatures_path = getattr(dataset, "signatures_path", None)
    papers_path = getattr(dataset, "papers_path", None)
    if not signatures_path or not papers_path:
        raise RuntimeError("Dataset does not expose signatures_path/papers_path for Rust JSON ingest")

    plan = _resolve_json_ingest_name_counts_plan(
        dataset,
        rust_featurizer_cls=rust_featurizer_cls,
        signature_name_counts_payload=signature_name_counts_payload,
    )
    dataset_signature_counts_payload = plan["dataset_signature_counts_payload"]
    signatures_total = int(plan["signatures_total"])
    signatures_with_counts = int(plan["signatures_with_counts"])
    dataset_has_signature_counts = bool(plan["dataset_has_signature_counts"])
    artifact_configured = bool(plan["artifact_configured"])
    rust_can_overlay_signature_counts = bool(plan["rust_can_overlay_signature_counts"])
    name_counts_source = str(plan["name_counts_source"])
    name_counts_path = plan["name_counts_path"]
    if dataset_has_signature_counts and not rust_can_overlay_signature_counts:
        logger.warning(
            "Rust JSON ingest: extension lacks update_signature_name_counts; "
            "cannot use precomputed signature name counts from dataset. "
            "name-count features will be NaN."
        )
    elif not dataset_has_signature_counts and name_counts_source == "none":
        logger.warning(
            "Rust JSON ingest: no signature name counts available and no name-count artifact selected; "
            "name-count features will be NaN."
        )

    # Normalization version validation is delegated to Rust when using artifact name counts.
    normalization_check_executed = False
    normalization_version_for_rust: str | None = None
    allow_mismatch_for_rust = False
    if name_counts_source == "artifact":
        normalization_check_executed = True
        logger.warning(
            "Rust JSON ingest: selected artifact name-count source path=%s; this can increase latency/RSS.",
            name_counts_path,
        )
        normalization_version_for_rust = _expected_normalization_version()
        allow_mismatch_for_rust = bool(allow_normalization_version_mismatch)

    logger.info(
        "Telemetry stage: stage=rust_json_ingest_name_counts_source "
        "name_counts_source=%s signatures_total=%d signatures_with_counts=%d "
        "overlay_api_available=%s artifact_configured=%s "
        "normalization_check_executed=%s normalization_version_delegated_to_rust=%s "
        "allow_normalization_version_mismatch=%s dataset=%s",
        name_counts_source,
        signatures_total,
        signatures_with_counts,
        rust_can_overlay_signature_counts,
        artifact_configured,
        normalization_check_executed,
        normalization_version_for_rust is not None,
        allow_mismatch_for_rust,
        _dataset_name_for_logs(dataset),
    )

    pre_build_seconds = time.perf_counter() - pre_build_start
    ffi_start = time.perf_counter()
    contract = build_rust_json_ingest_contract(
        dataset,
        name_counts_path=name_counts_path,
        cluster_seed_require_value=CLUSTER_SEEDS_LOOKUP["require"],
        cluster_seed_disallow_value=CLUSTER_SEEDS_LOOKUP["disallow"],
        num_threads=num_threads,
        name_tuples_path=None,
        expected_normalization_version=normalization_version_for_rust,
        allow_normalization_version_mismatch=allow_mismatch_for_rust,
    )
    featurizer = _from_json_paths_with_contract(rust_featurizer_cls, contract)
    ffi_seconds = time.perf_counter() - ffi_start
    get_json_ingest_telemetry = getattr(featurizer, "json_ingest_telemetry", None)
    if not callable(get_json_ingest_telemetry):
        raise RuntimeError(
            "s2and_rust RustFeaturizer.from_json_paths must expose json_ingest_telemetry; "
            f"rebuild/install s2and_rust>={min_supported_rust_extension_version_string()}."
        )
    telemetry = get_json_ingest_telemetry()
    if telemetry is None:
        raise RuntimeError(
            "s2and_rust RustFeaturizer.from_json_paths returned no json_ingest_telemetry; "
            f"rebuild/install s2and_rust>={min_supported_rust_extension_version_string()}."
        )
    stage_seconds = dict(telemetry.get("stage_seconds", {}))
    logger.info(
        "Telemetry stage: stage=rust_json_ingest_stage_seconds "
        "json_parse=%.3f paper_preprocess=%.3f signature_preprocess=%.3f "
        "reference_counter=%.3f cluster_seed=%.3f dataset=%s",
        float(stage_seconds.get("json_parse_seconds", 0.0)),
        float(stage_seconds.get("paper_preprocess_seconds", 0.0)),
        float(stage_seconds.get("signature_preprocess_seconds", 0.0)),
        float(stage_seconds.get("reference_counter_seconds", 0.0)),
        float(stage_seconds.get("cluster_seed_seconds", 0.0)),
        _dataset_name_for_logs(dataset),
    )
    counts = dict(telemetry.get("counts", {}))
    if counts:
        logger.info(
            "Telemetry stage: stage=rust_json_ingest_default_counts "
            "missing_specter_papers=%d defaulted_name_count_signatures=%d "
            "defaulted_name_count_first=%d defaulted_name_count_first_last=%d "
            "defaulted_name_count_last=%d defaulted_name_count_last_first_initial=%d "
            "defaulted_signature_author_positions=%d defaulted_paper_author_positions=%d dataset=%s",
            int(counts.get("missing_specter_paper_count", 0)),
            int(counts.get("defaulted_name_count_signature_count", 0)),
            int(counts.get("defaulted_name_count_first_count", 0)),
            int(counts.get("defaulted_name_count_first_last_count", 0)),
            int(counts.get("defaulted_name_count_last_count", 0)),
            int(counts.get("defaulted_name_count_last_first_initial_count", 0)),
            int(counts.get("defaulted_signature_author_position_count", 0)),
            int(counts.get("defaulted_paper_author_position_count", 0)),
            _dataset_name_for_logs(dataset),
        )

    post_build_seconds = 0.0
    if name_counts_source == "dataset":
        overlay_start = time.perf_counter()
        updated_count = featurizer.update_signature_name_counts(dataset_signature_counts_payload)
        post_build_seconds = time.perf_counter() - overlay_start
        logger.info(
            "Telemetry stage: stage=rust_json_signature_name_counts_overlay "
            "seconds=%.3f updated_signatures=%d dataset=%s",
            post_build_seconds,
            int(updated_count),
            _dataset_name_for_logs(dataset),
        )
    elif name_counts_source == "none":
        logger.warning(
            "Rust JSON ingest: using name_counts_source=none; name-count features will be NaN. "
            "signatures_total=%d signatures_with_counts=%d",
            signatures_total,
            signatures_with_counts,
        )

    return featurizer, {
        "pre_build_seconds": pre_build_seconds,
        "ffi_seconds": ffi_seconds,
        "post_build_seconds": post_build_seconds,
    }


def _rust_featurizer_method(method_name: str, purpose: str) -> Any:
    rust_module = _require_rust_runtime()
    rust_featurizer_cls = getattr(rust_module, "RustFeaturizer", None)
    method = None if rust_featurizer_cls is None else getattr(rust_featurizer_cls, method_name, None)
    if not callable(method):
        raise RuntimeError(f"RustFeaturizer.{method_name} is required for {purpose}")
    return method


def _resolve_direct_name_counts_path(load_name_counts: bool, name_counts_path: str | None) -> str | None:
    if not load_name_counts or name_counts_path is not None:
        return name_counts_path
    resolved_name_counts_path = _rust_name_counts_artifact_path()
    if resolved_name_counts_path is not None:
        return resolved_name_counts_path
    from s2and.consts import _PACKAGE_DATA_DIR

    return os.path.join(_PACKAGE_DATA_DIR, "name_counts_rust.json")


def normalize_arrow_paths(paths: Mapping[Any, Any]) -> dict[str, str]:
    """Return Arrow path mappings with explicit rejection of missing path values."""

    stringified: dict[str, str] = {}
    for key, value in paths.items():
        if value is None:
            raise ValueError(f"Arrow path for {key!r} is None")
        stringified[str(key)] = str(value)
    return stringified


def _stringified_arrow_paths(paths: Mapping[Any, Any]) -> dict[str, str]:
    return normalize_arrow_paths(paths)


def build_rust_featurizer_from_feature_block(
    feature_block: Any,
    *,
    name_tuples: Any = "filtered",
    load_name_counts: bool = False,
    name_counts_path: str | None = None,
    preprocess: bool = True,
    compute_reference_features: bool = False,
    cluster_seed_require_value: float = 0.0,
    cluster_seed_disallow_value: float = 10000.0,
    num_threads: int | None = None,
) -> Any:
    """Build a Rust featurizer directly from the narrow `FeatureBlock` contract."""

    method = _rust_featurizer_method("from_feature_block", "raw FeatureBlock scoring")
    resolved_name_counts_path = _resolve_direct_name_counts_path(load_name_counts, name_counts_path)
    return method(
        feature_block,
        name_tuples,
        resolved_name_counts_path,
        bool(preprocess),
        bool(compute_reference_features),
        float(cluster_seed_require_value),
        float(cluster_seed_disallow_value),
        None if num_threads is None else resolve_n_jobs(num_threads),
    )


def build_rust_featurizer_from_arrow_paths(
    paths: Mapping[str, Any],
    *,
    signature_ids: Sequence[Any] | None = None,
    name_tuples: Any = "filtered",
    load_name_counts: bool = False,
    name_counts_path: str | None = None,
    preprocess: bool = True,
    compute_reference_features: bool = False,
    cluster_seed_require_value: float = 0.0,
    cluster_seed_disallow_value: float = 10000.0,
    num_threads: int | None = None,
) -> Any:
    """Build a Rust featurizer directly from Arrow IPC FeatureBlock paths."""

    method = _rust_featurizer_method("from_arrow_paths", "raw Arrow scoring")
    resolved_name_counts_path = _resolve_direct_name_counts_path(load_name_counts, name_counts_path)
    return method(
        normalize_arrow_paths(paths),
        None if signature_ids is None else [str(value) for value in signature_ids],
        name_tuples,
        resolved_name_counts_path,
        bool(preprocess),
        bool(compute_reference_features),
        float(cluster_seed_require_value),
        float(cluster_seed_disallow_value),
        None if num_threads is None else resolve_n_jobs(num_threads),
    )


def build_rust_featurizer(
    dataset: ANDData,
    *,
    path: RustBuildPath,
    allow_normalization_version_mismatch: bool = False,
) -> tuple[Any, RustBuildPath, dict[str, float]]:
    """Build a Rust featurizer through the requested ingest path."""
    pre_build_start = time.perf_counter()
    rust_module = _require_rust_runtime()
    num_threads = resolve_n_jobs(getattr(dataset, "n_jobs", 1))
    selected_build_path = path
    signatures_path = dataset.signatures_path
    papers_path = dataset.papers_path
    if selected_build_path == "from_json_paths":
        if not signatures_path or not papers_path:
            raise RuntimeError(
                "Rust JSON ingest build path requested but signatures_path/papers_path are missing "
                f"(dataset={_dataset_name_for_logs(dataset)} signatures_path={signatures_path!r} "
                f"papers_path={papers_path!r})."
            )
        outer_pre_build_seconds = time.perf_counter() - pre_build_start
        featurizer, timings = _build_rust_featurizer_from_json_paths(
            dataset,
            num_threads,
            allow_normalization_version_mismatch=allow_normalization_version_mismatch,
        )
        timings["pre_build_seconds"] += outer_pre_build_seconds
        return featurizer, "from_json_paths", timings
    pre_build_seconds = time.perf_counter() - pre_build_start
    ffi_seconds = 0.0
    ffi_start = time.perf_counter()
    featurizer = rust_module.RustFeaturizer.from_dataset(
        dataset,
        CLUSTER_SEEDS_LOOKUP["require"],
        CLUSTER_SEEDS_LOOKUP["disallow"],
        num_threads,
    )
    ffi_seconds += time.perf_counter() - ffi_start
    return (
        featurizer,
        "from_dataset",
        {
            "pre_build_seconds": pre_build_seconds,
            "ffi_seconds": ffi_seconds,
            "post_build_seconds": 0.0,
        },
    )


def _resolve_requested_build_path(
    dataset: ANDData,
    *,
    dataset_mode: str,
    rust_build_path: RustBuildPath | None = None,
) -> RustBuildPath:
    if rust_build_path is not None:
        if rust_build_path not in {"from_dataset", "from_json_paths"}:
            raise ValueError(
                "rust_build_path must be one of from_dataset/from_json_paths; " f"got {rust_build_path!r}."
            )
        return rust_build_path

    dataset_mode_normalized = dataset_mode.strip().lower()
    has_signatures_path = bool(dataset.signatures_path)
    has_papers_path = bool(dataset.papers_path)
    legacy_requested_build_path: RustBuildPath = (
        "from_json_paths"
        if dataset_mode_normalized == "inference" and has_signatures_path and has_papers_path
        else "from_dataset"
    )
    rust_lifecycle_policy = _dataset_rust_lifecycle_policy(dataset)
    requested_build_path = (
        rust_lifecycle_policy.rust_build_path if rust_lifecycle_policy is not None else legacy_requested_build_path
    )
    return requested_build_path


def _build_rust_featurizer_strict(
    dataset: ANDData,
    *,
    requested_build_path: RustBuildPath,
    allow_normalization_version_mismatch: bool = False,
) -> tuple[Any, RustBuildPath, dict[str, float], int, float]:
    build_start = time.perf_counter()
    featurizer, build_path, build_timings = build_rust_featurizer(
        dataset,
        path=requested_build_path,
        allow_normalization_version_mismatch=allow_normalization_version_mismatch,
    )
    build_count = _increment_rust_featurizer_build_count(
        dataset,
        _rust_featurizer_cache_key(
            requested_build_path,
            allow_normalization_version_mismatch,
            _cluster_seeds_version_for_cache(dataset),
        ),
    )
    return featurizer, build_path, build_timings, build_count, time.perf_counter() - build_start


@dataclass(frozen=True)
class _RustFeaturizerBuildContext:
    operation: str
    run_id: str
    dataset_mode: str
    dataset_name_for_logs: str
    requested_build_path: RustBuildPath
    allow_normalization_version_mismatch: bool
    cluster_seeds_version: int
    cache_key: _RustFeaturizerCacheKey


def _get_or_wait_for_cached(
    dataset: ANDData,
    *,
    build_context: _RustFeaturizerBuildContext,
) -> tuple[Any | None, _InFlightFeaturizerBuild | None]:
    inflight_build: _InFlightFeaturizerBuild | None = None
    cache_key = build_context.cache_key
    with _RUST_FEATURIZER_CACHE_LOCK:
        entries = _RUST_FEATURIZER_CACHE.get(dataset)
        entry = None if entries is None else entries.get(cache_key)
        if entry is not None:
            logger.info(
                "Telemetry: rust_featurizer_cache cache=hit dataset=%s mode=%s op=%s run=%s builds=%d",
                build_context.dataset_name_for_logs,
                build_context.dataset_mode,
                build_context.operation,
                build_context.run_id,
                entry.build_count,
            )
            return entry.featurizer, None
        if entries:
            logger.info(
                "Telemetry: rust_featurizer_cache cache=option_miss dataset=%s mode=%s op=%s run=%s "
                "requested_path=%s requested_allow_normalization_mismatch=%s cached_options=%s",
                build_context.dataset_name_for_logs,
                build_context.dataset_mode,
                build_context.operation,
                build_context.run_id,
                build_context.requested_build_path,
                build_context.allow_normalization_version_mismatch,
                sorted(f"{path}:{allow}:seeds={seed_version}" for path, allow, seed_version in entries),
            )

        inflight_entries = _RUST_FEATURIZER_INFLIGHT_BUILDS.get(dataset)
        inflight_build = None if inflight_entries is None else inflight_entries.get(cache_key)
        if inflight_build is None:
            inflight_build = _InFlightFeaturizerBuild(build_context.requested_build_path)
            if inflight_entries is None:
                inflight_entries = {}
                _RUST_FEATURIZER_INFLIGHT_BUILDS[dataset] = inflight_entries
            inflight_entries[cache_key] = inflight_build
            logger.info(
                "Telemetry: rust_featurizer_cache cache=miss dataset=%s mode=%s op=%s run=%s builds=%d",
                build_context.dataset_name_for_logs,
                build_context.dataset_mode,
                build_context.operation,
                build_context.run_id,
                0,
            )
            return None, inflight_build

        logger.info(
            "Telemetry: rust_featurizer_cache cache=wait dataset=%s mode=%s op=%s run=%s builds=%d "
            "inflight_path=%s requested_path=%s",
            build_context.dataset_name_for_logs,
            build_context.dataset_mode,
            build_context.operation,
            build_context.run_id,
            0,
            inflight_build.build_path,
            build_context.requested_build_path,
        )

    inflight_build.event.wait()
    with _RUST_FEATURIZER_CACHE_LOCK:
        entries = _RUST_FEATURIZER_CACHE.get(dataset)
        entry = None if entries is None else entries.get(cache_key)
        if entry is not None:
            return entry.featurizer, None
        build_error = inflight_build.error
    if build_error is not None:
        raise RuntimeError(
            "Rust featurizer build failed for dataset="
            f"{build_context.dataset_name_for_logs} while waiting for concurrent builder"
        ) from build_error
    return None, None


def _build_and_cache_rust_featurizer(
    dataset: ANDData,
    *,
    inflight_build: _InFlightFeaturizerBuild,
    build_context: _RustFeaturizerBuildContext,
) -> Any:
    featurizer: Any | None = None
    build_path = build_context.requested_build_path
    cache_key = build_context.cache_key
    build_count = 0
    try:
        build_timings: dict[str, float] = {
            "pre_build_seconds": 0.0,
            "ffi_seconds": 0.0,
            "post_build_seconds": 0.0,
        }
        if featurizer is None:
            featurizer, build_path, build_timings, build_count, build_seconds = _build_rust_featurizer_strict(
                dataset,
                requested_build_path=build_context.requested_build_path,
                allow_normalization_version_mismatch=build_context.allow_normalization_version_mismatch,
            )
            logger.info(
                "Telemetry: rust_core_build seconds=%.3f dataset=%s path=%s count=%d pre=%.3f ffi=%.3f post=%.3f",
                build_seconds,
                build_context.dataset_name_for_logs,
                build_path,
                build_count,
                build_timings.get("pre_build_seconds", 0.0),
                build_timings.get("ffi_seconds", 0.0),
                build_timings.get("post_build_seconds", 0.0),
            )
        else:
            build_count = _rust_featurizer_build_count(dataset, cache_key)

        with _RUST_FEATURIZER_CACHE_LOCK:
            entries = _RUST_FEATURIZER_CACHE.get(dataset)
            if entries is None:
                entries = {}
                _RUST_FEATURIZER_CACHE[dataset] = entries
            entries[cache_key] = _CacheEntry(
                featurizer=featurizer,
                build_count=build_count,
            )
            logger.info(
                "Telemetry: rust_featurizer_cache_fill source=build dataset=%s path=%s count=%d",
                build_context.dataset_name_for_logs,
                build_path,
                build_count,
            )
            inflight_build.error = None
            inflight_build.event.set()
            inflight_entries = _RUST_FEATURIZER_INFLIGHT_BUILDS.get(dataset)
            if inflight_entries is not None and inflight_entries.get(cache_key) is inflight_build:
                del inflight_entries[cache_key]
                if not inflight_entries:
                    del _RUST_FEATURIZER_INFLIGHT_BUILDS[dataset]
    except Exception as build_error:
        with _RUST_FEATURIZER_CACHE_LOCK:
            inflight_build.error = build_error
            inflight_build.event.set()
            inflight_entries = _RUST_FEATURIZER_INFLIGHT_BUILDS.get(dataset)
            if inflight_entries is not None and inflight_entries.get(cache_key) is inflight_build:
                del inflight_entries[cache_key]
                if not inflight_entries:
                    del _RUST_FEATURIZER_INFLIGHT_BUILDS[dataset]
        raise

    if featurizer is None:
        raise RuntimeError("Rust featurizer was not initialized")
    return featurizer


def _get_rust_featurizer(
    dataset: ANDData,
    runtime_context: Any | None = None,
    rust_build_path: RustBuildPath | None = None,
    allow_normalization_version_mismatch: bool = False,
) -> Any:
    _require_rust_runtime()
    operation, run_id = _runtime_callsite_for_logs(dataset, runtime_context)
    dataset_mode = _dataset_mode_for_logs(dataset)
    requested_build_path = _resolve_requested_build_path(
        dataset,
        dataset_mode=dataset_mode,
        rust_build_path=rust_build_path,
    )
    ds_log = _dataset_name_for_logs(dataset)
    cluster_seeds_version = _cluster_seeds_version_for_cache(dataset)
    build_context = _RustFeaturizerBuildContext(
        operation=operation,
        run_id=run_id,
        dataset_mode=dataset_mode,
        dataset_name_for_logs=ds_log,
        requested_build_path=requested_build_path,
        allow_normalization_version_mismatch=bool(allow_normalization_version_mismatch),
        cluster_seeds_version=cluster_seeds_version,
        cache_key=_rust_featurizer_cache_key(
            requested_build_path,
            allow_normalization_version_mismatch,
            cluster_seeds_version,
        ),
    )
    max_empty_wait_retries = _rust_featurizer_empty_wait_max_retries()
    empty_wait_backoff_seconds = _rust_featurizer_empty_wait_backoff_seconds()
    empty_wait_attempt = 0

    while True:
        featurizer, inflight_build = _get_or_wait_for_cached(dataset, build_context=build_context)
        if featurizer is not None:
            return featurizer
        if inflight_build is None:
            empty_wait_attempt += 1
            if empty_wait_attempt > max_empty_wait_retries:
                raise RuntimeError(
                    "Rust featurizer cache resolution exhausted retries for empty wait state "
                    f"(dataset={ds_log}, mode={dataset_mode}, run={run_id}, "
                    f"attempts={empty_wait_attempt}, max_retries={max_empty_wait_retries})"
                )
            backoff_seconds = empty_wait_backoff_seconds * float(empty_wait_attempt)
            logger.warning(
                "Telemetry: rust_featurizer_cache cache=retry_empty dataset=%s mode=%s op=%s run=%s attempt=%d/%d "
                "backoff_seconds=%.3f",
                ds_log,
                dataset_mode,
                operation,
                run_id,
                empty_wait_attempt,
                max_empty_wait_retries,
                backoff_seconds,
            )
            if backoff_seconds > 0:
                time.sleep(backoff_seconds)
            continue
        return _build_and_cache_rust_featurizer(dataset, inflight_build=inflight_build, build_context=build_context)


def evict_rust_featurizer(dataset: ANDData) -> bool:
    """Evict a single dataset's Rust featurizer from the in-memory cache."""
    with _RUST_FEATURIZER_CACHE_LOCK:
        removed = False
        if dataset in _RUST_FEATURIZER_CACHE:
            del _RUST_FEATURIZER_CACHE[dataset]
            removed = True
        if dataset in _RUST_FEATURIZER_INFLIGHT_BUILDS:
            del _RUST_FEATURIZER_INFLIGHT_BUILDS[dataset]
        return removed


def clear_rust_featurizer_cache() -> int:
    """Clear all in-memory Rust featurizer cache entries."""
    with _RUST_FEATURIZER_CACHE_LOCK:
        count = _rust_featurizer_cache_entry_count_locked()
        _RUST_FEATURIZER_CACHE.clear()
        _RUST_FEATURIZER_INFLIGHT_BUILDS.clear()
        return count


def warm_rust_featurizer(
    dataset: ANDData,
    runtime_context: Any | None = None,
    rust_build_path: RustBuildPath | None = None,
    allow_normalization_version_mismatch: bool = False,
) -> None:
    """Preload the Rust featurizer into memory for low-latency inference."""
    _get_rust_featurizer(
        dataset,
        runtime_context=runtime_context,
        rust_build_path=rust_build_path,
        allow_normalization_version_mismatch=allow_normalization_version_mismatch,
    )


__all__ = [
    "build_rust_featurizer",
    "build_block_upper_triangle_feature_matrix_indexed_rust",
    "build_linker_pair_aggregate_stats_arrays_rust",
    "build_linker_pair_distance_accumulators_rust",
    "build_linker_pair_features_and_aggregate_stats_arrays_rust",
    "build_linker_pair_features_and_aggregate_stats_indexed_rust",
    "build_pair_feature_matrix_rust",
    "build_rust_featurizer_from_arrow_paths",
    "build_rust_featurizer_from_feature_block",
    "clear_rust_featurizer_cache",
    "evict_rust_featurizer",
    "featurize_pair_rust",
    "get_constraint_labels_index_arrays_rust",
    "get_constraint_rust",
    "get_constraints_block_upper_triangle_indexed_rust",
    "get_constraints_matrix_indexed_rust",
    "get_constraints_matrix_rust",
    "inspect_json_ingest_name_counts_source",
    "rust_featurizer_available",
    "rust_signature_preprocess_available",
    "s2and_rust",
    "signature_ngrams_batch_rust",
    "update_rust_cluster_seeds",
    "warm_rust_featurizer",
]


def rust_featurizer_available() -> bool:
    rust_module = _ensure_s2and_rust_loaded()
    capabilities = detect_rust_runtime_capabilities(extension_module=rust_module)
    return bool(capabilities.core_runtime_available)


def rust_signature_preprocess_available() -> bool:
    rust_module = _ensure_s2and_rust_loaded()
    if rust_module is None or not hasattr(rust_module, "signature_ngrams_batch"):
        return False
    capabilities = detect_rust_runtime_capabilities(extension_module=rust_module)
    return bool(capabilities.core_runtime_available)


def signature_ngrams_batch_rust(
    coauthor_texts: list[str],
    affiliation_texts: list[str],
    num_threads: int | None = None,
) -> tuple[list[Counter], list[Counter]]:
    rust_module = _require_rust_runtime()
    if not hasattr(rust_module, "signature_ngrams_batch"):
        raise RuntimeError("s2and_rust extension does not expose signature_ngrams_batch")
    resolved_num_threads = None if num_threads is None else resolve_n_jobs(num_threads)
    coauthor_raw, affiliation_raw = rust_module.signature_ngrams_batch(
        coauthor_texts,
        affiliation_texts,
        resolved_num_threads,
    )
    if len(coauthor_raw) != len(coauthor_texts) or len(affiliation_raw) != len(affiliation_texts):
        raise RuntimeError(
            "Rust signature_ngrams_batch returned unexpected output lengths: "
            f"coauthor={len(coauthor_raw)} expected={len(coauthor_texts)} "
            f"affiliation={len(affiliation_raw)} expected={len(affiliation_texts)}"
        )
    coauthor_counters = [Counter({k: int(v) for k, v in row.items()}) for row in coauthor_raw]
    affiliation_counters = [Counter({k: int(v) for k, v in row.items()}) for row in affiliation_raw]
    return coauthor_counters, affiliation_counters
