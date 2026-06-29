from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from s2and.runtime import Backend

RustBuildPath = Literal["from_dataset", "from_json_paths"]
RustLifecycleMode = Literal[
    "python_only",
    "rust_from_dataset_no_preprocess",
    "rust_inference_from_dataset",
    "rust_inference_json_raw",
    "rust_inference_json_raw_sinonym",
    "rust_inference_json",
    "rust_inference_json_sinonym",
    "rust_training_from_dataset",
    "rust_training_skip_preprocess",
]

_RUST_JSON_PATH_MODES: frozenset[RustLifecycleMode] = frozenset(
    {
        "rust_inference_json_raw",
        "rust_inference_json_raw_sinonym",
        "rust_inference_json",
        "rust_inference_json_sinonym",
    }
)
_SKIP_PYTHON_PAPER_PREPROCESS_MODES: frozenset[RustLifecycleMode] = frozenset(
    {
        "rust_inference_json",
        "rust_inference_json_sinonym",
        "rust_training_skip_preprocess",
    }
)
_DEFER_SIGNATURE_NGRAM_MODES: frozenset[RustLifecycleMode] = frozenset(
    {
        "rust_inference_from_dataset",
        "rust_inference_json",
        "rust_inference_json_sinonym",
        "rust_training_from_dataset",
        "rust_training_skip_preprocess",
    }
)
_DEFER_SIGNATURE_FIELD_MODES: frozenset[RustLifecycleMode] = frozenset(
    {
        "rust_training_from_dataset",
        "rust_training_skip_preprocess",
    }
)
_DEFER_RUST_JSON_INGEST_WRITE_MODES: frozenset[RustLifecycleMode] = frozenset(
    {
        "rust_inference_json_raw_sinonym",
        "rust_inference_json_sinonym",
    }
)


@dataclass(frozen=True)
class RustJsonIngestContract:
    """Canonical inference ingest contract for RustFeaturizer.from_json_paths."""

    signatures_path: str
    papers_path: str
    clusters_path: str | None
    cluster_seeds_path: str | None
    specter_embeddings: str | dict | None
    name_tuples_path: str | None
    name_counts_path: str | None
    preprocess: bool
    compute_reference_features: bool
    cluster_seed_require_value: float
    cluster_seed_disallow_value: float
    num_threads: int
    expected_normalization_version: str | None = None
    allow_normalization_version_mismatch: bool = False


def build_rust_json_ingest_contract(
    dataset: Any,
    *,
    name_counts_path: str | None,
    cluster_seed_require_value: float,
    cluster_seed_disallow_value: float,
    num_threads: int,
    name_tuples_path: str | None = None,
    expected_normalization_version: str | None = None,
    allow_normalization_version_mismatch: bool = False,
) -> RustJsonIngestContract:
    signatures_path = getattr(dataset, "signatures_path", None)
    papers_path = getattr(dataset, "papers_path", None)
    if not signatures_path or not papers_path:
        raise RuntimeError("Dataset does not expose signatures_path/papers_path for Rust JSON ingest")

    return RustJsonIngestContract(
        signatures_path=signatures_path,
        papers_path=papers_path,
        clusters_path=getattr(dataset, "clusters_path", None),
        cluster_seeds_path=getattr(dataset, "cluster_seeds_path", None),
        specter_embeddings=getattr(dataset, "specter_embeddings", None)
        or getattr(dataset, "specter_embeddings_path", None),
        name_tuples_path=name_tuples_path,
        name_counts_path=name_counts_path,
        preprocess=bool(getattr(dataset, "preprocess", True)),
        compute_reference_features=bool(getattr(dataset, "compute_reference_features", False)),
        cluster_seed_require_value=cluster_seed_require_value,
        cluster_seed_disallow_value=cluster_seed_disallow_value,
        num_threads=max(1, int(num_threads)),
        expected_normalization_version=expected_normalization_version,
        allow_normalization_version_mismatch=allow_normalization_version_mismatch,
    )


@dataclass(frozen=True)
class RustLifecyclePolicy:
    """Frozen canonical Rust lifecycle decision for a dataset."""

    mode: RustLifecycleMode

    @property
    def rust_build_path(self) -> RustBuildPath:
        """Return the Rust featurizer build path implied by this mode."""

        return "from_json_paths" if self.mode in _RUST_JSON_PATH_MODES else "from_dataset"

    @property
    def skip_python_paper_preprocess(self) -> bool:
        """Return whether Python paper preprocessing is deferred to Rust."""

        return self.mode in _SKIP_PYTHON_PAPER_PREPROCESS_MODES

    @property
    def defer_signature_ngrams_to_rust(self) -> bool:
        """Return whether signature n-gram computation is deferred to Rust."""

        return self.mode in _DEFER_SIGNATURE_NGRAM_MODES

    @property
    def defer_signature_fields_to_rust(self) -> bool:
        """Return whether normalized signature fields are deferred to Rust."""

        return self.mode in _DEFER_SIGNATURE_FIELD_MODES

    @property
    def defer_rust_json_ingest_write_for_sinonym(self) -> bool:
        """Return whether Sinonym-overwritten JSON payload writing is deferred."""

        return self.mode in _DEFER_RUST_JSON_INGEST_WRITE_MODES


PYTHON_ONLY_POLICY = RustLifecyclePolicy(mode="python_only")


def _is_inference_mode(mode: str) -> bool:
    return mode.strip().lower() == "inference"


def build_rust_lifecycle_policy(
    *,
    backend: Backend,
    mode: str,
    has_signatures_path: bool,
    has_papers_path: bool,
    preprocess: bool,
    compute_reference_features: bool = False,
    use_rust: bool,
    from_dataset_paper_preprocess_available: bool = False,
    use_sinonym_overwrite: bool = False,
) -> RustLifecyclePolicy:
    expected_use_rust = backend == "rust"
    if use_rust is not expected_use_rust:
        raise ValueError(
            "Inconsistent backend/use_rust configuration: "
            f"backend={backend!r} implies use_rust={expected_use_rust}, got use_rust={use_rust}."
        )

    if backend == "python":
        return PYTHON_ONLY_POLICY

    is_inference = _is_inference_mode(mode)

    has_json_paths = has_signatures_path and has_papers_path
    if is_inference and has_json_paths:
        if use_sinonym_overwrite:
            return RustLifecyclePolicy(
                mode="rust_inference_json_sinonym" if preprocess else "rust_inference_json_raw_sinonym"
            )
        return RustLifecyclePolicy(mode="rust_inference_json" if preprocess else "rust_inference_json_raw")

    if is_inference:
        return RustLifecyclePolicy(
            mode="rust_inference_from_dataset" if preprocess else "rust_from_dataset_no_preprocess"
        )

    if not preprocess:
        return RustLifecyclePolicy(mode="rust_from_dataset_no_preprocess")

    if from_dataset_paper_preprocess_available and not compute_reference_features:
        return RustLifecyclePolicy(mode="rust_training_skip_preprocess")
    return RustLifecyclePolicy(mode="rust_training_from_dataset")
