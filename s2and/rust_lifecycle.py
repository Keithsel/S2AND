from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from s2and.runtime import Backend

RustLifecycleMode = Literal[
    "python_only",
    "rust_from_dataset_no_preprocess",
    "rust_inference_from_dataset",
    "rust_training_from_dataset",
    "rust_training_skip_preprocess",
]

_SKIP_PYTHON_PAPER_PREPROCESS_MODES: frozenset[RustLifecycleMode] = frozenset(
    {
        "rust_training_skip_preprocess",
    }
)
_DEFER_SIGNATURE_NGRAM_MODES: frozenset[RustLifecycleMode] = frozenset(
    {
        "rust_inference_from_dataset",
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


@dataclass(frozen=True)
class RustLifecyclePolicy:
    """Frozen compatibility/training Rust lifecycle decision for a dataset."""

    mode: RustLifecycleMode

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


PYTHON_ONLY_POLICY = RustLifecyclePolicy(mode="python_only")


def _is_inference_mode(mode: str) -> bool:
    return mode.strip().lower() == "inference"


def build_rust_lifecycle_policy(
    *,
    backend: Backend,
    mode: str,
    preprocess: bool,
    compute_reference_features: bool = False,
    from_dataset_paper_preprocess_available: bool = False,
) -> RustLifecyclePolicy:
    if backend == "python":
        return PYTHON_ONLY_POLICY

    is_inference = _is_inference_mode(mode)

    if is_inference:
        return RustLifecyclePolicy(
            mode="rust_inference_from_dataset" if preprocess else "rust_from_dataset_no_preprocess"
        )

    if not preprocess:
        return RustLifecyclePolicy(mode="rust_from_dataset_no_preprocess")

    if from_dataset_paper_preprocess_available and not compute_reference_features:
        return RustLifecyclePolicy(mode="rust_training_skip_preprocess")
    return RustLifecyclePolicy(mode="rust_training_from_dataset")
