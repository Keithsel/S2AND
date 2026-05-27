from __future__ import annotations

from dataclasses import fields

import pytest

from s2and.rust_lifecycle import (
    PYTHON_ONLY_POLICY,
    RustLifecyclePolicy,
    build_rust_lifecycle_policy,
)


def test_python_backend_always_returns_python_only_policy():
    for mode in ("train", "inference"):
        policy = build_rust_lifecycle_policy(
            backend="python",
            mode=mode,
            preprocess=True,
            use_rust=False,
        )
        assert policy == PYTHON_ONLY_POLICY


@pytest.mark.parametrize(
    ("backend", "use_rust"),
    [
        ("python", True),
        ("rust", False),
    ],
)
def test_backend_use_rust_mismatch_raises(backend: str, use_rust: bool):
    with pytest.raises(ValueError, match="Inconsistent backend/use_rust configuration"):
        build_rust_lifecycle_policy(
            backend=backend,  # type: ignore[arg-type]
            mode="train",
            preprocess=True,
            use_rust=use_rust,
        )


def test_rust_inference_does_not_skip_python_paper_preprocess():
    policy = build_rust_lifecycle_policy(
        backend="rust",
        mode="inference",
        preprocess=True,
        use_rust=True,
    )
    assert policy.rust_build_path == "from_dataset"
    assert policy.skip_python_paper_preprocess is False


@pytest.mark.parametrize(
    ("compute_reference_features", "from_dataset_paper_preprocess_available", "expected_mode", "expected_skip"),
    [
        (False, True, "rust_training_skip_preprocess", True),
        (True, True, "rust_training_from_dataset", False),
        (False, False, "rust_training_from_dataset", False),
    ],
)
def test_rust_training_from_dataset_skip_preprocess_semantics(
    compute_reference_features: bool,
    from_dataset_paper_preprocess_available: bool,
    expected_mode: str,
    expected_skip: bool,
):
    policy = build_rust_lifecycle_policy(
        backend="rust",
        mode="train",
        preprocess=True,
        compute_reference_features=compute_reference_features,
        use_rust=True,
        from_dataset_paper_preprocess_available=from_dataset_paper_preprocess_available,
    )
    assert policy.mode == expected_mode
    assert policy.skip_python_paper_preprocess is expected_skip


@pytest.mark.parametrize("preprocess", [False, True])
@pytest.mark.parametrize("use_rust", [False, True])
def test_defer_signature_ngrams_requires_preprocess_and_rust(preprocess: bool, use_rust: bool):
    backend = "rust" if use_rust else "python"
    policy = build_rust_lifecycle_policy(
        backend=backend,
        mode="train",
        preprocess=preprocess,
        use_rust=use_rust,
    )
    assert policy.defer_signature_ngrams_to_rust is (preprocess and use_rust)


@pytest.mark.parametrize("mode", ["train", "inference"])
@pytest.mark.parametrize("use_rust", [False, True])
def test_defer_signature_fields_requires_rust_and_non_inference(
    mode: str,
    use_rust: bool,
):
    backend = "rust" if use_rust else "python"
    policy = build_rust_lifecycle_policy(
        backend=backend,
        mode=mode,
        preprocess=True,
        use_rust=use_rust,
    )
    expected = bool(mode == "train" and use_rust)
    assert policy.defer_signature_fields_to_rust is expected


def test_lifecycle_policy_stores_only_canonical_mode():
    assert [field.name for field in fields(RustLifecyclePolicy)] == ["mode"]
