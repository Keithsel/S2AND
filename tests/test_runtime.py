from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from s2and import feature_port, runtime, rust_calls
from s2and.featurizer import FeaturizationInfo, many_pairs_featurize
from tests.helpers import build_dummy_dataset


def _clear_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("S2AND_BACKEND",):
        monkeypatch.delenv(name, raising=False)
    runtime.reset_runtime_warning_state_for_tests()


def _runtime_capabilities(*, core_available: bool, reason: str) -> runtime.RustRuntimeCapabilities:
    return runtime.RustRuntimeCapabilities(
        extension_importable=core_available,
        core_runtime_available=core_available,
        from_dataset_paper_preprocess_available=core_available,
        reason=reason,
    )


def test_resolve_backend_unset_auto_falls_back_to_python_when_rust_core_missing(monkeypatch: pytest.MonkeyPatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setattr(
        runtime,
        "_auto_backend_capability_probe",
        lambda: (False, "rust_extension_unavailable"),
    )
    resolution = runtime.resolve_backend(emit_startup_warning=False)
    assert resolution.resolved_backend == "python"
    assert resolution.source == "default"
    assert resolution.requested_backend is None
    assert resolution.capability_reason == "rust_extension_unavailable"


def test_resolve_backend_unset_auto_uses_rust_when_core_capability_available(monkeypatch: pytest.MonkeyPatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setattr(
        runtime,
        "_auto_backend_capability_probe",
        lambda: (True, "rust_core_available"),
    )
    resolution = runtime.resolve_backend(emit_startup_warning=False)
    assert resolution.resolved_backend == "rust"
    assert resolution.source == "default"
    assert resolution.requested_backend is None
    assert resolution.capability_reason == "rust_core_available"


def test_resolve_backend_prefers_explicit_s2and_backend(monkeypatch: pytest.MonkeyPatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("S2AND_BACKEND", "python")
    resolution = runtime.resolve_backend(emit_startup_warning=False)
    assert resolution.resolved_backend == "python"
    assert resolution.source == "S2AND_BACKEND"


def test_resolve_backend_argument_overrides_env(monkeypatch: pytest.MonkeyPatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("S2AND_BACKEND", "python")
    monkeypatch.setattr(
        runtime,
        "detect_rust_runtime_capabilities",
        lambda: _runtime_capabilities(core_available=True, reason="rust_core_available"),
    )

    resolution = runtime.resolve_backend_for_request(backend="rust", emit_startup_warning=False)

    assert resolution.requested_backend == "rust"
    assert resolution.resolved_backend == "rust"
    assert resolution.source == "argument"


def test_build_runtime_context_accepts_backend_argument(monkeypatch: pytest.MonkeyPatch):
    _clear_runtime_env(monkeypatch)

    context = runtime.build_runtime_context("unit_test", backend="python", emit_startup_warning=False)

    assert context.requested_backend == "python"
    assert context.resolved_backend == "python"
    assert context.source == "argument"
    assert context.use_rust is False


def test_resolve_backend_explicit_rust_raises_when_runtime_unavailable(monkeypatch: pytest.MonkeyPatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("S2AND_BACKEND", "rust")
    monkeypatch.setattr(
        runtime,
        "detect_rust_runtime_capabilities",
        lambda: _runtime_capabilities(core_available=False, reason="rust_extension_unavailable"),
    )
    with pytest.raises(RuntimeError) as exc_info:
        runtime.resolve_backend(emit_startup_warning=False)
    message = str(exc_info.value)
    assert "reason=rust_extension_unavailable" in message
    assert f"s2and_rust (>= {runtime.min_supported_rust_extension_version_string()})" in message


def test_load_s2and_rust_extension_propagates_native_import_errors() -> None:
    def importer(name: str):
        if name == "s2and_rust":
            raise ImportError("GLIBC version mismatch")
        raise AssertionError(f"unexpected import: {name}")

    with pytest.raises(ImportError, match="GLIBC version mismatch"):
        runtime.load_s2and_rust_extension(import_module=importer)


def test_resolve_backend_explicit_rust_uses_capability_probe(monkeypatch: pytest.MonkeyPatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("S2AND_BACKEND", "rust")
    monkeypatch.setattr(
        runtime,
        "detect_rust_runtime_capabilities",
        lambda: _runtime_capabilities(core_available=True, reason="rust_core_available"),
    )
    resolution = runtime.resolve_backend(emit_startup_warning=False)
    assert resolution.requested_backend == "rust"
    assert resolution.resolved_backend == "rust"
    assert resolution.capability_reason == "rust_core_available"


def test_rust_calls_missing_matrix_reports_runtime_minimum() -> None:
    class MissingMatrixFeaturizer:
        pass

    with pytest.raises(RuntimeError) as exc_info:
        rust_calls.get_constraints_matrix_rust(
            dataset=cast(Any, object()),
            pairs=[],
            featurizer=MissingMatrixFeaturizer(),
        )

    assert f"s2and-rust>={runtime.min_supported_rust_extension_version_string()}" in str(exc_info.value)


def test_resolve_backend_auto_env_uses_capability_probe(monkeypatch: pytest.MonkeyPatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("S2AND_BACKEND", "auto")
    monkeypatch.setattr(
        runtime,
        "_auto_backend_capability_probe",
        lambda: (True, "rust_core_available"),
    )
    resolution = runtime.resolve_backend(emit_startup_warning=False)
    assert resolution.requested_backend == "auto"
    assert resolution.resolved_backend == "rust"
    assert resolution.source == "S2AND_BACKEND"
    assert resolution.capability_reason == "rust_core_available"


def test_resolve_backend_invalid_value_raises(monkeypatch: pytest.MonkeyPatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("S2AND_BACKEND", "invalid")
    with pytest.raises(ValueError, match="Invalid S2AND_BACKEND"):
        runtime.resolve_backend(emit_startup_warning=False)


def test_runtime_context_use_rust(monkeypatch: pytest.MonkeyPatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("S2AND_BACKEND", "python")
    python_context = runtime.build_runtime_context("unit_test", emit_startup_warning=False)
    assert python_context.use_rust is False
    assert python_context.stage_backend() == "python"

    monkeypatch.setenv("S2AND_BACKEND", "rust")
    monkeypatch.setattr(
        runtime,
        "detect_rust_runtime_capabilities",
        lambda: _runtime_capabilities(core_available=True, reason="rust_core_available"),
    )
    rust_context = runtime.build_runtime_context("unit_test", emit_startup_warning=False)
    assert rust_context.use_rust is True
    assert rust_context.stage_backend() == "rust"


def test_python_backend_pair_featurization_makes_zero_rust_calls(monkeypatch):
    monkeypatch.setenv("S2AND_BACKEND", "python")
    dataset = build_dummy_dataset("dummy_runtime_policy_python")
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    pairs = [("0", "1", 0), ("0", "2", 0)]

    monkeypatch.setattr(
        feature_port,
        "_get_rust_featurizer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected rust call")),
    )

    features, labels, _ = many_pairs_featurize(
        pairs,
        dataset,
        featurizer_info,
        n_jobs=1,
        use_cache=False,
        chunk_size=1,
        nan_value=np.nan,
    )

    assert features.shape[0] == len(pairs)
    assert labels.shape[0] == len(pairs)


def test_rust_backend_pair_featurization_fails_fast_on_rust_error(monkeypatch):
    if not feature_port.rust_featurizer_available():
        raise pytest.skip.Exception("s2and_rust extension is unavailable")

    monkeypatch.setenv("S2AND_BACKEND", "rust")
    monkeypatch.setattr(
        runtime,
        "detect_rust_runtime_capabilities",
        lambda: _runtime_capabilities(core_available=True, reason="rust_core_available"),
    )
    dataset = build_dummy_dataset("dummy_runtime_policy_rust")
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    pairs = [("0", "1", 0), ("0", "2", 0)]

    class FailingRustFeaturizer:
        def signature_ids(self):
            return []

        def featurize_pairs_matrix_indexed(self, _pairs, _indices, _threads, _nan):
            raise RuntimeError("synthetic rust batch failure")

        def featurize_pairs_matrix(self, *_args, **_kwargs):
            raise RuntimeError("synthetic rust batch failure")

        def featurize_pairs(self, _pairs, num_threads=None):
            del num_threads
            raise RuntimeError("synthetic rust batch failure")

    monkeypatch.setattr(feature_port, "s2and_rust", object())
    monkeypatch.setattr(feature_port, "_get_rust_featurizer", lambda *_args, **_kwargs: FailingRustFeaturizer())

    with pytest.raises(RuntimeError, match="strict rust backend"):
        many_pairs_featurize(
            pairs,
            dataset,
            featurizer_info,
            n_jobs=1,
            use_cache=False,
            chunk_size=1,
            nan_value=np.nan,
        )
