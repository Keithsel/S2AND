from __future__ import annotations

from typing import Any

import pytest

from s2and.text import detect_language


def _rust_language_audit() -> Any:
    s2and_rust = pytest.importorskip("s2and_rust")
    audit_fn = getattr(s2and_rust, "_debug_language_detector_audit", None)
    if audit_fn is None:
        raise pytest.skip.Exception("s2and_rust._debug_language_detector_audit is unavailable")
    return audit_fn


def test_debug_language_detector_audit_reports_python_compatible_final_decision() -> None:
    audit_fn = _rust_language_audit()
    title = "Genetic behavior of resistance to the beet cyst as a way to enchant"

    fasttext_label, cld2_label, predicted_language, is_reliable = audit_fn([title])[0]
    reference_reliable, _reference_english, reference_language = detect_language(title)

    assert isinstance(fasttext_label, str)
    assert isinstance(cld2_label, str)
    assert predicted_language == reference_language
    assert is_reliable == reference_reliable


def test_debug_language_detector_audit_errors_when_fasttext_model_cannot_load(monkeypatch: pytest.MonkeyPatch) -> None:
    audit_fn = _rust_language_audit()

    import s2and.file_cache as file_cache
    import s2and.text as text_module

    monkeypatch.delenv("S2AND_SKIP_FASTTEXT", raising=False)
    text_module.set_fasttext_loading_enabled(True)
    monkeypatch.setattr(file_cache, "cached_path", lambda _path: "missing-fasttext-model.bin")

    with pytest.raises(RuntimeError, match="failed to load fastText language model"):
        audit_fn(["hello world"])
