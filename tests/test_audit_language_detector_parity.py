from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest


def test_audit_language_detector_parity_defaults_to_small_limit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("pycld2")
    from scripts.verification import audit_language_detector_parity

    rows = [(f"p{i}", f"title {i}") for i in range(1005)]
    monkeypatch.setattr(sys, "argv", ["audit_language_detector_parity.py"])
    monkeypatch.delenv("S2AND_SKIP_FASTTEXT", raising=False)
    monkeypatch.setattr(audit_language_detector_parity, "_load_in_signature_titles", lambda _root, _dataset: rows)
    monkeypatch.setattr(
        audit_language_detector_parity,
        "_load_rust_module",
        lambda _path: SimpleNamespace(
            _debug_language_detector_audit=lambda titles: [("en", "en", "en", True) for _title in titles]
        ),
    )
    monkeypatch.setattr(
        audit_language_detector_parity,
        "_python_language_audit",
        lambda _title: ("en", "en", "en", True),
    )

    audit_language_detector_parity.main()

    summary = json.loads(capsys.readouterr().out)
    assert summary["config"]["limit"] == 1000
    assert summary["total_titles"] == 1000
