from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_run_ci_locally() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_ci_locally.py"
    spec = importlib.util.spec_from_file_location("run_ci_locally", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_py_only_lane_reports_expected_rust_skips(monkeypatch, capsys) -> None:
    run_ci = _load_run_ci_locally()
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    monkeypatch.setattr(run_ci, "sync_deps", lambda *, lock_present, lane: None)
    monkeypatch.setattr(run_ci, "run_ty_checks", lambda: None)
    monkeypatch.setattr(run_ci, "run_uv", lambda args, *, env=None: calls.append((args, env)))

    run_ci.run_typecheck_and_test_lane(lane="py-only", lock_present=True)

    assert len(calls) == 1
    assert calls[0][0] == [
        "run",
        "pytest",
        "-ra",
        "tests/",
        "--cov=s2and",
        "--cov-report=term-missing",
        "--cov-fail-under=40",
    ]
    assert calls[0][1] is not None
    assert calls[0][1]["S2AND_BACKEND"] == "python"
    assert "Rust-only tests are expected to report skips in the py-only lane" in capsys.readouterr().out


def test_rust_enabled_lane_reports_skip_reasons_for_all_pytest_runs(monkeypatch) -> None:
    run_ci = _load_run_ci_locally()
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    monkeypatch.delenv("S2AND_BACKEND", raising=False)
    monkeypatch.setattr(run_ci, "sync_deps", lambda *, lock_present, lane: None)
    monkeypatch.setattr(run_ci, "ensure_rust_on_path", lambda: None)
    monkeypatch.setattr(run_ci, "run_maturin_develop_with_retries", lambda: None)
    monkeypatch.setattr(run_ci, "run_ty_checks", lambda: None)
    monkeypatch.setattr(run_ci, "run_uv", lambda args, *, env=None: calls.append((args, env)))

    run_ci.run_typecheck_and_test_lane(lane="rust-enabled", lock_present=True)

    pytest_calls = [args for args, _env in calls if args[:2] == ["run", "pytest"]]
    assert len(pytest_calls) == len(run_ci.RUST_PARITY_TESTS) + 1
    assert all("-ra" in args for args in pytest_calls)
    assert pytest_calls[0] == ["run", "pytest", "-q", "-ra", run_ci.RUST_PARITY_TESTS[0]]
    assert pytest_calls[-1] == [
        "run",
        "pytest",
        "-ra",
        "tests/",
        "--cov=s2and",
        "--cov-report=term-missing",
        "--cov-fail-under=40",
    ]
    assert calls[-1][1] is not None
    assert "S2AND_BACKEND" not in calls[-1][1]
