from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import rust_suite
from scripts._rust_suite import prod_inference_cmd


def test_single_run_arrow_delegates_to_arrow_runner(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_single_arrow_run(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"input_format": "arrow", "backend_label": kwargs["run_label"]}

    monkeypatch.setattr(prod_inference_cmd, "_single_arrow_run", fake_single_arrow_run)

    result = prod_inference_cmd._single_run(  # noqa: SLF001
        backend="rust",
        dataset_name="qian",
        n_jobs=2,
        profile_output_path="profile.txt",
        model_path="model",
        arrow_data_root="arrow-root",
        specter_suffix="_specter2.pkl",
        run_label="rust_arrow",
        input_format="arrow",
    )

    assert result == {"input_format": "arrow", "backend_label": "rust_arrow"}
    assert captured["dataset_name"] == "qian"
    assert captured["arrow_data_root"] == "arrow-root"
    assert captured["specter_suffix"] == "_specter2.pkl"
    assert captured["profile_output_path"] == "profile.txt"


def test_single_run_arrow_requires_rust_backend() -> None:
    with pytest.raises(ValueError, match="requires --backend rust"):
        prod_inference_cmd._single_run(  # noqa: SLF001
            backend="python",
            dataset_name="qian",
            n_jobs=2,
            profile_output_path="profile.txt",
            model_path="model",
            input_format="arrow",
        )


def test_run_single_subprocess_forwards_arrow_flags(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(cmd, capture_output: bool, text: bool, check: bool):  # noqa: ANN001
        captured["cmd"] = list(cmd)

        class Completed:
            stdout = (
                f"{prod_inference_cmd.RESULT_JSON_START}\n" '{"ok": true}\n' f"{prod_inference_cmd.RESULT_JSON_END}\n"
            )

        return Completed()

    monkeypatch.setattr(prod_inference_cmd.subprocess, "run", fake_run)

    result = prod_inference_cmd._run_single_subprocess(  # noqa: SLF001
        script_path=Path("prod_inference_cmd.py"),
        backend="rust",
        dataset_name="qian",
        n_jobs=2,
        profile_output_path="profile.txt",
        arrow_data_root="arrow-root",
        specter_suffix="_specter2.pkl",
        input_format="arrow",
    )

    assert result == {"ok": True}
    cmd = captured["cmd"]
    assert "--input-format" in cmd
    assert cmd[cmd.index("--input-format") + 1] == "arrow"
    assert "--arrow-data-root" in cmd
    assert cmd[cmd.index("--arrow-data-root") + 1] == "arrow-root"
    assert "--specter-suffix" in cmd
    assert cmd[cmd.index("--specter-suffix") + 1] == "_specter2.pkl"


def test_main_single_arrow_emits_marked_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_json = tmp_path / "single.json"

    def fake_single_run(**kwargs: Any) -> dict[str, Any]:
        return {"input_format": kwargs["input_format"], "dataset": kwargs["dataset_name"]}

    monkeypatch.setattr(prod_inference_cmd, "_single_run", fake_single_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prod_inference_cmd.py",
            "--mode",
            "single",
            "--backend",
            "rust",
            "--input-format",
            "arrow",
            "--dataset-name",
            "qian",
            "--profile-output-path",
            str(tmp_path / "profile.txt"),
            "--single-write-json",
            str(output_json),
        ],
    )

    prod_inference_cmd.main()

    stdout = capsys.readouterr().out
    assert prod_inference_cmd.RESULT_JSON_START in stdout
    assert prod_inference_cmd.RESULT_JSON_END in stdout
    assert json.loads(output_json.read_text(encoding="utf-8")) == {"dataset": "qian", "input_format": "arrow"}


def test_compare_runs_defaults_to_arrow_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run_single_subprocess(**kwargs: Any) -> dict[str, Any]:
        calls.append(dict(kwargs))
        return {
            "backend_label": kwargs["run_label"],
            "total_latency_seconds": 1.0,
            "peak_rss_gb": 0.1,
            "b3": [1.0, 1.0, 1.0],
            "cluster": [1.0, 1.0, 1.0],
            "cluster_macro": [1.0, 1.0, 1.0],
        }

    monkeypatch.setattr(prod_inference_cmd, "_run_single_subprocess", fake_run_single_subprocess)
    prod_inference_cmd._compare_runs(  # noqa: SLF001
        argparse.Namespace(
            dataset_name="qian",
            n_jobs=1,
            model_path="model",
            data_root="data",
            arrow_data_root="arrow",
            specter_file="",
            specter_suffix="_specter2.pkl",
            rust_warm_featurizer_before_predict=0,
            require_rust_release=0,
            include_python_baseline=False,
            include_rust_from_dataset=False,
            write_json="",
        )
    )

    assert [call["input_format"] for call in calls] == ["arrow"]
    assert [call["run_label"] for call in calls] == ["rust_arrow"]


def test_rust_suite_prod_inference_dispatch_forwards_arrow_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_main() -> None:
        captured["argv"] = list(sys.argv)

    def fake_load_internal_module(module_key: str) -> Any:
        assert module_key == "prod_inference"
        return SimpleNamespace(main=fake_main)

    monkeypatch.setattr(rust_suite, "_load_internal_module", fake_load_internal_module)

    assert (
        rust_suite.main(
            [
                "prod-inference",
                "--mode",
                "single",
                "--backend",
                "rust",
                "--input-format",
                "arrow",
                "--profile-output-path",
                "profile.txt",
            ]
        )
        == 0
    )

    assert captured["argv"][1:] == [
        "--mode",
        "single",
        "--backend",
        "rust",
        "--input-format",
        "arrow",
        "--profile-output-path",
        "profile.txt",
    ]
