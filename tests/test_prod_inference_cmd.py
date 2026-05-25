from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

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
