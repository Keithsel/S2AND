from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts._rust_suite import largest_block_cmd


class _FakeRSSMonitor:
    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self.peak_gb = 0.5

    def __enter__(self) -> _FakeRSSMonitor:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        return False


def test_run_single_arrow_uses_predict_from_arrow_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from s2and import model as model_module
    from s2and import production_model, text
    from scripts import eval_prod_models

    captured: dict[str, Any] = {}

    class FakeClusterer:
        classifier = object()
        nameless_classifier = object()

        def predict_from_arrow_paths(self, block_dict: dict[str, list[str]], arrow_paths: dict[str, str], **kwargs):
            captured["block_dict"] = block_dict
            captured["arrow_paths"] = arrow_paths
            captured["kwargs"] = kwargs
            self._last_arrow_predict_telemetry = {"mode": "arrow"}
            return {"pred": ["s1", "s2"]}, None

    clusterer = FakeClusterer()
    monkeypatch.setattr(production_model, "load_production_model", lambda _path: clusterer)
    monkeypatch.setattr(model_module, "_ensure_lightgbm_fitted", lambda _model: None)
    monkeypatch.setattr(text, "set_fasttext_loading_enabled", lambda _enabled: None)
    monkeypatch.setattr(largest_block_cmd, "ProcessTreeRSSMonitor", _FakeRSSMonitor)
    monkeypatch.setattr(largest_block_cmd, "collect_rust_extension_identity", lambda **_kwargs: {"rust": True})
    monkeypatch.setattr(largest_block_cmd, "_write_profile_output", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(largest_block_cmd, "build_run_metadata", lambda **_kwargs: {"script": "largest"})
    monkeypatch.setattr(
        eval_prod_models,
        "resolve_arrow_dataset_paths",
        lambda arrow_root, dataset, suffix: {
            "signatures": str(tmp_path / "signatures.arrow"),
            "papers": str(tmp_path / "papers.arrow"),
            "paper_authors": str(tmp_path / "paper_authors.arrow"),
            "specter": str(tmp_path / "specter2.arrow"),
            "clusters": str(tmp_path / "clusters.json"),
            "name_counts_index": str(tmp_path / "name_counts_index"),
        },
    )
    monkeypatch.setattr(
        eval_prod_models,
        "read_arrow_s2_blocks",
        lambda _path: {"a smith": ["s3", "s1", "s2"]},
    )
    monkeypatch.setattr(
        eval_prod_models,
        "read_signature_to_cluster_id",
        lambda _path: {"s1": "truth", "s2": "truth", "s3": "other"},
    )

    result = largest_block_cmd._run_single(
        backend="rust",
        dataset_name="qian",
        block_key="a smith",
        n_jobs=2,
        profile_output_path=str(tmp_path / "profile.txt"),
        model_path="model",
        max_block_size=2,
        quality_check=True,
        input_format="arrow",
        arrow_data_root="arrow-root",
        specter_suffix="_specter2.pkl",
    )

    assert result["input_format"] == "arrow"
    assert result["effective_block_size"] == 2
    assert result["anddata_build_seconds"] == 0.0
    assert result["arrow_predict_telemetry"] == {"mode": "arrow"}
    assert result["quality_metrics"]["true_num_clusters"] == 1
    assert captured["block_dict"] == {"a smith": ["s1", "s2"]}
    assert "clusters" not in captured["arrow_paths"]
    assert captured["arrow_paths"]["name_counts_index"] == str(tmp_path / "name_counts_index")
    assert captured["kwargs"] == {
        "total_ram_bytes": largest_block_cmd.DEFAULT_ARROW_TOTAL_RAM_BYTES,
        "load_name_counts": True,
        "name_tuples": "filtered",
    }


def test_run_single_arrow_rejects_python_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires --backend rust"):
        largest_block_cmd._run_single(
            backend="python",
            dataset_name="qian",
            block_key="a smith",
            n_jobs=1,
            profile_output_path=str(tmp_path / "profile.txt"),
            input_format="arrow",
        )


def test_run_single_arrow_rejects_constraint_sampling(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires JSON/ANDData"):
        largest_block_cmd._run_single(
            backend="rust",
            dataset_name="qian",
            block_key="a smith",
            n_jobs=1,
            profile_output_path=str(tmp_path / "profile.txt"),
            input_format="arrow",
            constraint_sample=1,
        )


def test_run_single_subprocess_forwards_largest_block_arrow_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(cmd, capture_output: bool, text: bool, check: bool, timeout: int, cwd: str):  # noqa: ANN001
        captured["cmd"] = list(cmd)

        class Completed:
            returncode = 0
            stdout = (
                f"{largest_block_cmd.RESULT_JSON_START}\n" '{"ok": true}\n' f"{largest_block_cmd.RESULT_JSON_END}\n"
            )
            stderr = ""

        return Completed()

    monkeypatch.setattr(largest_block_cmd.subprocess, "run", fake_run)

    result = largest_block_cmd._run_single_subprocess(
        backend="rust",
        dataset_name="qian",
        block_key="a smith",
        n_jobs=1,
        profile_output_path="profile.txt",
        model_path="model",
        data_root="data",
        max_block_size=0,
        run_label="rust_arrow",
        timeout_seconds=10,
        quality_check=False,
        constraint_sample=0,
        constraint_sample_seed=42,
        emit_signature_map=False,
        require_rust_release=False,
        input_format="arrow",
        arrow_data_root="arrow-root",
        specter_suffix="_specter2.pkl",
    )

    assert result == {"ok": True}
    cmd = captured["cmd"]
    assert cmd[cmd.index("--input-format") + 1] == "arrow"
    assert cmd[cmd.index("--arrow-data-root") + 1] == "arrow-root"
    assert cmd[cmd.index("--specter-suffix") + 1] == "_specter2.pkl"
