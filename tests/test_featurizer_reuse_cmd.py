from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from scripts._rust_suite import featurizer_reuse_cmd


def _metrics() -> dict[str, tuple[float, float, float]]:
    return {
        "B3 (P, R, F1)": (0.1, 0.2, 0.345),
        "Cluster (P, R F1)": (0.4, 0.5, 0.678),
        "Cluster Macro (P, R, F1)": (0.7, 0.8, 0.901),
    }


class _FakeRSSMonitor:
    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self.peak_gb = 1.2345

    def __enter__(self) -> _FakeRSSMonitor:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        return False


def _install_common_runtime_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    clusterer: SimpleNamespace,
) -> dict[str, Any]:
    captured: dict[str, Any] = {"fasttext_enabled": []}

    consts_module = ModuleType("s2and.consts")
    consts_module.PROJECT_ROOT_PATH = str(tmp_path)

    production_model_module = ModuleType("s2and.production_model")

    def load_production_model(model_path: str) -> SimpleNamespace:
        captured["model_path"] = model_path
        return clusterer

    production_model_module.load_production_model = load_production_model

    text_module = ModuleType("s2and.text")
    text_module.set_fasttext_loading_enabled = lambda enabled: captured["fasttext_enabled"].append(enabled)

    monkeypatch.setitem(sys.modules, "s2and.consts", consts_module)
    monkeypatch.setitem(sys.modules, "s2and.production_model", production_model_module)
    monkeypatch.setitem(sys.modules, "s2and.text", text_module)
    monkeypatch.setattr(featurizer_reuse_cmd, "RSSMonitor", _FakeRSSMonitor)
    monkeypatch.setattr(
        featurizer_reuse_cmd,
        "collect_rust_extension_identity",
        lambda *, require_release, fail_if_unavailable: {
            "require_release": require_release,
            "fail_if_unavailable": fail_if_unavailable,
        },
    )
    return captured


def test_run_reuse_profile_defaults_to_arrow_bundle_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clusterer = SimpleNamespace()
    captured = _install_common_runtime_fakes(monkeypatch, tmp_path, clusterer)
    resolve_calls: list[tuple[str, str, str]] = []
    eval_calls: list[dict[str, Any]] = []

    eval_prod_models_module = ModuleType("scripts.eval_prod_models")

    def resolve_arrow_dataset_paths(arrow_root: str, dataset_name: str, specter_suffix: str) -> dict[str, str]:
        resolve_calls.append((arrow_root, dataset_name, specter_suffix))
        return {
            "signatures": f"signatures-{len(resolve_calls)}.arrow",
            "clusters": "clusters.json",
        }

    def cluster_eval_arrow(arrow_paths: dict[str, str], clusterer_arg: SimpleNamespace, **kwargs: Any):
        eval_calls.append({"arrow_paths": arrow_paths, "clusterer": clusterer_arg, "kwargs": kwargs})
        clusterer_arg._last_arrow_predict_telemetry = {
            "call_index": len(eval_calls),
            "signatures": arrow_paths["signatures"],
        }
        return _metrics(), {}

    eval_prod_models_module.resolve_arrow_dataset_paths = resolve_arrow_dataset_paths
    eval_prod_models_module.cluster_eval_arrow = cluster_eval_arrow
    monkeypatch.setitem(sys.modules, "scripts.eval_prod_models", eval_prod_models_module)

    result = featurizer_reuse_cmd.run_reuse_profile(dataset_name="kisti", n_jobs=2, repeats=2)

    expected_arrow_root = tmp_path / "s2and" / "data"
    assert result["input_format"] == "arrow"
    assert Path(result["arrow_data_root"]) == expected_arrow_root
    assert result["specter_suffix"] == "_specter2.pkl"
    assert captured["fasttext_enabled"] == [False]
    assert Path(captured["model_path"]) == tmp_path / "s2and" / "data" / "production_model_v1.21"
    assert clusterer.use_cache is False
    assert clusterer.n_jobs == 2

    assert resolve_calls == [
        (str(expected_arrow_root), "kisti", "_specter2.pkl"),
        (str(expected_arrow_root), "kisti", "_specter2.pkl"),
        (str(expected_arrow_root), "kisti", "_specter2.pkl"),
    ]
    assert len(eval_calls) == 4
    assert eval_calls[0]["arrow_paths"] is eval_calls[1]["arrow_paths"]
    assert eval_calls[2]["arrow_paths"] is not eval_calls[0]["arrow_paths"]
    assert eval_calls[3]["arrow_paths"] is not eval_calls[2]["arrow_paths"]
    assert all(call["clusterer"] is clusterer for call in eval_calls)
    assert all(call["kwargs"]["random_seed"] == 42 for call in eval_calls)
    assert all(call["kwargs"]["n_jobs"] == 2 for call in eval_calls)
    assert all(call["kwargs"]["split"] == "test" for call in eval_calls)
    assert all(
        call["kwargs"]["total_ram_bytes"] == featurizer_reuse_cmd.DEFAULT_ARROW_TOTAL_RAM_BYTES for call in eval_calls
    )
    assert result["same_object"]["iterations"][0]["arrow_predict_telemetry"]["call_index"] == 1
    assert result["reinstantiated_object"]["iterations"][1]["arrow_predict_telemetry"]["call_index"] == 4


def test_run_reuse_profile_json_mode_uses_legacy_anddata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clusterer = SimpleNamespace()
    _install_common_runtime_fakes(monkeypatch, tmp_path, clusterer)
    monkeypatch.setattr(featurizer_reuse_cmd.os.path, "exists", lambda _path: True)

    class FakeANDData:
        instances: list[FakeANDData] = []

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            FakeANDData.instances.append(self)

    eval_calls: list[dict[str, Any]] = []
    clear_calls: list[None] = []

    data_module = ModuleType("s2and.data")
    data_module.ANDData = FakeANDData

    eval_module = ModuleType("s2and.eval")

    def cluster_eval(dataset: FakeANDData, clusterer_arg: SimpleNamespace, **kwargs: Any):
        eval_calls.append({"dataset": dataset, "clusterer": clusterer_arg, "kwargs": kwargs})
        return _metrics(), {}

    eval_module.cluster_eval = cluster_eval

    feature_port_module = ModuleType("s2and.feature_port")
    feature_port_module.clear_rust_featurizer_cache = lambda: clear_calls.append(None)
    feature_port_module._rust_featurizer_build_count = lambda dataset: FakeANDData.instances.index(dataset) + 1

    monkeypatch.setitem(sys.modules, "s2and.data", data_module)
    monkeypatch.setitem(sys.modules, "s2and.eval", eval_module)
    monkeypatch.setitem(sys.modules, "s2and.feature_port", feature_port_module)

    result = featurizer_reuse_cmd.run_reuse_profile(
        dataset_name="qian",
        n_jobs=3,
        repeats=2,
        input_format="json",
    )

    assert result["input_format"] == "json"
    assert len(FakeANDData.instances) == 3
    assert len(eval_calls) == 4
    assert len(clear_calls) == 2
    assert eval_calls[0]["dataset"] is eval_calls[1]["dataset"]
    assert eval_calls[2]["dataset"] is not eval_calls[0]["dataset"]
    assert eval_calls[3]["dataset"] is not eval_calls[2]["dataset"]
    assert all(call["clusterer"] is clusterer for call in eval_calls)
    assert all(call["kwargs"] == {"split": "test", "use_s2_clusters": False} for call in eval_calls)

    first_dataset_kwargs = FakeANDData.instances[0].kwargs
    assert first_dataset_kwargs["name"] == "qian"
    assert first_dataset_kwargs["n_jobs"] == 3
    assert first_dataset_kwargs["mode"] == "train"
    assert first_dataset_kwargs["load_name_counts"] is True
    assert first_dataset_kwargs["name_tuples"] == "filtered"
    assert first_dataset_kwargs["use_orcid_id"] is True
    assert first_dataset_kwargs["use_sinonym_overwrite"] is True
    assert result["same_object"]["iterations"][0]["featurizer_build_count"] == 1
    assert result["reinstantiated_object"]["iterations"][1]["featurizer_build_count"] == 3


def test_run_reuse_profile_rejects_unknown_input_format() -> None:
    with pytest.raises(ValueError, match="Unsupported input_format"):
        featurizer_reuse_cmd.run_reuse_profile(
            dataset_name="kisti",
            n_jobs=1,
            repeats=1,
            input_format="parquet",
        )


def test_main_forwards_explicit_json_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def run_reuse_profile(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"ok": True}

    output_path = tmp_path / "reuse.json"
    monkeypatch.setattr(featurizer_reuse_cmd, "run_reuse_profile", run_reuse_profile)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "featurizer_reuse_cmd.py",
            "--dataset-name",
            "pubmed",
            "--n-jobs",
            "5",
            "--repeats",
            "3",
            "--require-rust-release",
            "1",
            "--input-format",
            "json",
            "--write-json",
            str(output_path),
        ],
    )

    featurizer_reuse_cmd.main()

    assert captured == {
        "dataset_name": "pubmed",
        "n_jobs": 5,
        "repeats": 3,
        "require_rust_release": True,
        "input_format": "json",
        "arrow_data_root": featurizer_reuse_cmd.DEFAULT_ARROW_DATA_ROOT,
        "specter_suffix": featurizer_reuse_cmd.DEFAULT_SPECTER_SUFFIX,
    }
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"ok": True}
