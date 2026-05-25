from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from scripts._rust_suite import stress_rebuild_cmd


class _FakeRSSMonitor:
    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self.peak_gb = 0.25

    def __enter__(self) -> _FakeRSSMonitor:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def test_parse_args_defaults_to_arrow_build_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["stress_rebuild_cmd.py", "--dataset", "qian"])

    args = stress_rebuild_cmd._parse_args()

    assert args.build_path == "from_arrow_paths"
    assert args.arrow_data_root == stress_rebuild_cmd.DEFAULT_ARROW_DATA_ROOT
    assert args.specter_suffix == stress_rebuild_cmd.DEFAULT_ARROW_SPECTER_SUFFIX


@pytest.mark.parametrize("build_path", ["from_json_paths", "from_dataset"])
def test_parse_args_keeps_legacy_build_paths_explicit(
    monkeypatch: pytest.MonkeyPatch,
    build_path: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["stress_rebuild_cmd.py", "--dataset", "qian", "--build-path", build_path],
    )

    args = stress_rebuild_cmd._parse_args()

    assert args.build_path == build_path


def test_arrow_dataset_paths_delegates_to_eval_prod_models(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import eval_prod_models

    calls: list[tuple[str, str, str]] = []

    def fake_resolve_arrow_dataset_paths(arrow_root: str, dataset_name: str, specter_suffix: str) -> dict[str, str]:
        calls.append((arrow_root, dataset_name, specter_suffix))
        return {"signatures": "signatures.arrow"}

    monkeypatch.setattr(eval_prod_models, "resolve_arrow_dataset_paths", fake_resolve_arrow_dataset_paths)

    result = stress_rebuild_cmd._arrow_dataset_paths("QIAN", "relative/arrow-root", "_specter2.pkl")

    assert result == {"signatures": "signatures.arrow"}
    assert calls == [(str(stress_rebuild_cmd._PROJECT_ROOT / Path("relative/arrow-root")), "qian", "_specter2.pkl")]


def test_build_from_arrow_paths_delegates_to_feature_port(monkeypatch: pytest.MonkeyPatch) -> None:
    from s2and import feature_port

    calls: list[tuple[dict[str, str], dict[str, Any]]] = []

    def fake_build_rust_featurizer_from_arrow_paths(paths: dict[str, str], **kwargs: Any) -> object:
        calls.append((paths, kwargs))
        return "featurizer"

    monkeypatch.setattr(
        feature_port,
        "build_rust_featurizer_from_arrow_paths",
        fake_build_rust_featurizer_from_arrow_paths,
    )

    paths = {"signatures": "signatures.arrow", "name_counts_index": "name-counts"}
    result = stress_rebuild_cmd._build_from_arrow_paths(
        paths=paths,
        compute_reference_features=True,
        preprocess=False,
        num_threads=0,
    )

    assert result == "featurizer"
    assert calls == [
        (
            paths,
            {
                "name_tuples": "filtered",
                "load_name_counts": True,
                "preprocess": False,
                "compute_reference_features": True,
                "cluster_seed_require_value": 0.0,
                "cluster_seed_disallow_value": 10000.0,
                "num_threads": 1,
            },
        )
    ]


def test_run_rebuild_stress_uses_arrow_default_without_legacy_dataset_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_text = ModuleType("s2and.text")
    text_calls: list[bool] = []
    fake_text.set_fasttext_loading_enabled = lambda enabled: text_calls.append(bool(enabled))
    monkeypatch.setitem(sys.modules, "s2and.text", fake_text)

    arrow_paths = {
        "signatures": "signatures.arrow",
        "papers": "papers.arrow",
        "paper_authors": "paper_authors.arrow",
        "specter": "specter2.arrow",
        "clusters": "qian_clusters.json",
        "name_counts_index": "name-counts",
    }
    resolver_calls: list[tuple[str, str, str]] = []
    builder_calls: list[dict[str, Any]] = []

    def fake_arrow_dataset_paths(dataset_name: str, arrow_data_root: str, specter_suffix: str) -> dict[str, str]:
        resolver_calls.append((dataset_name, arrow_data_root, specter_suffix))
        return dict(arrow_paths)

    def fake_build_from_arrow_paths(**kwargs: Any) -> object:
        builder_calls.append(kwargs)
        return object()

    def fail_legacy_dataset_paths(_dataset_name: str) -> dict[str, str | None]:
        raise AssertionError("legacy dataset paths should not be resolved for Arrow builds")

    monkeypatch.setattr(stress_rebuild_cmd, "_arrow_dataset_paths", fake_arrow_dataset_paths)
    monkeypatch.setattr(stress_rebuild_cmd, "_build_from_arrow_paths", fake_build_from_arrow_paths)
    monkeypatch.setattr(stress_rebuild_cmd, "_dataset_paths", fail_legacy_dataset_paths)
    monkeypatch.setattr(stress_rebuild_cmd, "_import_rust_module", lambda: object())
    monkeypatch.setattr(stress_rebuild_cmd, "ProcessTreeRSSMonitor", _FakeRSSMonitor)
    monkeypatch.setattr(
        stress_rebuild_cmd,
        "collect_rust_extension_identity",
        lambda **_kwargs: {"extension": "fake"},
    )
    monkeypatch.setattr(stress_rebuild_cmd, "build_run_metadata", lambda **_kwargs: {"script": "fake"})

    result = stress_rebuild_cmd.run_rebuild_stress(
        dataset="QIAN",
        build_path="from_arrow_paths",
        repeats=1,
        num_threads=4,
        compute_reference_features=True,
        preprocess=False,
        rss_sample_ms=25,
        arrow_data_root="arrow-root",
        specter_suffix="_specter.pickle",
    )

    assert text_calls == [False]
    assert resolver_calls == [("qian", "arrow-root", "_specter.pickle")]
    assert builder_calls == [
        {
            "paths": arrow_paths,
            "compute_reference_features": True,
            "preprocess": False,
            "num_threads": 4,
        }
    ]
    assert result["dataset"] == "qian"
    assert result["build_path"] == "from_arrow_paths"
    assert result["arrow_data_root"] == str(stress_rebuild_cmd._PROJECT_ROOT / "arrow-root")
    assert result["specter_suffix"] == "_specter.pickle"
    assert result["success_count"] == 1
    assert result["failure_count"] == 0
