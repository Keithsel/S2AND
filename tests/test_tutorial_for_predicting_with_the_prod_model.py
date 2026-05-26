from __future__ import annotations

import sys
from typing import Any

import pytest

from scripts import tutorial_for_predicting_with_the_prod_model as tutorial


def test_tutorial_help_exposes_arrow_flags(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["tutorial_for_predicting_with_the_prod_model.py", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        tutorial.main()

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--input-format" in help_text
    assert "--arrow-data-root" in help_text
    assert "--arrow-total-ram-bytes" in help_text


def test_tutorial_desired_memory_requires_batched_json_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["tutorial_for_predicting_with_the_prod_model.py", "--desired-memory-use", "1000"],
    )

    with pytest.raises(ValueError, match="--desired-memory-use requires --batching-threshold"):
        tutorial.main()


def test_tutorial_selects_arrow_route_without_anddata_fallback() -> None:
    calls: list[tuple[str, str, str]] = []

    def resolve_arrow_dataset_paths(arrow_root: str, dataset_name: str, specter_suffix: str) -> dict[str, str]:
        calls.append((arrow_root, dataset_name, specter_suffix))
        return {"signatures": "signatures.arrow"}

    input_format, arrow_paths = tutorial._select_input_route(  # noqa: SLF001
        requested_input_format="auto",
        dataset_name="pubmed",
        arrow_data_root="arrow-root",
        specter_suffix="_specter2.pkl",
        batching_threshold=None,
        desired_memory_use=None,
        warm_rust_featurizer_before_predict=0,
        resolve_arrow_dataset_paths=resolve_arrow_dataset_paths,
    )

    assert input_format == "arrow"
    assert arrow_paths == {"signatures": "signatures.arrow"}
    assert calls == [("arrow-root", "pubmed", "_specter2.pkl")]


def test_tutorial_auto_keeps_json_route_for_json_only_subblocking_knobs() -> None:
    def resolve_arrow_dataset_paths(*_args: Any, **_kwargs: Any) -> dict[str, str]:
        return {"signatures": "signatures.arrow"}

    input_format, arrow_paths = tutorial._select_input_route(  # noqa: SLF001
        requested_input_format="auto",
        dataset_name="pubmed",
        arrow_data_root="arrow-root",
        specter_suffix="_specter2.pkl",
        batching_threshold=5000,
        desired_memory_use=None,
        warm_rust_featurizer_before_predict=0,
        resolve_arrow_dataset_paths=resolve_arrow_dataset_paths,
    )

    assert input_format == "json"
    assert arrow_paths == {"signatures": "signatures.arrow"}


def test_tutorial_arrow_route_rejects_json_only_knobs() -> None:
    def resolve_arrow_dataset_paths(*_args: Any, **_kwargs: Any) -> dict[str, str]:
        return {"signatures": "signatures.arrow"}

    with pytest.raises(ValueError, match="JSON/ANDData tutorial knobs"):
        tutorial._select_input_route(  # noqa: SLF001
            requested_input_format="arrow",
            dataset_name="pubmed",
            arrow_data_root="arrow-root",
            specter_suffix="_specter2.pkl",
            batching_threshold=5000,
            desired_memory_use=None,
            warm_rust_featurizer_before_predict=0,
            resolve_arrow_dataset_paths=resolve_arrow_dataset_paths,
        )
