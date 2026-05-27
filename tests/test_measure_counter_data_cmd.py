from __future__ import annotations

from pathlib import Path

import pytest

from scripts._rust_suite import measure_counter_data_cmd


def test_build_anddata_resolves_exact_dataset_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import s2and.data as data_module

    dataset_root = tmp_path / "demo"
    dataset_root.mkdir()
    for filename in (
        "demo_signatures.json",
        "demo_papers.json",
        "demo_cluster_seeds.json",
        "demo_clusters.json",
    ):
        (dataset_root / filename).write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}

    class FakeANDData:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(data_module, "ANDData", FakeANDData)

    measure_counter_data_cmd._build_anddata("demo", str(tmp_path))  # noqa: SLF001

    assert captured["signatures"] == str(dataset_root / "demo_signatures.json")
    assert captured["papers"] == str(dataset_root / "demo_papers.json")
    assert captured["clusters"] == str(dataset_root / "demo_clusters.json")
