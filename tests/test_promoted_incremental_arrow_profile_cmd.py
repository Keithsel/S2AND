from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import rust_suite
from scripts._rust_suite import promoted_incremental_arrow_profile_cmd as cmd


def test_promoted_incremental_arrow_profile_is_canonical_command() -> None:
    assert "promoted-incremental-arrow-profile" in rust_suite._COMMANDS  # noqa: SLF001
    assert "big-block-incremental" not in rust_suite._COMMANDS  # noqa: SLF001


def test_select_workload_uses_largest_block_and_stable_seed_queries() -> None:
    workload = cmd._select_workload(
        blocks={
            "small": ["x"],
            "large": ["a", "b", "c", "d", "e"],
        },
        signature_to_cluster_id={
            "a": "cluster-1",
            "b": "cluster-1",
            "c": "cluster-2",
            "d": "cluster-3",
            "e": "cluster-3",
        },
        target_block="",
        query_limit=2,
        max_seed_clusters=2,
    )

    assert workload.target_block == "large"
    assert workload.block_signature_count == 5
    assert workload.seed_signature_to_cluster == {"a": "cluster-1", "c": "cluster-2"}
    assert workload.query_signature_ids == ["b", "d"]
    assert workload.block_signatures == ["a", "c", "b", "d"]


def test_run_profiles_arrow_only_incremental_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    arrow_root = tmp_path / "arrow_bundle"
    signatures_path = arrow_root / "datasets" / "dummy" / "signatures.arrow"
    clusters_path = arrow_root / "datasets" / "dummy" / "dummy_clusters.json"
    output_json = tmp_path / "profile" / "summary.json"
    written_seed_maps: dict[Path, dict[str, str]] = {}
    captured_calls: list[dict[str, Any]] = []

    rows = [
        cmd.ArrowSignatureRow("a", "pa", "large", "Ada", "", "Lovelace", None),
        cmd.ArrowSignatureRow("b", "pb", "large", "Ada", "", "Lovelace", None),
        cmd.ArrowSignatureRow("c", "pc", "large", "Grace", "", "Hopper", None),
        cmd.ArrowSignatureRow("d", "pd", "large", "Katherine", "", "Johnson", None),
        cmd.ArrowSignatureRow("x", "px", "small", "Alan", "", "Turing", None),
    ]

    class FakeClusterer:
        use_cache = True
        n_jobs = 0

        def predict_incremental(
            self,
            block_signatures: list[str],
            dataset: cmd.ArrowProfileDataset,
            **kwargs: Any,
        ) -> dict[str, Any]:
            captured_calls.append(
                {
                    "block_signatures": list(block_signatures),
                    "arrow_paths": dict(dataset.arrow_paths),
                    "dataset_name": dataset.name,
                    "signatures": set(dataset.signatures),
                    "kwargs": kwargs,
                    "use_cache": self.use_cache,
                    "n_jobs": self.n_jobs,
                }
            )
            return {
                "clusters": {"cluster-1": block_signatures},
                "incremental_linker_telemetry": {
                    "arrow_promoted_incremental": 1,
                    "candidate_row_count": 2,
                },
            }

    fake_clusterer = FakeClusterer()
    monkeypatch.setattr(
        cmd,
        "_resolve_arrow_dataset_paths",
        lambda root, dataset: {
            "signatures": str(signatures_path),
            "papers": str(arrow_root / "datasets" / "dummy" / "papers.arrow"),
            "paper_authors": str(arrow_root / "datasets" / "dummy" / "paper_authors.arrow"),
            "specter": str(arrow_root / "datasets" / "dummy" / "specter2.arrow"),
            "name_counts_index": str(arrow_root / "name_counts_index"),
            "clusters": str(clusters_path),
        },
    )
    monkeypatch.setattr(cmd, "_read_signature_rows", lambda _path: rows)
    monkeypatch.setattr(
        cmd,
        "_read_signature_to_cluster_id",
        lambda _path: {
            "a": "cluster-1",
            "b": "cluster-1",
            "c": "cluster-2",
            "d": "cluster-3",
            "x": "cluster-4",
        },
    )
    monkeypatch.setattr(
        cmd,
        "collect_rust_extension_identity",
        lambda *, require_release: {"available": True, "require_release": require_release},
    )

    import s2and.incremental_linking.feature_block as feature_block
    import s2and.production_model as production_model

    def fake_write_cluster_seeds_arrow(path: Path, seeds: dict[str, str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake-arrow", encoding="utf-8")
        written_seed_maps[path] = dict(seeds)

    monkeypatch.setattr(feature_block, "write_cluster_seeds_arrow", fake_write_cluster_seeds_arrow)
    monkeypatch.setattr(production_model, "load_production_model", lambda _path: fake_clusterer)

    args = cmd.parse_args(
        [
            "--arrow-root",
            str(arrow_root),
            "--dataset",
            "dummy",
            "--model-path",
            str(tmp_path / "model"),
            "--query-limit",
            "1",
            "--max-seed-clusters",
            "2",
            "--runs",
            "1",
            "--n-jobs",
            "2",
            "--output-dir",
            str(tmp_path / "profile"),
            "--write-json",
            str(output_json),
        ]
    )

    payload = cmd.run(args)

    assert payload["runner"] == "promoted_incremental_arrow_profile"
    assert payload["target_block"] == "large"
    assert payload["seed_signature_count"] == 2
    assert payload["query_signature_count"] == 1
    assert payload["profile_signature_count"] == 3
    assert payload["runs"][0]["telemetry"]["arrow_promoted_incremental"] == 1
    assert output_json.exists()
    assert json.loads(output_json.read_text(encoding="utf-8"))["query_signature_count"] == 1

    assert len(captured_calls) == 1
    assert captured_calls[0]["block_signatures"] == ["a", "c", "b"]
    assert captured_calls[0]["arrow_paths"]["signatures"] == str(signatures_path)
    assert captured_calls[0]["arrow_paths"]["cluster_seeds"].endswith("cluster_seeds_run_0.arrow")
    assert "clusters" not in captured_calls[0]["arrow_paths"]
    assert captured_calls[0]["signatures"] == {"a", "b", "c", "d", "x"}
    assert captured_calls[0]["use_cache"] is False
    assert captured_calls[0]["n_jobs"] == 2
    assert captured_calls[0]["kwargs"]["prevent_new_incompatibilities"] is False
    assert written_seed_maps == {
        Path(captured_calls[0]["arrow_paths"]["cluster_seeds"]): {"a": "cluster-1", "c": "cluster-2"}
    }


def test_run_refuses_unbounded_query_batch_without_full_run() -> None:
    args = cmd.parse_args(["--dataset", "dummy", "--query-limit", "0"])

    with pytest.raises(ValueError, match="--full-run"):
        cmd.run(args)
