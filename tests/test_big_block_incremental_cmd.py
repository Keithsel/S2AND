from __future__ import annotations

import json
import os
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts._rust_suite.big_block_incremental_cmd as big_block_incremental_cmd


def _base_args(**overrides):
    args = {
        "mode": "single",
        "backend": "rust",
        "subset_dir": "",
        "target_block": "",
        "total_signatures": 3,
        "seed_signatures": 2,
        "seed_cluster_count": 1,
        "batching_threshold": 12,
        "n_jobs": 20,
        "random_seed": 7,
        "use_orcid_id": 1,
        "specter_path": "",
        "cluster_seeds_path": "",
        "altered_cluster_signatures_path": "",
        "truth_bundle_root": "",
        "truth_dataset": "",
        "truth_table_key": "",
        "truth_split": "test",
        "truth_query_limit": 20,
        "truth_max_candidates_per_query": 25,
        "truth_max_component_members": 20,
        "total_ram_bytes": 34,
        "model_path": "model.pickle",
        "emit_signature_map": 0,
        "write_json": "",
        "single_write_json": "",
        "fail_on_cluster_mismatch": 0,
        "require_rust_release": 0,
        "full_run": True,
    }
    args.update(overrides)
    return Namespace(**args)


def _write_truth_bundle(
    tmp_path: Path,
    *,
    labels: list[dict[str, object]],
    assignments: list[dict[str, object]],
    members: list[dict[str, object]],
) -> Path:
    bundle_root = tmp_path / "truth_bundle"
    labels_dir = bundle_root / "featureless_rows"
    splits_dir = bundle_root / "splits"
    raw_dir = bundle_root / "raw" / "a_khan"
    components_dir = bundle_root / "components"
    labels_dir.mkdir(parents=True)
    splits_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    components_dir.mkdir(parents=True)

    labels_path = labels_dir / "a_khan.parquet"
    pd.DataFrame(labels).to_parquet(labels_path, index=False)
    pd.DataFrame(assignments, columns=["query_group_id", "source_key", "split"]).to_csv(
        splits_dir / "combined_query_split_assignments.csv",
        index=False,
    )
    pd.DataFrame(members).to_parquet(components_dir / "a_khan_members.parquet", index=False)

    signatures = {
        "q1": {"paper_id": "p1", "author_info": {"block": "a khan"}},
        "s1": {"paper_id": "p2", "author_info": {"block": "a khan"}},
    }
    papers = {"p1": {"title": "query"}, "p2": {"title": "seed"}}
    (raw_dir / "signatures.json").write_text(json.dumps(signatures), encoding="utf-8")
    (raw_dir / "papers.json").write_text(json.dumps(papers), encoding="utf-8")
    (bundle_root / "bundle.json").write_text(
        json.dumps(
            {
                "assets": {
                    "featureless_rows": {
                        "files": {
                            "extra_eval_paths.a_khan": str(labels_path.relative_to(bundle_root)),
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return bundle_root


def test_run_single_uses_synthetic_cluster_seeds_when_no_path_is_supplied(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("S2AND_BACKEND", "python")
    monkeypatch.setenv("S2AND_SKIP_FASTTEXT", "0")
    monkeypatch.setenv("OMP_NUM_THREADS", "3")

    subset_dir = tmp_path / "subset"
    subset_dir.mkdir()
    model_path = tmp_path / "model.pickle"
    model_path.write_text("model", encoding="utf-8")

    signatures = {
        "s1": {"paper_id": "p1", "author_info": {"block": "h wang"}},
        "s2": {"paper_id": "p2", "author_info": {"block": "h wang"}},
        "s3": {"paper_id": "p3", "author_info": {"block": "h wang"}},
    }
    papers = {
        "p1": {"authors": [{"author_name": "A"}]},
        "p2": {"authors": [{"author_name": "B"}]},
        "p3": {"authors": [{"author_name": "C"}]},
    }

    synthetic_cluster_seeds = {"s1": {"s2": "require"}}
    monkeypatch.setattr(
        big_block_incremental_cmd,
        "_load_subset_payload",
        lambda _subset_dir: (signatures, papers, "h wang"),
    )
    cluster_seed_build_calls = {}

    def fake_build_cluster_seeds(seed_signature_ids, seed_cluster_count):
        cluster_seed_build_calls["seed_signature_ids"] = list(seed_signature_ids)
        cluster_seed_build_calls["seed_cluster_count"] = seed_cluster_count
        return synthetic_cluster_seeds

    monkeypatch.setattr(big_block_incremental_cmd, "_build_cluster_seeds", fake_build_cluster_seeds)
    monkeypatch.setattr(big_block_incremental_cmd, "collect_rust_extension_identity", lambda **_kwargs: {"rust": 1})
    monkeypatch.setattr(
        big_block_incremental_cmd,
        "ProcessTreeRSSMonitor",
        type(
            "FakeMonitor",
            (),
            {
                "__init__": lambda self, interval_seconds=0.05: setattr(self, "peak_gb", 1.25),
                "__enter__": lambda self: self,
                "__exit__": lambda self, exc_type, exc, tb: False,
            },
        ),
    )

    captured_anddata_kwargs = {}

    class FakeANDData:
        def __init__(self, **kwargs):
            captured_anddata_kwargs.update(kwargs)
            self.cluster_seeds_require = {"s1": 0}
            self.cluster_seeds_disallow = set()
            self.altered_cluster_signatures = None
            self.max_seed_cluster_id = 1
            self._rust_cluster_seeds_sync_calls = 0
            self._rust_cluster_seeds_sync_attempted = 0
            self._rust_cluster_seeds_sync_succeeded = 0
            self._rust_cluster_seeds_sync_skipped_unchanged = 0
            self._rust_cluster_seeds_sync_seconds_total = 0.0
            self._rust_cluster_seeds_sync_seconds_max = 0.0

    monkeypatch.setattr("s2and.data.ANDData", FakeANDData)
    monkeypatch.setattr(
        "s2and.production_model.load_production_model",
        lambda _path: SimpleNamespace(
            classifier=SimpleNamespace(),
            nameless_classifier=SimpleNamespace(),
            use_cache=False,
            n_jobs=0,
            predict_incremental=lambda block, dataset, **kwargs: {
                "clusters": {"0": list(block)},
                "phase_b_mode": "exact",
                "phase_b_budget_bytes": 1,
                "phase_b_required_bytes": 1,
                "phase_b_residual_count": 1,
                "incremental_linker_telemetry": {
                    "candidate_row_count": 11,
                    "pair_count": 22,
                    "link_count": 1,
                    "abstain_count": 1,
                    "query_batch_count": 2,
                    "query_batch_size_max": 3,
                    "memory_predicted_peak_delta_bytes_max": 444,
                    "memory_observed_peak_delta_bytes_max": 555,
                },
            },
        ),
    )
    monkeypatch.setattr("s2and.model._ensure_lightgbm_fitted", lambda *_args, **_kwargs: None)

    result = big_block_incremental_cmd._run_single(
        _base_args(
            subset_dir=str(subset_dir),
            model_path=str(model_path),
        )
    )

    assert len(cluster_seed_build_calls["seed_signature_ids"]) == 2
    assert set(cluster_seed_build_calls["seed_signature_ids"]).issubset(signatures)
    assert cluster_seed_build_calls["seed_cluster_count"] == 1
    assert captured_anddata_kwargs["specter_embeddings"] is None
    assert captured_anddata_kwargs["cluster_seeds"] == synthetic_cluster_seeds
    assert captured_anddata_kwargs["altered_cluster_signatures"] is None
    assert captured_anddata_kwargs["load_name_counts"] is True
    assert captured_anddata_kwargs["use_orcid_id"] is True
    assert result["cluster_seeds_source"] == "synthetic"
    assert result["specter_embeddings_source"] == "unset"
    assert result["altered_cluster_signatures_source"] == "unset"
    assert result["seed_signatures_requested"] == 2
    assert result["seed_signatures"] == 1
    assert result["unassigned_signatures"] == 2
    assert result["estimated_incremental_pairs"] == 2
    assert result["measurement_contract"] == "promoted_rust_predict_incremental"
    assert result["promoted_measurement_available"] is True
    assert result["broad_seed_query_pairs"] == 2
    assert result["phase_b_residual_count"] == 1
    assert result["residual_tail_pair_count"] == 0
    assert result["residual_tail_matrix_bytes"] == 1
    assert result["promoted_candidate_rows"] == 11
    assert result["promoted_scored_pairs"] == 22
    assert result["promoted_scored_pair_reduction"] == -20
    assert result["promoted_scored_pair_reduction_fraction"] == -10.0
    assert result["promoted_candidate_row_ratio_vs_broad_pairs"] == 5.5
    assert result["promoted_scored_pair_ratio_vs_broad_pairs"] == 11.0
    assert result["promoted_link_count"] == 1
    assert result["promoted_abstain_count"] == 1
    assert result["promoted_query_batch_count"] == 2
    assert result["promoted_query_batch_size_max"] == 3
    assert result["promoted_memory_predicted_peak_delta_bytes_max"] == 444
    assert result["promoted_memory_observed_peak_delta_bytes_max"] == 555
    assert os.environ["S2AND_BACKEND"] == "python"
    assert os.environ["S2AND_SKIP_FASTTEXT"] == "0"
    assert os.environ["OMP_NUM_THREADS"] == "3"


def test_run_single_rejects_external_seed_signatures_outside_selected_subset(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("S2AND_BACKEND", "python")
    monkeypatch.delenv("S2AND_SKIP_FASTTEXT", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

    subset_dir = tmp_path / "subset"
    subset_dir.mkdir()
    cluster_seeds_path = tmp_path / "cluster_seeds.json"
    cluster_seeds_path.write_text("{}", encoding="utf-8")
    model_path = tmp_path / "model.pickle"
    model_path.write_text("model", encoding="utf-8")

    signatures = {
        "s1": {"paper_id": "p1", "author_info": {"block": "h wang"}},
        "s2": {"paper_id": "p2", "author_info": {"block": "h wang"}},
        "s3": {"paper_id": "p3", "author_info": {"block": "h wang"}},
    }
    papers = {
        "p1": {"authors": [{"author_name": "A"}]},
        "p2": {"authors": [{"author_name": "B"}]},
        "p3": {"authors": [{"author_name": "C"}]},
    }

    monkeypatch.setattr(
        big_block_incremental_cmd,
        "_load_subset_payload",
        lambda _subset_dir: (signatures, papers, "h wang"),
    )
    monkeypatch.setattr(big_block_incremental_cmd, "collect_rust_extension_identity", lambda **_kwargs: {"rust": 1})
    monkeypatch.setattr(
        big_block_incremental_cmd,
        "ProcessTreeRSSMonitor",
        type(
            "FakeMonitor",
            (),
            {
                "__init__": lambda self, interval_seconds=0.05: setattr(self, "peak_gb", 1.25),
                "__enter__": lambda self: self,
                "__exit__": lambda self, exc_type, exc, tb: False,
            },
        ),
    )

    class FakeANDData:
        def __init__(self, **kwargs):
            self.cluster_seeds_require = {"outside": 0, "s1": 0}
            self.cluster_seeds_disallow = set()
            self.altered_cluster_signatures = None
            self.max_seed_cluster_id = 1
            self._rust_cluster_seeds_sync_calls = 0
            self._rust_cluster_seeds_sync_attempted = 0
            self._rust_cluster_seeds_sync_succeeded = 0
            self._rust_cluster_seeds_sync_skipped_unchanged = 0
            self._rust_cluster_seeds_sync_seconds_total = 0.0
            self._rust_cluster_seeds_sync_seconds_max = 0.0

    monkeypatch.setattr("s2and.data.ANDData", FakeANDData)
    monkeypatch.setattr(
        "s2and.production_model.load_production_model",
        lambda _path: SimpleNamespace(
            classifier=SimpleNamespace(),
            nameless_classifier=SimpleNamespace(),
            use_cache=False,
            n_jobs=0,
            predict_incremental=lambda block, dataset, **kwargs: {
                "clusters": {"0": list(block)},
                "phase_b_mode": "exact",
                "phase_b_budget_bytes": 1,
                "phase_b_required_bytes": 1,
            },
        ),
    )
    monkeypatch.setattr("s2and.model._ensure_lightgbm_fitted", lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="outside the selected subset"):
        big_block_incremental_cmd._run_single(
            _base_args(
                subset_dir=str(subset_dir),
                cluster_seeds_path=str(cluster_seeds_path),
                model_path=str(model_path),
            )
        )
    assert os.environ["S2AND_BACKEND"] == "python"
    assert "S2AND_SKIP_FASTTEXT" not in os.environ
    assert "OMP_NUM_THREADS" not in os.environ


def test_compare_promoted_reports_partition_diff_without_failing(monkeypatch, tmp_path: Path):
    call_kwargs = []

    def fake_run_subprocess_single(_script_path, _args, **kwargs):
        call_kwargs.append(dict(kwargs))
        backend = kwargs["backend_override"]
        common = {
            "target_block": "h wang",
            "seed_clusters_effective": 1,
            "seed_signatures": 2,
            "unassigned_signatures": 2,
            "broad_seed_query_pairs": 4,
            "phase_b_residual_count": 0,
            "residual_tail_pair_count": 0,
            "residual_tail_matrix_bytes": 0,
            "total_runtime_seconds": 10.0 if backend == "python" else 2.0,
            "predict_seconds": 8.0 if backend == "python" else 1.0,
            "peak_rss_gb": 2.0 if backend == "python" else 0.5,
            "cluster_membership_digest": "python-digest" if backend == "python" else "rust-digest",
            "signature_to_cluster_fingerprint": {
                "s1": "a",
                "s2": "a" if backend == "python" else "b",
            },
            "measurement_contract": "legacy_predict_incremental"
            if backend == "python"
            else "promoted_rust_predict_incremental",
            "promoted_measurement_available": backend == "rust",
            "promoted_candidate_rows": 0 if backend == "python" else 2,
            "promoted_scored_pairs": 0 if backend == "python" else 1,
            "promoted_scored_pair_reduction": None if backend == "python" else 3,
            "promoted_scored_pair_reduction_fraction": None if backend == "python" else 0.75,
            "promoted_link_count": 0 if backend == "python" else 1,
            "promoted_abstain_count": 0 if backend == "python" else 1,
            "promoted_query_batch_count": 0 if backend == "python" else 1,
        }
        return {"backend": backend, **common}

    monkeypatch.setattr(big_block_incremental_cmd, "_run_subprocess_single", fake_run_subprocess_single)
    output_path = tmp_path / "compare.json"

    summary = big_block_incremental_cmd._run_compare_promoted(
        _base_args(
            mode="compare_promoted",
            write_json=str(output_path),
            fail_on_cluster_mismatch=1,
        )
    )

    assert summary["legacy_output_parity_is_release_gate"] is False
    assert call_kwargs[0]["backend_override"] == "python"
    assert call_kwargs[0]["batching_threshold_override"] == 13
    assert call_kwargs[1]["backend_override"] == "rust"
    assert "batching_threshold_override" not in call_kwargs[1]
    assert summary["cluster_equivalent"] is False
    assert summary["signature_partition_diff_count"] == 1
    assert summary["signature_partition_diff_fraction"] == 0.5
    assert summary["predict_speedup_vs_baseline"] == 8.0
    assert summary["total_speedup_vs_baseline"] == 5.0
    assert summary["promoted_rust"]["promoted_scored_pair_reduction_fraction"] == 0.75
    assert output_path.exists()


def test_evaluate_truth_link_quality_scores_link_and_abstain_outcomes():
    pred_clusters = {
        "c1": ["q_correct", "seed_pos"],
        "c2": ["q_wrong", "seed_neg"],
        "c3": ["q_false_link", "seed_no_pos"],
        "c4": ["q_false_abstain"],
        "c5": ["q_correct_abstain"],
    }
    truth_context = {
        "seed_signature_to_component": {
            "seed_pos": "component_positive",
            "seed_neg": "component_negative",
            "seed_no_pos": "component_no_positive",
        },
        "truth_queries": {
            "q_correct": {
                "query_group_id": "g_correct",
                "candidate_components": ["component_positive", "component_negative"],
                "positive_components": ["component_positive"],
            },
            "q_wrong": {
                "query_group_id": "g_wrong",
                "candidate_components": ["component_positive", "component_negative"],
                "positive_components": ["component_positive"],
            },
            "q_false_link": {
                "query_group_id": "g_false_link",
                "candidate_components": ["component_no_positive"],
                "positive_components": [],
            },
            "q_false_abstain": {
                "query_group_id": "g_false_abstain",
                "candidate_components": ["component_positive"],
                "positive_components": ["component_positive"],
            },
            "q_correct_abstain": {
                "query_group_id": "g_correct_abstain",
                "candidate_components": ["component_no_positive"],
                "positive_components": [],
            },
        },
        "truth_dataset": "a_khan",
        "truth_table_key": "extra_eval_paths.a_khan",
        "truth_split": "test",
        "target_block": "a khan",
    }

    quality = big_block_incremental_cmd._evaluate_truth_link_quality(pred_clusters, truth_context)

    assert quality["evaluated_queries"] == 5
    assert quality["correct_link"] == 1
    assert quality["wrong_link"] == 1
    assert quality["false_link"] == 1
    assert quality["false_abstain"] == 1
    assert quality["correct_abstain"] == 1
    assert quality["link_precision"] == 0.333333
    assert quality["link_recall"] == 0.333333
    assert quality["link_f1"] == 0.333333
    assert quality["accuracy"] == 0.4


def test_load_truth_bundle_rejects_labels_missing_split_assignments(tmp_path: Path):
    bundle_root = _write_truth_bundle(
        tmp_path,
        labels=[
            {
                "dataset": "a_khan",
                "query_group_id": "g1",
                "query_signature_id": "q1",
                "candidate_component_key": "component_a",
                "label": 1,
                "retrieval_rank": 1,
                "source_key": "source_a",
            }
        ],
        assignments=[],
        members=[{"candidate_component_key": "component_a", "member_index": 0, "signature_id": "s1"}],
    )

    with pytest.raises(ValueError, match="missing split assignments"):
        big_block_incremental_cmd._load_truth_bundle_inputs(
            _base_args(
                truth_bundle_root=str(bundle_root),
                truth_dataset="a_khan",
                truth_table_key="extra_eval_paths.a_khan",
            )
        )


def test_load_truth_bundle_rejects_invalid_retrieval_rank(tmp_path: Path):
    bundle_root = _write_truth_bundle(
        tmp_path,
        labels=[
            {
                "dataset": "a_khan",
                "query_group_id": "g1",
                "query_signature_id": "q1",
                "candidate_component_key": "component_a",
                "label": 1,
                "retrieval_rank": "not-a-rank",
                "source_key": "source_a",
            }
        ],
        assignments=[{"query_group_id": "g1", "source_key": "source_a", "split": "test"}],
        members=[{"candidate_component_key": "component_a", "member_index": 0, "signature_id": "s1"}],
    )

    with pytest.raises(ValueError, match="invalid retrieval_rank"):
        big_block_incremental_cmd._load_truth_bundle_inputs(
            _base_args(
                truth_bundle_root=str(bundle_root),
                truth_dataset="a_khan",
                truth_table_key="extra_eval_paths.a_khan",
            )
        )


def test_load_truth_bundle_rejects_invalid_component_member_index(tmp_path: Path):
    bundle_root = _write_truth_bundle(
        tmp_path,
        labels=[
            {
                "dataset": "a_khan",
                "query_group_id": "g1",
                "query_signature_id": "q1",
                "candidate_component_key": "component_a",
                "label": 1,
                "retrieval_rank": 1,
                "source_key": "source_a",
            }
        ],
        assignments=[{"query_group_id": "g1", "source_key": "source_a", "split": "test"}],
        members=[{"candidate_component_key": "component_a", "member_index": "bad", "signature_id": "s1"}],
    )

    with pytest.raises(ValueError, match="invalid member_index"):
        big_block_incremental_cmd._load_truth_bundle_inputs(
            _base_args(
                truth_bundle_root=str(bundle_root),
                truth_dataset="a_khan",
                truth_table_key="extra_eval_paths.a_khan",
            )
        )
