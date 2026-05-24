from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import s2and.eval as eval_module
import s2and.model as model_module
import s2and.subblocking as subblocking_module
from s2and.data import ANDData
from s2and.eval import incremental_cluster_eval
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer
from s2and.runtime import RuntimeContext
from s2and.sampling import sampling


def _as_anddata(dataset: object) -> ANDData:
    return cast(ANDData, dataset)


def _subblocking_signature(first_name: str, *, middle_name: str = "", orcid: str | None = None):
    return SimpleNamespace(
        author_info_first_normalized_without_apostrophe=first_name,
        author_info_middle_normalized_without_apostrophe=middle_name,
        author_info_first=first_name,
        author_info_middle=middle_name,
        author_info_orcid=orcid,
    )


VALID_ORCID_1 = "0000-0001-2345-6789"
VALID_ORCID_2 = "0000-0001-2345-679X"


def test_model_predict_class0_does_not_require_num_threads_keyword_support():
    class RejectsNumThreads:
        def predict_proba(self, features):
            return np.column_stack((np.zeros(len(features)), np.ones(len(features))))

    predictions, seconds, backend = model_module._predict_class0_with_runtime(
        RejectsNumThreads(),
        np.zeros((2, 1), dtype=np.float64),
        num_threads=2,
    )

    assert np.array_equal(predictions, np.zeros(2, dtype=np.float64))
    assert isinstance(seconds, float)
    assert backend == "python"


def test_cacheable_value_preserves_list_order_but_sorts_sets():
    assert model_module._cacheable_value(["year_diff", "name_counts"]) != model_module._cacheable_value(
        ["name_counts", "year_diff"]
    )
    assert model_module._cacheable_value({"year_diff", "name_counts"}) == model_module._cacheable_value(
        {"name_counts", "year_diff"}
    )


def _expected_upper_triangle_pairs_for_range(
    block_size: int,
    start_offset: int,
    max_pairs: int | None,
) -> list[tuple[int, int]]:
    total_pairs = block_size * (block_size - 1) // 2
    count = total_pairs - start_offset if max_pairs is None else min(max_pairs, total_pairs - start_offset)
    row = 0
    remaining_offset = start_offset
    while row < block_size - 1:
        row_len = block_size - row - 1
        if remaining_offset < row_len:
            break
        remaining_offset -= row_len
        row += 1
    col = row + 1 + remaining_offset
    pairs = []
    for _ in range(count):
        pairs.append((row, col))
        col += 1
        if col >= block_size:
            row += 1
            col = row + 1
    return pairs


@pytest.mark.parametrize(
    ("block_size", "start_offset", "max_pairs"),
    [
        (6, 0, 4),
        (6, 1, 4),
        (6, 14, 4),
        (6, 15, 4),
        (2000, 0, 7),
        (2000, 1998, 7),
        (2000, 1999, 7),
        (2000, 999_500, 7),
        (2000, 1_998_995, 7),
        (2000, 1_999_000, 7),
    ],
)
def test_upper_triangle_indices_for_range_matches_row_major_order(
    block_size: int,
    start_offset: int,
    max_pairs: int | None,
):
    left, right = model_module._upper_triangle_indices_for_range(block_size, start_offset, max_pairs)
    assert list(zip(left.tolist(), right.tolist(), strict=True)) == _expected_upper_triangle_pairs_for_range(
        block_size,
        start_offset,
        max_pairs,
    )


def test_python_predicted_batches_use_effective_pair_chunk_size(monkeypatch):
    dataset = _as_anddata(SimpleNamespace(cluster_seeds_require={}, cluster_seeds_disallow=set()))
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=None,
        n_jobs=1,
        use_cache=False,
        batch_size=10,
    )
    helper_items = [((f"s{i}", f"s{i + 1}", float("nan")), (0, i + 1), "block") for i in range(5)]

    def fake_distance_matrix_helper(self, *_args, **_kwargs):
        del self, _args, _kwargs
        yield from helper_items

    def fake_many_pairs_featurize(pairs, *_args, **_kwargs):
        row_count = len(pairs)
        return np.zeros((row_count, 1), dtype=np.float64), np.zeros(row_count, dtype=np.float64), None

    def fake_predict_and_combine(
        _classifier,
        _nameless_classifier,
        features,
        labels,
        _nameless_features,
        _batch_label,
        **_kwargs,
    ):
        del labels, _kwargs
        return np.arange(len(features), dtype=np.float64), 0.0

    monkeypatch.setattr(Clusterer, "distance_matrix_helper", fake_distance_matrix_helper)
    monkeypatch.setattr(model_module, "many_pairs_featurize", fake_many_pairs_featurize)
    monkeypatch.setattr(model_module, "_predict_and_combine", fake_predict_and_combine)

    batches = list(
        clusterer._iter_python_predicted_distance_matrix_batches(
            {"block": ["s0", "s1", "s2", "s3", "s4", "s5"]},
            dataset,
            {},
            incremental_dont_use_cluster_seeds=False,
            runtime_context=RuntimeContext(
                operation="model_predict",
                requested_backend="python",
                resolved_backend="python",
                use_rust=False,
                run_id="test-python-batches",
                source="default",
            ),
            num_pairs=len(helper_items),
            pair_chunk_size=2,
        )
    )

    assert [len(batch.predictions) for batch in batches] == [2, 2, 1]


def test_fused_constraint_fallback_resumes_at_failed_offset(monkeypatch):
    dataset = _as_anddata(SimpleNamespace(cluster_seeds_require={}, cluster_seeds_disallow=set()))
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=None,
        n_jobs=1,
        use_cache=False,
        batch_size=2,
        use_default_constraints_as_supervision=True,
    )
    signatures = ["s0", "s1", "s2", "s3"]
    signature_index_by_id = {signature_id: idx for idx, signature_id in enumerate(signatures)}

    class FakeRustFeaturizer:
        def get_constraints_block_upper_triangle_indexed(self):
            raise AssertionError("only checked with hasattr")

        def featurize_block_upper_triangle_matrix_indexed(self):
            raise AssertionError("only checked with hasattr")

    backend = model_module._IncrementalConstraintBackend(
        rust_featurizer=FakeRustFeaturizer(),
        use_rust_constraints=True,
        constraint_api_mode="indexed",
        signature_index_by_id=signature_index_by_id,
        suppress_orcid=False,
    )

    def fake_build_backend(*_args, **_kwargs):
        return backend

    def fake_constraints(
        _dataset,
        _block_signature_indices,
        *,
        start_offset,
        max_pairs,
        **_kwargs,
    ):
        if start_offset == 0:
            local_i, local_j = model_module._upper_triangle_indices_for_range(4, start_offset, max_pairs)
            return local_i.tolist(), local_j.tolist(), [None] * len(local_i)
        raise RuntimeError("optional fused failure")

    def fake_resolve_constraint_batch(self, _dataset, pair_batch_ids, **_kwargs):
        del self, _dataset, _kwargs
        return [float("nan")] * len(pair_batch_ids), model_module._ConstraintBatchTelemetry(
            total_pairs=len(pair_batch_ids),
            partial_supervision_hits=0,
            unresolved_pairs=len(pair_batch_ids),
            rust_batch_call_count=0,
            api_mode="test",
            elapsed_seconds=0.0,
        )

    monkeypatch.setattr(model_module, "_build_incremental_constraint_backend", fake_build_backend)
    monkeypatch.setattr(model_module, "get_constraints_block_upper_triangle_indexed_rust", fake_constraints)
    monkeypatch.setattr(model_module, "_handle_optional_rust_exception", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(Clusterer, "_resolve_constraint_batch", fake_resolve_constraint_batch)

    chunks = list(
        clusterer._distance_matrix_chunk_helper_rust(
            {"block": signatures},
            dataset,
            {},
            runtime_context=RuntimeContext(
                operation="constraints",
                requested_backend="rust",
                resolved_backend="rust",
                use_rust=True,
                run_id="test-fused-fallback",
                source="default",
            ),
        )
    )

    assert [chunk.start_offset for chunk in chunks] == [0, 2, 4]


def test_predict_from_arrow_paths_rejects_disallows_with_precomputed_dists_before_build(monkeypatch):
    def fake_build_from_arrow_paths(*_args, **_kwargs):
        raise AssertionError("precomputed dists with disallows should be rejected before Rust featurizer build")

    monkeypatch.setattr(model_module, "build_rust_featurizer_from_arrow_paths", fake_build_from_arrow_paths)

    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=None,
        n_jobs=1,
        use_cache=False,
    )
    dists = {"block": np.asarray([0.5], dtype=np.float64)}
    with pytest.raises(ValueError, match="cluster_seeds_disallow cannot be used with precomputed dists"):
        clusterer.predict_from_arrow_paths(
            {"block": ["s0", "s1"]},
            {"signatures": "signatures.arrow", "papers": "papers.arrow", "paper_authors": "paper_authors.arrow"},
            dists=dists,
            cluster_seeds_disallow={("s0", "s1")},
        )


def test_predict_from_arrow_paths_rejects_none_path_before_rust_builder(monkeypatch):
    def fail_build_from_arrow_paths(*_args, **_kwargs):
        raise AssertionError("invalid Arrow paths should be rejected before Rust featurizer build")

    monkeypatch.setattr(model_module, "build_rust_featurizer_from_arrow_paths", fail_build_from_arrow_paths)

    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=None,
        n_jobs=1,
        use_cache=False,
    )

    with pytest.raises(ValueError, match="signatures.*None"):
        clusterer.predict_from_arrow_paths(
            {"block": ["s0", "s1"]},
            {"signatures": None, "papers": "papers.arrow", "paper_authors": "paper_authors.arrow"},
        )


@pytest.mark.parametrize("bad_path", ["", "   ", Path()])
def test_predict_from_arrow_paths_rejects_empty_path_before_rust_builder(monkeypatch, bad_path):
    def fail_build_from_arrow_paths(*_args, **_kwargs):
        raise AssertionError("invalid Arrow paths should be rejected before Rust featurizer build")

    monkeypatch.setattr(model_module, "build_rust_featurizer_from_arrow_paths", fail_build_from_arrow_paths)

    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=None,
        n_jobs=1,
        use_cache=False,
    )

    with pytest.raises(ValueError, match="signatures"):
        clusterer.predict_from_arrow_paths(
            {"block": ["s0", "s1"]},
            {"signatures": bad_path, "papers": "papers.arrow", "paper_authors": "paper_authors.arrow"},
        )


def test_rust_featurizer_distance_matrix_guards_allocation_before_matrix_build():
    class FakeRustFeaturizer:
        def signature_ids(self):
            return [str(i) for i in range(100)]

        def get_constraints_block_upper_triangle_indexed(self, *_args, **_kwargs):
            raise AssertionError("guard should run before constraint evaluation")

        def featurize_block_upper_triangle_matrix_indexed(self, *_args, **_kwargs):
            raise AssertionError("guard should run before feature evaluation")

    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=None,
        n_jobs=1,
        use_cache=False,
    )

    with pytest.raises(MemoryError, match="Predict exact block exceeds memory budget"):
        clusterer.make_distance_matrices_from_rust_featurizer(
            {"block": [str(i) for i in range(100)]},
            FakeRustFeaturizer(),
            total_ram_bytes=1,
        )


def test_make_distance_matrices_guards_allocation_before_pair_featurization(monkeypatch):
    dataset = _as_anddata(SimpleNamespace(cluster_seeds_require={}, cluster_seeds_disallow=set()))
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=None,
        n_jobs=1,
        use_cache=False,
    )

    monkeypatch.setattr(
        model_module,
        "build_runtime_context",
        lambda _operation: RuntimeContext(
            operation="model_predict",
            requested_backend="python",
            resolved_backend="python",
            use_rust=False,
            run_id="test-matrix-guard",
            source="default",
        ),
    )
    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        model_module,
        "many_pairs_featurize",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("guard should run before pair featurization")),
    )

    with pytest.raises(MemoryError, match="Predict exact block exceeds memory budget"):
        clusterer.make_distance_matrices(
            {"block": [str(i) for i in range(100)]},
            dataset,
            total_ram_bytes=1,
        )


def test_subblocked_altered_presplit_telemetry_is_reset_on_failure(monkeypatch):
    dataset = _as_anddata(SimpleNamespace(cluster_seeds_require={}, cluster_seeds_disallow=set()))
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=None,
        n_jobs=1,
        use_cache=False,
    )
    clusterer._last_subblocked_altered_presplit_telemetry = {"stale": 1}

    def fail_build_subblocks(self, *_args, **_kwargs):
        del self, _args, _kwargs
        raise RuntimeError("subblock failure")

    monkeypatch.setattr(Clusterer, "_build_subblocked_block_dict", fail_build_subblocks)

    with pytest.raises(RuntimeError, match="subblock failure"):
        clusterer._predict_subblocked(
            {"block": ["s0", "s1"]},
            dataset,
            cluster_model_params=None,
            partial_supervision={},
            use_s2_clusters=False,
            incremental_dont_use_cluster_seeds=False,
            batching_threshold=1,
            desired_memory_use=None,
            runtime_context=RuntimeContext(
                operation="cluster_predict",
                requested_backend="python",
                resolved_backend="python",
                use_rust=False,
                run_id="test-subblocked-telemetry",
                source="default",
            ),
            dists=None,
            total_ram_bytes=None,
            restore_rust_cluster_seeds_on_exit=True,
            arrow_paths=None,
        )

    assert clusterer._last_subblocked_altered_presplit_telemetry == {
        "bulk_altered_presplit_applied": 0,
        "bulk_altered_presplit_seconds": 0.0,
    }


def test_residual_first_initial_groups_union_normalized_orcids():
    dataset = SimpleNamespace(
        signatures={
            "s1": _subblocking_signature("alice", orcid="0000-0000-0000-0001"),
            "s2": _subblocking_signature("bob", orcid="0000000000000001"),
            "s3": _subblocking_signature("carol", orcid=None),
        }
    )
    clusterer = SimpleNamespace(use_default_constraints_as_supervision=True, suppress_orcid=False)

    groups = model_module._residual_phase_b_first_initial_groups(clusterer, dataset, ["s1", "s2", "s3"], {})

    assert {frozenset(group) for group in groups} == {frozenset({"s1", "s2"}), frozenset({"s3"})}


def _run_make_subblocks_with_fixed_first_pass(monkeypatch, signatures, first_pass_output, *, maximum_size: int):
    anddata = SimpleNamespace(signatures=signatures, random_seed=0)
    call_count = {"value": 0}

    def fake_subdivide_helper(names, sig_ids, maximum_size, starting_k=2):
        del names, sig_ids, maximum_size, starting_k
        call_count["value"] += 1
        if call_count["value"] == 1:
            return {key: np.array(value) for key, value in first_pass_output.items()}, {}
        raise AssertionError("Unexpected extra call to subdivide_helper")

    def fail_if_specter_called(*_args, **_kwargs):
        raise AssertionError("cluster_with_specter should not be called in this regression test")

    monkeypatch.setattr(subblocking_module, "subdivide_helper", fake_subdivide_helper)
    monkeypatch.setattr(subblocking_module, "cluster_with_specter", fail_if_specter_called)
    return subblocking_module.make_subblocks(
        list(signatures),
        anddata,
        maximum_size=maximum_size,
        first_k_letter_counts_sorted={},
    )


def test_sampling_balanced_homonym_synonym_respects_sample_size():
    all_pairs = [
        ("a", "b", 0),
        ("c", "d", 1),
        ("e", "f", 1),
        ("g", "h", 0),
    ]
    sampled = sampling(
        same_name_different_cluster=[all_pairs[0]],
        different_name_same_cluster=[all_pairs[1]],
        same_name_same_cluster=[all_pairs[2]],
        different_name_different_cluster=[all_pairs[3]],
        sample_size=1,
        balanced_homonyms_and_synonyms=True,
        random_seed=3,
    )
    assert len(sampled) == 1
    assert sampled[0] in all_pairs


def test_incremental_cluster_eval_val_uses_val_block_for_pairwise_metrics(monkeypatch):
    class DummyDataset:
        def __init__(self):
            self.signature_to_cluster_id = {"s_train": "c_train", "s_val": "c_val", "s_test": "c_test"}

        def get_blocks(self):
            return {"b": ["s_train", "s_val", "s_test"]}

        def split_cluster_signatures(self):
            return {"b": ["s_train"]}, {"b": ["s_val"]}, {"b": ["s_test"]}

        def construct_cluster_to_signatures(self, block_dict):
            output = {}
            for signatures in block_dict.values():
                for signature in signatures:
                    cluster_id = self.signature_to_cluster_id[signature]
                    output.setdefault(cluster_id, []).append(signature)
            return output

    class DummyClusterer:
        def predict(self, block_dict, dataset, partial_supervision=None):
            all_signatures = []
            for signatures in block_dict.values():
                all_signatures.extend(signatures)
            return {"pred_cluster": all_signatures}, None

    captured_test_blocks = []

    def fake_pairwise_precision_recall_fscore(true_clus, pred_clus, test_block, strategy="clusters"):
        captured_test_blocks.append(test_block)
        return 0.0, 0.0, 0.0

    monkeypatch.setattr(eval_module, "pairwise_precision_recall_fscore", fake_pairwise_precision_recall_fscore)

    dataset = DummyDataset()
    clusterer = DummyClusterer()
    incremental_cluster_eval(cast(ANDData, dataset), cast(Clusterer, clusterer), split="val")

    assert len(captured_test_blocks) == 2
    assert captured_test_blocks[0] == {"b": ["s_val"]}
    assert captured_test_blocks[1] == {"b": ["s_val"]}


def test_make_subblocks_handles_specter_edge_case_without_unbound_local(monkeypatch):
    class Signature:
        def __init__(self, first_name, middle_name, orcid=None):
            self.author_info_first_normalized_without_apostrophe = first_name
            self.author_info_middle_normalized_without_apostrophe = middle_name
            self.author_info_orcid = orcid

    anddata = SimpleNamespace(signatures={"s1": Signature("ab", "cd")})

    call_count = {"value": 0}

    def fake_subdivide_helper(names, sig_ids, maximum_size, starting_k=2):
        call_count["value"] += 1
        if call_count["value"] == 1:
            return {}, {"ab": np.array(["s1"])}
        if call_count["value"] == 2:
            return {}, {"cd": np.array(["s1"])}
        raise AssertionError("Unexpected extra call to subdivide_helper")

    monkeypatch.setattr(subblocking_module, "subdivide_helper", fake_subdivide_helper)
    monkeypatch.setattr(subblocking_module, "cluster_with_specter", lambda *args, **kwargs: {"0": ["s1"]})

    output = subblocking_module.make_subblocks(["s1"], anddata, maximum_size=2, first_k_letter_counts_sorted={})
    assert output == {"ab|middle=cd": ["s1"]}


def test_make_subblocks_skips_orcid_merge_when_all_targets_are_full(monkeypatch):
    signatures = {
        "s1": _subblocking_signature("aa", orcid=VALID_ORCID_1),
        "s2": _subblocking_signature("aa", orcid=VALID_ORCID_2),
        "s3": _subblocking_signature("bb", orcid=VALID_ORCID_1),
        "s4": _subblocking_signature("bb"),
        "s5": _subblocking_signature("cc", orcid=VALID_ORCID_2),
        "s6": _subblocking_signature("cc"),
    }

    output = _run_make_subblocks_with_fixed_first_pass(
        monkeypatch,
        signatures,
        {
            "a": ["s1", "s2"],
            "b": ["s3", "s4"],
            "c": ["s5", "s6"],
        },
        maximum_size=2,
    )

    assert sorted(sorted(signature_ids) for signature_ids in output.values()) == [
        ["s1", "s2"],
        ["s3", "s4"],
        ["s5", "s6"],
    ]
    assert all(len(signature_ids) <= 2 for signature_ids in output.values())


def test_make_subblocks_skips_orcid_merge_without_enough_capacity(monkeypatch):
    signatures = {
        "s1": _subblocking_signature("aa", orcid=VALID_ORCID_1),
        "x1": _subblocking_signature("aa"),
        "s2": _subblocking_signature("bb", orcid=VALID_ORCID_1),
        "x2": _subblocking_signature("bb"),
        "s3": _subblocking_signature("cc", orcid=VALID_ORCID_1),
        "x3": _subblocking_signature("cc"),
    }

    output = _run_make_subblocks_with_fixed_first_pass(
        monkeypatch,
        signatures,
        {
            "a": ["s1", "x1"],
            "b": ["s2", "x2"],
            "c": ["s3", "x3"],
        },
        maximum_size=3,
    )

    assert sorted(sorted(signature_ids) for signature_ids in output.values()) == [
        ["s1", "x1"],
        ["s2", "x2"],
        ["s3", "x3"],
    ]
    assert all(len(signature_ids) <= 3 for signature_ids in output.values())


def test_make_subblocks_merges_orcid_when_target_has_capacity(monkeypatch):
    signatures = {
        "s1": _subblocking_signature("aa", orcid=VALID_ORCID_1),
        "x1": _subblocking_signature("aa"),
        "s2": _subblocking_signature("bb", orcid=VALID_ORCID_1),
        "s3": _subblocking_signature("cc", orcid=VALID_ORCID_1),
    }

    output = _run_make_subblocks_with_fixed_first_pass(
        monkeypatch,
        signatures,
        {
            "a": ["s1", "x1"],
            "b": ["s2"],
            "c": ["s3"],
        },
        maximum_size=4,
    )

    target_subblock_id = next(subblock_id for subblock_id, sig_ids in output.items() if "s1" in sig_ids)
    assert set(output[target_subblock_id]) == {"s1", "x1", "s2", "s3"}
    assert all(len(signature_ids) <= 4 for signature_ids in output.values())


def test_clusterer_predict_uses_minimum_one_for_incremental_batch_threshold(monkeypatch):
    class Signature:
        def __init__(self, first_name):
            self.author_info_first_normalized_without_apostrophe = first_name

    dataset = _as_anddata(
        SimpleNamespace(
            signatures={
                "m1": Signature("alex"),
                "m2": Signature("alex"),
                "m3": Signature("alex"),
                "m4": Signature("alex"),
                "m5": Signature("alex"),
                "m6": Signature("alex"),
                "s1": Signature("a"),
                "s2": Signature("a"),
            },
            cluster_seeds_require={},
        )
    )

    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    clusterer = Clusterer(featurizer_info=featurizer_info, classifier=None, n_jobs=1, use_cache=False)

    monkeypatch.setattr(
        model_module,
        "_sync_rust_cluster_seeds",
        lambda _dataset, runtime_context=None: None,
    )
    monkeypatch.setattr(
        model_module,
        "make_subblocks",
        lambda block_signatures, _dataset, maximum_size, **_kwargs: {
            "multi_1": ["m1", "m2"],
            "multi_2": ["m3", "m4"],
            "multi_3": ["m5", "m6"],
            "single_1": ["s1", "s2"],
        },
    )

    def fake_predict_helper(self, block_dict, _dataset, *args, **kwargs):
        predicted = {}
        for block_key, signatures in block_dict.items():
            predicted[f"cluster_{block_key}"] = list(signatures)
        return predicted, None

    captured = {"batching_threshold": None}

    def fake_predict_incremental(self, block_signatures, dataset, *args, **kwargs):
        captured["batching_threshold"] = kwargs["batching_threshold"]
        return {
            "clusters": {"merged": list(dataset.cluster_seeds_require.keys()) + list(block_signatures)},
            "phase_b_mode": "exact",
            "phase_b_budget_bytes": 0,
            "phase_b_required_bytes": 0,
        }

    monkeypatch.setattr(Clusterer, "predict_helper", fake_predict_helper)
    monkeypatch.setattr(Clusterer, "predict_incremental", fake_predict_incremental)

    clusterer.predict(
        {"block": ["m1", "m2", "m3", "m4", "m5", "m6", "s1", "s2"]},
        dataset,
        batching_threshold=2,
        desired_memory_use=4,
    )

    assert captured["batching_threshold"] == 1


@pytest.mark.parametrize(
    ("restore_rust_cluster_seeds_on_exit", "expected_sync_calls", "expected_evict_calls"),
    [
        (True, 3, 0),
        (False, 2, 1),
    ],
)
def test_clusterer_predict_optionally_skips_final_rust_seed_restore(
    monkeypatch,
    restore_rust_cluster_seeds_on_exit,
    expected_sync_calls,
    expected_evict_calls,
):
    class Signature:
        def __init__(self, first_name):
            self.author_info_first_normalized_without_apostrophe = first_name

    original_cluster_seeds = {"orig_seed": "orig_cluster"}
    dataset = _as_anddata(
        SimpleNamespace(
            signatures={
                "m1": Signature("alex"),
                "m2": Signature("alex"),
                "s1": Signature("a"),
                "s2": Signature("a"),
            },
            cluster_seeds_require=dict(original_cluster_seeds),
            cluster_seeds_disallow=set(),
        )
    )

    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    clusterer = Clusterer(featurizer_info=featurizer_info, classifier=None, n_jobs=1, use_cache=False)

    sync_snapshots: list[dict[str, str]] = []
    evict_snapshots: list[dict[str, str]] = []

    monkeypatch.setattr(
        model_module,
        "_sync_rust_cluster_seeds",
        lambda dataset_arg, runtime_context=None: sync_snapshots.append(dict(dataset_arg.cluster_seeds_require)),
    )
    monkeypatch.setattr(
        model_module,
        "evict_rust_featurizer",
        lambda dataset_arg: evict_snapshots.append(dict(dataset_arg.cluster_seeds_require)) or True,
    )
    monkeypatch.setattr(
        model_module,
        "make_subblocks",
        lambda block_signatures, _dataset, maximum_size, **_kwargs: {
            "multi_1": ["m1", "m2"],
            "single_1": ["s1", "s2"],
        },
    )

    def fake_predict_helper(self, block_dict, _dataset, *args, **kwargs):
        del self, _dataset, args, kwargs
        block_key = next(iter(block_dict))
        return {"cluster_multi": list(block_dict[block_key])}, None

    def fake_predict_incremental(self, block_signatures, dataset_arg, *args, **kwargs):
        del self, args, kwargs
        return {
            "clusters": {"merged": list(dataset_arg.cluster_seeds_require.keys()) + list(block_signatures)},
            "phase_b_mode": "exact",
            "phase_b_budget_bytes": 0,
            "phase_b_required_bytes": 0,
        }

    monkeypatch.setattr(Clusterer, "predict_helper", fake_predict_helper)
    monkeypatch.setattr(Clusterer, "predict_incremental", fake_predict_incremental)

    pred_clusters, _ = clusterer.predict(
        {"block": ["m1", "m2", "s1", "s2"]},
        dataset,
        batching_threshold=2,
        restore_rust_cluster_seeds_on_exit=restore_rust_cluster_seeds_on_exit,
    )

    assert pred_clusters == {"merged": ["m1", "m2", "s1", "s2"]}
    assert sync_snapshots[:2] == [
        {"m1": "cluster_multi", "m2": "cluster_multi"},
        {"m1": "merged", "m2": "merged", "s1": "merged", "s2": "merged"},
    ]
    assert len(sync_snapshots) == expected_sync_calls
    assert len(evict_snapshots) == expected_evict_calls
    assert dict(dataset.cluster_seeds_require) == original_cluster_seeds

    if restore_rust_cluster_seeds_on_exit:
        assert sync_snapshots[-1] == original_cluster_seeds
        assert evict_snapshots == []
    else:
        assert evict_snapshots == [original_cluster_seeds]

    version_before_mutation = int(dataset._cluster_seeds_version)
    dataset.cluster_seeds_require["new_seed"] = "new_cluster"
    assert int(dataset._cluster_seeds_version) == version_before_mutation + 1


def test_clusterer_predict_subblocked_single_letter_ignores_original_altered_profiles(monkeypatch):
    class Signature:
        def __init__(self, first_name):
            self.author_info_first_normalized_without_apostrophe = first_name

    original_altered = ["m1"]
    dataset = _as_anddata(
        SimpleNamespace(
            signatures={
                "m1": Signature("alex"),
                "m2": Signature("alex"),
                "s1": Signature("a"),
                "s2": Signature("a"),
            },
            cluster_seeds_require={},
            cluster_seeds_disallow=set(),
            altered_cluster_signatures=list(original_altered),
        )
    )

    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    clusterer = Clusterer(featurizer_info=featurizer_info, classifier=None, n_jobs=1, use_cache=False)

    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda _dataset, runtime_context=None: None)
    monkeypatch.setattr(
        model_module,
        "make_subblocks",
        lambda block_signatures, _dataset, maximum_size, **_kwargs: {
            "multi": ["m1", "m2"],
            "single": ["s1", "s2"],
        },
    )

    def fake_predict_helper(self, block_dict, _dataset, *args, **kwargs):
        del self, _dataset, args, kwargs
        block_key = next(iter(block_dict))
        return {"cluster_multi": list(block_dict[block_key])}, None

    observed_altered: list[list[str]] = []

    def fake_predict_incremental(self, block_signatures, dataset_arg, *args, **kwargs):
        del self, args, kwargs
        observed_altered.append(list(dataset_arg.altered_cluster_signatures or []))
        return {
            "clusters": {"merged": list(dataset_arg.cluster_seeds_require.keys()) + list(block_signatures)},
            "phase_b_mode": "exact",
            "phase_b_budget_bytes": 0,
            "phase_b_required_bytes": 0,
        }

    monkeypatch.setattr(Clusterer, "predict_helper", fake_predict_helper)
    monkeypatch.setattr(Clusterer, "predict_incremental", fake_predict_incremental)

    runtime_context = RuntimeContext(
        operation="cluster_predict",
        requested_backend="python",
        resolved_backend="python",
        use_rust=False,
        run_id="test-no-altered-arrow-leak",
        source="default",
    )
    clusterer.predict(
        {"block": ["m1", "m2", "s1", "s2"]},
        dataset,
        batching_threshold=2,
        runtime_context=runtime_context,
    )

    assert observed_altered == [[]]
    assert dataset.altered_cluster_signatures == original_altered


def test_clusterer_predict_subblocked_single_letter_restores_absent_altered_attr(monkeypatch):
    dataset = _as_anddata(SimpleNamespace(cluster_seeds_require={}, cluster_seeds_disallow=set()))
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    clusterer = Clusterer(featurizer_info=featurizer_info, classifier=None, n_jobs=1, use_cache=False)
    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda _dataset, runtime_context=None: None)

    def fake_predict_incremental(self, block_signatures, dataset_arg, *args, **kwargs):
        del self, dataset_arg, args, kwargs
        return {
            "clusters": {"merged": list(block_signatures)},
            "phase_b_mode": "exact",
            "phase_b_budget_bytes": 0,
            "phase_b_required_bytes": 0,
        }

    monkeypatch.setattr(Clusterer, "predict_incremental", fake_predict_incremental)
    runtime_context = RuntimeContext(
        operation="cluster_predict",
        requested_backend="python",
        resolved_backend="python",
        use_rust=False,
        run_id="test-absent-altered-restore",
        source="default",
    )

    clusterer._predict_subblocked_single_letter_incremental_groups(
        {"single": ["s1"]},
        pred_clusters={"seed_cluster": ["seed"]},
        desired_memory_use=100,
        dataset=dataset,
        partial_supervision={},
        runtime_context=runtime_context,
        restore_rust_cluster_seeds_on_exit=True,
    )

    assert not hasattr(dataset, "altered_cluster_signatures")


def test_clusterer_predict_subblocked_single_letter_suppresses_altered_arrow_path(monkeypatch, tmp_path):
    import pyarrow as pa

    class Signature:
        def __init__(self, first_name):
            self.author_info_first_normalized_without_apostrophe = first_name

    altered_path = tmp_path / "altered_cluster_signatures.arrow"
    table = pa.table({"signature_id": pa.array(["m1"], type=pa.string())})
    with pa.OSFile(str(altered_path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    arrow_paths = {"altered_cluster_signatures": str(altered_path)}
    for key in ("signatures", "papers", "paper_authors", "cluster_seeds"):
        path = tmp_path / f"{key}.arrow"
        path.touch()
        arrow_paths[key] = str(path)

    dataset = _as_anddata(
        SimpleNamespace(
            signatures={
                "m1": Signature("alex"),
                "m2": Signature("alex"),
                "s1": Signature("a"),
                "s2": Signature("a"),
            },
            cluster_seeds_require={},
            cluster_seeds_disallow=set(),
            altered_cluster_signatures=None,
            arrow_paths=arrow_paths,
        )
    )

    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    clusterer = Clusterer(featurizer_info=featurizer_info, classifier=None, n_jobs=1, use_cache=False)

    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda _dataset, runtime_context=None: None)
    monkeypatch.setattr(
        model_module,
        "make_subblocks",
        lambda block_signatures, _dataset, maximum_size, **_kwargs: {
            "multi": ["m1", "m2"],
            "single": ["s1", "s2"],
        },
    )

    def fake_predict_helper(self, block_dict, _dataset, *args, **kwargs):
        del self, _dataset, args, kwargs
        block_key = next(iter(block_dict))
        return {"cluster_multi": list(block_dict[block_key])}, None

    observed_model_altered: list[list[str]] = []

    def fake_predict_incremental(self, block_signatures, dataset_arg, *args, **kwargs):
        del self, args, kwargs
        observed_model_altered.append(
            model_module._dataset_altered_cluster_signatures(dataset_arg, dataset_arg.arrow_paths)
        )
        return {
            "clusters": {"merged": list(dataset_arg.cluster_seeds_require.keys()) + list(block_signatures)},
            "phase_b_mode": "exact",
            "phase_b_budget_bytes": 0,
            "phase_b_required_bytes": 0,
        }

    monkeypatch.setattr(Clusterer, "predict_helper", fake_predict_helper)
    monkeypatch.setattr(Clusterer, "predict_incremental", fake_predict_incremental)

    runtime_context = RuntimeContext(
        operation="cluster_predict",
        requested_backend="python",
        resolved_backend="python",
        use_rust=False,
        run_id="test-no-altered-arrow-leak",
        source="default",
    )
    clusterer.predict(
        {"block": ["m1", "m2", "s1", "s2"]},
        dataset,
        batching_threshold=2,
        runtime_context=runtime_context,
    )

    assert observed_model_altered == [[]]
    assert dataset.altered_cluster_signatures is None


def test_clusterer_predict_subblocked_single_letter_skips_sync_for_arrow_promoted_incremental(
    monkeypatch,
    tmp_path,
):
    arrow_paths = {}
    for key in ("signatures", "papers", "paper_authors", "cluster_seeds"):
        path = tmp_path / f"{key}.arrow"
        path.touch()
        arrow_paths[key] = str(path)

    original_cluster_seeds = {"orig_seed": "orig_cluster"}
    dataset = _as_anddata(
        SimpleNamespace(
            signatures={
                "m1": _subblocking_signature("alex"),
                "m2": _subblocking_signature("alex"),
                "s1": _subblocking_signature("a"),
                "s2": _subblocking_signature("a"),
            },
            cluster_seeds_require=dict(original_cluster_seeds),
            cluster_seeds_disallow=set(),
            altered_cluster_signatures=None,
            arrow_paths=arrow_paths,
        )
    )

    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    clusterer = Clusterer(featurizer_info=featurizer_info, classifier=None, n_jobs=1, use_cache=False)

    sync_snapshots: list[dict[str, str]] = []
    evict_snapshots: list[dict[str, str]] = []
    observed_seed_maps: list[dict[str, str]] = []

    monkeypatch.setattr(
        model_module,
        "_sync_rust_cluster_seeds",
        lambda dataset_arg, runtime_context=None: sync_snapshots.append(dict(dataset_arg.cluster_seeds_require)),
    )
    monkeypatch.setattr(
        model_module,
        "evict_rust_featurizer",
        lambda dataset_arg: evict_snapshots.append(dict(dataset_arg.cluster_seeds_require)) or True,
    )

    def fake_predict_incremental(self, block_signatures, dataset_arg, *args, **kwargs):
        del self, args
        assert kwargs["runtime_context"].use_rust is True
        observed_seed_maps.append(dict(dataset_arg.cluster_seeds_require))
        return {
            "clusters": {"merged": list(dataset_arg.cluster_seeds_require.keys()) + list(block_signatures)},
            "phase_b_mode": "exact",
            "phase_b_budget_bytes": 0,
            "phase_b_required_bytes": 0,
        }

    monkeypatch.setattr(Clusterer, "predict_incremental", fake_predict_incremental)

    runtime_context = RuntimeContext(
        operation="cluster_predict",
        requested_backend="rust",
        resolved_backend="rust",
        use_rust=True,
        run_id="test-subblocked-single-letter-arrow-sync-skip",
        source="default",
    )
    result = clusterer._predict_subblocked_single_letter_incremental_groups(
        {"single_a": ["s1"], "single_b": ["s2"]},
        pred_clusters={"cluster_multi": ["m1", "m2"]},
        desired_memory_use=1_000_000,
        dataset=dataset,
        partial_supervision={},
        runtime_context=runtime_context,
        restore_rust_cluster_seeds_on_exit=False,
    )

    assert result == {"merged": ["m1", "m2", "s1", "s2"]}
    assert observed_seed_maps == [
        {"m1": "cluster_multi", "m2": "cluster_multi"},
        {"m1": "merged", "m2": "merged", "s1": "merged"},
    ]
    assert sync_snapshots == []
    assert evict_snapshots == [original_cluster_seeds]
    assert dict(dataset.cluster_seeds_require) == original_cluster_seeds


def test_clusterer_predict_subblocked_single_letter_restores_state_after_incremental_failure(monkeypatch):
    class Signature:
        def __init__(self, first_name):
            self.author_info_first_normalized_without_apostrophe = first_name

    original_cluster_seeds = {"orig_seed": "orig_cluster"}
    original_altered = ["orig_seed"]
    dataset = _as_anddata(
        SimpleNamespace(
            signatures={
                "m1": Signature("alex"),
                "m2": Signature("alex"),
                "s1": Signature("a"),
                "s2": Signature("a"),
            },
            cluster_seeds_require=dict(original_cluster_seeds),
            cluster_seeds_disallow=set(),
            altered_cluster_signatures=list(original_altered),
        )
    )

    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    clusterer = Clusterer(featurizer_info=featurizer_info, classifier=None, n_jobs=1, use_cache=False)

    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda _dataset, runtime_context=None: None)
    monkeypatch.setattr(
        model_module,
        "make_subblocks",
        lambda block_signatures, _dataset, maximum_size, **_kwargs: {
            "multi": ["m1", "m2"],
            "single": ["s1", "s2"],
        },
    )

    def fake_predict_helper(self, block_dict, _dataset, *args, **kwargs):
        del self, _dataset, args, kwargs
        block_key = next(iter(block_dict))
        return {"cluster_multi": list(block_dict[block_key])}, None

    def fake_predict_incremental(self, block_signatures, dataset_arg, *args, **kwargs):
        del self, block_signatures, dataset_arg, args, kwargs
        raise RuntimeError("synthetic incremental failed")

    monkeypatch.setattr(Clusterer, "predict_helper", fake_predict_helper)
    monkeypatch.setattr(Clusterer, "predict_incremental", fake_predict_incremental)

    with pytest.raises(RuntimeError, match="synthetic incremental failed"):
        clusterer.predict({"block": ["m1", "m2", "s1", "s2"]}, dataset, batching_threshold=2)

    assert dict(dataset.cluster_seeds_require) == original_cluster_seeds
    assert dataset.altered_cluster_signatures == original_altered


def test_clusterer_predict_subblocked_arrow_presplits_altered_profile_seeds(
    monkeypatch,
    tmp_path,
):
    import pyarrow as pa

    dataset = _as_anddata(
        SimpleNamespace(
            signatures={
                "seed0": _subblocking_signature("alex"),
                "seed1": _subblocking_signature("alex"),
                "seed2": _subblocking_signature("alex"),
            },
            cluster_seeds_require={"seed0": "7", "seed1": "7", "seed2": "8"},
            cluster_seeds_disallow=set(),
            altered_cluster_signatures=["seed0"],
            name_tuples="filtered",
        )
    )
    original_seeds = dict(dataset.cluster_seeds_require)
    arrow_paths = {
        "signatures": str(tmp_path / "signatures.arrow"),
        "papers": str(tmp_path / "papers.arrow"),
        "paper_authors": str(tmp_path / "paper_authors.arrow"),
        "cluster_seeds": str(tmp_path / "cluster_seeds.arrow"),
    }
    for path in arrow_paths.values():
        Path(path).touch()

    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    clusterer = Clusterer(featurizer_info=featurizer_info, classifier=None, n_jobs=1, use_cache=False)

    def fake_predict_from_arrow_paths(block_dict, arrow_paths_arg, **kwargs):
        assert dict(block_dict) == {"altered_profile_0": ["seed0", "seed1"]}
        assert "cluster_seeds" not in arrow_paths_arg
        assert kwargs["incremental_dont_use_cluster_seeds"] is True
        return {"split0": ["seed0"], "split1": ["seed1"]}, None

    clusterer.predict_from_arrow_paths = cast(Any, fake_predict_from_arrow_paths)
    rust_sentinel = object()
    captured: dict[str, Any] = {}

    def fake_build_rust_featurizer_from_arrow_paths(paths, **kwargs):
        del kwargs
        seed_path = Path(paths["cluster_seeds"])
        assert seed_path != Path(arrow_paths["cluster_seeds"])
        captured["cluster_seeds_path"] = seed_path
        with pa.memory_map(str(seed_path), "r") as source:
            table = pa.ipc.open_file(source).read_all()
        captured["cluster_seeds_arrow"] = {
            str(signature_id): str(cluster_id)
            for signature_id, cluster_id in zip(
                table["signature_id"].to_pylist(),
                table["cluster_id"].to_pylist(),
                strict=True,
            )
        }
        return rust_sentinel

    def fake_predict_multiple(self, block_dict_multiple_letter, **kwargs):
        del self
        assert dict(block_dict_multiple_letter) == {"block": ["seed0", "seed1", "seed2"]}
        assert kwargs["rust_featurizer"] is rust_sentinel
        assert dict(kwargs["dataset"].cluster_seeds_require) == {
            "seed0": "7_0",
            "seed1": "7_1",
            "seed2": "8",
        }
        return {
            "cluster0": ["seed0"],
            "cluster1": ["seed1"],
            "cluster2": ["seed2"],
        }

    def fake_predict_single(self, block_dict_single_letter, *, pred_clusters, dataset, **kwargs):
        del self, kwargs
        assert block_dict_single_letter == {}
        assert dict(dataset.cluster_seeds_require) == {"seed0": "7_0", "seed1": "7_1", "seed2": "8"}
        return pred_clusters

    sync_snapshots: list[dict[str, Any]] = []
    monkeypatch.setattr(
        model_module,
        "build_rust_featurizer_from_arrow_paths",
        fake_build_rust_featurizer_from_arrow_paths,
    )
    monkeypatch.setattr(Clusterer, "_predict_subblocked_multiple_letter_groups", fake_predict_multiple)
    monkeypatch.setattr(Clusterer, "_predict_subblocked_single_letter_incremental_groups", fake_predict_single)
    monkeypatch.setattr(
        model_module,
        "_sync_rust_cluster_seeds",
        lambda dataset_arg, runtime_context=None: sync_snapshots.append(dict(dataset_arg.cluster_seeds_require)),
    )

    runtime_context = RuntimeContext(
        operation="cluster_predict",
        requested_backend="rust",
        resolved_backend="rust",
        use_rust=True,
        run_id="test-bulk-altered-presplit",
        source="default",
    )
    clusters, dists = clusterer._predict_subblocked(
        {"block": ["seed0", "seed1", "seed2"]},
        dataset,
        cluster_model_params=None,
        partial_supervision={},
        use_s2_clusters=False,
        incremental_dont_use_cluster_seeds=False,
        batching_threshold=3,
        desired_memory_use=None,
        runtime_context=runtime_context,
        dists=None,
        total_ram_bytes=None,
        restore_rust_cluster_seeds_on_exit=True,
        arrow_paths=arrow_paths,
    )

    assert clusters == {"cluster0": ["seed0"], "cluster1": ["seed1"], "cluster2": ["seed2"]}
    assert dists is None
    assert captured["cluster_seeds_arrow"] == {"seed0": "7_0", "seed1": "7_1", "seed2": "8"}
    assert not captured["cluster_seeds_path"].exists()
    assert dict(dataset.cluster_seeds_require) == original_seeds
    assert sync_snapshots[-1] == original_seeds
    assert clusterer._last_subblocked_altered_presplit_telemetry["bulk_altered_presplit_applied"] == 1
    assert clusterer._last_subblocked_altered_presplit_telemetry["bulk_altered_presplit_cache_miss_count"] == 1


def test_sync_rust_cluster_seeds_skips_when_unchanged(monkeypatch):
    calls = {"count": 0}

    def fake_update(_dataset, runtime_context=None, **_kwargs):
        del runtime_context
        calls["count"] += 1

    monkeypatch.setattr(model_module, "update_rust_cluster_seeds", fake_update)

    dataset = _as_anddata(
        SimpleNamespace(
            cluster_seeds_require={},
            cluster_seeds_disallow=set(),
            _cluster_seeds_version=1,
        )
    )
    runtime_context = RuntimeContext(
        operation="constraints",
        requested_backend="rust",
        resolved_backend="rust",
        use_rust=True,
        run_id="run-1",
        source="default",
    )

    model_module._sync_rust_cluster_seeds(dataset, runtime_context=runtime_context)
    model_module._sync_rust_cluster_seeds(dataset, runtime_context=runtime_context)
    assert calls["count"] == 1
    assert int(getattr(dataset, "_rust_cluster_seeds_sync_calls", 0)) == 2
    assert int(getattr(dataset, "_rust_cluster_seeds_sync_attempted", 0)) == 1
    assert int(getattr(dataset, "_rust_cluster_seeds_sync_succeeded", 0)) == 1
    assert int(getattr(dataset, "_rust_cluster_seeds_sync_skipped_unchanged", 0)) == 1
    sync_seconds_total = float(getattr(dataset, "_rust_cluster_seeds_sync_seconds_total", 0.0))
    sync_seconds_max = float(getattr(dataset, "_rust_cluster_seeds_sync_seconds_max", 0.0))
    assert isinstance(sync_seconds_total, float)
    assert sync_seconds_max <= sync_seconds_total

    dataset._cluster_seeds_version += 1
    model_module._sync_rust_cluster_seeds(dataset, runtime_context=runtime_context)
    assert calls["count"] == 2
    assert int(getattr(dataset, "_rust_cluster_seeds_sync_calls", 0)) == 3
    assert int(getattr(dataset, "_rust_cluster_seeds_sync_attempted", 0)) == 2
    assert int(getattr(dataset, "_rust_cluster_seeds_sync_succeeded", 0)) == 2
    assert int(getattr(dataset, "_rust_cluster_seeds_sync_skipped_unchanged", 0)) == 1


def test_sync_rust_cluster_seeds_detects_in_place_seed_mutation(monkeypatch):
    calls = {"count": 0}

    def fake_update(_dataset, runtime_context=None, **_kwargs):
        del runtime_context
        calls["count"] += 1

    monkeypatch.setattr(model_module, "update_rust_cluster_seeds", fake_update)

    dataset = _as_anddata(
        SimpleNamespace(
            cluster_seeds_require={"s1": "c1", "s2": "c1"},
            cluster_seeds_disallow={("s1", "s3")},
            _cluster_seeds_version=1,
        )
    )
    runtime_context = RuntimeContext(
        operation="constraints",
        requested_backend="rust",
        resolved_backend="rust",
        use_rust=True,
        run_id="run-2",
        source="default",
    )

    model_module._sync_rust_cluster_seeds(dataset, runtime_context=runtime_context)
    assert calls["count"] == 1

    dataset.cluster_seeds_require["s2"] = "c2"
    model_module._sync_rust_cluster_seeds(dataset, runtime_context=runtime_context)
    assert calls["count"] == 2

    dataset.cluster_seeds_disallow.remove(("s1", "s3"))
    dataset.cluster_seeds_disallow.add(("s2", "s3"))
    model_module._sync_rust_cluster_seeds(dataset, runtime_context=runtime_context)
    assert calls["count"] == 3
    assert int(getattr(dataset, "_rust_cluster_seeds_sync_calls", 0)) == 3
    assert int(getattr(dataset, "_rust_cluster_seeds_sync_attempted", 0)) == 3
    assert int(getattr(dataset, "_rust_cluster_seeds_sync_succeeded", 0)) == 3
    assert int(getattr(dataset, "_rust_cluster_seeds_sync_skipped_unchanged", 0)) == 0


def test_make_distance_matrices_fastcluster_cross_batch_preserves_per_block_order(monkeypatch):
    dataset = _as_anddata(SimpleNamespace(cluster_seeds_require={}, cluster_seeds_disallow=set()))
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    clusterer = Clusterer(
        featurizer_info=featurizer_info,
        classifier=None,
        n_jobs=1,
        use_cache=False,
        batch_size=2,
    )

    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(model_module, "stage_uses_rust", lambda _runtime_context: False)

    batches = [
        model_module._PredictedDistanceMatrixBatch(
            batch_num=0,
            blocks=["a", "a"],
            indices=[(0, 1), (0, 2)],
            predictions=np.asarray([0.1, 0.2], dtype=np.float64),
            batch_seconds=0.0,
        ),
        model_module._PredictedDistanceMatrixBatch(
            batch_num=1,
            blocks=["b", "a"],
            indices=[(0, 1), (0, 3)],
            predictions=np.asarray([9.9, 0.3], dtype=np.float64),
            batch_seconds=0.0,
        ),
        model_module._PredictedDistanceMatrixBatch(
            batch_num=2,
            blocks=["a", "a", "a"],
            indices=[(1, 2), (1, 3), (2, 3)],
            predictions=np.asarray([0.4, 0.5, 0.6], dtype=np.float64),
            batch_seconds=0.0,
        ),
    ]

    def fake_iter_python_batches(self, *_args, **_kwargs):
        del self, _args, _kwargs
        yield from batches

    monkeypatch.setattr(
        model_module.Clusterer,
        "_iter_python_predicted_distance_matrix_batches",
        fake_iter_python_batches,
    )

    output = clusterer.make_distance_matrices(
        {"a": ["s1", "s2", "s3", "s4"], "b": ["t1", "t2"]},
        dataset,
        partial_supervision={},
    )

    expected_a = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)
    expected_b = np.asarray([9.9], dtype=np.float64)
    np.testing.assert_array_equal(output["a"], expected_a.astype(output["a"].dtype))
    np.testing.assert_array_equal(output["b"], expected_b.astype(output["b"].dtype))


def test_propagate_n_jobs_re_raises_unexpected_set_params_error():
    class _ExplodingEstimator:
        def set_params(self, **_kwargs):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        model_module._propagate_n_jobs(_ExplodingEstimator(), 4)
