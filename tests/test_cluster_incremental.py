from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import numpy as np
import pytest
from lightgbm import LGBMClassifier

import s2and.incremental_linking.production as production_module
import s2and.model as model_module
from s2and.consts import LARGE_DISTANCE
from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.feature_block import write_cluster_seeds_arrow
from s2and.model import Clusterer, IncrementalDistStats
from tests.helpers import tiny_name_counts


def _same_partition(a: dict[str, list[str]], b: dict[str, list[str]]) -> bool:
    """Check that two cluster dicts encode the same partition (same groupings, ignoring cluster IDs)."""

    def _to_partition(clusters: dict[str, list[str]]) -> frozenset:
        return frozenset(frozenset(sigs) for sigs in clusters.values() if sigs)

    return _to_partition(a) == _to_partition(b)


def _clusters(result: dict[str, Any]) -> dict[str, list[str]]:
    return dict(result["clusters"])


def test_raw_arrow_plan_windows_isolate_seed_overlap_queries() -> None:
    windows = production_module._raw_arrow_plan_windows(
        ["query-1", "seed-1", "query-2", "query-3", "seed-2", "query-4"],
        window_size=2,
        seed_signature_ids={"seed-1", "seed-2"},
    )

    assert windows == [["query-1"], ["seed-1"], ["query-2", "query-3"], ["seed-2"], ["query-4"]]


def _patch_fake_raw_arrow_planner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    captured: dict[str, Any] | None = None,
) -> None:
    """Install a planner-shaped Rust fake for promoted raw Arrow orchestration tests."""

    class FakePlanner:
        def __init__(self, _paths: object, query_signature_ids: list[str], **_kwargs: object):
            self._query_signature_ids = tuple(query_signature_ids)
            if captured is not None:
                captured.setdefault("planner_inits", []).append(self._query_signature_ids)

        def build_telemetry(self):
            return {
                "query_signature_count": len(self._query_signature_ids),
                "signature_count": len(self._query_signature_ids),
            }

        def plan(self, query_signature_ids: list[str], **_kwargs: object):
            query_ids = tuple(query_signature_ids)
            if captured is not None:
                captured.setdefault("planner_plans", []).append(query_ids)
            return {"query_signature_ids": query_ids}

    class FakeRustModule:
        RawBlockQueryCandidatePlanner = FakePlanner

    monkeypatch.setattr(production_module.feature_port, "_require_rust_runtime", lambda: FakeRustModule())
    monkeypatch.setattr(
        production_module.feature_port,
        "build_rust_featurizer_from_arrow_paths",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        production_module.runtime_module,
        "feature_block_signature_order_from_raw_candidate_plan",
        lambda raw_plan: SimpleNamespace(signature_ids=tuple(raw_plan.get("query_signature_ids", ()))),
    )
    monkeypatch.setattr(
        production_module.runtime_module,
        "subset_raw_candidate_plan_for_query_ids",
        lambda _raw_plan, query_ids, **_kwargs: {"query_signature_ids": tuple(query_ids)},
    )


def test_finish_incremental_uses_split_inverse_for_altered_incompatibility_check() -> None:
    """A split altered profile should compare new names only against the linked split."""

    def signature(first: str) -> SimpleNamespace:
        return SimpleNamespace(
            author_info_first=first,
            author_info_first_normalized_without_apostrophe=first,
            author_info_last="Jones",
            paper_id=f"p-{first}",
        )

    dataset = SimpleNamespace(
        signatures={
            "seed_david": signature("David"),
            "seed_initial": signature("D"),
            "new_donald": signature("Donald"),
        },
        name_tuples=set(),
        max_seed_cluster_id=0,
    )
    clusterer = SimpleNamespace(
        use_default_constraints_as_supervision=True,
        suppress_orcid=False,
    )

    clusters = Clusterer._finish_incremental_with_seed_links(
        cast(Any, clusterer),
        ["new_donald"],
        cast(Any, dataset),
        {"new_donald": "0_1"},
        {"0_0": "0", "0_1": "0"},
        {"0": ["seed_david", "seed_initial"]},
        prevent_new_incompatibilities=True,
        partial_supervision={},
        runtime_context=cast(Any, SimpleNamespace()),
        split_cluster_seeds_require_inverse={
            "0_0": ["seed_david"],
            "0_1": ["seed_initial"],
        },
    )

    assert clusters == {"0": ["seed_david", "seed_initial", "new_donald"]}


def test_subblocked_altered_presplit_failure_refreshes_telemetry(monkeypatch) -> None:
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=[]),
        classifier=None,
        cluster_model=None,
        n_jobs=1,
    )
    clusterer._last_subblocked_altered_presplit_telemetry = {"stale": 1}
    dataset = SimpleNamespace(cluster_seeds_require={"s1": "c1"})

    monkeypatch.setattr(model_module, "_dataset_altered_cluster_signatures", lambda _dataset, _paths: ["s1"])

    def fail_seed_setup(self, *_args, **_kwargs):
        del self
        raise RuntimeError("seed setup failed")

    monkeypatch.setattr(Clusterer, "_build_incremental_seed_setup", fail_seed_setup)

    with pytest.raises(RuntimeError, match="seed setup failed"):
        clusterer._predict_subblocked(
            {"block": ["s1"]},
            cast(ANDData, dataset),
            cluster_model_params=None,
            partial_supervision={},
            use_s2_clusters=False,
            incremental_dont_use_cluster_seeds=False,
            batching_threshold=10,
            desired_memory_use=None,
            runtime_context=cast(Any, SimpleNamespace(run_id="test")),
            dists=None,
            total_ram_bytes=None,
            restore_rust_cluster_seeds_on_exit=True,
            arrow_paths={"signatures": "signatures.arrow"},
        )

    assert clusterer._last_subblocked_altered_presplit_telemetry == {
        "bulk_altered_presplit_applied": 0,
        "bulk_altered_presplit_seconds": 0.0,
    }


def test_model_presplit_cache_fingerprint_drops_cluster_model_identity() -> None:
    class DummyClusterModel:
        def get_params(self, *, deep: bool = False) -> dict[str, float]:
            return {"eps": 0.5}

    classifier = object()
    nameless_classifier = object()
    base = {
        "classifier": classifier,
        "nameless_classifier": nameless_classifier,
        "featurizer_info": SimpleNamespace(features_to_use=("year_diff",)),
        "nameless_featurizer_info": SimpleNamespace(features_to_use=()),
        "use_default_constraints_as_supervision": True,
        "dont_merge_cluster_seeds": True,
        "suppress_orcid": False,
    }
    first = SimpleNamespace(**base, cluster_model=DummyClusterModel())
    second = SimpleNamespace(**base, cluster_model=DummyClusterModel())

    assert model_module._model_presplit_cache_fingerprint(first) == model_module._model_presplit_cache_fingerprint(
        second
    )


def test_predict_from_rust_featurizer_proxy_exposes_signature_rule_metadata() -> None:
    captured: dict[str, Any] = {}

    class DummyClusterer:
        predict_from_rust_featurizer = Clusterer.predict_from_rust_featurizer

        def predict_helper(self, block_dict, dataset, **kwargs):
            captured["dataset"] = dataset
            captured["kwargs"] = kwargs
            return {"block_0": list(block_dict["block"])}, kwargs["dists"]

    class FakeRustFeaturizer:
        def signature_rule_metadata(self):
            return [
                ("s_alice", "Alice", "0000-0000-0000-0001"),
                ("s_bob", "Bob", None),
                ("s_alicia", "Alicia", "0000-0000-0000-0001"),
            ]

    clusterer = DummyClusterer()
    cast(Any, clusterer).predict_from_rust_featurizer(
        {"block": ["s_alice", "s_bob", "s_alicia"]},
        FakeRustFeaturizer(),
        dists={"block": np.asarray([0.1, 0.2, 0.3], dtype=np.float64)},
        cluster_seeds_require={},
    )

    proxy_dataset = captured["dataset"]
    assert proxy_dataset.signatures["s_alice"].author_info_first == "Alice"
    groups = model_module._residual_phase_b_first_initial_groups(
        SimpleNamespace(use_default_constraints_as_supervision=True, suppress_orcid=False),
        proxy_dataset,
        ["s_alice", "s_bob"],
        {},
    )
    assert groups == [["s_alice"], ["s_bob"]]
    assert model_module._can_skip_orcid_homogeneous_altered_presplit(
        SimpleNamespace(use_default_constraints_as_supervision=True, suppress_orcid=False),
        proxy_dataset,
        ["s_alice", "s_alicia"],
        {},
        set(),
    )


def _seeds_preserved(clusters: dict[str, list[str]], seed_groups: list[list[str]]) -> bool:
    """Each seed group must be entirely contained in one predicted cluster."""
    cluster_sets = [set(sigs) for sigs in clusters.values() if sigs]
    for group in seed_groups:
        group_set = set(group)
        if not any(group_set.issubset(cluster_set) for cluster_set in cluster_sets):
            return False
    return True


def _build_dummy_clusterer_and_dataset(*, name: str = "dummy_chunked") -> tuple[Clusterer, ANDData]:
    dataset = ANDData(
        "tests/dummy/signatures.json",
        "tests/dummy/papers.json",
        clusters="tests/dummy/clusters.json",
        cluster_seeds={"6": {"7": "require"}, "3": {"4": "require"}},
        name=name,
        load_name_counts=tiny_name_counts(),
    )

    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    rng = np.random.RandomState(1)
    X_random = rng.random((10, 6))
    y_random = rng.randint(0, 6, 10)
    clusterer = Clusterer(
        featurizer_info=featurizer_info,
        classifier=LGBMClassifier(random_state=1, data_random_seed=1, feature_fraction_seed=1, verbosity=-1).fit(
            X_random, y_random
        ),
        n_jobs=1,
        use_cache=False,
        use_default_constraints_as_supervision=True,
    )
    return clusterer, dataset


@pytest.fixture
def clusterer_dataset_factory():
    def _factory(*, name: str = "dummy_chunked") -> tuple[Clusterer, ANDData]:
        return _build_dummy_clusterer_and_dataset(name=name)

    return _factory


@pytest.fixture(autouse=True)
def _use_python_backend_by_default(monkeypatch):
    monkeypatch.setenv("S2AND_BACKEND", "python")


def test_predict_incremental(clusterer_dataset_factory):
    # base clustering of the random model would be
    # {'0': ['0', '1', '2'], '1': ['3', '4', '5', '8'], '2': ['6', '7']}
    dummy_clusterer, dummy_dataset = clusterer_dataset_factory(name="dummy")
    block = ["3", "4", "5", "6", "7", "8"]

    # Non-subblocked (monolithic) is the reference output.
    output_monolithic = _clusters(dummy_clusterer.predict_incremental(block, dummy_dataset, batching_threshold=None))
    expected_output = {"0": ["6", "7"], "1": ["3", "4", "5", "8"]}
    assert _same_partition(output_monolithic, expected_output)

    with pytest.raises(ValueError, match="batching_threshold is only supported for promoted Rust"):
        dummy_clusterer.predict_incremental(block, dummy_dataset, batching_threshold=3)

    dummy_dataset.cluster_seeds_disallow = {("5", "7"), ("8", "4"), ("5", "4"), ("8", "7")}
    output = _clusters(dummy_clusterer.predict_incremental(block, dummy_dataset))
    expected_output = {"0": ["6", "7"], "1": ["3", "4"], "2": ["5", "8"]}
    assert _same_partition(output, expected_output)

    dummy_dataset.altered_cluster_signatures = ["1", "5"]
    dummy_dataset.cluster_seeds_require = {"1": 0, "2": 0, "5": 0, "6": 1, "7": 1}
    block = ["3", "4", "8"]
    output = _clusters(dummy_clusterer.predict_incremental(block, dummy_dataset, batching_threshold=None))
    expected_output = {"0": ["1", "2", "5", "8"], "1": ["6", "7", "3", "4"]}
    assert _same_partition(output, expected_output)


def test_predict_incremental_return_contract(clusterer_dataset_factory, monkeypatch):
    block = ["3", "4", "5", "6", "7", "8"]
    clusterer, dataset = clusterer_dataset_factory(name="dummy_incremental_contract")
    canned = {
        "clusters": {"0": ["3", "4"], "1": ["5", "6", "7", "8"]},
        "phase_b_mode": "exact",
        "phase_b_budget_bytes": 123,
        "phase_b_required_bytes": 120,
    }

    def _fake_predict_incremental_helper(self, *args, **kwargs):
        del self, args, kwargs
        return dict(canned)

    monkeypatch.setattr(Clusterer, "_predict_incremental_helper", _fake_predict_incremental_helper)

    payload = clusterer.predict_incremental(block, dataset, batching_threshold=None)
    assert payload == canned

    clusters_only = clusterer.predict_incremental(
        block,
        dataset,
        batching_threshold=None,
        return_clusters_only=True,
    )
    assert clusters_only == canned["clusters"]


def test_promoted_incremental_orcid_fanout_by_query_counts_matching_components() -> None:
    dataset = SimpleNamespace(
        signatures={
            "q": SimpleNamespace(author_info_orcid=" 0000-0000-0000-0001 "),
            "blank": SimpleNamespace(author_info_orcid="   "),
            "other": SimpleNamespace(author_info_orcid="0000-0000-0000-0002"),
            "seed_a": SimpleNamespace(author_info_orcid=" 0000-0000-0000-0001 "),
            "seed_b": SimpleNamespace(author_info_orcid="0000-0000-0000-0001"),
            "seed_c": SimpleNamespace(author_info_orcid="0000-0000-0000-0003"),
            "seed_blank": SimpleNamespace(author_info_orcid="   "),
        }
    )
    fanout = production_module.promoted_incremental_orcid_fanout_by_query(
        dataset,  # type: ignore[arg-type]
        ["q", "blank", "other"],
        {"seed_a": "cluster_a", "seed_b": "cluster_b", "seed_c": "cluster_b", "seed_blank": "cluster_blank"},
        orcid_enabled=True,
    )

    assert fanout == {"q": (2, 3)}
    assert (
        production_module.promoted_incremental_orcid_fanout_by_query(
            dataset,  # type: ignore[arg-type]
            ["q"],
            {"seed_a": "cluster_a"},
            orcid_enabled=False,
        )
        == {}
    )


def test_predict_incremental_rust_promoted_linker_uses_seed_link_seam(clusterer_dataset_factory, monkeypatch):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_rust_incremental_linker")
    block = ["3", "4", "5", "6", "7", "8"]
    residual_blocks: list[list[str]] = []
    residual_total_ram_bytes: list[int | None] = []
    runtime_context = SimpleNamespace(
        operation="cluster_predict_incremental",
        requested_backend="rust",
        resolved_backend="rust",
        use_rust=True,
        run_id="test-rust-promoted-seed-link-seam",
        source="S2AND_BACKEND",
    )

    def fake_predict_helper(block_dict, dataset_arg, partial_supervision, runtime_context, total_ram_bytes=None):
        del dataset_arg, partial_supervision, runtime_context
        residual_blocks.append(list(block_dict["block"]))
        residual_total_ram_bytes.append(total_ram_bytes)
        return {"residual_cluster": list(block_dict["block"])}, None

    clusterer.predict_helper = cast(Any, fake_predict_helper)
    monkeypatch.setattr(
        model_module,
        "_resolve_total_ram_bytes_for_incremental",
        lambda _total=None: (1_000_000_000, "test"),
    )
    monkeypatch.setattr(model_module, "build_runtime_context", lambda _operation, **_kwargs: runtime_context)
    monkeypatch.setattr(model_module.memory_budget, "current_rss_bytes_best_effort", lambda _total: (1_000, "rss:test"))
    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_module, "_get_rust_featurizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        model_module,
        "_build_incremental_constraint_backend",
        lambda *args, **kwargs: SimpleNamespace(rust_featurizer=None),
    )

    import s2and.incremental_linking.artifact as artifact_module
    import s2and.incremental_linking.query_adapter as query_adapter_module
    import s2and.incremental_linking.runtime as runtime_module

    artifact = SimpleNamespace(metadata=SimpleNamespace(retrieval_top_k=25))
    retriever = object()
    captured_inputs: dict[str, Any] = {}
    captured_runtime: dict[str, Any] = {}
    monkeypatch.setattr(artifact_module, "load_incremental_linking_artifact", lambda _path: artifact)

    def fake_build_inputs(**kwargs):
        captured_inputs.update(kwargs)
        query_by_signature_id = {
            str(signature_id): f"query-{signature_id}" for signature_id in kwargs["query_signature_ids"]
        }
        query_view_by_signature_id = {str(signature_id): "full" for signature_id in kwargs["query_signature_ids"]}
        return SimpleNamespace(
            queries=tuple(query_by_signature_id[signature_id] for signature_id in kwargs["query_signature_ids"]),
            query_by_signature_id=query_by_signature_id,
            query_views=tuple(
                query_view_by_signature_id[signature_id] for signature_id in kwargs["query_signature_ids"]
            ),
            query_view_by_signature_id=query_view_by_signature_id,
            retriever=retriever,
            summary_by_component={},
        )

    monkeypatch.setattr(query_adapter_module, "build_incremental_linker_inputs", fake_build_inputs)
    monkeypatch.setattr(
        query_adapter_module,
        "build_name_count_rarity_row_signals",
        lambda *args, **kwargs: {},
    )

    def fake_private_runtime(clusterer_arg, artifact_arg, **kwargs):
        captured_runtime["clusterer"] = clusterer_arg
        captured_runtime["artifact"] = artifact_arg
        captured_runtime.update(kwargs)
        return SimpleNamespace(
            linked_signature_clusters={"5": "1"},
            telemetry={"query_count": 2, "link_count": 1, "abstain_count": 1},
        )

    monkeypatch.setattr(
        runtime_module,
        "_predict_incremental_link_or_abstain_production_private",
        fake_private_runtime,
    )

    result = clusterer.predict_incremental(
        block,
        dataset,
        batching_threshold=None,
    )

    assert captured_inputs["query_signature_ids"] == ["5", "8"]
    assert captured_inputs["query_view"] is None
    assert captured_inputs["orcid_enabled"] is True
    assert captured_runtime["query_signature_ids"] == ["5", "8"]
    assert captured_runtime["query_view"] == ("full", "full")
    assert captured_runtime["queries"] == ("query-5", "query-8")
    assert captured_runtime["retriever"] is retriever
    assert captured_runtime["artifact"] is artifact
    assert captured_runtime["total_ram_bytes"] == 1_000_000_000
    assert captured_runtime["seed_setup"][0] == captured_inputs["cluster_seeds_require"]
    assert residual_blocks == []
    assert residual_total_ram_bytes == []
    assert any(set(signatures) == {"3", "4", "5"} for signatures in result["clusters"].values())
    assert any(set(signatures) == {"8"} for signatures in result["clusters"].values())
    assert result["incremental_linker_query_view"] == "full"
    assert result["incremental_linker_telemetry"]["query_view_full_count"] == 2
    assert result["incremental_linker_telemetry"]["link_count"] == 1


def test_predict_incremental_promoted_linker_respects_suppress_orcid(
    clusterer_dataset_factory,
    monkeypatch,
):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_rust_incremental_linker_suppress_orcid")
    clusterer.suppress_orcid = True
    runtime_context = SimpleNamespace(
        operation="cluster_predict_incremental",
        requested_backend="rust",
        resolved_backend="rust",
        use_rust=True,
        run_id="test-rust-promoted-suppress-orcid",
        source="S2AND_BACKEND",
    )
    captured_inputs: dict[str, Any] = {}

    def fake_predict_helper(block_dict, dataset_arg, partial_supervision, runtime_context, total_ram_bytes=None):
        del dataset_arg, partial_supervision, runtime_context, total_ram_bytes
        return {"residual_cluster": list(block_dict["block"])}, None

    clusterer.predict_helper = cast(Any, fake_predict_helper)
    monkeypatch.setattr(
        model_module,
        "_resolve_total_ram_bytes_for_incremental",
        lambda _total=None: (1_000_000_000, "test"),
    )
    monkeypatch.setattr(model_module, "build_runtime_context", lambda _operation, **_kwargs: runtime_context)
    monkeypatch.setattr(model_module.memory_budget, "current_rss_bytes_best_effort", lambda _total: (1_000, "rss:test"))
    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_module, "_get_rust_featurizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        model_module,
        "_build_incremental_constraint_backend",
        lambda *args, **kwargs: SimpleNamespace(rust_featurizer=None),
    )

    import s2and.incremental_linking.artifact as artifact_module
    import s2and.incremental_linking.query_adapter as query_adapter_module
    import s2and.incremental_linking.runtime as runtime_module

    monkeypatch.setattr(
        artifact_module,
        "load_incremental_linking_artifact",
        lambda _path: SimpleNamespace(metadata=SimpleNamespace(retrieval_top_k=25)),
    )

    def fake_build_inputs(**kwargs):
        captured_inputs.update(kwargs)
        query_by_signature_id = {
            str(signature_id): f"query-{signature_id}" for signature_id in kwargs["query_signature_ids"]
        }
        return SimpleNamespace(
            queries=tuple(query_by_signature_id[signature_id] for signature_id in kwargs["query_signature_ids"]),
            query_by_signature_id=query_by_signature_id,
            query_views=tuple("full" for _signature_id in kwargs["query_signature_ids"]),
            query_view_by_signature_id={str(signature_id): "full" for signature_id in kwargs["query_signature_ids"]},
            retriever=object(),
            summary_by_component={},
        )

    monkeypatch.setattr(query_adapter_module, "build_incremental_linker_inputs", fake_build_inputs)
    monkeypatch.setattr(query_adapter_module, "build_name_count_rarity_row_signals", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        runtime_module,
        "_predict_incremental_link_or_abstain_production_private",
        lambda *args, **kwargs: SimpleNamespace(
            linked_signature_clusters={},
            telemetry={"query_count": 1, "link_count": 0, "abstain_count": 1},
        ),
    )

    clusterer.predict_incremental(["3", "4", "5"], dataset, batching_threshold=None)

    assert captured_inputs["orcid_enabled"] is False


def test_predict_incremental_promoted_linker_passes_orcid_fanout_floor_to_limits(
    clusterer_dataset_factory,
    monkeypatch,
):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_rust_incremental_linker_orcid_fanout_floor")
    block = ["3", "4", "5", "6", "7", "8"]
    for signature_id in ("3", "5", "6"):
        dataset.signatures[signature_id] = dataset.signatures[signature_id]._replace(
            author_info_orcid="0000-0000-0000-0001"
        )
    runtime_context = SimpleNamespace(
        operation="cluster_predict_incremental",
        requested_backend="rust",
        resolved_backend="rust",
        use_rust=True,
        run_id="test-rust-promoted-orcid-fanout-floor",
        source="S2AND_BACKEND",
    )
    limit_calls: list[dict[str, Any]] = []

    def fake_predict_helper(block_dict, dataset_arg, partial_supervision, runtime_context, total_ram_bytes=None):
        del dataset_arg, partial_supervision, runtime_context, total_ram_bytes
        return {"residual_cluster": list(block_dict["block"])}, None

    def fake_limits(**kwargs):
        limit_calls.append(dict(kwargs))
        query_count = int(kwargs["query_count"])
        return _mock_promoted_limits(query_count=query_count, query_batch_size=max(1, query_count))

    clusterer.predict_helper = cast(Any, fake_predict_helper)
    monkeypatch.setattr(
        model_module,
        "_resolve_total_ram_bytes_for_incremental",
        lambda _total=None: (1_000_000_000, "test"),
    )
    monkeypatch.setattr(model_module, "build_runtime_context", lambda _operation, **_kwargs: runtime_context)
    monkeypatch.setattr(model_module.memory_budget, "current_rss_bytes_best_effort", lambda _total: (1_000, "rss:test"))
    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_module, "_get_rust_featurizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        model_module,
        "_build_incremental_constraint_backend",
        lambda *args, **kwargs: SimpleNamespace(rust_featurizer=None),
    )
    monkeypatch.setattr(production_module, "compute_promoted_incremental_limits", fake_limits)

    import s2and.incremental_linking.artifact as artifact_module
    import s2and.incremental_linking.query_adapter as query_adapter_module
    import s2and.incremental_linking.runtime as runtime_module

    monkeypatch.setattr(
        artifact_module,
        "load_incremental_linking_artifact",
        lambda _path: SimpleNamespace(metadata=SimpleNamespace(retrieval_top_k=1)),
    )

    def fake_build_inputs(**kwargs):
        query_by_signature_id = {
            str(signature_id): f"query-{signature_id}" for signature_id in kwargs["query_signature_ids"]
        }
        return SimpleNamespace(
            queries=tuple(query_by_signature_id[signature_id] for signature_id in kwargs["query_signature_ids"]),
            query_by_signature_id=query_by_signature_id,
            query_views=tuple("full" for _signature_id in kwargs["query_signature_ids"]),
            query_view_by_signature_id={str(signature_id): "full" for signature_id in kwargs["query_signature_ids"]},
            retriever=object(),
            summary_by_component={},
        )

    monkeypatch.setattr(query_adapter_module, "build_incremental_linker_inputs", fake_build_inputs)
    monkeypatch.setattr(query_adapter_module, "build_name_count_rarity_row_signals", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        runtime_module,
        "_predict_incremental_link_or_abstain_production_private",
        lambda *args, **kwargs: SimpleNamespace(
            linked_signature_clusters={},
            telemetry={"query_count": len(kwargs["query_signature_ids"]), "link_count": 0, "abstain_count": 0},
        ),
    )

    clusterer.predict_incremental(block, dataset, batching_threshold=None)

    assert limit_calls[0]["retrieval_top_k"] == 1
    assert limit_calls[0]["candidate_rows_per_query_floor"] == 2
    assert limit_calls[0]["pairs_per_query_floor"] == 4
    assert limit_calls[0]["candidate_rows_total_floor"] == 3
    assert limit_calls[0]["pairs_total_floor"] == 6
    assert limit_calls[1]["candidate_rows_per_query_floor"] == 2
    assert limit_calls[1]["pairs_per_query_floor"] == 4
    assert limit_calls[1]["candidate_rows_total_floor"] == 3
    assert limit_calls[1]["pairs_total_floor"] == 6


def test_promoted_incremental_orcid_fanout_skips_seed_scan_without_query_orcids(monkeypatch):
    calls: list[str] = []

    def fake_signature_orcid(_dataset, signature_id):
        calls.append(str(signature_id))
        if str(signature_id).startswith("seed"):
            raise AssertionError("seed scan should be skipped when no query has an ORCID")
        return None

    monkeypatch.setattr(production_module, "_signature_orcid", fake_signature_orcid)

    fanout = production_module.promoted_incremental_orcid_fanout_by_query(
        cast(ANDData, SimpleNamespace()),
        ["query-1", "query-2"],
        {"seed-1": "component-1"},
        orcid_enabled=True,
    )

    assert fanout == {}
    assert calls == ["query-1", "query-2"]


def test_predict_incremental_explicit_rust_backend_uses_promoted_linker_by_default(
    clusterer_dataset_factory,
    monkeypatch,
):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_rust_incremental_linker_default")
    block = ["3", "4", "5", "6", "7", "8"]
    runtime_context = SimpleNamespace(
        operation="cluster_predict_incremental",
        requested_backend="rust",
        resolved_backend="rust",
        use_rust=True,
        run_id="test-rust-promoted-incremental",
        source="S2AND_BACKEND",
    )
    captured: dict[str, Any] = {}
    promoted_payload = {
        "clusters": {"promoted": list(block)},
        "phase_b_mode": "exact",
        "phase_b_budget_bytes": 0,
        "phase_b_required_bytes": 0,
    }

    monkeypatch.setattr(model_module, "build_runtime_context", lambda _operation, **_kwargs: runtime_context)
    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *args, **kwargs: None)

    def fake_promoted_mode(self, block_signatures, dataset_arg, **kwargs):
        captured["self"] = self
        captured["block_signatures"] = list(block_signatures)
        captured["dataset"] = dataset_arg
        captured.update(kwargs)
        return dict(promoted_payload)

    monkeypatch.setattr(Clusterer, "_predict_incremental_promoted_linker", fake_promoted_mode)

    result = clusterer.predict_incremental(block, dataset, batching_threshold=None)

    assert result == promoted_payload
    assert captured["self"] is clusterer
    assert captured["block_signatures"] == block
    assert captured["dataset"] is dataset
    assert captured["runtime_context"] is runtime_context


def test_predict_incremental_auto_backend_uses_promoted_linker_when_auto_resolves_to_rust(
    clusterer_dataset_factory,
    monkeypatch,
):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_auto_incremental_linker_default")
    block = ["3", "4", "5", "6", "7", "8"]
    runtime_context = SimpleNamespace(
        operation="cluster_predict_incremental",
        requested_backend="auto",
        resolved_backend="rust",
        use_rust=True,
        run_id="test-auto-promoted-incremental",
        source="S2AND_BACKEND",
    )
    promoted_payload = {
        "clusters": {"promoted": list(block)},
        "phase_b_mode": "exact",
        "phase_b_budget_bytes": 0,
        "phase_b_required_bytes": 0,
    }

    monkeypatch.setattr(model_module, "build_runtime_context", lambda _operation, **_kwargs: runtime_context)
    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        Clusterer,
        "_predict_incremental_promoted_linker",
        lambda *args, **kwargs: dict(promoted_payload),
    )

    assert clusterer.predict_incremental(block, dataset, batching_threshold=None) == promoted_payload


def test_predict_incremental_auto_uses_arrow_promoted_linker_when_dataset_seed_map_exists(
    clusterer_dataset_factory,
    monkeypatch,
    tmp_path,
):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_auto_incremental_arrow")
    dataset.cluster_seeds_require = {"6": "0", "7": "0", "3": "1", "4": "1"}
    dataset.cluster_seeds_disallow = {("5", "6")}
    block = ["3", "4", "5", "6", "7", "8"]
    runtime_context = SimpleNamespace(
        operation="cluster_predict_incremental",
        requested_backend="rust",
        resolved_backend="rust",
        use_rust=True,
        run_id="test-rust-promoted-incremental-arrow",
        source="S2AND_BACKEND",
    )
    arrow_paths = {}
    for key, filename in {
        "signatures": "signatures.arrow",
        "papers": "papers.arrow",
        "paper_authors": "paper_authors.arrow",
    }.items():
        path = tmp_path / filename
        path.touch()
        arrow_paths[key] = str(path)
    dataset.arrow_paths = arrow_paths
    captured: dict[str, Any] = {}
    sync_calls: list[object] = []

    class FakeArtifact:
        metadata = SimpleNamespace(retrieval_top_k=25)

    def fake_raw_arrow_linker(clusterer_arg, artifact_arg, **kwargs):
        import pyarrow as pa

        captured["clusterer"] = clusterer_arg
        captured["artifact"] = artifact_arg
        captured["arrow_paths"] = dict(kwargs["arrow_paths"])
        captured["query_signature_ids"] = tuple(kwargs["query_signature_ids"])
        with pa.memory_map(captured["arrow_paths"]["cluster_seeds"], "r") as source:
            seed_table = pa.ipc.open_file(source).read_all()
        with pa.memory_map(captured["arrow_paths"]["cluster_seed_disallows"], "r") as source:
            disallow_table = pa.ipc.open_file(source).read_all()
        captured["raw_seed_rows"] = dict(
            zip(
                seed_table["signature_id"].to_pylist(),
                seed_table["cluster_id"].to_pylist(),
                strict=True,
            )
        )
        captured["raw_disallow_rows"] = list(
            zip(
                disallow_table["signature_id_1"].to_pylist(),
                disallow_table["signature_id_2"].to_pylist(),
                strict=True,
            )
        )
        return SimpleNamespace(
            linked_signature_clusters={str(kwargs["query_signature_ids"][0]): "6"}
            if kwargs["query_signature_ids"]
            else {},
            telemetry={"candidate_row_count": 1, "pair_count": 1, "query_count": len(kwargs["query_signature_ids"])},
        )

    def fake_finish_incremental(
        self,
        unassigned_signature_ids,
        dataset_arg,
        linked_signature_clusters,
        recluster_map,
        cluster_seeds_require_inverse,
        prevent_new_incompatibilities,
        partial_supervision,
        runtime_context_arg,
        total_ram_bytes=None,
        arrow_paths=None,
        split_cluster_seeds_require_inverse=None,
    ):
        del self, recluster_map, cluster_seeds_require_inverse, prevent_new_incompatibilities, partial_supervision
        del split_cluster_seeds_require_inverse
        captured["finish_unassigned"] = list(unassigned_signature_ids)
        captured["finish_dataset"] = dataset_arg
        captured["finish_linked"] = dict(linked_signature_clusters)
        captured["finish_runtime_context"] = runtime_context_arg
        captured["finish_total_ram_bytes"] = total_ram_bytes
        captured["finish_arrow_paths"] = None if arrow_paths is None else dict(arrow_paths)
        return {"finished": list(unassigned_signature_ids)}

    monkeypatch.setattr(model_module, "build_runtime_context", lambda _operation, **_kwargs: runtime_context)
    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *args, **kwargs: sync_calls.append(args))
    monkeypatch.setattr(
        production_module.artifact_module,
        "load_incremental_linking_artifact",
        lambda _path: FakeArtifact(),
    )
    monkeypatch.setattr(
        production_module,
        "compute_promoted_incremental_limits",
        lambda **kwargs: _mock_promoted_limits(
            query_count=int(kwargs["query_count"]),
            query_batch_size=max(1, int(kwargs["query_count"])),
        ),
    )
    monkeypatch.setattr(
        production_module.runtime_module,
        "predict_incremental_link_or_abstain_from_raw_arrow_paths",
        fake_raw_arrow_linker,
    )
    _patch_fake_raw_arrow_planner(monkeypatch, captured=captured)
    monkeypatch.setattr(Clusterer, "_finish_incremental_with_seed_links", fake_finish_incremental)

    result = clusterer.predict_incremental(block, dataset, batching_threshold=None)

    assert result["clusters"] == {"finished": captured["finish_unassigned"]}
    assert result["incremental_linker_query_view"] == "raw_arrow"
    assert result["incremental_linker_telemetry"]["arrow_promoted_incremental"] == 1
    assert result["incremental_linker_telemetry"]["seed_setup_altered_signature_count"] == 0
    assert result["incremental_linker_telemetry"]["seed_setup_cluster_seeds_source"] == "dataset"
    assert result["incremental_linker_telemetry"]["seed_arrow_reused_source"] == 0
    assert result["incremental_linker_telemetry"]["seed_arrow_disallow_count"] == 1
    assert isinstance(result["incremental_linker_telemetry"]["incremental_finish_seconds"], float)
    assert sync_calls == []
    generated_path_keys = {"cluster_seeds", "cluster_seed_disallows", "name_counts_index"}
    assert {key: value for key, value in captured["arrow_paths"].items() if key not in generated_path_keys} == {
        key: value for key, value in arrow_paths.items() if key not in generated_path_keys
    }
    assert captured["arrow_paths"]["cluster_seeds"].endswith(".arrow")
    assert captured["arrow_paths"]["cluster_seed_disallows"].endswith(".arrow")
    assert captured["raw_seed_rows"] == dataset.cluster_seeds_require
    assert captured["raw_disallow_rows"] == [("5", "6")]
    assert captured["finish_dataset"] is dataset
    assert captured["finish_runtime_context"] is runtime_context
    assert captured["finish_linked"]
    assert captured["finish_arrow_paths"] == captured["arrow_paths"]


def test_predict_incremental_arrow_promoted_linker_cleans_up_temp_seed_context_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    closed: list[bool] = []

    class FakeArtifact:
        metadata = SimpleNamespace(retrieval_top_k=25)

    @contextmanager
    def fake_temporary_arrow_paths_with_cluster_seeds(*_args: object, **_kwargs: object):
        try:
            yield {
                "signatures": "signatures.arrow",
                "papers": "papers.arrow",
                "paper_authors": "paper_authors.arrow",
                "cluster_seeds": "temp_cluster_seeds.arrow",
            }
        finally:
            closed.append(True)

    class FakeClusterer:
        n_jobs = 1
        suppress_orcid = False
        _last_incremental_seed_setup_telemetry: dict[str, Any] = {}

        def _build_incremental_seed_setup(self, *_args: object, **_kwargs: object):
            self._last_incremental_seed_setup_telemetry = {"seed_setup_cluster_seeds_source": "python"}
            return {"seed": "c_seed"}, {}, {"c_seed": ["seed"]}, {"c_seed": ["seed"]}

    def fail_raw_arrow_linker(*_args: object, **_kwargs: object):
        raise RuntimeError("raw Arrow linker failed")

    monkeypatch.setattr(
        production_module.artifact_module,
        "load_incremental_linking_artifact",
        lambda _path: FakeArtifact(),
    )
    monkeypatch.setattr(
        production_module,
        "require_arrow_name_counts_index_for_clusterer",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(production_module, "clusterer_uses_name_count_features", lambda _clusterer: False)
    monkeypatch.setattr(
        production_module.runtime_module,
        "predict_incremental_link_or_abstain_from_raw_arrow_paths",
        fail_raw_arrow_linker,
    )
    monkeypatch.setattr(
        production_module,
        "temporary_arrow_paths_with_cluster_seeds",
        fake_temporary_arrow_paths_with_cluster_seeds,
    )
    _patch_fake_raw_arrow_planner(monkeypatch)

    with pytest.raises(RuntimeError, match="raw Arrow linker failed"):
        production_module.predict_incremental_promoted_linker_from_arrow_paths(
            FakeClusterer(),
            ["seed", "query"],
            cast(ANDData, SimpleNamespace(name_tuples=set())),
            arrow_paths={
                "signatures": "signatures.arrow",
                "papers": "papers.arrow",
                "paper_authors": "paper_authors.arrow",
            },
            artifact_dir=tmp_path,
            prevent_new_incompatibilities=False,
            partial_supervision={},
            runtime_context=cast(Any, SimpleNamespace(run_id="test")),
            total_ram_bytes=None,
            batching_threshold=None,
            resolve_total_ram_bytes=lambda value: (value, None),
            build_incremental_result=lambda *_args, **_kwargs: {},
        )

    assert closed == [True]


def test_predict_incremental_arrow_promoted_linker_fails_closed_when_single_query_exceeds_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeArtifact:
        metadata = SimpleNamespace(retrieval_top_k=25)

    class FakeClusterer:
        n_jobs = 1
        suppress_orcid = True
        _last_incremental_seed_setup_telemetry: dict[str, Any] = {}

        def _build_incremental_seed_setup(self, *_args: object, **_kwargs: object):
            return {"seed": "c_seed"}, {}, {"c_seed": ["seed"]}

    raw_calls: list[object] = []
    monkeypatch.setattr(
        production_module.artifact_module,
        "load_incremental_linking_artifact",
        lambda _path: FakeArtifact(),
    )
    monkeypatch.setattr(
        production_module,
        "compute_promoted_incremental_limits",
        lambda **_kwargs: _mock_promoted_limits(single_query_exceeds_budget=True),
    )
    monkeypatch.setattr(
        production_module.runtime_module,
        "predict_incremental_link_or_abstain_from_raw_arrow_paths",
        lambda *args, **kwargs: raw_calls.append((args, kwargs)),
    )

    with pytest.raises(MemoryError, match="cannot fit a single query"):
        production_module.predict_incremental_promoted_linker_from_arrow_paths(
            FakeClusterer(),
            ["seed", "query"],
            cast(ANDData, SimpleNamespace(name_tuples=set())),
            arrow_paths={
                "signatures": "signatures.arrow",
                "papers": "papers.arrow",
                "paper_authors": "paper_authors.arrow",
            },
            artifact_dir=tmp_path,
            prevent_new_incompatibilities=False,
            partial_supervision={},
            runtime_context=cast(Any, SimpleNamespace(run_id="test")),
            total_ram_bytes=None,
            batching_threshold=None,
            resolve_total_ram_bytes=lambda _value: (100_000, "test"),
            build_incremental_result=lambda *_args, **_kwargs: {},
        )

    assert raw_calls == []


def test_predict_incremental_arrow_promoted_linker_uses_budget_batch_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeArtifact:
        metadata = SimpleNamespace(retrieval_top_k=25)

    class FakeClusterer:
        n_jobs = 1
        suppress_orcid = True
        _last_incremental_seed_setup_telemetry = {"seed_setup_cluster_seeds_source": "arrow"}
        _last_incremental_residual_phase_b_telemetry: dict[str, Any] = {}

        def _build_incremental_seed_setup(self, *_args: object, **_kwargs: object):
            return {"seed": "c_seed"}, {}, {"c_seed": ["seed"]}

        def _finish_incremental_with_seed_links(self, unassigned_signature_ids, *_args: object, **_kwargs: object):
            return {"finished": list(unassigned_signature_ids)}

    class FakePlanner:
        def __init__(self, _paths: object, _query_ids: list[str], **_kwargs: object):
            pass

        def plan(self, query_ids: list[str], **_kwargs: object):
            raw_windows.append(tuple(query_ids))
            return {"query_signature_ids": tuple(query_ids)}

    class FakeRustModule:
        RawBlockQueryCandidatePlanner = FakePlanner

    raw_windows: list[tuple[str, ...]] = []
    raw_batches: list[tuple[str, ...]] = []
    limit_calls: list[dict[str, Any]] = []

    def fake_limits(**kwargs: Any):
        limit_calls.append(dict(kwargs))
        return _mock_promoted_limits(query_count=int(kwargs["query_count"]), query_batch_size=1)

    def fake_raw_arrow_linker(*_args: object, **kwargs: Any):
        query_ids = tuple(str(signature_id) for signature_id in kwargs["query_signature_ids"])
        raw_batches.append(query_ids)
        return SimpleNamespace(
            linked_signature_clusters={signature_id: "c_seed" for signature_id in query_ids},
            telemetry={"candidate_row_count": 1, "pair_count": 1, "query_count": len(query_ids)},
        )

    monkeypatch.setattr(
        production_module.artifact_module,
        "load_incremental_linking_artifact",
        lambda _path: FakeArtifact(),
    )
    monkeypatch.setattr(production_module, "compute_promoted_incremental_limits", fake_limits)
    monkeypatch.setattr(production_module.feature_port, "_require_rust_runtime", lambda: FakeRustModule())
    monkeypatch.setattr(
        production_module.feature_port,
        "build_rust_featurizer_from_arrow_paths",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        production_module.runtime_module,
        "feature_block_signature_order_from_raw_candidate_plan",
        lambda raw_plan: SimpleNamespace(signature_ids=("seed", *raw_plan["query_signature_ids"])),
    )
    monkeypatch.setattr(
        production_module.runtime_module,
        "subset_raw_candidate_plan_for_query_ids",
        lambda _raw_plan, query_ids, **_kwargs: {"query_signature_ids": tuple(query_ids)},
    )
    monkeypatch.setattr(
        production_module.runtime_module,
        "predict_incremental_link_or_abstain_from_raw_arrow_paths",
        fake_raw_arrow_linker,
    )

    result = production_module.predict_incremental_promoted_linker_from_arrow_paths(
        FakeClusterer(),
        ["seed", "query-1", "query-2"],
        cast(ANDData, SimpleNamespace(name_tuples="filtered")),
        arrow_paths={
            "signatures": "signatures.arrow",
            "papers": "papers.arrow",
            "paper_authors": "paper_authors.arrow",
            "cluster_seeds": "cluster_seeds.arrow",
        },
        artifact_dir=tmp_path,
        prevent_new_incompatibilities=False,
        partial_supervision={},
        runtime_context=cast(Any, SimpleNamespace(run_id="test")),
        total_ram_bytes=None,
        batching_threshold=None,
        resolve_total_ram_bytes=lambda _value: (100_000, "test"),
        build_incremental_result=lambda clusters, **kwargs: {"clusters": clusters, **kwargs},
    )

    assert limit_calls[0]["query_count"] == 2
    assert raw_windows == [("query-1", "query-2")]
    assert raw_batches == [("query-1",), ("query-2",)]
    assert result["incremental_linker_telemetry"]["query_batch_size_max"] == 1
    assert result["incremental_linker_telemetry"]["memory_initial_query_batch_size"] == 1


def test_predict_incremental_arrow_promoted_linker_reuses_raw_planner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeArtifact:
        metadata = SimpleNamespace(retrieval_top_k=25)

    class FakeClusterer:
        n_jobs = 1
        suppress_orcid = True
        _last_incremental_seed_setup_telemetry = {"seed_setup_cluster_seeds_source": "arrow"}
        _last_incremental_residual_phase_b_telemetry: dict[str, Any] = {}

        def _build_incremental_seed_setup(self, *_args: object, **_kwargs: object):
            return {"seed": "c_seed"}, {}, {"c_seed": ["seed"]}

        def _finish_incremental_with_seed_links(self, unassigned_signature_ids, *_args: object, **_kwargs: object):
            return {"finished": list(unassigned_signature_ids)}

    planner_inits: list[tuple[str, ...]] = []
    planner_plans: list[tuple[str, ...]] = []
    raw_windows: list[tuple[str, ...]] = []
    raw_batches: list[tuple[str, ...]] = []
    featurizer_signature_ids: list[tuple[str, ...]] = []
    fake_featurizer = object()

    class FakePlanner:
        def __init__(self, _paths: object, query_ids: list[str], **_kwargs: object):
            planner_inits.append(tuple(query_ids))
            self._query_ids = tuple(query_ids)

        def build_telemetry(self):
            return {"query_signature_count": len(self._query_ids), "signature_count": len(self._query_ids) + 1}

        def signature_ids(self):
            return ("seed", *self._query_ids)

        def plan(self, query_ids: list[str], **_kwargs: object):
            planner_plans.append(tuple(query_ids))
            return {"query_signature_ids": tuple(query_ids)}

    class FakeRustModule:
        RawBlockQueryCandidatePlanner = FakePlanner

        def raw_block_query_candidate_plan_arrow(self, _paths: object, query_ids: list[str], **_kwargs: object):
            raw_windows.append(tuple(query_ids))
            raise AssertionError("planner path should not call one-shot raw planner")

    def fake_raw_arrow_linker(*_args: object, **kwargs: Any):
        query_ids = tuple(str(signature_id) for signature_id in kwargs["query_signature_ids"])
        raw_batches.append(query_ids)
        assert kwargs["raw_candidate_plan"] == {"query_signature_ids": query_ids}
        assert kwargs["rust_featurizer"] is fake_featurizer
        return SimpleNamespace(
            linked_signature_clusters={signature_id: "c_seed" for signature_id in query_ids},
            telemetry={
                "candidate_row_count": 1,
                "pair_count": 1,
                "query_count": len(query_ids),
                "raw_arrow_featurizer_reused": 1,
            },
        )

    def fake_build_rust_featurizer_from_arrow_paths(_paths: object, **kwargs: Any):
        featurizer_signature_ids.append(tuple(str(value) for value in kwargs["signature_ids"]))
        return fake_featurizer

    monkeypatch.setattr(
        production_module.artifact_module,
        "load_incremental_linking_artifact",
        lambda _path: FakeArtifact(),
    )
    monkeypatch.setattr(
        production_module,
        "compute_promoted_incremental_limits",
        lambda **kwargs: _mock_promoted_limits(query_count=int(kwargs["query_count"]), query_batch_size=1),
    )
    monkeypatch.setattr(production_module.feature_port, "_require_rust_runtime", lambda: FakeRustModule())
    monkeypatch.setattr(
        production_module.feature_port,
        "build_rust_featurizer_from_arrow_paths",
        fake_build_rust_featurizer_from_arrow_paths,
    )
    monkeypatch.setattr(
        production_module.runtime_module,
        "feature_block_signature_order_from_raw_candidate_plan",
        lambda raw_plan: SimpleNamespace(signature_ids=("seed", *raw_plan["query_signature_ids"])),
    )
    monkeypatch.setattr(
        production_module.runtime_module,
        "subset_raw_candidate_plan_for_query_ids",
        lambda _raw_plan, query_ids, **_kwargs: {"query_signature_ids": tuple(query_ids)},
    )
    monkeypatch.setattr(
        production_module.runtime_module,
        "predict_incremental_link_or_abstain_from_raw_arrow_paths",
        fake_raw_arrow_linker,
    )

    result = production_module.predict_incremental_promoted_linker_from_arrow_paths(
        FakeClusterer(),
        ["seed", "query-1", "query-2"],
        cast(ANDData, SimpleNamespace(name_tuples="filtered")),
        arrow_paths={
            "signatures": "signatures.arrow",
            "papers": "papers.arrow",
            "paper_authors": "paper_authors.arrow",
            "cluster_seeds": "cluster_seeds.arrow",
        },
        artifact_dir=tmp_path,
        prevent_new_incompatibilities=False,
        partial_supervision={},
        runtime_context=cast(Any, SimpleNamespace(run_id="test")),
        total_ram_bytes=None,
        batching_threshold=None,
        resolve_total_ram_bytes=lambda _value: (100_000, "test"),
        build_incremental_result=lambda clusters, **kwargs: {"clusters": clusters, **kwargs},
    )

    assert planner_inits == [("query-1", "query-2")]
    assert planner_plans == [("query-1", "query-2")]
    assert featurizer_signature_ids == [("seed", "query-1", "query-2")]
    assert raw_windows == []
    assert raw_batches == [("query-1",), ("query-2",)]
    telemetry = result["incremental_linker_telemetry"]
    assert telemetry["raw_arrow_window_plan_enabled"] == 1
    assert telemetry["raw_arrow_window_plan_size"] == 2
    assert telemetry["raw_arrow_window_plan_multiplier"] == 4
    assert telemetry["raw_arrow_window_planner_count"] == 1
    assert telemetry["raw_arrow_window_planner_batch_plan_count"] == 2
    assert telemetry["raw_arrow_window_planner_plan_call_count"] == 1
    assert telemetry["raw_arrow_window_plan_signature_count"] == 3.0
    assert telemetry["raw_arrow_window_featurizer_count"] == 1
    assert telemetry["raw_arrow_window_featurizer_reused_batch_count"] == 2
    assert telemetry["raw_arrow_window_featurizer_signature_count"] == 3
    assert telemetry["raw_arrow_featurizer_reused"] == 2
    assert telemetry["raw_arrow_reusable_planner_enabled"] == 1


def test_predict_incremental_arrow_promoted_linker_requires_raw_planner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeArtifact:
        metadata = SimpleNamespace(retrieval_top_k=25)

    class FakeClusterer:
        n_jobs = 1
        suppress_orcid = True
        _last_incremental_seed_setup_telemetry = {"seed_setup_cluster_seeds_source": "arrow"}

        def _build_incremental_seed_setup(self, *_args: object, **_kwargs: object):
            return {"seed": "c_seed"}, {}, {"c_seed": ["seed"]}

    class FakeRustModule:
        pass

    monkeypatch.setattr(
        production_module.artifact_module,
        "load_incremental_linking_artifact",
        lambda _path: FakeArtifact(),
    )
    monkeypatch.setattr(
        production_module,
        "compute_promoted_incremental_limits",
        lambda **kwargs: _mock_promoted_limits(query_count=int(kwargs["query_count"]), query_batch_size=1),
    )
    monkeypatch.setattr(production_module.feature_port, "_require_rust_runtime", lambda: FakeRustModule())

    with pytest.raises(RuntimeError, match="RawBlockQueryCandidatePlanner"):
        production_module.predict_incremental_promoted_linker_from_arrow_paths(
            FakeClusterer(),
            ["seed", "query"],
            cast(ANDData, SimpleNamespace(name_tuples="filtered")),
            arrow_paths={
                "signatures": "signatures.arrow",
                "papers": "papers.arrow",
                "paper_authors": "paper_authors.arrow",
                "cluster_seeds": "cluster_seeds.arrow",
            },
            artifact_dir=tmp_path,
            prevent_new_incompatibilities=False,
            partial_supervision={},
            runtime_context=cast(Any, SimpleNamespace(run_id="test")),
            total_ram_bytes=None,
            batching_threshold=None,
            resolve_total_ram_bytes=lambda _value: (100_000, "test"),
            build_incremental_result=lambda clusters, **kwargs: {"clusters": clusters, **kwargs},
        )


def test_resolve_dataset_arrow_paths_discovers_raw_planner_batch_indexes(tmp_path: Path) -> None:
    arrow_paths = {}
    for key, filename in {
        "signatures": "signatures.arrow",
        "papers": "papers.arrow",
        "paper_authors": "paper_authors.arrow",
        "cluster_seeds": "cluster_seeds.arrow",
    }.items():
        path = tmp_path / filename
        path.touch()
        arrow_paths[key] = str(path)
    for filename in (
        "signatures.signatures_batch_index.bin",
        "papers.papers_batch_index.bin",
        "paper_authors.paper_authors_batch_index.bin",
    ):
        (tmp_path / filename).touch()
    (tmp_path / "cluster_seed_disallows.arrow").touch()
    (tmp_path / "altered_cluster_signatures.arrow").touch()

    dataset = SimpleNamespace(arrow_paths=arrow_paths)
    resolved = model_module._resolve_dataset_arrow_paths(
        dataset,
        require_specter=False,
        require_cluster_seeds=True,
    )

    assert resolved is not None
    assert Path(resolved["signatures_batch_index"]).name == "signatures.signatures_batch_index.bin"
    assert Path(resolved["papers_batch_index"]).name == "papers.papers_batch_index.bin"
    assert Path(resolved["paper_authors_batch_index"]).name == "paper_authors.paper_authors_batch_index.bin"
    assert Path(resolved["cluster_seed_disallows"]).name == "cluster_seed_disallows.arrow"
    assert Path(resolved["altered_cluster_signatures"]).name == "altered_cluster_signatures.arrow"


def test_cluster_seeds_arrow_matches_surfaces_arrow_invalid_by_type(tmp_path: Path, monkeypatch) -> None:
    pa = pytest.importorskip("pyarrow")
    path = tmp_path / "cluster_seeds.arrow"
    path.write_bytes(b"not-arrow")

    def raise_arrow_invalid(_path):
        raise pa.lib.ArrowInvalid("invalid arrow")

    monkeypatch.setattr(production_module, "read_cluster_seeds_arrow", raise_arrow_invalid)

    with pytest.raises(pa.lib.ArrowInvalid, match="invalid arrow"):
        production_module._cluster_seeds_arrow_matches(path, {"s1": "c1"})


def test_cluster_seeds_arrow_matches_surfaces_arrow_invalid_by_name(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "cluster_seeds.arrow"
    path.write_bytes(b"not-arrow")
    arrow_invalid = type("ArrowInvalid", (Exception,), {})

    def raise_arrow_invalid(_path):
        raise arrow_invalid("invalid arrow")

    monkeypatch.setattr(production_module, "read_cluster_seeds_arrow", raise_arrow_invalid)

    with pytest.raises(arrow_invalid, match="invalid arrow"):
        production_module._cluster_seeds_arrow_matches(path, {"s1": "c1"})


def test_specter_arrow_name_uses_declared_suffix_not_substring() -> None:
    specter2_paths = [
        "/tmp/pubmed_specter2.pkl",
        "/tmp/specter2.tar.gz",
        "/tmp/specter2.embeddings.pkl",
        "/tmp/pubmed_specter2.pkl.gz",
    ]
    for specter_path in specter2_paths:
        assert (
            model_module._specter_arrow_name_for_dataset(SimpleNamespace(specter_embeddings_path=specter_path))
            == "specter2.arrow"
        )

    assert (
        model_module._specter_arrow_name_for_dataset(
            SimpleNamespace(specter_embeddings_path="/tmp/specter2backup.json")
        )
        == "specter.arrow"
    )


def test_resolve_dataset_arrow_paths_discovers_name_counts_index_from_manifest(tmp_path: Path) -> None:
    arrow_paths = {}
    for key, filename in {
        "signatures": "signatures.arrow",
        "papers": "papers.arrow",
        "paper_authors": "paper_authors.arrow",
    }.items():
        path = tmp_path / filename
        path.touch()
        arrow_paths[key] = str(path)
    name_counts_index = tmp_path / "shared" / "name_counts_index"
    name_counts_index.mkdir(parents=True)
    (tmp_path / "manifest.json").write_text(
        '{"paths": {"name_counts_index": "shared/name_counts_index"}}',
        encoding="utf-8",
    )

    dataset = SimpleNamespace(arrow_paths=arrow_paths)
    resolved = model_module._resolve_dataset_arrow_paths(
        dataset,
        require_specter=False,
        require_cluster_seeds=False,
        require_name_counts_index=True,
    )

    assert resolved is not None
    assert resolved["name_counts_index"] == str(name_counts_index)


def test_resolve_dataset_arrow_paths_rejects_bad_manifest_name_counts_index(tmp_path: Path) -> None:
    arrow_paths = {}
    for key, filename in {
        "signatures": "signatures.arrow",
        "papers": "papers.arrow",
        "paper_authors": "paper_authors.arrow",
    }.items():
        path = tmp_path / filename
        path.touch()
        arrow_paths[key] = str(path)
    (tmp_path / "name_counts_index").mkdir()
    (tmp_path / "manifest.json").write_text(
        '{"paths": {"name_counts_index": "missing/name_counts_index"}}',
        encoding="utf-8",
    )

    dataset = SimpleNamespace(arrow_paths=arrow_paths)
    with pytest.raises(FileNotFoundError, match="specifies name_counts_index path that does not exist"):
        model_module._resolve_dataset_arrow_paths(
            dataset,
            require_specter=False,
            require_cluster_seeds=False,
            require_name_counts_index=True,
        )


def test_resolve_dataset_arrow_paths_declines_missing_required_name_counts_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(model_module, "PROJECT_ROOT_PATH", str(tmp_path / "project"))
    arrow_paths = {}
    for key, filename in {
        "signatures": "signatures.arrow",
        "papers": "papers.arrow",
        "paper_authors": "paper_authors.arrow",
    }.items():
        path = tmp_path / filename
        path.touch()
        arrow_paths[key] = str(path)

    dataset = SimpleNamespace(arrow_paths=arrow_paths)
    resolved = model_module._resolve_dataset_arrow_paths(
        dataset,
        require_specter=False,
        require_cluster_seeds=False,
        require_name_counts_index=True,
    )

    assert resolved is None


def test_predict_from_arrow_paths_requires_name_counts_index_for_name_count_features(tmp_path: Path) -> None:
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["name_counts"]),
        classifier=object(),
        n_jobs=1,
    )
    arrow_paths = {}
    for key, filename in {
        "signatures": "signatures.arrow",
        "papers": "papers.arrow",
        "paper_authors": "paper_authors.arrow",
    }.items():
        path = tmp_path / filename
        path.touch()
        arrow_paths[key] = str(path)

    with pytest.raises(model_module.MissingArrowArtifactError, match="missing mapping keys: name_counts_index"):
        clusterer.predict_from_arrow_paths(
            {"block": ["s1"]},
            arrow_paths,
        )

    with pytest.raises(model_module.MissingArrowArtifactError, match="name_counts_index"):
        clusterer.predict_from_arrow_paths(
            {"block": ["s1"]},
            {
                **arrow_paths,
                "name_counts_index": "missing_name_counts_index",
            },
        )


def test_predict_from_arrow_paths_loads_name_counts_by_default_for_name_count_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["name_counts"]),
        classifier=object(),
        n_jobs=1,
    )
    name_counts_index = tmp_path / "name_counts_index"
    name_counts_index.mkdir()
    arrow_paths = {}
    for key, filename in {
        "signatures": "signatures.arrow",
        "papers": "papers.arrow",
        "paper_authors": "paper_authors.arrow",
    }.items():
        path = tmp_path / filename
        path.touch()
        arrow_paths[key] = str(path)
    captured: dict[str, Any] = {}

    def fake_build_rust_featurizer_from_arrow_paths(*_args: Any, **kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        model_module, "build_rust_featurizer_from_arrow_paths", fake_build_rust_featurizer_from_arrow_paths
    )
    monkeypatch.setattr(
        Clusterer,
        "predict_from_rust_featurizer",
        lambda *_args, **_kwargs: ({"block": ["s1"]}, None),
    )

    clusterer.predict_from_arrow_paths(
        {"block": ["s1"]},
        {
            **arrow_paths,
            "name_counts_index": str(name_counts_index),
        },
    )

    assert captured["load_name_counts"] is True


def test_raw_arrow_runtime_requires_name_counts_index_for_name_count_features() -> None:
    clusterer = SimpleNamespace(featurizer_info=FeaturizationInfo(features_to_use=["name_counts"]))

    with pytest.raises(ValueError, match="requires name_counts_index"):
        production_module.runtime_module.require_arrow_name_counts_index_for_clusterer(
            clusterer,
            {},
            context="Raw Arrow scoring",
        )

    with pytest.raises(ValueError, match="requires name_counts_index"):
        production_module.runtime_module.require_arrow_name_counts_index_for_clusterer(
            clusterer,
            {"name_counts_index": "missing_name_counts_index"},
            context="Raw Arrow scoring",
        )


def test_predict_incremental_arrow_promoted_linker_uses_seed_arrow_without_python_seed_map(
    clusterer_dataset_factory,
    monkeypatch,
    tmp_path,
):
    import pyarrow as pa

    clusterer, dataset = clusterer_dataset_factory(name="dummy_auto_incremental_arrow_seed_only")
    block = ["3", "4", "5", "6", "7", "8"]
    dataset.cluster_seeds_require = {}
    dataset.altered_cluster_signatures = None
    runtime_context = SimpleNamespace(
        operation="cluster_predict_incremental",
        requested_backend="rust",
        resolved_backend="rust",
        use_rust=True,
        run_id="test-rust-promoted-incremental-arrow-seed-only",
        source="S2AND_BACKEND",
    )
    arrow_paths = {}
    for key, filename in {
        "signatures": "signatures.arrow",
        "papers": "papers.arrow",
        "paper_authors": "paper_authors.arrow",
    }.items():
        path = tmp_path / filename
        path.touch()
        arrow_paths[key] = str(path)
    cluster_seeds_path = tmp_path / "cluster_seeds.arrow"
    seed_table = pa.table(
        {
            "signature_id": pa.array(["6", "7", "3", "4"], type=pa.string()),
            "cluster_id": pa.array(["0", "0", "1", "1"], type=pa.string()),
        }
    )
    with pa.OSFile(str(cluster_seeds_path), "wb") as sink:
        with pa.ipc.new_file(sink, seed_table.schema) as writer:
            writer.write_table(seed_table)
    arrow_paths["cluster_seeds"] = str(cluster_seeds_path)
    dataset.arrow_paths = arrow_paths
    captured: dict[str, Any] = {}
    sync_calls: list[object] = []

    class FakeArtifact:
        metadata = SimpleNamespace(retrieval_top_k=25)

    def fail_python_incremental(*_args, **_kwargs):
        raise AssertionError("cluster_seeds.arrow should keep promoted Arrow incremental active")

    def fake_raw_arrow_linker(clusterer_arg, artifact_arg, **kwargs):
        del clusterer_arg, artifact_arg
        captured["raw_arrow_paths"] = dict(kwargs["arrow_paths"])
        captured["query_signature_ids"] = tuple(kwargs["query_signature_ids"])
        return SimpleNamespace(
            linked_signature_clusters={},
            telemetry={"candidate_row_count": 0, "pair_count": 0, "query_count": len(kwargs["query_signature_ids"])},
        )

    def fake_finish_incremental(
        self,
        unassigned_signature_ids,
        dataset_arg,
        linked_signature_clusters,
        recluster_map,
        cluster_seeds_require_inverse,
        prevent_new_incompatibilities,
        partial_supervision,
        runtime_context_arg,
        total_ram_bytes=None,
        arrow_paths=None,
        split_cluster_seeds_require_inverse=None,
    ):
        del self, dataset_arg, linked_signature_clusters, recluster_map, prevent_new_incompatibilities
        del partial_supervision, runtime_context_arg, total_ram_bytes, arrow_paths, split_cluster_seeds_require_inverse
        captured["finish_unassigned"] = list(unassigned_signature_ids)
        captured["finish_seed_inverse"] = {
            str(cluster_id): list(signature_ids) for cluster_id, signature_ids in cluster_seeds_require_inverse.items()
        }
        return {"finished": list(unassigned_signature_ids)}

    monkeypatch.setattr(model_module, "build_runtime_context", lambda _operation, **_kwargs: runtime_context)
    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *args, **kwargs: sync_calls.append(args))
    monkeypatch.setattr(Clusterer, "_predict_incremental_helper", fail_python_incremental)
    monkeypatch.setattr(
        production_module.artifact_module,
        "load_incremental_linking_artifact",
        lambda _path: FakeArtifact(),
    )
    monkeypatch.setattr(
        production_module,
        "compute_promoted_incremental_limits",
        lambda **kwargs: _mock_promoted_limits(
            query_count=int(kwargs["query_count"]),
            query_batch_size=max(1, int(kwargs["query_count"])),
        ),
    )
    monkeypatch.setattr(
        production_module.runtime_module,
        "predict_incremental_link_or_abstain_from_raw_arrow_paths",
        fake_raw_arrow_linker,
    )
    _patch_fake_raw_arrow_planner(monkeypatch, captured=captured)
    monkeypatch.setattr(Clusterer, "_finish_incremental_with_seed_links", fake_finish_incremental)

    result = clusterer.predict_incremental(block, dataset, batching_threshold=None)

    assert result["clusters"] == {"finished": ["5", "8"]}
    assert result["incremental_linker_query_view"] == "raw_arrow"
    assert result["incremental_linker_telemetry"]["seed_setup_seed_signature_count"] == 4
    assert result["incremental_linker_telemetry"]["seed_setup_component_count"] == 2
    assert result["incremental_linker_telemetry"]["seed_setup_cluster_seeds_source"] == "arrow"
    assert result["incremental_linker_telemetry"]["seed_arrow_reused_source"] == 1
    assert sync_calls == []
    assert captured["query_signature_ids"] == ("5", "8")
    assert captured["finish_unassigned"] == ["5", "8"]
    assert captured["finish_seed_inverse"] == {"0": ["6", "7"], "1": ["3", "4"]}
    assert captured["raw_arrow_paths"]["cluster_seeds"] == str(cluster_seeds_path)


def test_predict_incremental_arrow_promoted_linker_rewrites_stale_seed_arrow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cluster_seeds_path = tmp_path / "cluster_seeds.arrow"
    write_cluster_seeds_arrow(cluster_seeds_path, {"seed": "old"})
    arrow_paths = {}
    for key, filename in {
        "signatures": "signatures.arrow",
        "papers": "papers.arrow",
        "paper_authors": "paper_authors.arrow",
    }.items():
        path = tmp_path / filename
        path.touch()
        arrow_paths[key] = str(path)
    arrow_paths["cluster_seeds"] = str(cluster_seeds_path)
    captured: dict[str, Any] = {}

    class FakeArtifact:
        metadata = SimpleNamespace(retrieval_top_k=25)

    class FakeClusterer:
        n_jobs = 1
        suppress_orcid = False
        _last_incremental_seed_setup_telemetry = {"seed_setup_cluster_seeds_source": "arrow"}

        def _build_incremental_seed_setup(self, *_args: object, **_kwargs: object):
            self._last_incremental_seed_setup_telemetry = {"seed_setup_cluster_seeds_source": "arrow"}
            return {"seed": "new"}, {}, {"new": ["seed"]}, {"new": ["seed"]}

        def _finish_incremental_with_seed_links(self, *args: object, **kwargs: object):
            del args, kwargs
            return {"new": ["seed", "query"]}

    @contextmanager
    def fake_temporary_arrow_paths_with_cluster_seeds(paths_arg, cluster_seeds_require, **_kwargs):
        captured["rewritten_paths_input"] = dict(paths_arg)
        captured["rewritten_seed_map"] = dict(cluster_seeds_require)
        try:
            yield {**arrow_paths, "cluster_seeds": "rewritten_cluster_seeds.arrow"}
        finally:
            captured["closed"] = True

    def fake_raw_arrow_linker(_clusterer, _artifact, **kwargs):
        captured["raw_arrow_paths"] = dict(kwargs["arrow_paths"])
        return SimpleNamespace(
            linked_signature_clusters={},
            telemetry={"candidate_row_count": 0, "pair_count": 0, "query_count": len(kwargs["query_signature_ids"])},
        )

    monkeypatch.setattr(
        production_module.artifact_module,
        "load_incremental_linking_artifact",
        lambda _path: FakeArtifact(),
    )
    monkeypatch.setattr(
        production_module,
        "compute_promoted_incremental_limits",
        lambda **kwargs: _mock_promoted_limits(
            query_count=int(kwargs["query_count"]),
            query_batch_size=max(1, int(kwargs["query_count"])),
        ),
    )
    monkeypatch.setattr(
        production_module,
        "temporary_arrow_paths_with_cluster_seeds",
        fake_temporary_arrow_paths_with_cluster_seeds,
    )
    monkeypatch.setattr(
        production_module.runtime_module,
        "predict_incremental_link_or_abstain_from_raw_arrow_paths",
        fake_raw_arrow_linker,
    )
    _patch_fake_raw_arrow_planner(monkeypatch, captured=captured)

    result = production_module.predict_incremental_promoted_linker_from_arrow_paths(
        FakeClusterer(),
        ["seed", "query"],
        cast(ANDData, SimpleNamespace(name_tuples=set(), cluster_seeds_disallow=set())),
        arrow_paths=arrow_paths,
        artifact_dir=tmp_path,
        prevent_new_incompatibilities=False,
        partial_supervision={},
        runtime_context=cast(Any, SimpleNamespace(run_id="test")),
        total_ram_bytes=None,
        batching_threshold=None,
        resolve_total_ram_bytes=lambda value: (value, None),
        build_incremental_result=lambda clusters, **kwargs: {"clusters": clusters, **kwargs},
    )

    assert result["incremental_linker_telemetry"]["seed_arrow_reused_source"] == 0
    assert captured["rewritten_seed_map"] == {"seed": "new"}
    assert captured["rewritten_paths_input"]["cluster_seeds"] == str(cluster_seeds_path)
    assert captured["raw_arrow_paths"]["cluster_seeds"] == "rewritten_cluster_seeds.arrow"
    assert captured["closed"] is True


def test_predict_incremental_arrow_promoted_linker_rejects_none_arrow_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeArtifact:
        metadata = SimpleNamespace(retrieval_top_k=25)

    def fail_raw_arrow_linker(*_args: object, **_kwargs: object):
        raise AssertionError("invalid Arrow paths should be rejected before raw Arrow linker")

    monkeypatch.setattr(
        production_module.artifact_module,
        "load_incremental_linking_artifact",
        lambda _path: FakeArtifact(),
    )
    monkeypatch.setattr(
        production_module.runtime_module,
        "predict_incremental_link_or_abstain_from_raw_arrow_paths",
        fail_raw_arrow_linker,
    )

    with pytest.raises(ValueError, match="signatures.*None"):
        production_module.predict_incremental_promoted_linker_from_arrow_paths(
            SimpleNamespace(),
            ["seed", "query"],
            cast(ANDData, SimpleNamespace(name_tuples=set(), cluster_seeds_disallow=set())),
            arrow_paths={"signatures": None, "papers": "papers.arrow", "paper_authors": "paper_authors.arrow"},
            artifact_dir=tmp_path,
            prevent_new_incompatibilities=False,
            partial_supervision={},
            runtime_context=cast(Any, SimpleNamespace(run_id="test")),
            total_ram_bytes=None,
            batching_threshold=None,
            resolve_total_ram_bytes=lambda value: (value, None),
            build_incremental_result=lambda clusters, **kwargs: {"clusters": clusters, **kwargs},
        )


def test_predict_incremental_arrow_promoted_linker_keeps_scoring_batches_within_budget(
    clusterer_dataset_factory,
    monkeypatch,
    tmp_path,
):
    import pyarrow as pa

    clusterer, dataset = clusterer_dataset_factory(name="dummy_auto_incremental_arrow_window_featurizer")
    block = ["3", "4", "5", "6", "7", "8"]
    dataset.cluster_seeds_require = {}
    dataset.altered_cluster_signatures = None
    runtime_context = SimpleNamespace(
        operation="cluster_predict_incremental",
        requested_backend="rust",
        resolved_backend="rust",
        use_rust=True,
        run_id="test-rust-promoted-incremental-arrow-window-featurizer",
        source="S2AND_BACKEND",
    )
    arrow_paths = {}
    for key, filename in {
        "signatures": "signatures.arrow",
        "papers": "papers.arrow",
        "paper_authors": "paper_authors.arrow",
    }.items():
        path = tmp_path / filename
        path.touch()
        arrow_paths[key] = str(path)
    cluster_seeds_path = tmp_path / "cluster_seeds.arrow"
    seed_table = pa.table(
        {
            "signature_id": pa.array(["6", "7", "3", "4"], type=pa.string()),
            "cluster_id": pa.array(["0", "0", "1", "1"], type=pa.string()),
        }
    )
    with pa.OSFile(str(cluster_seeds_path), "wb") as sink:
        with pa.ipc.new_file(sink, seed_table.schema) as writer:
            writer.write_table(seed_table)
    arrow_paths["cluster_seeds"] = str(cluster_seeds_path)
    dataset.arrow_paths = arrow_paths
    captured: dict[str, Any] = {
        "planner_inits": [],
        "planner_plans": [],
        "featurizer_signature_ids": [],
        "runtime_featurizers": [],
        "runtime_batches": [],
        "runtime_raw_plans": [],
    }

    class FakeArtifact:
        metadata = SimpleNamespace(retrieval_top_k=25)

    class FakeFeaturizer:
        def __init__(self, signature_ids):
            self._signature_ids = list(signature_ids)

        def signature_ids(self):
            return list(self._signature_ids)

    class FakePlanner:
        def __init__(self, paths_arg, query_signature_ids, **kwargs):
            del paths_arg, kwargs
            captured["planner_inits"].append(tuple(query_signature_ids))

        def plan(self, query_signature_ids, **kwargs):
            del kwargs
            captured["planner_plans"].append(tuple(query_signature_ids))
            return {"query_signature_ids": tuple(query_signature_ids)}

    class FakeRustModule:
        RawBlockQueryCandidatePlanner = FakePlanner

    def fake_build_rust_featurizer_from_arrow_paths(paths_arg, **kwargs):
        del paths_arg
        captured["featurizer_signature_ids"].append(tuple(kwargs["signature_ids"]))
        return FakeFeaturizer(kwargs["signature_ids"])

    def fake_raw_arrow_linker(clusterer_arg, artifact_arg, **kwargs):
        del clusterer_arg, artifact_arg
        query_batch = tuple(kwargs["query_signature_ids"])
        captured["runtime_batches"].append(query_batch)
        captured["runtime_featurizers"].append(kwargs["rust_featurizer"])
        captured["runtime_raw_plans"].append(kwargs["raw_candidate_plan"])
        return SimpleNamespace(
            linked_signature_clusters={str(signature_id): "0" for signature_id in query_batch},
            telemetry={
                "candidate_row_count": 0,
                "pair_count": 0,
                "query_count": len(query_batch),
                "raw_arrow_featurizer_reused": int(kwargs["rust_featurizer"] is not None),
                "raw_arrow_seed_signature_count": 4,
                "raw_arrow_seed_component_count": 2,
                "raw_arrow_plan_seed_signature_count": 0,
                "raw_arrow_plan_cluster_count": 0,
            },
        )

    monkeypatch.setattr(model_module, "build_runtime_context", lambda _operation, **_kwargs: runtime_context)
    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        production_module.artifact_module,
        "load_incremental_linking_artifact",
        lambda _path: FakeArtifact(),
    )
    monkeypatch.setattr(production_module.feature_port, "_require_rust_runtime", lambda: FakeRustModule)
    monkeypatch.setattr(
        production_module.feature_port,
        "build_rust_featurizer_from_arrow_paths",
        fake_build_rust_featurizer_from_arrow_paths,
    )
    monkeypatch.setattr(
        production_module.runtime_module,
        "feature_block_signature_order_from_raw_candidate_plan",
        lambda raw_plan: SimpleNamespace(signature_ids=("6", "7", *raw_plan["query_signature_ids"])),
    )
    monkeypatch.setattr(
        production_module.runtime_module,
        "subset_raw_candidate_plan_for_query_ids",
        lambda _raw_plan, query_ids, **_kwargs: {"query_signature_ids": tuple(query_ids)},
    )
    monkeypatch.setattr(
        production_module.runtime_module,
        "predict_incremental_link_or_abstain_from_raw_arrow_paths",
        fake_raw_arrow_linker,
    )

    result = clusterer.predict_incremental(block, dataset, batching_threshold=1)

    assert result["clusters"] == {"0": ["6", "7", "5", "8"], "1": ["3", "4"]}
    assert captured["planner_inits"] == [("5", "8")]
    assert captured["planner_plans"] == [("5", "8")]
    assert captured["featurizer_signature_ids"] == [("6", "7", "5", "8")]
    assert captured["runtime_batches"] == [("5",), ("8",)]
    assert [featurizer.signature_ids() for featurizer in captured["runtime_featurizers"]] == [
        ["6", "7", "5", "8"],
        ["6", "7", "5", "8"],
    ]
    assert captured["runtime_raw_plans"] == [{"query_signature_ids": ("5",)}, {"query_signature_ids": ("8",)}]
    telemetry = result["incremental_linker_telemetry"]
    assert telemetry["raw_arrow_window_plan_count"] == 1
    assert telemetry["raw_arrow_window_plan_enabled"] == 1
    assert telemetry["raw_arrow_window_plan_size"] == 2
    assert telemetry["raw_arrow_window_plan_multiplier"] == 4
    assert telemetry["raw_arrow_window_plan_query_count"] == 2
    assert telemetry["raw_arrow_window_featurizer_count"] == 1
    assert telemetry["raw_arrow_window_featurizer_signature_count"] == 4
    assert "raw_arrow_window_subset_seconds" in telemetry
    assert "raw_arrow_window_plan_signature_count" not in telemetry
    assert "raw_arrow_window_plan_seed_signature_count" not in telemetry
    assert telemetry["raw_arrow_featurizer_reused"] == 2
    assert telemetry["seed_signature_count"] == 4
    assert telemetry["seed_component_count"] == 2
    assert telemetry["raw_arrow_seed_signature_count"] == 4
    assert telemetry["raw_arrow_seed_component_count"] == 2
    assert telemetry["raw_arrow_plan_seed_signature_count"] == 0
    assert telemetry["raw_arrow_plan_cluster_count"] == 0


def test_predict_incremental_arrow_promoted_linker_transforms_altered_seed_arrow(
    clusterer_dataset_factory,
    monkeypatch,
    tmp_path,
):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_auto_incremental_arrow_altered")
    block = ["3", "4", "5", "6", "7", "8"]
    dataset.altered_cluster_signatures = ["6"]
    runtime_context = SimpleNamespace(
        operation="cluster_predict_incremental",
        requested_backend="rust",
        resolved_backend="rust",
        use_rust=True,
        run_id="test-rust-promoted-incremental-arrow-altered",
        source="S2AND_BACKEND",
    )
    arrow_paths = {}
    for key, filename in {
        "signatures": "signatures.arrow",
        "papers": "papers.arrow",
        "paper_authors": "paper_authors.arrow",
        "cluster_seeds": "cluster_seeds.arrow",
    }.items():
        path = tmp_path / filename
        path.touch()
        arrow_paths[key] = str(path)
    dataset.arrow_paths = arrow_paths
    captured: dict[str, Any] = {}
    sync_calls: list[object] = []

    class FakeArtifact:
        metadata = SimpleNamespace(retrieval_top_k=25)

    class FakePlanner:
        def __init__(self, _paths, query_signature_ids, **_kwargs):
            captured["planner_query_ids"] = tuple(query_signature_ids)

        def plan(self, query_signature_ids, **_kwargs):
            captured["planner_window_query_ids"] = tuple(query_signature_ids)
            return {"query_signature_ids": tuple(query_signature_ids)}

    class FakeRustModule:
        RawBlockQueryCandidatePlanner = FakePlanner

    def fake_predict_from_arrow_paths(block_dict, arrow_paths_arg, **kwargs):
        captured["presplit_block_dict"] = dict(block_dict)
        captured["presplit_arrow_paths"] = dict(arrow_paths_arg)
        captured["presplit_incremental_dont_use_cluster_seeds"] = kwargs["incremental_dont_use_cluster_seeds"]
        captured["presplit_runtime_context"] = kwargs["runtime_context"]
        return {"split0": ["6"], "split1": ["7"]}, None

    def fake_raw_arrow_linker(clusterer_arg, artifact_arg, **kwargs):
        import pyarrow as pa

        del clusterer_arg, artifact_arg
        captured["raw_arrow_paths"] = dict(kwargs["arrow_paths"])
        captured["query_signature_ids"] = tuple(kwargs["query_signature_ids"])
        cluster_seeds_path = captured["raw_arrow_paths"]["cluster_seeds"]
        with pa.memory_map(cluster_seeds_path, "r") as source:
            table = pa.ipc.open_file(source).read_all()
        captured["raw_seed_rows"] = dict(
            zip(table["signature_id"].to_pylist(), table["cluster_id"].to_pylist(), strict=True)
        )
        return SimpleNamespace(
            linked_signature_clusters={str(signature_id): "0_0" for signature_id in kwargs["query_signature_ids"]},
            telemetry={"candidate_row_count": 2, "pair_count": 2, "query_count": len(kwargs["query_signature_ids"])},
        )

    monkeypatch.setattr(model_module, "build_runtime_context", lambda _operation, **_kwargs: runtime_context)
    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *args, **kwargs: sync_calls.append(args))
    monkeypatch.setattr(
        model_module,
        "_resolve_total_ram_bytes_for_incremental",
        lambda _total=None: (1_000_000_000, "test"),
    )
    monkeypatch.setattr(
        production_module.artifact_module,
        "load_incremental_linking_artifact",
        lambda _path: FakeArtifact(),
    )
    monkeypatch.setattr(
        production_module,
        "compute_promoted_incremental_limits",
        lambda **kwargs: _mock_promoted_limits(
            query_count=int(kwargs["query_count"]),
            query_batch_size=max(1, int(kwargs["query_count"])),
        ),
    )
    monkeypatch.setattr(
        production_module.runtime_module,
        "predict_incremental_link_or_abstain_from_raw_arrow_paths",
        fake_raw_arrow_linker,
    )
    monkeypatch.setattr(production_module.feature_port, "_require_rust_runtime", lambda: FakeRustModule())
    monkeypatch.setattr(
        production_module.feature_port,
        "build_rust_featurizer_from_arrow_paths",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        production_module.runtime_module,
        "feature_block_signature_order_from_raw_candidate_plan",
        lambda raw_plan: SimpleNamespace(signature_ids=("6", "7", "3", "4", *raw_plan["query_signature_ids"])),
    )
    clusterer.predict_from_arrow_paths = cast(Any, fake_predict_from_arrow_paths)

    result = clusterer.predict_incremental(
        block,
        dataset,
        prevent_new_incompatibilities=False,
        batching_threshold=None,
    )

    assert result["incremental_linker_query_view"] == "raw_arrow"
    assert result["incremental_linker_telemetry"]["arrow_promoted_incremental"] == 1
    assert result["incremental_linker_telemetry"]["seed_setup_altered_signature_count"] == 1
    assert result["incremental_linker_telemetry"]["seed_setup_altered_presplit_block_count"] == 1
    assert result["incremental_linker_telemetry"]["seed_setup_altered_presplit_signature_count"] == 2
    assert result["incremental_linker_telemetry"]["seed_arrow_reused_source"] == 0
    assert sync_calls == []
    assert result["clusters"] == {"0": ["6", "7", "5", "8"], "1": ["3", "4"]}
    assert captured["presplit_block_dict"] == {"altered_profile_0": ["6", "7"]}
    assert "cluster_seeds" not in captured["presplit_arrow_paths"]
    assert captured["presplit_incremental_dont_use_cluster_seeds"] is True
    assert captured["presplit_runtime_context"] is runtime_context
    assert captured["planner_query_ids"] == ("5", "8")
    assert captured["planner_window_query_ids"] == ("5", "8")
    assert captured["query_signature_ids"] == ("5", "8")
    assert captured["raw_seed_rows"] == {"6": "0_0", "7": "0_1", "3": "1", "4": "1"}


def test_predict_incremental_rust_empty_seeds_requires_seed_source(clusterer_dataset_factory, monkeypatch):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_rust_empty_seeds")
    dataset.cluster_seeds_require = {}
    block = ["3", "4", "5"]
    runtime_context = SimpleNamespace(
        operation="cluster_predict_incremental",
        requested_backend="rust",
        resolved_backend="rust",
        use_rust=True,
        run_id="test-rust-empty-seeds",
        source="S2AND_BACKEND",
    )

    monkeypatch.setattr(model_module, "build_runtime_context", lambda _operation, **_kwargs: runtime_context)
    monkeypatch.setattr(
        model_module,
        "_sync_rust_cluster_seeds",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("seed sync should not run")),
    )
    monkeypatch.setattr(
        Clusterer,
        "_predict_incremental_helper",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("fallback helper should not run")),
    )
    monkeypatch.setattr(
        Clusterer,
        "_predict_incremental_promoted_linker",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("promoted linker should not run")),
    )

    with pytest.raises(model_module.MissingArrowArtifactError, match="cluster_seeds_source"):
        clusterer.predict_incremental(block, dataset, batching_threshold=None)


def test_predict_incremental_rust_empty_seeds_rejects_batching_threshold_before_routing(
    clusterer_dataset_factory,
    monkeypatch,
):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_rust_empty_seeds_batching")
    dataset.cluster_seeds_require = {}
    block = ["3", "4", "5"]
    runtime_context = SimpleNamespace(
        operation="cluster_predict_incremental",
        requested_backend="rust",
        resolved_backend="rust",
        use_rust=True,
        run_id="test-rust-empty-seeds-batching",
        source="S2AND_BACKEND",
    )

    monkeypatch.setattr(model_module, "build_runtime_context", lambda _operation, **_kwargs: runtime_context)
    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        Clusterer,
        "_predict_incremental_helper",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("fallback helper should not run")),
    )

    with pytest.raises(model_module.MissingArrowArtifactError, match="cluster_seeds_source"):
        clusterer.predict_incremental(block, dataset, batching_threshold=2)


def test_predict_incremental_promoted_linker_batches_queries(
    clusterer_dataset_factory,
    monkeypatch,
):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_promoted_incremental_linker_batch")
    block = ["3", "4", "5", "6", "7", "8"]
    residual_blocks: list[list[str]] = []
    runtime_context = SimpleNamespace(
        operation="cluster_predict_incremental",
        requested_backend="rust",
        resolved_backend="rust",
        use_rust=True,
        run_id="test-rust-promoted-incremental-batch",
        source="S2AND_BACKEND",
    )
    monkeypatch.setattr(model_module, "build_runtime_context", lambda _operation, **_kwargs: runtime_context)
    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_module, "_get_rust_featurizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        model_module,
        "_build_incremental_constraint_backend",
        lambda *args, **kwargs: SimpleNamespace(rust_featurizer=None),
    )

    def fake_predict_helper(block_dict, dataset_arg, partial_supervision, runtime_context, total_ram_bytes=None):
        del dataset_arg, partial_supervision, runtime_context, total_ram_bytes
        residual_blocks.append(list(block_dict["block"]))
        return {"residual_cluster": list(block_dict["block"])}, None

    clusterer.predict_helper = cast(Any, fake_predict_helper)

    import s2and.incremental_linking.artifact as artifact_module
    import s2and.incremental_linking.query_adapter as query_adapter_module
    import s2and.incremental_linking.runtime as runtime_module

    artifact = SimpleNamespace(metadata=SimpleNamespace(retrieval_top_k=25))
    captured_inputs: dict[str, Any] = {}
    runtime_batches: list[list[str]] = []
    monkeypatch.setattr(artifact_module, "load_incremental_linking_artifact", lambda _path: artifact)

    def fake_build_inputs(**kwargs):
        captured_inputs.update(kwargs)
        query_by_signature_id = {
            str(signature_id): f"query-{signature_id}" for signature_id in kwargs["query_signature_ids"]
        }
        query_view_by_signature_id = {str(signature_id): "full" for signature_id in kwargs["query_signature_ids"]}
        return SimpleNamespace(
            queries=tuple(query_by_signature_id[signature_id] for signature_id in kwargs["query_signature_ids"]),
            query_by_signature_id=query_by_signature_id,
            query_views=tuple(
                query_view_by_signature_id[signature_id] for signature_id in kwargs["query_signature_ids"]
            ),
            query_view_by_signature_id=query_view_by_signature_id,
            retriever=object(),
            summary_by_component={},
        )

    monkeypatch.setattr(query_adapter_module, "build_incremental_linker_inputs", fake_build_inputs)
    monkeypatch.setattr(
        query_adapter_module,
        "build_name_count_rarity_row_signals",
        lambda *args, **kwargs: {},
    )

    def fake_private_runtime(clusterer_arg, artifact_arg, **kwargs):
        del clusterer_arg, artifact_arg
        batch = [str(signature_id) for signature_id in kwargs["query_signature_ids"]]
        runtime_batches.append(batch)
        assert kwargs["query_view"] == tuple("full" for _signature_id in batch)
        return SimpleNamespace(
            linked_signature_clusters={"5": "1"} if batch == ["5"] else {},
            telemetry={
                "query_count": len(batch),
                "candidate_row_count": 10 + len(batch),
                "pair_count": 20 + len(batch),
                "link_count": 1 if batch == ["5"] else 0,
                "abstain_count": 0 if batch == ["5"] else len(batch),
                "retrieval_top_k": 25,
                "seed_signature_count": 4,
                "seed_component_count": 2,
            },
        )

    monkeypatch.setattr(
        runtime_module,
        "_predict_incremental_link_or_abstain_production_private",
        fake_private_runtime,
    )

    result = clusterer.predict_incremental(block, dataset, batching_threshold=1)

    assert captured_inputs["query_signature_ids"] == ["5", "8"]
    assert captured_inputs["query_view"] is None
    assert captured_inputs["orcid_enabled"] is True
    assert runtime_batches == [["5"], ["8"]]
    assert residual_blocks == []
    assert any(set(signatures) == {"3", "4", "5"} for signatures in result["clusters"].values())
    telemetry = result["incremental_linker_telemetry"]
    assert telemetry["query_count"] == 2
    assert telemetry["candidate_row_count"] == 22
    assert telemetry["pair_count"] == 42
    assert telemetry["link_count"] == 1
    assert telemetry["abstain_count"] == 1
    assert telemetry["retrieval_top_k"] == 25
    assert telemetry["seed_signature_count"] == 4
    assert telemetry["seed_component_count"] == 2
    assert telemetry["query_batch_count"] == 2
    assert telemetry["query_batch_size_configured"] == 1
    assert telemetry["query_batch_size_min"] == 1
    assert telemetry["query_batch_size_max"] == 1
    assert telemetry["query_view_full_count"] == 2
    assert result["incremental_linker_query_view"] == "full"


def test_predict_incremental_promoted_linker_recalibrates_query_batch_size(
    clusterer_dataset_factory,
    monkeypatch,
    caplog,
):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_promoted_incremental_linker_calibration")
    block = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
    runtime_context = SimpleNamespace(
        operation="cluster_predict_incremental",
        requested_backend="rust",
        resolved_backend="rust",
        use_rust=True,
        run_id="test-rust-promoted-incremental-calibration",
        source="S2AND_BACKEND",
    )
    monkeypatch.setattr(model_module, "build_runtime_context", lambda _operation, **_kwargs: runtime_context)
    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_module, "_get_rust_featurizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        model_module,
        "_build_incremental_constraint_backend",
        lambda *args, **kwargs: SimpleNamespace(rust_featurizer=None),
    )
    monkeypatch.setattr(model_module.memory_budget, "current_rss_bytes_best_effort", lambda _total: (1_200, "rss:test"))

    def fake_limits(**kwargs):
        query_count = int(kwargs["query_count"])
        cap = kwargs.get("max_query_batch_size")
        max_batch = query_count if cap is None else min(query_count, int(cap))
        observed = int(kwargs.get("observed_query_count", 0) or 0) > 0
        query_batch_size = 0 if query_count == 0 else max(1, min(max_batch, 3 if observed else 1))
        return _mock_promoted_limits(
            query_count=query_count,
            query_batch_size=query_batch_size,
            predicted_peak_delta_bytes=2_000 + query_batch_size,
            predicted_peak_rss_bytes=3_000 + query_batch_size,
            operational_estimate_source="observed_probe" if observed else "top_k_largest_components",
            predicted_pairs_per_batch=40 * query_batch_size,
            predicted_candidate_rows_per_batch=10 * query_batch_size,
            pair_chunk_count=1 if query_batch_size else 0,
        )

    monkeypatch.setattr(
        production_module, "compute_promoted_incremental_limits", lambda **kwargs: fake_limits(**kwargs)
    )

    def fake_predict_helper(block_dict, dataset_arg, partial_supervision, runtime_context, total_ram_bytes=None):
        del dataset_arg, partial_supervision, runtime_context, total_ram_bytes
        return {"residual_cluster": list(block_dict["block"])}, None

    clusterer.predict_helper = cast(Any, fake_predict_helper)

    import s2and.incremental_linking.artifact as artifact_module
    import s2and.incremental_linking.query_adapter as query_adapter_module
    import s2and.incremental_linking.runtime as runtime_module

    monkeypatch.setattr(
        artifact_module,
        "load_incremental_linking_artifact",
        lambda _path: SimpleNamespace(metadata=SimpleNamespace(retrieval_top_k=25)),
    )

    def fake_build_inputs(**kwargs):
        assert kwargs["orcid_enabled"] is True
        query_by_signature_id = {
            str(signature_id): f"query-{signature_id}" for signature_id in kwargs["query_signature_ids"]
        }
        query_view_by_signature_id = {str(signature_id): "full" for signature_id in kwargs["query_signature_ids"]}
        return SimpleNamespace(
            queries=tuple(query_by_signature_id[signature_id] for signature_id in kwargs["query_signature_ids"]),
            query_by_signature_id=query_by_signature_id,
            query_views=tuple(
                query_view_by_signature_id[signature_id] for signature_id in kwargs["query_signature_ids"]
            ),
            query_view_by_signature_id=query_view_by_signature_id,
            retriever=object(),
            summary_by_component={},
        )

    monkeypatch.setattr(query_adapter_module, "build_incremental_linker_inputs", fake_build_inputs)
    monkeypatch.setattr(
        query_adapter_module,
        "build_name_count_rarity_row_signals",
        lambda *args, **kwargs: {},
    )

    runtime_batches: list[list[str]] = []

    def fake_private_runtime(clusterer_arg, artifact_arg, **kwargs):
        del clusterer_arg, artifact_arg
        batch = [str(signature_id) for signature_id in kwargs["query_signature_ids"]]
        runtime_batches.append(batch)
        assert kwargs["query_view"] == tuple("full" for _signature_id in batch)
        return SimpleNamespace(
            linked_signature_clusters={},
            telemetry={
                "query_count": len(batch),
                "candidate_row_count": 2 * len(batch),
                "pair_count": 4 * len(batch),
                "link_count": 0,
                "abstain_count": len(batch),
                "retrieval_top_k": 25,
                "seed_signature_count": 4,
                "seed_component_count": 2,
            },
        )

    monkeypatch.setattr(
        runtime_module,
        "_predict_incremental_link_or_abstain_production_private",
        fake_private_runtime,
    )

    with caplog.at_level("INFO", logger="s2and"):
        result = clusterer.predict_incremental(block, dataset, batching_threshold=None)

    assert runtime_batches == [["0"], ["1", "2", "5"], ["8"]]
    telemetry = result["incremental_linker_telemetry"]
    assert telemetry["query_batch_count"] == 3
    assert telemetry["query_batch_size_max"] == 3
    assert telemetry["memory_initial_query_batch_size"] == 1
    assert telemetry["memory_final_query_batch_size"] == 3
    assert telemetry["memory_observed_calibration_applied"] == 1
    assert telemetry["memory_final_operational_estimate_source"] == "observed_probe"
    assert telemetry["memory_predicted_peak_delta_bytes_max"] == 2003
    assert any(
        record.message.startswith("Telemetry: incremental_promoted_query_batch_calibration ")
        for record in caplog.records
    )


def test_promoted_incremental_batch_telemetry_does_not_sum_absolute_memory_fields() -> None:
    merged = production_module.merge_promoted_incremental_batch_telemetry(
        [
            {
                "query_count": 1,
                "memory_total_ram_bytes": 100,
                "memory_available_bytes": 40,
                "memory_stage_budget_bytes": 20,
            },
            {
                "query_count": 1,
                "memory_total_ram_bytes": 100,
                "memory_available_bytes": 35,
                "memory_stage_budget_bytes": 20,
            },
        ],
        batch_sizes=[1, 1],
        configured_batch_size=1,
    )

    assert merged["query_count"] == 2
    assert merged["memory_total_ram_bytes"] == 100
    assert merged["memory_available_bytes"] == 40
    assert merged["memory_stage_budget_bytes"] == 20
    assert merged["memory_available_bytes_batch_conflict_count"] == 1


def test_promoted_incremental_batch_telemetry_does_not_sum_raw_plan_seed_counts() -> None:
    merged = production_module.merge_promoted_incremental_batch_telemetry(
        [
            {"query_count": 1, "raw_arrow_plan_seed_signature_count": 10, "raw_arrow_plan_cluster_count": 2},
            {"query_count": 1, "raw_arrow_plan_seed_signature_count": 10, "raw_arrow_plan_cluster_count": 2},
        ],
        batch_sizes=[1, 1],
        configured_batch_size=1,
    )

    assert merged["query_count"] == 2
    assert merged["raw_arrow_plan_seed_signature_count"] == 10
    assert merged["raw_arrow_plan_cluster_count"] == 2


def test_promoted_incremental_batch_telemetry_keeps_numeric_after_mixed_type_conflict() -> None:
    merged = production_module.merge_promoted_incremental_batch_telemetry(
        [
            {"query_count": 1, "custom_metric": "unregistered"},
            {"query_count": 1, "custom_metric": 3},
            {"query_count": 1, "custom_metric": 4},
        ],
        batch_sizes=[1, 1, 1],
        configured_batch_size=1,
    )

    assert merged["query_count"] == 3
    assert merged["custom_metric"] == 7
    assert merged["custom_metric_batch_conflict_count"] == 1


def test_raw_window_plan_telemetry_marks_conflicting_string_values() -> None:
    merged: dict[str, int | float | str] = {}

    production_module._merge_raw_window_plan_telemetry(
        merged,
        {
            "raw_arrow_window_plan_query_view": "full",
            "raw_arrow_window_plan_signature_count": 3,
        },
    )
    production_module._merge_raw_window_plan_telemetry(
        merged,
        {
            "raw_arrow_window_plan_query_view": "initial_only",
            "raw_arrow_window_plan_signature_count": 4,
        },
    )

    assert merged["raw_arrow_window_plan_query_view"] == "__mixed__"
    assert merged["raw_arrow_window_plan_signature_count"] == 7.0


def test_predict_incremental_promoted_linker_fails_closed_when_single_query_exceeds_budget(
    clusterer_dataset_factory,
    monkeypatch,
):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_promoted_incremental_linker_budget_fail")
    runtime_context = SimpleNamespace(
        operation="cluster_predict_incremental",
        requested_backend="rust",
        resolved_backend="rust",
        use_rust=True,
        run_id="test-rust-promoted-incremental-budget-fail",
        source="S2AND_BACKEND",
    )
    monkeypatch.setattr(model_module, "build_runtime_context", lambda _operation, **_kwargs: runtime_context)
    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        model_module,
        "_get_rust_featurizer",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should fail before featurizer build")),
    )

    import s2and.incremental_linking.artifact as artifact_module

    monkeypatch.setattr(
        artifact_module,
        "load_incremental_linking_artifact",
        lambda _path: SimpleNamespace(metadata=SimpleNamespace(retrieval_top_k=25)),
    )
    monkeypatch.setattr(
        production_module,
        "compute_promoted_incremental_limits",
        lambda **_kwargs: _mock_promoted_limits(single_query_exceeds_budget=True),
    )

    with pytest.raises(MemoryError, match="cannot fit a single query"):
        clusterer.predict_incremental(["3", "4", "5"], dataset, batching_threshold=None)


def test_predict_incremental_dont_use_cluster_seeds_flag(clusterer_dataset_factory):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_incremental_alias")
    block = {"block": ["3", "4", "5", "6"]}

    expected_clusters, _ = clusterer.predict(block, dataset)
    explicit_default_clusters, _ = clusterer.predict(
        block,
        dataset,
        incremental_dont_use_cluster_seeds=False,
    )

    assert _same_partition(expected_clusters, explicit_default_clusters)


def test_clusterer_init_prefers_legacy_seed_flag_name():
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    rng = np.random.RandomState(7)
    X_random = rng.random((10, 6))
    y_random = rng.randint(0, 2, 10)

    clusterer = Clusterer(
        featurizer_info=featurizer_info,
        classifier=LGBMClassifier(random_state=7, data_random_seed=7, feature_fraction_seed=7, verbosity=-1).fit(
            X_random, y_random
        ),
        dont_merge_cluster_seeds=False,
        n_jobs=1,
        use_cache=False,
    )

    assert clusterer.dont_merge_cluster_seeds is False


def _mock_promoted_limits(
    *,
    query_count: int = 1,
    query_batch_size: int = 1,
    single_query_exceeds_budget: bool = False,
    operational_estimate_source: str = "top_k_largest_components",
    predicted_peak_delta_bytes: int = 2_000,
    predicted_peak_rss_bytes: int = 3_000,
    predicted_pairs_per_batch: int = 40,
    predicted_candidate_rows_per_batch: int = 10,
    pair_chunk_pairs: int = 100,
    pair_chunk_count: int = 1,
) -> model_module.memory_budget.PromotedPhaseALimits:
    return model_module.memory_budget.PromotedPhaseALimits(
        total_ram_bytes=100_000,
        total_ram_source="test",
        current_rss_bytes=1_000,
        current_rss_source="rss:test",
        available_bytes=90_000,
        effective_available_fraction=0.9,
        safety_margin_bytes=1_000,
        stage_budget_fraction=0.5,
        stage_budget_bytes=10_000,
        query_count=int(query_count),
        max_query_batch_size=max(1, int(query_count)),
        query_batch_size=int(query_batch_size),
        component_count=4,
        retrieval_top_k=25,
        candidate_rows_per_query=2,
        conservative_pairs_per_query=4,
        hard_query_batch_size=int(query_batch_size),
        observed_query_count=0,
        observed_candidate_rows_per_query=0,
        observed_pairs_per_query=0,
        observed_safety_multiplier=2.0,
        operational_candidate_rows_per_query=2,
        operational_pairs_per_query=4,
        operational_estimate_source=operational_estimate_source,
        max_component_size=2,
        predicted_candidate_rows_per_batch=int(predicted_candidate_rows_per_batch),
        predicted_pairs_per_batch=int(predicted_pairs_per_batch),
        hard_predicted_candidate_rows_per_batch=int(predicted_candidate_rows_per_batch),
        hard_predicted_pairs_per_batch=int(predicted_pairs_per_batch),
        retrieval_pair_bytes=16,
        retrieval_row_bytes=512,
        pair_label_bytes=8,
        distance_row_bytes=96,
        final_matrix_feature_count=70,
        aggregate_feature_count=31,
        fixed_overhead_bytes=16 * (1 << 20),
        predicted_retrieval_pair_arrays_bytes=0,
        predicted_retrieval_row_bytes=0,
        predicted_pair_label_bytes=0,
        predicted_aggregate_bytes=0,
        predicted_distance_row_bytes=0,
        predicted_final_matrix_bytes=0,
        predicted_fixed_overhead_bytes=16 * (1 << 20),
        predicted_persistent_bytes=0,
        predicted_pair_chunk_bytes=0,
        predicted_peak_delta_bytes=int(predicted_peak_delta_bytes),
        predicted_peak_rss_bytes=int(predicted_peak_rss_bytes),
        pair_chunk_pairs=int(pair_chunk_pairs),
        pair_chunk_count=int(pair_chunk_count),
        pair_chunk_stage_budget_bytes=10_000,
        single_query_predicted_persistent_bytes=100 if not single_query_exceeds_budget else 20_000,
        single_query_exceeds_budget=bool(single_query_exceeds_budget),
    )


def _build_minimal_incremental_clusterer() -> Clusterer:
    return Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff"]),
        classifier=object(),
        n_jobs=1,
        use_cache=False,
    )


def test_next_unused_cluster_id_prevents_overwrite():
    pred_clusters = {
        "0": ["s0"],
        "1": ["s1"],
        "2": ["existing_singleton_cluster"],
    }
    start = model_module._next_unused_cluster_id(pred_clusters, 2)
    assert start == 3

    # Simulate the singleton recluster append loop in _predict_incremental_helper.
    for signatures in (["new_a"], ["new_b"]):
        cluster_id = model_module._next_unused_cluster_id(pred_clusters, start)
        pred_clusters[str(cluster_id)] = signatures
        start = cluster_id + 1

    assert pred_clusters["2"] == ["existing_singleton_cluster"]
    assert pred_clusters["3"] == ["new_a"]
    assert pred_clusters["4"] == ["new_b"]


def test_predict_incremental_without_seeds_covers_all_signatures(clusterer_dataset_factory):
    clusterer, dataset = clusterer_dataset_factory()
    dataset.cluster_seeds_require = {}
    block = ["3", "4", "5", "6", "7", "8"]

    output_no_subblock = _clusters(clusterer.predict_incremental(block, dataset, batching_threshold=None))
    assigned_no_subblock = {signature for signatures in output_no_subblock.values() for signature in signatures}
    assert assigned_no_subblock == set(block)

    # Re-create to get fresh state (dataset.cluster_seeds_require was mutated above).
    clusterer2, dataset2 = clusterer_dataset_factory()
    dataset2.cluster_seeds_require = {}
    with pytest.raises(ValueError, match="batching_threshold is only supported for promoted Rust"):
        clusterer2.predict_incremental(block, dataset2, batching_threshold=3)


def test_predict_incremental_batch_constraint_path_parity(clusterer_dataset_factory, monkeypatch):
    block = ["3", "4", "5", "6", "7", "8"]

    baseline_clusterer, baseline_dataset = clusterer_dataset_factory()
    baseline = _clusters(baseline_clusterer.predict_incremental(block, baseline_dataset, batching_threshold=None))

    batch_clusterer, batch_dataset = clusterer_dataset_factory()

    sig_ids = list(batch_dataset.signatures.keys())

    class _FakeIndexedFeaturizer:
        def signature_ids(self):
            return sig_ids

        def get_constraints_matrix_indexed(self, *_args, **_kwargs):
            return [None]

    calls = {"batch": 0}
    monkeypatch.setattr(
        model_module,
        "_initialize_incremental_constraint_backend",
        lambda *_args, **_kwargs: (_FakeIndexedFeaturizer(), True),
    )

    def _fake_get_constraints_matrix_indexed_rust(dataset, indexed_pairs, **kwargs):
        calls["batch"] += 1
        dont_merge = kwargs.get("dont_merge_cluster_seeds", True)
        incremental_flag = kwargs.get("incremental_dont_use_cluster_seeds", False)
        return [
            dataset.get_constraint(
                sig_ids[i1],
                sig_ids[i2],
                dont_merge_cluster_seeds=dont_merge,
                incremental_dont_use_cluster_seeds=incremental_flag,
            )
            for i1, i2 in indexed_pairs
        ]

    monkeypatch.setattr(model_module, "get_constraints_matrix_indexed_rust", _fake_get_constraints_matrix_indexed_rust)
    monkeypatch.setattr(model_module, "get_constraint_rust", lambda *_args, **_kwargs: None)

    batch_output = _clusters(batch_clusterer.predict_incremental(block, batch_dataset, batching_threshold=None))
    assert _same_partition(
        batch_output, baseline
    ), f"Batch-constraint and baseline partitions differ:\n  batch={batch_output}\n  baseline={baseline}"
    assert calls["batch"] > 0


def test_predict_subblocked_processes_subblocks_in_sorted_key_order(clusterer_dataset_factory, monkeypatch):
    clusterer, dataset = clusterer_dataset_factory()
    block_signatures = ["3", "4", "5", "6"]
    observed_order: list[str] = []

    def _fake_make_subblocks(signatures, anddata, maximum_size=7500, first_k_letter_counts_sorted=None, **kwargs):
        del signatures, anddata, maximum_size, first_k_letter_counts_sorted, kwargs
        # Intentionally unsorted insertion order to verify deterministic processing order in predict().
        return {"zeta": ["3", "4"], "alpha": ["5", "6"]}

    def _fake_predict_helper(
        self,
        block_dict,
        dataset,
        dists=None,
        cluster_model_params=None,
        partial_supervision=None,
        use_s2_clusters=False,
        incremental_dont_use_cluster_seeds=False,
        runtime_context=None,
        total_ram_bytes=None,
    ):
        del self, dataset, dists, cluster_model_params, partial_supervision
        del use_s2_clusters, incremental_dont_use_cluster_seeds, runtime_context, total_ram_bytes
        key = next(iter(block_dict))
        observed_order.append(key)
        return {f"cluster_{len(observed_order)}": list(block_dict[key])}, None

    monkeypatch.setattr(model_module, "make_subblocks", _fake_make_subblocks)
    monkeypatch.setattr(model_module, "_signature_first_for_rules", lambda _: "john")
    monkeypatch.setattr(Clusterer, "predict_helper", _fake_predict_helper)

    clusterer.predict({"block": block_signatures}, dataset, batching_threshold=3)
    assert observed_order == ["block|subblock=alpha", "block|subblock=zeta"]


def test_predict_subblocked_arrow_path_wires_graph_fallback(clusterer_dataset_factory, monkeypatch):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_arrow_graph_subblocking")
    fake_arrow_paths = {
        "signatures": "signatures.arrow",
        "papers": "papers.arrow",
        "paper_authors": "paper_authors.arrow",
        "specter": "specter.arrow",
    }
    captured: dict[str, Any] = {}

    class FakeFallback:
        load_seconds = 1.25
        stats = [{"input_signature_count": 4, "packed_component_count": 2}]

        def __call__(self, *_args: object, **_kwargs: object) -> dict[str, list[str]]:
            raise AssertionError("make_subblocks fake should only receive the callback")

    fake_fallback = FakeFallback()

    def fake_factory(paths, signature_ids, *, config, random_seed):
        captured["factory_paths"] = dict(paths)
        captured["factory_signature_ids"] = list(signature_ids)
        captured["factory_config"] = config
        captured["factory_random_seed"] = random_seed
        return fake_fallback

    def fake_make_subblocks(signatures, anddata, maximum_size=7500, first_k_letter_counts_sorted=None, **kwargs):
        del anddata, first_k_letter_counts_sorted
        captured["make_subblocks_signatures"] = list(signatures)
        captured["make_subblocks_maximum_size"] = maximum_size
        captured["specter_cluster_fn"] = kwargs.get("specter_cluster_fn")
        return {"zeta": ["3", "4"], "alpha": ["5", "6"]}

    def fake_partition(self, block_dict_subblocked, dataset):
        del self, dataset
        captured["subblocked_keys"] = sorted(block_dict_subblocked)
        return {}, {}, False

    def fake_predict_multiple(self, block_dict_multiple_letter, **kwargs):
        del self, block_dict_multiple_letter, kwargs
        return {"cluster": ["3", "4", "5", "6"]}

    def fake_predict_single(self, block_dict_single_letter, *, pred_clusters, **kwargs):
        del self, block_dict_single_letter, kwargs
        return pred_clusters

    monkeypatch.setattr(model_module, "make_arrow_graph_subblocking_cluster_fn", fake_factory)
    monkeypatch.setattr(model_module, "make_subblocks", fake_make_subblocks)
    monkeypatch.setattr(Clusterer, "_partition_subblocked_first_name_groups", fake_partition)
    monkeypatch.setattr(Clusterer, "_predict_subblocked_multiple_letter_groups", fake_predict_multiple)
    monkeypatch.setattr(Clusterer, "_predict_subblocked_single_letter_incremental_groups", fake_predict_single)

    pred_clusters, dists = clusterer._predict_subblocked(
        {"block": ["3", "4", "5", "6"]},
        dataset,
        cluster_model_params={},
        partial_supervision={},
        use_s2_clusters=False,
        incremental_dont_use_cluster_seeds=False,
        batching_threshold=3,
        desired_memory_use=None,
        runtime_context=cast(Any, SimpleNamespace(run_id="test", use_rust=True)),
        dists=None,
        total_ram_bytes=None,
        restore_rust_cluster_seeds_on_exit=True,
        arrow_paths=fake_arrow_paths,
    )

    assert pred_clusters == {"cluster": ["3", "4", "5", "6"]}
    assert dists is None
    assert captured["factory_paths"] == fake_arrow_paths
    assert captured["factory_signature_ids"] == ["3", "4", "5", "6"]
    assert captured["factory_random_seed"] == clusterer.random_state
    assert captured["specter_cluster_fn"].graph_fallback is fake_fallback
    assert captured["subblocked_keys"] == ["block|subblock=alpha", "block|subblock=zeta"]
    assert clusterer._last_arrow_graph_subblocking_telemetry == {
        "enabled": 1,
        "mode": "graph",
        "source": "arrow",
        "candidate_signature_count": 4,
        "arrow_load_seconds": 1.25,
        "arrow_load_metrics": {},
        "fallback_invocation_count": 1,
        "fallback_stats": [{"input_signature_count": 4, "packed_component_count": 2}],
        "legacy_fallback_invocation_count": 0,
        "graph_prepare_failed": 0,
        "graph_prepare_error": None,
        "graph_fallback_errors": [],
    }


def test_build_subblocked_block_dict_uses_indexed_arrow_rust_subblocking_when_enabled(
    clusterer_dataset_factory, monkeypatch
):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_arrow_rust_subblocking")
    fake_arrow_paths = {"signatures": "signatures.arrow", "signatures_batch_index": "signatures.index"}
    captured: dict[str, Any] = {}

    def fake_make_subblocks_arrow_rust(arrow_paths, signatures, anddata, **kwargs):
        captured["arrow_paths"] = dict(arrow_paths)
        captured["signatures"] = list(signatures)
        captured["anddata"] = anddata
        captured["kwargs"] = dict(kwargs)
        return {"beta": ["3", "4"], "alpha": ["5", "6"]}

    def fail_make_subblocks(*_args, **_kwargs):
        raise AssertionError("Python make_subblocks should not run when Arrow Rust subblocking is enabled")

    fallback = object()
    monkeypatch.setattr(model_module, "make_subblocks_arrow_rust", fake_make_subblocks_arrow_rust)
    monkeypatch.setattr(model_module, "make_subblocks", fail_make_subblocks)
    monkeypatch.setattr(model_module, "rust_arrow_subblocking_available", lambda: True)

    output = clusterer._build_subblocked_block_dict(
        {"block": ["3", "4", "5", "6"]},
        dataset,
        batching_threshold=3,
        specter_cluster_fn=cast(Any, fallback),
        subblocking_arrow_paths=fake_arrow_paths,
        use_rust_subblocking=True,
    )

    assert output == {
        "block|subblock=alpha": ["5", "6"],
        "block|subblock=beta": ["3", "4"],
    }
    assert captured["arrow_paths"] == fake_arrow_paths
    assert captured["signatures"] == ["3", "4", "5", "6"]
    assert captured["anddata"] is dataset
    assert captured["kwargs"]["maximum_size"] == 3
    assert captured["kwargs"]["specter_cluster_fn"] is fallback


def test_build_subblocked_block_dict_falls_back_when_arrow_rust_subblocking_unavailable(
    clusterer_dataset_factory, monkeypatch
):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_arrow_rust_subblocking_unavailable")
    fake_arrow_paths = {"signatures": "signatures.arrow", "signatures_batch_index": "signatures.index"}
    captured: dict[str, Any] = {}

    def fail_make_subblocks_arrow_rust(*_args, **_kwargs):
        raise AssertionError("Rust Arrow subblocking should not run when the capability is missing")

    def fake_make_subblocks(signatures, anddata, **kwargs):
        captured["signatures"] = list(signatures)
        captured["anddata"] = anddata
        captured["kwargs"] = dict(kwargs)
        return {"alpha": ["3", "4"], "beta": ["5", "6"]}

    fallback = object()
    monkeypatch.setattr(model_module, "make_subblocks_arrow_rust", fail_make_subblocks_arrow_rust)
    monkeypatch.setattr(model_module, "make_subblocks", fake_make_subblocks)
    monkeypatch.setattr(model_module, "rust_arrow_subblocking_available", lambda: False)

    output = clusterer._build_subblocked_block_dict(
        {"block": ["3", "4", "5", "6"]},
        dataset,
        batching_threshold=3,
        specter_cluster_fn=cast(Any, fallback),
        subblocking_arrow_paths=fake_arrow_paths,
        use_rust_subblocking=True,
    )

    assert output == {
        "block|subblock=alpha": ["3", "4"],
        "block|subblock=beta": ["5", "6"],
    }
    assert captured["signatures"] == ["3", "4", "5", "6"]
    assert captured["anddata"] is dataset
    assert captured["kwargs"]["maximum_size"] == 3
    assert captured["kwargs"]["specter_cluster_fn"] is fallback


def test_predict_accepts_backend_argument(clusterer_dataset_factory, monkeypatch):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_predict_backend_argument")
    captured: dict[str, Any] = {}
    runtime_context = SimpleNamespace(
        operation="cluster_predict",
        requested_backend="python",
        resolved_backend="python",
        use_rust=False,
        run_id="test",
        source="argument",
    )

    def fake_build_runtime_context(operation, *, backend=None, **kwargs):
        del kwargs
        captured["operation"] = operation
        captured["backend"] = backend
        return runtime_context

    def fake_predict_subblocked(self, block_dict, dataset, **kwargs):
        del self, dataset
        captured["block_dict"] = dict(block_dict)
        captured["runtime_context"] = kwargs["runtime_context"]
        return {"cluster": ["3", "4"]}, None

    monkeypatch.setattr(model_module, "build_runtime_context", fake_build_runtime_context)
    monkeypatch.setattr(Clusterer, "_predict_subblocked", fake_predict_subblocked)

    pred_clusters, dists = clusterer.predict(
        {"block": ["3", "4", "5", "6"]},
        dataset,
        batching_threshold=3,
        backend="python",
    )

    assert pred_clusters == {"cluster": ["3", "4"]}
    assert dists is None
    assert captured["operation"] == "cluster_predict"
    assert captured["backend"] == "python"
    assert captured["runtime_context"] is runtime_context


def test_predict_rejects_backend_with_runtime_context(clusterer_dataset_factory):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_predict_backend_conflict")
    runtime_context = SimpleNamespace(
        operation="cluster_predict",
        requested_backend="python",
        resolved_backend="python",
        use_rust=False,
        run_id="test",
        source="argument",
    )

    with pytest.raises(ValueError, match="Pass either runtime_context or backend"):
        clusterer.predict(
            {"block": ["3", "4"]},
            dataset,
            runtime_context=cast(Any, runtime_context),
            backend="python",
        )


def test_predict_subblocked_python_path_wires_graph_fallback(clusterer_dataset_factory, monkeypatch):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_python_graph_subblocking")
    captured: dict[str, Any] = {}

    class FakeFallback:
        load_seconds = 0.0
        stats = [{"input_signature_count": 4, "packed_component_count": 2}]

    fake_fallback = FakeFallback()

    def fake_factory(*, config):
        captured["factory_config"] = config
        return fake_fallback

    def fake_make_subblocks(signatures, anddata, maximum_size=7500, first_k_letter_counts_sorted=None, **kwargs):
        del anddata, first_k_letter_counts_sorted
        captured["make_subblocks_signatures"] = list(signatures)
        captured["make_subblocks_maximum_size"] = maximum_size
        captured["specter_cluster_fn"] = kwargs.get("specter_cluster_fn")
        return {"zeta": ["3", "4"], "alpha": ["5", "6"]}

    def fake_partition(self, block_dict_subblocked, dataset):
        del self, dataset
        captured["subblocked_keys"] = sorted(block_dict_subblocked)
        return {}, {}, False

    def fake_predict_multiple(self, block_dict_multiple_letter, **kwargs):
        del self, block_dict_multiple_letter, kwargs
        return {"cluster": ["3", "4", "5", "6"]}

    def fake_predict_single(self, block_dict_single_letter, *, pred_clusters, **kwargs):
        del self, block_dict_single_letter, kwargs
        return pred_clusters

    monkeypatch.setattr(model_module, "make_dataset_graph_subblocking_cluster_fn", fake_factory)
    monkeypatch.setattr(model_module, "make_subblocks", fake_make_subblocks)
    monkeypatch.setattr(Clusterer, "_partition_subblocked_first_name_groups", fake_partition)
    monkeypatch.setattr(Clusterer, "_predict_subblocked_multiple_letter_groups", fake_predict_multiple)
    monkeypatch.setattr(Clusterer, "_predict_subblocked_single_letter_incremental_groups", fake_predict_single)

    pred_clusters, dists = clusterer._predict_subblocked(
        {"block": ["3", "4", "5", "6"]},
        dataset,
        cluster_model_params={},
        partial_supervision={},
        use_s2_clusters=False,
        incremental_dont_use_cluster_seeds=False,
        batching_threshold=3,
        desired_memory_use=None,
        runtime_context=cast(Any, SimpleNamespace(run_id="test", use_rust=False)),
        dists=None,
        total_ram_bytes=None,
        restore_rust_cluster_seeds_on_exit=True,
        arrow_paths=None,
    )

    assert pred_clusters == {"cluster": ["3", "4", "5", "6"]}
    assert dists is None
    assert captured["specter_cluster_fn"].graph_fallback is fake_fallback
    assert captured["subblocked_keys"] == ["block|subblock=alpha", "block|subblock=zeta"]
    assert clusterer._last_graph_subblocking_telemetry == {
        "enabled": 1,
        "mode": "graph",
        "source": "anddata",
        "candidate_signature_count": 4,
        "arrow_load_seconds": 0.0,
        "arrow_load_metrics": {},
        "fallback_invocation_count": 1,
        "fallback_stats": [{"input_signature_count": 4, "packed_component_count": 2}],
        "legacy_fallback_invocation_count": 0,
        "graph_prepare_failed": 0,
        "graph_prepare_error": None,
        "graph_fallback_errors": [],
    }


def test_graph_subblocking_is_default(clusterer_dataset_factory):
    clusterer, _dataset = clusterer_dataset_factory(name="dummy_default_graph_subblocking")

    assert clusterer.subblocking_fallback_mode == "graph"


def test_legacy_subblocking_telemetry_has_graph_contract_keys(clusterer_dataset_factory):
    clusterer, _dataset = clusterer_dataset_factory(name="dummy_legacy_graph_subblocking_telemetry")
    clusterer.subblocking_fallback_mode = "legacy"

    fallback = clusterer._subblocking_specter_cluster_fn(None, ["3", "4", "3"])

    assert fallback is None
    assert clusterer._last_graph_subblocking_telemetry == {
        "enabled": 0,
        "mode": "legacy",
        "source": "legacy",
        "candidate_signature_count": 2,
        "arrow_load_seconds": 0.0,
        "arrow_load_metrics": {},
        "fallback_invocation_count": 0,
        "fallback_stats": [],
        "legacy_fallback_invocation_count": 0,
        "graph_prepare_failed": 0,
        "graph_prepare_error": None,
        "graph_fallback_errors": [],
    }
    assert clusterer._last_arrow_graph_subblocking_telemetry is clusterer._last_graph_subblocking_telemetry


def test_graph_subblocking_value_errors_propagate(clusterer_dataset_factory, monkeypatch):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_graph_value_error_propagates")

    class FailingFallback:
        load_seconds = 0.0
        load_metrics = {}
        stats: list[dict[str, Any]] = []

        def __call__(self, *_args: object, **_kwargs: object) -> dict[str, list[str]]:
            raise ValueError("graph failed")

    def fake_factory(*, config):
        del config
        return FailingFallback()

    monkeypatch.setattr(model_module, "make_dataset_graph_subblocking_cluster_fn", fake_factory)
    monkeypatch.setattr(
        model_module,
        "cluster_with_specter",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy fallback should not hide graph ValueError")
        ),
    )

    fallback = clusterer._subblocking_specter_cluster_fn(None, ["3", "4", "5"])
    assert fallback is not None

    with pytest.raises(ValueError, match="graph failed"):
        fallback(["3", "4", "5"], dataset, target_subblock_size=2, compute_block_fn=str)

    assert fallback.legacy_fallback_invocation_count == 0
    assert fallback.graph_fallback_errors == []


def test_graph_subblocking_falls_back_to_legacy_specter_for_io_errors(clusterer_dataset_factory, monkeypatch):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_graph_io_legacy_fallback")
    captured: dict[str, Any] = {}

    class FailingFallback:
        load_seconds = 0.0
        load_metrics = {}
        stats: list[dict[str, Any]] = []

        def __call__(self, *_args: object, **_kwargs: object) -> dict[str, list[str]]:
            raise OSError("arrow mmap failed")

    def fake_factory(*, config):
        del config
        return FailingFallback()

    def fake_legacy(signature_ids, anddata, target_subblock_size=10000, **kwargs):
        del anddata, target_subblock_size, kwargs
        captured["legacy_signature_ids"] = list(signature_ids)
        return {"legacy": list(signature_ids)}

    monkeypatch.setattr(model_module, "make_dataset_graph_subblocking_cluster_fn", fake_factory)
    monkeypatch.setattr(model_module, "cluster_with_specter", fake_legacy)

    fallback = clusterer._subblocking_specter_cluster_fn(None, ["3", "4", "5"])
    assert fallback is not None

    output = fallback(["3", "4", "5"], dataset, target_subblock_size=2)

    assert output == {"legacy": ["3", "4", "5"]}
    assert captured["legacy_signature_ids"] == ["3", "4", "5"]
    assert fallback.legacy_fallback_invocation_count == 1
    assert fallback.graph_fallback_errors == [
        {
            "stage": "call",
            "type": "OSError",
            "message": "arrow mmap failed",
            "signature_count": 3,
        }
    ]


def test_graph_subblocking_prepare_io_errors_switch_to_legacy_specter(clusterer_dataset_factory, monkeypatch):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_graph_prepare_io_legacy_fallback")
    captured: dict[str, Any] = {}

    class FailingPrepareFallback:
        load_seconds = 0.0
        load_metrics = {}
        stats: list[dict[str, Any]] = []

        def prepare(self, _signature_groups):
            raise OSError("arrow footer is invalid")

        def __call__(self, *_args: object, **_kwargs: object) -> dict[str, list[str]]:
            raise AssertionError("graph fallback should not be called after prepare failure")

    def fake_factory(*, config):
        del config
        return FailingPrepareFallback()

    def fake_legacy(signature_ids, anddata, target_subblock_size=10000, **kwargs):
        del anddata, target_subblock_size, kwargs
        captured["legacy_signature_ids"] = list(signature_ids)
        return {"legacy": list(signature_ids)}

    monkeypatch.setattr(model_module, "make_dataset_graph_subblocking_cluster_fn", fake_factory)
    monkeypatch.setattr(model_module, "cluster_with_specter", fake_legacy)

    fallback = clusterer._subblocking_specter_cluster_fn(None, ["3", "4", "5"])
    assert fallback is not None
    fallback.prepare([["3", "4"], ["5"]])

    output = fallback(["3", "4", "5"], dataset, target_subblock_size=2)

    assert output == {"legacy": ["3", "4", "5"]}
    assert captured["legacy_signature_ids"] == ["3", "4", "5"]
    assert fallback.graph_prepare_failed is True
    assert fallback.graph_prepare_error == {
        "stage": "prepare",
        "type": "OSError",
        "message": "arrow footer is invalid",
        "signature_count": 3,
        "group_count": 2,
    }


def test_graph_subblocking_unexpected_errors_propagate(clusterer_dataset_factory, monkeypatch):
    clusterer, dataset = clusterer_dataset_factory(name="dummy_graph_unexpected_failure")

    class FailingFallback:
        def __call__(self, *_args: object, **_kwargs: object) -> dict[str, list[str]]:
            raise RuntimeError("graph bug")

    def fake_factory(*, config):
        del config
        return FailingFallback()

    monkeypatch.setattr(model_module, "make_dataset_graph_subblocking_cluster_fn", fake_factory)

    fallback = clusterer._subblocking_specter_cluster_fn(None, ["3", "4", "5"])
    assert fallback is not None

    with pytest.raises(RuntimeError, match="graph bug"):
        fallback(["3", "4", "5"], dataset, target_subblock_size=2)


def test_best_incremental_cluster_respects_seed_score_mode():
    clusterer = _build_minimal_incremental_clusterer()
    cluster_dists = {
        "mean_favorite": (0.20, 2, 0.20),
        "min_favorite": (0.29, 2, 0.01),
    }

    clusterer.incremental_seed_score_mode = "mean"
    best_mean, best_mean_score, _ = clusterer._best_incremental_cluster(
        cluster_dists,
        config=clusterer._incremental_experiment_config(),
    )
    assert best_mean == "mean_favorite"
    assert best_mean_score == pytest.approx(0.20)

    clusterer.incremental_seed_score_mode = "min"
    best_min, best_min_score, _ = clusterer._best_incremental_cluster(
        cluster_dists,
        config=clusterer._incremental_experiment_config(),
    )
    assert best_min == "min_favorite"
    assert best_min_score == pytest.approx(0.01)

    clusterer.incremental_seed_score_mode = "mean_min_hybrid"
    clusterer.incremental_mean_min_hybrid_weight = 0.25
    best_hybrid_low, best_hybrid_low_score, _ = clusterer._best_incremental_cluster(
        cluster_dists,
        config=clusterer._incremental_experiment_config(),
    )
    assert best_hybrid_low == "mean_favorite"
    assert best_hybrid_low_score == pytest.approx(0.20)

    clusterer.incremental_mean_min_hybrid_weight = 0.75
    best_hybrid_high, best_hybrid_high_score, _ = clusterer._best_incremental_cluster(
        cluster_dists,
        config=clusterer._incremental_experiment_config(),
    )
    assert best_hybrid_high == "min_favorite"
    assert best_hybrid_high_score == pytest.approx(0.08)


def test_finish_incremental_with_seed_links_reclusters_only_abstains():
    clusterer = _build_minimal_incremental_clusterer()
    residual_blocks: list[list[str]] = []
    residual_total_ram_bytes: list[int | None] = []

    def fake_predict_helper(block_dict, dataset, partial_supervision, runtime_context, total_ram_bytes=None):
        del dataset, partial_supervision, runtime_context
        residual_blocks.append(list(block_dict["block"]))
        residual_total_ram_bytes.append(total_ram_bytes)
        return {"residual_cluster": list(block_dict["block"])}, None

    clusterer.predict_helper = cast(Any, fake_predict_helper)
    dataset = cast(
        ANDData,
        type(
            "IncrementalDataset",
            (),
            {
                "cluster_seeds_require": {"seed0": "7", "seed1": "8"},
                "max_seed_cluster_id": 8,
                "signatures": {},
                "name_tuples": set(),
            },
        )(),
    )

    result = clusterer._finish_incremental_with_seed_links(
        ["u1", "u2"],
        dataset,
        {"u1": "7_0"},
        {"7_0": "7"},
        {"7": ["seed0"], "8": ["seed1"]},
        False,
        {},
        runtime_context=cast(Any, object()),
        total_ram_bytes=123_456,
    )

    assert result == {"7": ["seed0", "u1"], "8": ["seed1"], "9": ["u2"]}
    assert residual_blocks == []
    assert residual_total_ram_bytes == []


def test_finish_incremental_with_seed_links_uses_seed_setup_when_dataset_seed_map_is_empty():
    clusterer = _build_minimal_incremental_clusterer()
    dataset = cast(
        ANDData,
        SimpleNamespace(
            cluster_seeds_require={},
            signatures={},
            max_seed_cluster_id=0,
            name_tuples="filtered",
        ),
    )

    result = clusterer._finish_incremental_with_seed_links(
        ["q1"],
        dataset,
        {"q1": "c1"},
        {},
        {"c1": ["s1", "s2"]},
        prevent_new_incompatibilities=False,
        partial_supervision={},
        runtime_context=cast(Any, object()),
    )

    assert result == {"c1": ["s1", "s2", "q1"]}


def test_finish_incremental_with_seed_links_reclusters_abstains_from_arrow_paths():
    clusterer = _build_minimal_incremental_clusterer()
    captured: dict[str, Any] = {}

    def fail_predict_helper(*_args, **_kwargs):
        raise AssertionError("Arrow residual Phase B should not call legacy predict_helper")

    def fake_predict_from_arrow_paths(block_dict, arrow_paths, **kwargs):
        captured["block_dict"] = dict(block_dict)
        captured["arrow_paths"] = dict(arrow_paths)
        captured["partial_supervision"] = dict(kwargs["partial_supervision"])
        captured["runtime_context"] = kwargs["runtime_context"]
        captured["total_ram_bytes"] = kwargs["total_ram_bytes"]
        return {"residual_cluster": list(block_dict["block"])}, None

    clusterer.predict_helper = cast(Any, fail_predict_helper)
    clusterer.predict_from_arrow_paths = cast(Any, fake_predict_from_arrow_paths)
    dataset = cast(
        ANDData,
        SimpleNamespace(
            cluster_seeds_require={"seed0": "7", "seed1": "8"},
            cluster_seeds_disallow={("u2", "u3")},
            max_seed_cluster_id=8,
            signatures={},
            name_tuples=set(),
        ),
    )
    arrow_paths = {"signatures": "signatures.arrow", "papers": "papers.arrow", "paper_authors": "paper_authors.arrow"}
    runtime_context = cast(Any, object())

    result = clusterer._finish_incremental_with_seed_links(
        ["u1", "u2", "u3"],
        dataset,
        {"u1": "7_0"},
        {"7_0": "7"},
        {"7": ["seed0"], "8": ["seed1"]},
        False,
        {},
        runtime_context=runtime_context,
        total_ram_bytes=123_456,
        arrow_paths=arrow_paths,
    )

    assert result == {"7": ["seed0", "u1"], "8": ["seed1"], "9": ["u2", "u3"]}
    assert captured["block_dict"] == {"block": ["u2", "u3"]}
    assert captured["arrow_paths"] == arrow_paths
    assert captured["partial_supervision"] == {("u2", "u3"): LARGE_DISTANCE}
    assert captured["runtime_context"] is runtime_context
    assert captured["total_ram_bytes"] == 123_456


def test_finish_incremental_with_seed_links_splits_residual_phase_b_by_first_initial():
    clusterer = _build_minimal_incremental_clusterer()
    residual_blocks: list[list[str]] = []

    def fake_predict_from_arrow_paths(block_dict, arrow_paths, **kwargs):
        del arrow_paths, kwargs
        residual_block = list(block_dict["block"])
        residual_blocks.append(residual_block)
        return {"residual_cluster": residual_block}, None

    clusterer.predict_from_arrow_paths = cast(Any, fake_predict_from_arrow_paths)
    dataset = cast(
        ANDData,
        SimpleNamespace(
            cluster_seeds_require={"seed": "7"},
            cluster_seeds_disallow=set(),
            max_seed_cluster_id=7,
            signatures={
                "u_a1": SimpleNamespace(
                    author_info_first_normalized_without_apostrophe="alice",
                    author_info_first="Alice",
                    author_info_orcid=None,
                ),
                "u_b1": SimpleNamespace(
                    author_info_first_normalized_without_apostrophe="bob",
                    author_info_first="Bob",
                    author_info_orcid=None,
                ),
                "u_a2": SimpleNamespace(
                    author_info_first_normalized_without_apostrophe="alan",
                    author_info_first="Alan",
                    author_info_orcid=None,
                ),
                "u_b2": SimpleNamespace(
                    author_info_first_normalized_without_apostrophe="bea",
                    author_info_first="Bea",
                    author_info_orcid=None,
                ),
            },
            name_tuples=set(),
        ),
    )

    result = clusterer._finish_incremental_with_seed_links(
        ["u_a1", "u_b1", "u_a2", "u_b2"],
        dataset,
        {},
        {},
        {"7": ["seed"]},
        False,
        {},
        runtime_context=cast(Any, object()),
        arrow_paths={"signatures": "signatures.arrow"},
    )

    assert result == {"7": ["seed"], "8": ["u_a1", "u_a2"], "9": ["u_b1", "u_b2"]}
    assert residual_blocks == [["u_a1", "u_a2"], ["u_b1", "u_b2"]]
    assert clusterer._last_incremental_residual_phase_b_telemetry == {
        "residual_phase_b_signature_count": 4,
        "residual_phase_b_group_count": 2,
        "residual_phase_b_pair_count_before": 6,
        "residual_phase_b_pair_count_after": 2,
        "residual_phase_b_pair_count_saved": 4,
    }


def test_finish_incremental_with_seed_links_residual_phase_b_preserves_same_orcid_group():
    clusterer = _build_minimal_incremental_clusterer()
    residual_blocks: list[list[str]] = []

    def fake_predict_helper(block_dict, dataset, partial_supervision, runtime_context, total_ram_bytes=None):
        del dataset, partial_supervision, runtime_context, total_ram_bytes
        residual_block = list(block_dict["block"])
        residual_blocks.append(residual_block)
        return {"residual_cluster": residual_block}, None

    clusterer.predict_helper = cast(Any, fake_predict_helper)
    dataset = cast(
        ANDData,
        SimpleNamespace(
            cluster_seeds_require={"seed": "7"},
            cluster_seeds_disallow=set(),
            max_seed_cluster_id=7,
            signatures={
                "u_a": SimpleNamespace(
                    author_info_first_normalized_without_apostrophe="alice",
                    author_info_first="Alice",
                    author_info_orcid="0000-0000-0000-0001",
                ),
                "u_b": SimpleNamespace(
                    author_info_first_normalized_without_apostrophe="bob",
                    author_info_first="Bob",
                    author_info_orcid="0000-0000-0000-0001",
                ),
            },
            name_tuples=set(),
        ),
    )

    result = clusterer._finish_incremental_with_seed_links(
        ["u_a", "u_b"],
        dataset,
        {},
        {},
        {"7": ["seed"]},
        False,
        {},
        runtime_context=cast(Any, object()),
    )

    assert result == {"7": ["seed"], "8": ["u_a", "u_b"]}
    assert residual_blocks == [["u_a", "u_b"]]
    assert clusterer._last_incremental_residual_phase_b_telemetry["residual_phase_b_group_count"] == 1
    assert clusterer._last_incremental_residual_phase_b_telemetry["residual_phase_b_pair_count_saved"] == 0


def test_finish_incremental_with_seed_links_accepts_legacy_name_tuple_forms():
    def _finish_with_name_tuples(name_tuples: set[tuple[str, str]]) -> dict[str, list[str]]:
        clusterer = _build_minimal_incremental_clusterer()

        def fake_predict_helper(block_dict, dataset, partial_supervision, runtime_context, total_ram_bytes=None):
            del dataset, partial_supervision, runtime_context, total_ram_bytes
            return {"residual_cluster": list(block_dict["block"])}, None

        clusterer.predict_helper = cast(Any, fake_predict_helper)
        dataset = cast(
            ANDData,
            SimpleNamespace(
                cluster_seeds_require={"seed": "7"},
                max_seed_cluster_id=7,
                signatures={
                    "seed": SimpleNamespace(
                        author_info_first="Qi-Xin",
                        author_info_first_normalized_without_apostrophe="qi xin",
                        author_info_last="Ou Yang",
                        paper_id=1,
                    ),
                    "candidate": SimpleNamespace(
                        author_info_first="Qadir",
                        author_info_first_normalized_without_apostrophe="qadir",
                        author_info_last="Ou Yang",
                        paper_id=2,
                    ),
                },
                name_tuples=name_tuples,
            ),
        )
        return clusterer._finish_incremental_with_seed_links(
            ["candidate"],
            dataset,
            {"candidate": "7_0"},
            {"7_0": "7"},
            {"7": ["seed"]},
            True,
            {},
            runtime_context=cast(Any, object()),
        )

    exact_name_tuples = {("qi xin", "qadir")}
    joined_name_tuples = {("qixin", "qadir")}
    first_token_name_tuples = {("qi", "qadir")}

    exact_alias = _finish_with_name_tuples(exact_name_tuples)
    joined_legacy_alias = _finish_with_name_tuples(joined_name_tuples)
    first_token_legacy_alias = _finish_with_name_tuples(first_token_name_tuples)

    assert exact_alias == {"7": ["seed", "candidate"]}
    assert ("qi xin", "qadir") not in joined_name_tuples
    assert joined_legacy_alias == {"7": ["seed", "candidate"]}
    assert ("qi xin", "qadir") not in first_token_name_tuples
    assert first_token_legacy_alias == {"7": ["seed", "candidate"]}


def test_build_incremental_seed_setup_passes_total_ram_to_altered_profile_reclustering():
    clusterer = _build_minimal_incremental_clusterer()
    recluster_blocks: list[list[str]] = []
    recluster_total_ram_bytes: list[int | None] = []
    recluster_partial_supervision: list[dict[tuple[str, str], int | float]] = []

    def fake_predict_helper(
        block_dict,
        dataset,
        *,
        incremental_dont_use_cluster_seeds,
        partial_supervision,
        runtime_context,
        total_ram_bytes=None,
    ):
        del dataset, runtime_context
        assert incremental_dont_use_cluster_seeds is True
        recluster_blocks.append(list(block_dict["block"]))
        recluster_total_ram_bytes.append(total_ram_bytes)
        recluster_partial_supervision.append(dict(partial_supervision))
        return {"split0": ["seed0"], "split1": ["seed1"]}, None

    clusterer.predict_helper = cast(Any, fake_predict_helper)
    dataset = cast(
        ANDData,
        type(
            "IncrementalDataset",
            (),
            {
                "cluster_seeds_require": {"seed0": "7", "seed1": "7", "seed2": "8"},
                "cluster_seeds_disallow": {("seed0", "seed1")},
                "altered_cluster_signatures": ["seed0"],
            },
        )(),
    )

    cluster_seeds_require, recluster_map, cluster_seeds_require_inverse, _split_inverse = (
        clusterer._build_incremental_seed_setup(
            dataset,
            {},
            runtime_context=cast(Any, object()),
            total_ram_bytes=123_456,
        )
    )

    assert recluster_blocks == [["seed0", "seed1"]]
    assert recluster_total_ram_bytes == [123_456]
    assert recluster_partial_supervision == [{("seed0", "seed1"): LARGE_DISTANCE}]
    assert cluster_seeds_require == {"seed0": "7_0", "seed1": "7_1", "seed2": "8"}
    assert recluster_map == {"7_0": "7", "7_1": "7"}
    assert cluster_seeds_require_inverse == {"7": ["seed0", "seed1"], "8": ["seed2"]}


def test_build_incremental_seed_setup_uses_arrow_paths_for_altered_profile_reclustering():
    clusterer = _build_minimal_incremental_clusterer()
    captured: dict[str, Any] = {}

    def fail_predict_helper(*_args, **_kwargs):
        raise AssertionError("Arrow altered-profile pre-splitting should not call legacy predict_helper")

    def fake_predict_from_arrow_paths(block_dict, arrow_paths, **kwargs):
        captured["block_dict"] = dict(block_dict)
        captured["arrow_paths"] = dict(arrow_paths)
        captured["partial_supervision"] = dict(kwargs["partial_supervision"])
        captured["incremental_dont_use_cluster_seeds"] = kwargs["incremental_dont_use_cluster_seeds"]
        captured["runtime_context"] = kwargs["runtime_context"]
        captured["total_ram_bytes"] = kwargs["total_ram_bytes"]
        return {
            "altered_profile_0_0": ["seed0"],
            "altered_profile_0_1": ["seed1"],
            "altered_profile_1_0": ["seed2"],
            "altered_profile_1_1": ["seed3"],
        }, None

    clusterer.predict_helper = cast(Any, fail_predict_helper)
    clusterer.predict_from_arrow_paths = cast(Any, fake_predict_from_arrow_paths)
    dataset = cast(
        ANDData,
        SimpleNamespace(
            cluster_seeds_require={"seed0": "7", "seed1": "7", "seed2": "8", "seed3": "8", "seed4": "9"},
            cluster_seeds_disallow={("seed0", "seed1"), ("seed2", "seed3"), ("seed3", "seed4")},
            altered_cluster_signatures=["seed0", "seed2", "seed4"],
            name_tuples="filtered",
        ),
    )
    arrow_paths = {
        "signatures": "signatures.arrow",
        "papers": "papers.arrow",
        "paper_authors": "paper_authors.arrow",
        "cluster_seeds": "cluster_seeds.arrow",
    }
    runtime_context = cast(Any, object())

    cluster_seeds_require, recluster_map, cluster_seeds_require_inverse, _split_inverse = (
        clusterer._build_incremental_seed_setup(
            dataset,
            {},
            runtime_context=runtime_context,
            total_ram_bytes=123_456,
            arrow_paths=arrow_paths,
        )
    )

    assert captured["block_dict"] == {
        "altered_profile_0": ["seed0", "seed1"],
        "altered_profile_1": ["seed2", "seed3"],
    }
    assert captured["arrow_paths"] == {
        "signatures": "signatures.arrow",
        "papers": "papers.arrow",
        "paper_authors": "paper_authors.arrow",
    }
    assert captured["partial_supervision"] == {
        ("seed0", "seed1"): LARGE_DISTANCE,
        ("seed2", "seed3"): LARGE_DISTANCE,
    }
    assert captured["incremental_dont_use_cluster_seeds"] is True
    assert captured["runtime_context"] is runtime_context
    assert captured["total_ram_bytes"] == 123_456
    assert cluster_seeds_require == {
        "seed0": "7_0",
        "seed1": "7_1",
        "seed2": "8_0",
        "seed3": "8_1",
        "seed4": "9",
    }
    assert recluster_map == {"7_0": "7", "7_1": "7", "8_0": "8", "8_1": "8"}
    assert cluster_seeds_require_inverse == {"7": ["seed0", "seed1"], "8": ["seed2", "seed3"], "9": ["seed4"]}


def test_predict_subblocked_restores_seed_state_when_presplit_setup_raises(monkeypatch):
    clusterer = _build_minimal_incremental_clusterer()
    dataset = cast(
        ANDData,
        SimpleNamespace(
            cluster_seeds_require={"seed0": "7"},
            cluster_seeds_disallow=set(),
            altered_cluster_signatures=["seed0"],
            name_tuples="filtered",
        ),
    )

    def fake_seed_setup(*_args, **_kwargs):
        return {"seed0": "7_0"}, {}, {"7": ["seed0"]}, {"7_0": ["seed0"]}

    def fail_temporary_arrow_paths(current_dataset, _arrow_paths):
        assert current_dataset.cluster_seeds_require == {"seed0": "7_0"}
        raise RuntimeError("temporary arrow path setup failed")

    monkeypatch.setattr(clusterer, "_build_incremental_seed_setup", fake_seed_setup)
    monkeypatch.setattr(model_module, "_temporary_arrow_paths_with_current_cluster_seeds", fail_temporary_arrow_paths)

    with pytest.raises(RuntimeError, match="temporary arrow path setup failed"):
        clusterer._predict_subblocked(
            {"block": ["seed0"]},
            dataset,
            cluster_model_params=None,
            partial_supervision={},
            use_s2_clusters=False,
            incremental_dont_use_cluster_seeds=False,
            batching_threshold=2,
            desired_memory_use=None,
            runtime_context=cast(Any, object()),
            dists=None,
            total_ram_bytes=None,
            restore_rust_cluster_seeds_on_exit=False,
            arrow_paths={
                "signatures": "signatures.arrow",
                "papers": "papers.arrow",
                "paper_authors": "paper_authors.arrow",
            },
        )

    assert dataset.cluster_seeds_require == {"seed0": "7"}


def test_predict_subblocked_arrow_forwards_disallows_to_multiple_letter_rust_path(monkeypatch):
    clusterer = _build_minimal_incremental_clusterer()
    clusterer.subblocking_fallback_mode = "legacy"
    dataset = cast(
        ANDData,
        SimpleNamespace(
            cluster_seeds_require={},
            cluster_seeds_disallow={("s1", "s2")},
            altered_cluster_signatures=[],
            name_tuples="filtered",
            signatures={
                "s1": SimpleNamespace(
                    author_info_first="Alice",
                    author_info_first_normalized_without_apostrophe="Alice",
                ),
                "s2": SimpleNamespace(
                    author_info_first="Alicia",
                    author_info_first_normalized_without_apostrophe="Alicia",
                ),
            },
        ),
    )
    captured: dict[str, Any] = {}

    monkeypatch.setattr(model_module, "build_rust_featurizer_from_arrow_paths", lambda *_args, **_kwargs: object())

    def fake_predict_from_rust_featurizer(_block_dict, _rust_featurizer, **kwargs):
        captured["partial_supervision"] = dict(kwargs["partial_supervision"])
        captured["cluster_seeds_disallow"] = set(kwargs["cluster_seeds_disallow"])
        return {"block_0": ["s1"], "block_1": ["s2"]}, None

    clusterer.predict_from_rust_featurizer = cast(Any, fake_predict_from_rust_featurizer)

    result, dists = clusterer._predict_subblocked(
        {"block": ["s1", "s2"]},
        dataset,
        cluster_model_params=None,
        partial_supervision={},
        use_s2_clusters=False,
        incremental_dont_use_cluster_seeds=False,
        batching_threshold=10,
        desired_memory_use=None,
        runtime_context=cast(Any, SimpleNamespace(use_rust=False)),
        dists=None,
        total_ram_bytes=None,
        restore_rust_cluster_seeds_on_exit=False,
        arrow_paths={
            "signatures": "signatures.arrow",
            "papers": "papers.arrow",
            "paper_authors": "paper_authors.arrow",
        },
    )

    assert result == {"block_0": ["s1"], "block_1": ["s2"]}
    assert dists is None
    assert captured["partial_supervision"] == {("s1", "s2"): LARGE_DISTANCE}
    assert captured["cluster_seeds_disallow"] == {("s1", "s2")}


def test_build_incremental_seed_setup_caches_arrow_altered_profile_reclustering():
    clusterer = _build_minimal_incremental_clusterer()
    call_count = 0

    def fake_predict_from_arrow_paths(block_dict, arrow_paths, **kwargs):
        nonlocal call_count
        del arrow_paths, kwargs
        call_count += 1
        return {f"{block_key}_0": [signatures[0]] for block_key, signatures in block_dict.items()} | {
            f"{block_key}_1": list(signatures[1:]) for block_key, signatures in block_dict.items()
        }, None

    clusterer.predict_from_arrow_paths = cast(Any, fake_predict_from_arrow_paths)
    dataset = cast(
        ANDData,
        SimpleNamespace(
            cluster_seeds_require={"seed0": "7", "seed1": "7", "seed2": "8", "seed3": "8"},
            cluster_seeds_disallow=set(),
            altered_cluster_signatures=["seed0", "seed2"],
            name_tuples="filtered",
        ),
    )
    arrow_paths = {
        "signatures": "signatures.arrow",
        "papers": "papers.arrow",
        "paper_authors": "paper_authors.arrow",
        "cluster_seeds": "cluster_seeds.arrow",
    }

    first_require, first_recluster_map, _cluster_seeds_inverse, _split_inverse = (
        clusterer._build_incremental_seed_setup(
            dataset,
            {},
            runtime_context=cast(Any, object()),
            arrow_paths=arrow_paths,
        )
    )
    first_telemetry = dict(clusterer._last_incremental_seed_setup_telemetry)
    second_require, second_recluster_map, _cluster_seeds_inverse, _split_inverse = (
        clusterer._build_incremental_seed_setup(
            dataset,
            {},
            runtime_context=cast(Any, object()),
            arrow_paths=arrow_paths,
        )
    )
    second_telemetry = dict(clusterer._last_incremental_seed_setup_telemetry)

    assert call_count == 1
    assert second_require == first_require
    assert second_recluster_map == first_recluster_map
    assert first_telemetry["seed_setup_altered_presplit_cache_hit_count"] == 0
    assert first_telemetry["seed_setup_altered_presplit_cache_miss_count"] == 2
    assert second_telemetry["seed_setup_altered_presplit_cache_hit_count"] == 2
    assert second_telemetry["seed_setup_altered_presplit_cache_miss_count"] == 0
    assert second_telemetry["seed_setup_altered_presplit_predict_seconds"] == 0.0


def test_build_incremental_seed_setup_cache_includes_partial_supervision():
    clusterer = _build_minimal_incremental_clusterer()
    call_count = 0

    def fake_predict_from_arrow_paths(block_dict, arrow_paths, **kwargs):
        nonlocal call_count
        del block_dict, arrow_paths, kwargs
        call_count += 1
        return {"split0": ["seed0"], "split1": ["seed1"]}, None

    clusterer.predict_from_arrow_paths = cast(Any, fake_predict_from_arrow_paths)
    dataset = cast(
        ANDData,
        SimpleNamespace(
            cluster_seeds_require={"seed0": "7", "seed1": "7"},
            cluster_seeds_disallow=set(),
            altered_cluster_signatures=["seed0"],
            name_tuples="filtered",
        ),
    )
    arrow_paths = {
        "signatures": "signatures.arrow",
        "papers": "papers.arrow",
        "paper_authors": "paper_authors.arrow",
    }

    clusterer._build_incremental_seed_setup(
        dataset,
        {},
        runtime_context=cast(Any, object()),
        arrow_paths=arrow_paths,
    )
    clusterer._build_incremental_seed_setup(
        dataset,
        {("seed0", "seed1"): LARGE_DISTANCE},
        runtime_context=cast(Any, object()),
        arrow_paths=arrow_paths,
    )

    assert call_count == 2
    assert clusterer._last_incremental_seed_setup_telemetry["seed_setup_altered_presplit_cache_miss_count"] == 1


def test_build_incremental_seed_setup_skips_same_orcid_altered_profile_reclustering():
    clusterer = _build_minimal_incremental_clusterer()
    clusterer.predict_from_arrow_paths = cast(
        Any,
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("same-ORCID altered profile should not need exact reclustering")
        ),
    )
    dataset = cast(
        ANDData,
        SimpleNamespace(
            cluster_seeds_require={"seed0": "7", "seed1": "7", "seed2": "7"},
            cluster_seeds_disallow=set(),
            altered_cluster_signatures=["seed0"],
            signatures={
                "seed0": SimpleNamespace(author_info_orcid="https://orcid.org/0000-0002-1825-009X"),
                "seed1": SimpleNamespace(author_info_orcid="0000-0002-1825-009x"),
                "seed2": SimpleNamespace(author_info_orcid="ORCID: 000000021825009X"),
            },
            name_tuples="filtered",
        ),
    )

    cluster_seeds_require, recluster_map, _cluster_seeds_require_inverse, _split_inverse = (
        clusterer._build_incremental_seed_setup(
            dataset,
            {},
            runtime_context=cast(Any, object()),
            arrow_paths={
                "signatures": "signatures.arrow",
                "papers": "papers.arrow",
                "paper_authors": "paper_authors.arrow",
            },
        )
    )

    assert cluster_seeds_require == {"seed0": "7", "seed1": "7", "seed2": "7"}
    assert recluster_map == {}
    assert clusterer._last_incremental_seed_setup_telemetry["seed_setup_altered_presplit_orcid_skip_count"] == 1
    assert clusterer._last_incremental_seed_setup_telemetry["seed_setup_altered_presplit_cache_miss_count"] == 0


def test_build_incremental_seed_setup_same_orcid_skip_respects_explicit_disallow():
    clusterer = _build_minimal_incremental_clusterer()
    call_count = 0

    def fake_predict_from_arrow_paths(block_dict, arrow_paths, **kwargs):
        nonlocal call_count
        del block_dict, arrow_paths, kwargs
        call_count += 1
        return {"split0": ["seed0"], "split1": ["seed1"]}, None

    clusterer.predict_from_arrow_paths = cast(Any, fake_predict_from_arrow_paths)
    dataset = cast(
        ANDData,
        SimpleNamespace(
            cluster_seeds_require={"seed0": "7", "seed1": "7"},
            cluster_seeds_disallow={("seed0", "seed1")},
            altered_cluster_signatures=["seed0"],
            signatures={
                "seed0": SimpleNamespace(author_info_orcid="0000-0002-1825-009X"),
                "seed1": SimpleNamespace(author_info_orcid="ORCID: 000000021825009X"),
            },
            name_tuples="filtered",
        ),
    )

    clusterer._build_incremental_seed_setup(
        dataset,
        {},
        runtime_context=cast(Any, object()),
        arrow_paths={
            "signatures": "signatures.arrow",
            "papers": "papers.arrow",
            "paper_authors": "paper_authors.arrow",
        },
    )

    assert call_count == 1
    assert clusterer._last_incremental_seed_setup_telemetry["seed_setup_altered_presplit_orcid_skip_count"] == 0


def test_build_incremental_seed_setup_same_orcid_skip_rejects_invalid_orcid_pair():
    clusterer = _build_minimal_incremental_clusterer()
    call_count = 0

    def fake_predict_from_arrow_paths(block_dict, arrow_paths, **kwargs):
        nonlocal call_count
        del block_dict, arrow_paths, kwargs
        call_count += 1
        return {"split0": ["seed0"], "split1": ["seed1"]}, None

    clusterer.predict_from_arrow_paths = cast(Any, fake_predict_from_arrow_paths)
    dataset = cast(
        ANDData,
        SimpleNamespace(
            cluster_seeds_require={"seed0": "7", "seed1": "7"},
            cluster_seeds_disallow=set(),
            altered_cluster_signatures=["seed0"],
            signatures={
                "seed0": SimpleNamespace(author_info_orcid="not-an-orcid"),
                "seed1": SimpleNamespace(author_info_orcid="not-an-orcid"),
            },
            name_tuples="filtered",
        ),
    )

    clusterer._build_incremental_seed_setup(
        dataset,
        {},
        runtime_context=cast(Any, object()),
        arrow_paths={
            "signatures": "signatures.arrow",
            "papers": "papers.arrow",
            "paper_authors": "paper_authors.arrow",
        },
    )

    assert call_count == 1
    assert clusterer._last_incremental_seed_setup_telemetry["seed_setup_altered_presplit_orcid_skip_count"] == 0


def test_build_incremental_seed_setup_loads_altered_signatures_from_arrow_path(tmp_path: Path):
    import pyarrow as pa

    clusterer = _build_minimal_incremental_clusterer()
    captured: dict[str, Any] = {}

    def fake_predict_from_arrow_paths(block_dict, arrow_paths, **kwargs):
        del arrow_paths
        captured["block_dict"] = dict(block_dict)
        captured["partial_supervision"] = dict(kwargs.get("partial_supervision", {}))
        return {"split0": ["seed0"], "split1": ["seed1"]}, None

    clusterer.predict_from_arrow_paths = cast(Any, fake_predict_from_arrow_paths)
    altered_path = tmp_path / "altered_cluster_signatures.arrow"
    table = pa.table({"signature_id": pa.array(["seed0"], type=pa.string())})
    with pa.OSFile(str(altered_path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    dataset = cast(
        ANDData,
        SimpleNamespace(
            cluster_seeds_require={"seed0": "7", "seed1": "7", "seed2": "8"},
            cluster_seeds_disallow=set(),
            altered_cluster_signatures=None,
            name_tuples="filtered",
        ),
    )

    cluster_seeds_require, recluster_map, _cluster_seeds_require_inverse, _split_inverse = (
        clusterer._build_incremental_seed_setup(
            dataset,
            {},
            runtime_context=cast(Any, object()),
            arrow_paths={"altered_cluster_signatures": str(altered_path)},
        )
    )

    assert captured["block_dict"] == {"altered_profile_0": ["seed0", "seed1"]}
    assert cluster_seeds_require == {"seed0": "7_0", "seed1": "7_1", "seed2": "8"}
    assert recluster_map == {"7_0": "7", "7_1": "7"}
    assert clusterer._last_incremental_seed_setup_telemetry["seed_setup_altered_signature_count"] == 1


def test_build_incremental_seed_setup_loads_seed_and_altered_signatures_from_arrow_paths(tmp_path: Path):
    import pyarrow as pa

    clusterer = _build_minimal_incremental_clusterer()
    captured: dict[str, Any] = {}

    def fake_predict_from_arrow_paths(block_dict, arrow_paths, **kwargs):
        del arrow_paths
        captured["block_dict"] = dict(block_dict)
        captured["partial_supervision"] = dict(kwargs.get("partial_supervision", {}))
        return {"split0": ["seed0"], "split1": ["seed1"]}, None

    clusterer.predict_from_arrow_paths = cast(Any, fake_predict_from_arrow_paths)
    cluster_seeds_path = tmp_path / "cluster_seeds.arrow"
    seed_table = pa.table(
        {
            "signature_id": pa.array(["seed0", "seed1", "seed2"], type=pa.string()),
            "cluster_id": pa.array(["7", "7", "8"], type=pa.string()),
        }
    )
    with pa.OSFile(str(cluster_seeds_path), "wb") as sink:
        with pa.ipc.new_file(sink, seed_table.schema) as writer:
            writer.write_table(seed_table)
    altered_path = tmp_path / "altered_cluster_signatures.arrow"
    altered_table = pa.table({"signature_id": pa.array(["seed0"], type=pa.string())})
    with pa.OSFile(str(altered_path), "wb") as sink:
        with pa.ipc.new_file(sink, altered_table.schema) as writer:
            writer.write_table(altered_table)
    disallow_path = tmp_path / "cluster_seed_disallows.arrow"
    disallow_table = pa.table(
        {
            "signature_id_1": pa.array(["seed0"], type=pa.string()),
            "signature_id_2": pa.array(["seed1"], type=pa.string()),
        }
    )
    with pa.OSFile(str(disallow_path), "wb") as sink:
        with pa.ipc.new_file(sink, disallow_table.schema) as writer:
            writer.write_table(disallow_table)
    dataset = cast(
        ANDData,
        SimpleNamespace(
            cluster_seeds_require={},
            cluster_seeds_disallow=set(),
            altered_cluster_signatures=None,
            name_tuples="filtered",
        ),
    )

    cluster_seeds_require, recluster_map, cluster_seeds_require_inverse, _split_inverse = (
        clusterer._build_incremental_seed_setup(
            dataset,
            {},
            runtime_context=cast(Any, object()),
            arrow_paths={
                "cluster_seeds": str(cluster_seeds_path),
                "cluster_seed_disallows": str(disallow_path),
                "altered_cluster_signatures": str(altered_path),
            },
        )
    )

    assert captured["block_dict"] == {"altered_profile_0": ["seed0", "seed1"]}
    assert captured["partial_supervision"] == {("seed0", "seed1"): LARGE_DISTANCE}
    assert cluster_seeds_require == {"seed0": "7_0", "seed1": "7_1", "seed2": "8"}
    assert recluster_map == {"7_0": "7", "7_1": "7"}
    assert cluster_seeds_require_inverse == {"7": ["seed0", "seed1"], "8": ["seed2"]}


def test_cluster_seeds_arrow_read_cache_reuses_parse_and_returns_copy(monkeypatch, tmp_path: Path):
    import pyarrow as pa

    model_module._CLUSTER_SEEDS_ARROW_CACHE.clear()
    cluster_seeds_path = tmp_path / "cluster_seeds.arrow"
    write_cluster_seeds_arrow(cluster_seeds_path, {"seed0": "7", "seed1": "7"})

    open_file_call_count = 0
    original_open_file = pa.ipc.open_file

    def counting_open_file(*args, **kwargs):
        nonlocal open_file_call_count
        open_file_call_count += 1
        return original_open_file(*args, **kwargs)

    monkeypatch.setattr(pa.ipc, "open_file", counting_open_file)

    def fail_read_bytes(self: Path):
        raise AssertionError(f"cache fingerprint should not read Arrow bytes: {self}")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    first = model_module._cluster_seeds_require_from_arrow_paths({"cluster_seeds": str(cluster_seeds_path)})
    first["seed0"] = "mutated"
    second = model_module._cluster_seeds_require_from_arrow_paths({"cluster_seeds": str(cluster_seeds_path)})

    assert second == {"seed0": "7", "seed1": "7"}
    assert open_file_call_count == 1


def test_cluster_seeds_arrow_rejects_missing_explicit_path(tmp_path: Path):
    missing_path = tmp_path / "missing_cluster_seeds.arrow"

    with pytest.raises(FileNotFoundError, match="cluster_seeds"):
        model_module._cluster_seeds_require_from_arrow_paths({"cluster_seeds": str(missing_path)})


def test_arrow_paths_need_current_cluster_seeds_rewrites_missing_seed_sidecar(tmp_path: Path):
    dataset = SimpleNamespace(cluster_seeds_require={"seed0": "7"})

    assert model_module._arrow_paths_need_current_cluster_seeds(dataset, {}) is True
    assert (
        model_module._arrow_paths_need_current_cluster_seeds(
            dataset,
            {"cluster_seeds": str(tmp_path / "missing_cluster_seeds.arrow")},
        )
        is True
    )


def test_cluster_seeds_arrow_rejects_duplicate_and_empty_rows(tmp_path: Path):
    import pyarrow as pa

    cluster_seeds_path = tmp_path / "cluster_seeds.arrow"
    duplicate_table = pa.table(
        {
            "signature_id": pa.array(["seed0", "seed0"], type=pa.string()),
            "cluster_id": pa.array(["7", "8"], type=pa.string()),
        }
    )
    with pa.OSFile(str(cluster_seeds_path), "wb") as sink:
        with pa.ipc.new_file(sink, duplicate_table.schema) as writer:
            writer.write_table(duplicate_table)
    with pytest.raises(ValueError, match="duplicate signature_id"):
        model_module._cluster_seeds_require_from_arrow_paths({"cluster_seeds": str(cluster_seeds_path)})

    empty_cluster_table = pa.table(
        {
            "signature_id": pa.array(["seed0"], type=pa.string()),
            "cluster_id": pa.array([""], type=pa.string()),
        }
    )
    with pa.OSFile(str(cluster_seeds_path), "wb") as sink:
        with pa.ipc.new_file(sink, empty_cluster_table.schema) as writer:
            writer.write_table(empty_cluster_table)
    with pytest.raises(ValueError, match="empty cluster_id"):
        model_module._cluster_seeds_require_from_arrow_paths({"cluster_seeds": str(cluster_seeds_path)})


def test_cluster_seed_disallows_arrow_deduplicates_bidirectional_pairs(tmp_path: Path):
    import pyarrow as pa

    disallow_path = tmp_path / "cluster_seed_disallows.arrow"
    disallow_table = pa.table(
        {
            "signature_id_1": pa.array(["seed0", "seed1"], type=pa.string()),
            "signature_id_2": pa.array(["seed1", "seed0"], type=pa.string()),
        }
    )
    with pa.OSFile(str(disallow_path), "wb") as sink:
        with pa.ipc.new_file(sink, disallow_table.schema) as writer:
            writer.write_table(disallow_table)

    assert model_module._read_cluster_seed_disallows_arrow(disallow_path) == {("seed0", "seed1")}


def test_partial_supervision_disallow_merge_respects_reverse_existing_pair():
    dataset = SimpleNamespace(cluster_seeds_disallow={("q", "s1")})

    merged = model_module._partial_supervision_with_cluster_seed_disallows(
        ["q", "s1"],
        dataset,
        {("s1", "q"): 42.0},
        cluster_seed_disallows={("q", "s1")},
    )

    assert merged == {("s1", "q"): 42.0}


def test_build_incremental_seed_setup_empty_altered_list_overrides_arrow_path(tmp_path: Path):
    import pyarrow as pa

    clusterer = _build_minimal_incremental_clusterer()
    clusterer.predict_from_arrow_paths = cast(
        Any,
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("empty Python altered list is authoritative")),
    )
    altered_path = tmp_path / "altered_cluster_signatures.arrow"
    table = pa.table({"signature_id": pa.array(["seed0"], type=pa.string())})
    with pa.OSFile(str(altered_path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    dataset = cast(
        ANDData,
        SimpleNamespace(
            cluster_seeds_require={"seed0": "7", "seed1": "7", "seed2": "8"},
            cluster_seeds_disallow=set(),
            altered_cluster_signatures=[],
            name_tuples="filtered",
        ),
    )

    cluster_seeds_require, recluster_map, _cluster_seeds_require_inverse, _split_inverse = (
        clusterer._build_incremental_seed_setup(
            dataset,
            {},
            runtime_context=cast(Any, object()),
            arrow_paths={"altered_cluster_signatures": str(altered_path)},
        )
    )

    assert cluster_seeds_require == {"seed0": "7", "seed1": "7", "seed2": "8"}
    assert recluster_map == {}
    assert clusterer._last_incremental_seed_setup_telemetry["seed_setup_altered_signature_count"] == 0


def test_build_incremental_seed_setup_rejects_arrow_altered_signature_missing_seed(tmp_path: Path):
    import pyarrow as pa

    clusterer = _build_minimal_incremental_clusterer()
    altered_path = tmp_path / "altered_cluster_signatures.arrow"
    altered_table = pa.table({"signature_id": pa.array(["missing_seed"], type=pa.string())})
    with pa.OSFile(str(altered_path), "wb") as sink:
        with pa.ipc.new_file(sink, altered_table.schema) as writer:
            writer.write_table(altered_table)
    dataset = cast(
        ANDData,
        SimpleNamespace(
            cluster_seeds_require={"seed0": "7"},
            cluster_seeds_disallow=set(),
            altered_cluster_signatures=None,
            name_tuples="filtered",
        ),
    )

    with pytest.raises(ValueError, match="must all be present in cluster_seeds_require"):
        clusterer._build_incremental_seed_setup(
            dataset,
            {},
            runtime_context=cast(Any, object()),
            arrow_paths={"altered_cluster_signatures": str(altered_path)},
        )


def test_read_altered_cluster_signatures_arrow_rejects_null_and_duplicates(tmp_path: Path):
    import pyarrow as pa

    null_path = tmp_path / "altered_null.arrow"
    null_table = pa.table({"signature_id": pa.array(["seed0", None], type=pa.string())})
    with pa.OSFile(str(null_path), "wb") as sink:
        with pa.ipc.new_file(sink, null_table.schema) as writer:
            writer.write_table(null_table)

    with pytest.raises(ValueError, match="null or empty"):
        model_module._read_altered_cluster_signatures_arrow(null_path)

    duplicate_path = tmp_path / "altered_duplicate.arrow"
    duplicate_table = pa.table({"signature_id": pa.array(["seed0", "seed0"], type=pa.string())})
    with pa.OSFile(str(duplicate_path), "wb") as sink:
        with pa.ipc.new_file(sink, duplicate_table.schema) as writer:
            writer.write_table(duplicate_table)

    with pytest.raises(ValueError, match="duplicate"):
        model_module._read_altered_cluster_signatures_arrow(duplicate_path)


def test_top1_consensus_broadcast_only_applies_when_cluster_members_agree():
    def _run(
        mode: Literal["always", "never", "top1_consensus"],
        signature_dists: dict[str, dict[int, tuple[float, int, float]]],
    ) -> dict[str, list[str]]:
        clusterer = _build_minimal_incremental_clusterer()
        clusterer.incremental_precluster_broadcast_mode = mode

        def fake_predict_helper(block_dict, dataset, partial_supervision, runtime_context, total_ram_bytes=None):
            del dataset, partial_supervision, runtime_context, total_ram_bytes
            if "incremental_unassigned" in block_dict:
                return {"incremental_cluster": list(block_dict["incremental_unassigned"])}, None
            if "block" in block_dict:
                return {"singleton_cluster": list(block_dict["block"])}, None
            raise AssertionError(f"Unexpected block_dict={block_dict}")

        clusterer.predict_helper = cast(Any, fake_predict_helper)
        dataset = cast(
            ANDData,
            type(
                "IncrementalDataset",
                (),
                {
                    "cluster_seeds_require": {"seed0": 0, "seed1": 1},
                    "max_seed_cluster_id": 2,
                    "signatures": {},
                    "name_tuples": set(),
                },
            )(),
        )
        signature_to_cluster_to_average_dist = cast(
            dict[str, dict[int | str, IncrementalDistStats]],
            {signature_id: cluster_dists.copy() for signature_id, cluster_dists in signature_dists.items()},
        )
        return clusterer._run_incremental_phases_bcd(
            ["u1", "u2"],
            dataset,
            signature_to_cluster_to_average_dist,
            dict(dataset.cluster_seeds_require),
            {},
            {0: ["seed0"], 1: ["seed1"]},
            False,
            {},
            runtime_context=cast(Any, object()),
        )

    divergent_top1_dists = {
        "u1": {0: (0.10, 1, 0.10), 1: (0.60, 1, 0.60)},
        "u2": {0: (0.60, 1, 0.60), 1: (0.20, 1, 0.20)},
    }
    always_divergent = _run("always", divergent_top1_dists)
    never_divergent = _run("never", divergent_top1_dists)
    consensus_divergent = _run("top1_consensus", divergent_top1_dists)
    assert always_divergent == {"0": ["seed0", "u1", "u2"], "1": ["seed1"]}
    assert never_divergent == {"0": ["seed0", "u1"], "1": ["seed1", "u2"]}
    assert consensus_divergent == never_divergent

    consensus_top1_dists = {
        "u1": {0: (0.10, 1, 0.10), 1: (0.60, 1, 0.60)},
        "u2": {0: (0.70, 1, 0.70), 1: (0.80, 1, 0.80)},
    }
    never_consensus = _run("never", consensus_top1_dists)
    consensus_enabled = _run("top1_consensus", consensus_top1_dists)
    assert never_consensus == {"0": ["seed0", "u1"], "1": ["seed1"], "2": ["u2"]}
    assert consensus_enabled == {"0": ["seed0", "u1", "u2"], "1": ["seed1"]}


def test_precluster_broadcast_preserves_min_score_semantics():
    def _run(
        *,
        seed_score_mode: Literal["min", "mean_min_hybrid"],
        mean_min_hybrid_weight: float = 0.5,
    ) -> dict[str, list[str]]:
        clusterer = _build_minimal_incremental_clusterer()
        clusterer.incremental_precluster_broadcast_mode = "always"
        clusterer.incremental_seed_score_mode = seed_score_mode
        clusterer.incremental_mean_min_hybrid_weight = mean_min_hybrid_weight

        def fake_predict_helper(block_dict, dataset, partial_supervision, runtime_context, total_ram_bytes=None):
            del dataset, partial_supervision, runtime_context, total_ram_bytes
            if "incremental_unassigned" in block_dict:
                return {"incremental_cluster": list(block_dict["incremental_unassigned"])}, None
            if "block" in block_dict:
                return {"singleton_cluster": list(block_dict["block"])}, None
            raise AssertionError(f"Unexpected block_dict={block_dict}")

        clusterer.predict_helper = cast(Any, fake_predict_helper)
        dataset = cast(
            ANDData,
            type(
                "IncrementalDataset",
                (),
                {
                    "cluster_seeds_require": {"seed0": 0, "seed1": 1},
                    "max_seed_cluster_id": 2,
                    "signatures": {},
                    "name_tuples": set(),
                },
            )(),
        )
        signature_to_cluster_to_average_dist = cast(
            dict[str, dict[int | str, IncrementalDistStats]],
            {
                "u1": {0: (0.40, 1, 0.01), 1: (0.20, 1, 0.20)},
                "u2": {0: (0.40, 1, 0.80), 1: (0.20, 1, 0.20)},
            },
        )
        return clusterer._run_incremental_phases_bcd(
            ["u1", "u2"],
            dataset,
            signature_to_cluster_to_average_dist,
            dict(dataset.cluster_seeds_require),
            {},
            {0: ["seed0"], 1: ["seed1"]},
            False,
            {},
            runtime_context=cast(Any, object()),
        )

    min_result = _run(seed_score_mode="min")
    assert min_result == {"0": ["seed0", "u1", "u2"], "1": ["seed1"]}

    hybrid_result = _run(seed_score_mode="mean_min_hybrid", mean_min_hybrid_weight=0.75)
    assert hybrid_result == {"0": ["seed0", "u1", "u2"], "1": ["seed1"]}
