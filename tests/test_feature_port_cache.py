import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import s2and.feature_port as feature_port
import s2and.runtime as runtime
from s2and.arrow_inputs import MissingArrowArtifactError
from s2and.data import ANDData, NameCounts
from s2and.incremental_linking.feature_block import write_name_counts_index
from tests.helpers import patch_tiny_name_counts_loader


def _missing_module(name: str) -> ModuleNotFoundError:
    return ModuleNotFoundError(f"No module named {name!r}", name=name)


class DummyRustFeaturizer:
    created = []
    from_json_created = []
    signature_overlay_payloads = []

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def signature_ids(self):
        return []

    def get_constraint(self, *_args, **_kwargs):
        return None

    def get_constraints_matrix_indexed(self, *_args, **_kwargs):
        return []

    def json_ingest_telemetry(self):
        return {"stage_seconds": {}, "counts": {}}

    @classmethod
    def from_dataset(cls, dataset, _require_value, _disallow_value, _num_threads=None):
        cls.created.append(dataset.name)
        return cls(dataset.name)

    @classmethod
    def from_json_paths(cls, *_args, **_kwargs):
        cls.from_json_created.append((_args, _kwargs))
        return cls("json")

    @classmethod
    def from_arrow_paths(cls, *_args, **_kwargs):
        return cls("arrow")

    def update_signature_name_counts(self, signatures):
        self.__class__.signature_overlay_payloads.append(signatures)
        return len(signatures)

    def featurize_pairs_matrix_indexed(self, *_args, **_kwargs):
        return []

    @classmethod
    def load(cls, _path):
        raise AssertionError("Disk cache path should not be used in this test")

    def update_cluster_seeds(self, _require_map, _disallow_set):
        return None


class DummyRustModule:
    __version__ = "0.51.0"
    RustFeaturizer = DummyRustFeaturizer


class DummyDataset(ANDData):
    def __init__(self, name: str, mode: str = "train"):
        self.name = name
        self.mode = mode
        self.signatures = {}
        self.papers = {}
        self.name_tuples = set()
        self.compute_reference_features = False
        self.preprocess = True
        self.n_jobs = 1
        self.cluster_seeds_require = {}
        self.cluster_seeds_disallow = set()
        self.signatures_path = None
        self.papers_path = None
        self.clusters_path = None
        self.cluster_seeds_path = None
        self.specter_embeddings_path = None
        self.name_counts_last_first_initial_semantics = "legacy_compat"


class SleepyCounter:
    """Counter that yields mid-increment to expose race windows in tests."""

    def __init__(self, value: int = 0):
        self.value = value

    def __iadd__(self, increment: int):
        current = self.value
        time.sleep(0.0005)
        self.value = current + int(increment)
        return self

    def __int__(self) -> int:
        return self.value


def _cache_size() -> int:
    return sum(len(entries) for entries in feature_port._RUST_FEATURIZER_CACHE.values())


@pytest.fixture(autouse=True)
def _reset_feature_port_state(monkeypatch):
    feature_port.clear_rust_featurizer_cache()
    DummyRustFeaturizer.created = []
    DummyRustFeaturizer.from_json_created = []
    DummyRustFeaturizer.signature_overlay_payloads = []
    monkeypatch.setattr(feature_port, "s2and_rust", DummyRustModule)
    yield
    feature_port.clear_rust_featurizer_cache()


def test_rust_featurizer_in_memory_cache_keeps_train_entries():
    """Same-process cache keeps all live dataset entries."""
    d1 = DummyDataset("d1", mode="train")
    d2 = DummyDataset("d2", mode="train")

    feature_port._get_rust_featurizer(d1)
    assert _cache_size() == 1

    feature_port._get_rust_featurizer(d2)
    assert _cache_size() == 2
    assert feature_port._RUST_FEATURIZER_CACHE.get(d1) is not None
    assert feature_port._RUST_FEATURIZER_CACHE.get(d2) is not None

    # Re-access d1 — should be a cache hit, no rebuild.
    feature_port._get_rust_featurizer(d1)
    assert DummyRustFeaturizer.created == ["d1", "d2"]


def test_rust_featurizer_available_reloads_extension_when_missing(monkeypatch):
    sentinel_module = object()
    load_calls = {"count": 0}

    def _load_stub():
        load_calls["count"] += 1
        return sentinel_module

    def _detect_stub(*, extension_module):
        return SimpleNamespace(core_runtime_available=extension_module is sentinel_module)

    monkeypatch.setattr(feature_port, "s2and_rust", None)
    monkeypatch.setattr(feature_port, "load_s2and_rust_extension", _load_stub)
    monkeypatch.setattr(feature_port, "detect_rust_runtime_capabilities", _detect_stub)

    assert feature_port.rust_featurizer_available() is True
    assert feature_port.s2and_rust is sentinel_module
    assert load_calls["count"] == 1


def test_rust_signature_preprocess_available_reloads_extension_when_missing(monkeypatch):
    sentinel_module = SimpleNamespace(signature_ngrams_batch=lambda *_args, **_kwargs: None)
    load_calls = {"count": 0}

    def _load_stub():
        load_calls["count"] += 1
        return sentinel_module

    def _detect_stub(*, extension_module):
        return SimpleNamespace(core_runtime_available=extension_module is sentinel_module)

    monkeypatch.setattr(feature_port, "s2and_rust", None)
    monkeypatch.setattr(feature_port, "load_s2and_rust_extension", _load_stub)
    monkeypatch.setattr(feature_port, "detect_rust_runtime_capabilities", _detect_stub)

    assert feature_port.rust_signature_preprocess_available() is True
    assert feature_port.s2and_rust is sentinel_module
    assert load_calls["count"] == 1


def test_rust_featurizer_in_memory_cache_keeps_inference_entries():
    d1 = DummyDataset("d1", mode="inference")
    d2 = DummyDataset("d2", mode="inference")
    d3 = DummyDataset("d3", mode="inference")

    feature_port._get_rust_featurizer(d1)
    feature_port._get_rust_featurizer(d2)
    feature_port._get_rust_featurizer(d3)

    assert _cache_size() == 3


def test_rust_featurizer_reuses_live_dataset_entry():
    dataset = DummyDataset("no_cache_dataset", mode="train")

    feature_port._get_rust_featurizer(dataset)
    feature_port._get_rust_featurizer(dataset)

    assert DummyRustFeaturizer.created == ["no_cache_dataset"]
    assert _cache_size() == 1


def test_warm_rust_featurizer_populates_in_memory_cache():
    dataset = DummyDataset("warm_dataset", mode="train")

    feature_port.warm_rust_featurizer(dataset)
    feature_port._get_rust_featurizer(dataset)

    assert DummyRustFeaturizer.created == ["warm_dataset"]
    assert _cache_size() == 1


def test_rust_featurizer_cache_keeps_distinct_build_options(monkeypatch):
    dataset = DummyDataset("option_cache_dataset", mode="train")
    build_calls: list[tuple[str, bool]] = []

    def _build_stub(
        dataset_arg,
        *,
        requested_build_path,
        allow_normalization_version_mismatch,
        name_counts_path=None,
        expected_normalization_version=None,
    ):
        assert name_counts_path is None
        if requested_build_path == "from_dataset":
            assert expected_normalization_version is None
        build_calls.append((requested_build_path, bool(allow_normalization_version_mismatch)))
        return (
            DummyRustFeaturizer(f"{dataset_arg.name}:{requested_build_path}:{allow_normalization_version_mismatch}"),
            requested_build_path,
            {"pre_build_seconds": 0.0, "ffi_seconds": 0.0, "post_build_seconds": 0.0},
            1,
            0.0,
        )

    monkeypatch.setattr(feature_port, "_build_rust_featurizer_strict", _build_stub)

    first = feature_port._get_rust_featurizer(dataset, rust_build_path="from_dataset")
    feature_port.warm_rust_featurizer(
        dataset,
        rust_build_path="from_json_paths",
        allow_normalization_version_mismatch=True,
    )
    first_again = feature_port._get_rust_featurizer(dataset, rust_build_path="from_dataset")
    json_entry = feature_port._get_rust_featurizer(
        dataset,
        rust_build_path="from_json_paths",
        allow_normalization_version_mismatch=True,
    )

    assert first_again is first
    assert json_entry.dataset_name == "option_cache_dataset:from_json_paths:True"
    assert build_calls == [("from_dataset", False), ("from_json_paths", True)]
    assert _cache_size() == 2


def test_rust_featurizer_cache_key_normalizes_blank_name_counts_path():
    none_key = feature_port._rust_featurizer_cache_key("from_json_paths", False, 0, None)
    blank_key = feature_port._rust_featurizer_cache_key("from_json_paths", False, 0, "  ")
    configured_key = feature_port._rust_featurizer_cache_key("from_json_paths", False, 0, " names.json ")

    assert blank_key == none_key
    assert configured_key[-2] == "names.json"


def test_rust_featurizer_cache_tracks_cluster_seed_version():
    dataset = DummyDataset("seed_version_cache_dataset", mode="train")

    first = feature_port._get_rust_featurizer(dataset)
    dataset._cluster_seeds_version = 1
    second = feature_port._get_rust_featurizer(dataset)

    assert second is not first
    assert DummyRustFeaturizer.created == ["seed_version_cache_dataset", "seed_version_cache_dataset"]
    assert _cache_size() == 1
    assert list(feature_port._RUST_FEATURIZER_CACHE[dataset]) == [
        feature_port._rust_featurizer_cache_key("from_dataset", False, 1)
    ]


def test_rust_featurizer_cache_retries_when_seed_version_changes_during_lookup(monkeypatch):
    dataset = DummyDataset("seed_version_race_dataset", mode="train")
    versions = iter([0, 1, 1, 1, 1])
    monkeypatch.setattr(feature_port, "_cluster_seeds_version_for_cache", lambda _dataset: next(versions))
    monkeypatch.setattr(feature_port, "RUST_FEATURIZER_EMPTY_WAIT_BACKOFF_SECONDS", 0.0)
    monkeypatch.setattr(feature_port, "RUST_FEATURIZER_EMPTY_WAIT_MAX_RETRIES", 3)

    featurizer = feature_port._get_rust_featurizer(dataset)

    assert featurizer.dataset_name == "seed_version_race_dataset"
    assert DummyRustFeaturizer.created == ["seed_version_race_dataset"]
    assert list(feature_port._RUST_FEATURIZER_CACHE[dataset]) == [
        feature_port._rust_featurizer_cache_key("from_dataset", False, 1)
    ]


def test_rust_featurizer_build_count_continues_across_seed_versions():
    dataset = DummyDataset("seed_version_build_count_dataset", mode="train")
    dataset._cluster_seeds_version = 1

    first = feature_port._get_rust_featurizer(dataset)
    dataset._cluster_seeds_version = 2
    second = feature_port._get_rust_featurizer(dataset)

    assert second is not first
    assert (
        feature_port._rust_featurizer_build_count(
            dataset,
            feature_port._rust_featurizer_cache_key("from_dataset", False, 2),
        )
        == 2
    )


def test_update_rust_cluster_seeds_reuses_cached_featurizer_after_version_bump():
    from s2and.rust_calls import update_rust_cluster_seeds

    dataset = DummyDataset("direct_seed_update_dataset", mode="train")
    dataset._cluster_seeds_version = 1
    first = feature_port._get_rust_featurizer(dataset)
    dataset.cluster_seeds_require["s1"] = "c1"

    update_rust_cluster_seeds(dataset)

    assert int(dataset._cluster_seeds_version) == 2
    assert DummyRustFeaturizer.created == ["direct_seed_update_dataset"]
    assert feature_port._get_rust_featurizer(dataset) is first
    assert list(feature_port._RUST_FEATURIZER_CACHE[dataset]) == [
        feature_port._rust_featurizer_cache_key("from_dataset", False, 2)
    ]


def test_update_rust_cluster_seeds_leaves_version_unchanged_on_ffi_failure():
    from s2and.rust_calls import update_rust_cluster_seeds

    dataset = DummyDataset("failed_seed_update_dataset", mode="train")
    dataset._cluster_seeds_version = 1
    featurizer = feature_port._get_rust_featurizer(dataset)

    def fail_update(_require_map, _disallow_set):
        raise RuntimeError("ffi failed")

    featurizer.update_cluster_seeds = fail_update

    with pytest.raises(RuntimeError, match="ffi failed"):
        update_rust_cluster_seeds(dataset)

    assert int(dataset._cluster_seeds_version) == 1
    assert list(feature_port._RUST_FEATURIZER_CACHE[dataset]) == [
        feature_port._rust_featurizer_cache_key("from_dataset", False, 1)
    ]


def test_update_rust_cluster_seeds_promotes_externally_bumped_cache_key():
    from s2and.rust_calls import update_rust_cluster_seeds

    dataset = DummyDataset("external_seed_update_dataset", mode="train")
    first = feature_port._get_rust_featurizer(dataset)
    dataset._cluster_seeds_version = 1
    dataset.cluster_seeds_require["s1"] = "c1"

    update_rust_cluster_seeds(dataset, bump_version=False)

    assert int(dataset._cluster_seeds_version) == 1
    assert DummyRustFeaturizer.created == ["external_seed_update_dataset"]
    assert feature_port._get_rust_featurizer(dataset) is first
    assert list(feature_port._RUST_FEATURIZER_CACHE[dataset]) == [
        feature_port._rust_featurizer_cache_key("from_dataset", False, 1)
    ]


def test_promote_cluster_seed_version_preserves_unrelated_cache_families(monkeypatch):
    dataset = DummyDataset("seed_update_cache_family_dataset", mode="train")
    build_calls: list[tuple[str, bool]] = []

    def _build_stub(
        dataset_arg,
        *,
        requested_build_path,
        allow_normalization_version_mismatch,
        name_counts_path=None,
        expected_normalization_version=None,
    ):
        assert name_counts_path is None
        if requested_build_path == "from_dataset":
            assert expected_normalization_version is None
        build_calls.append((requested_build_path, bool(allow_normalization_version_mismatch)))
        return (
            DummyRustFeaturizer(f"{dataset_arg.name}:{requested_build_path}:{allow_normalization_version_mismatch}"),
            requested_build_path,
            {"pre_build_seconds": 0.0, "ffi_seconds": 0.0, "post_build_seconds": 0.0},
            1,
            0.0,
        )

    monkeypatch.setattr(feature_port, "_build_rust_featurizer_strict", _build_stub)

    primary = feature_port._get_rust_featurizer(dataset, rust_build_path="from_dataset")
    other_family = feature_port._get_rust_featurizer(
        dataset,
        rust_build_path="from_json_paths",
        allow_normalization_version_mismatch=True,
    )

    assert feature_port._promote_cached_rust_featurizer_cluster_seed_version(  # noqa: SLF001
        dataset,
        primary,
        target_seed_version=1,
    )

    entries = feature_port._RUST_FEATURIZER_CACHE[dataset]
    assert entries[feature_port._rust_featurizer_cache_key("from_dataset", False, 1)].featurizer is primary
    assert entries[feature_port._rust_featurizer_cache_key("from_json_paths", True, 0)].featurizer is other_family
    assert build_calls == [("from_dataset", False), ("from_json_paths", True)]


def test_rust_featurizer_cache_rejects_invalid_cluster_seed_version():
    dataset = DummyDataset("bad_seed_version_cache_dataset", mode="train")

    first = feature_port._get_rust_featurizer(dataset)
    dataset._cluster_seeds_version = "bad"

    with pytest.raises(ValueError, match="invalid literal"):
        feature_port._get_rust_featurizer(dataset)

    assert first.dataset_name == "bad_seed_version_cache_dataset"
    assert DummyRustFeaturizer.created == ["bad_seed_version_cache_dataset"]
    assert _cache_size() == 1


def test_rust_featurizer_cache_rejects_none_cluster_seed_version():
    dataset = DummyDataset("none_seed_version_cache_dataset", mode="train")

    first = feature_port._get_rust_featurizer(dataset)
    dataset._cluster_seeds_version = None

    with pytest.raises(TypeError):
        feature_port._get_rust_featurizer(dataset)

    assert first.dataset_name == "none_seed_version_cache_dataset"
    assert DummyRustFeaturizer.created == ["none_seed_version_cache_dataset"]
    assert _cache_size() == 1


def test_build_rust_featurizer_from_arrow_paths_rejects_none_path(monkeypatch):
    class ArrowRustFeaturizer(DummyRustFeaturizer):
        @classmethod
        def from_arrow_paths(cls, *_args, **_kwargs):
            raise AssertionError("from_arrow_paths should not be called for invalid paths")

    class ArrowRustModule:
        __version__ = "0.51.0"
        RustFeaturizer = ArrowRustFeaturizer

    monkeypatch.setattr(feature_port, "s2and_rust", ArrowRustModule)

    with pytest.raises(ValueError, match="papers.*None"):
        feature_port.build_rust_featurizer_from_arrow_paths(
            {
                "signatures": "signatures.arrow",
                "papers": None,
            }
        )


def test_build_rust_featurizer_from_arrow_paths_requires_index_for_name_counts(monkeypatch, tmp_path):
    calls: list[dict[str, Any]] = []

    class ArrowRustFeaturizer(DummyRustFeaturizer):
        @classmethod
        def from_arrow_paths(cls, paths, _signature_ids, _name_tuples, *_args):
            calls.append({"paths": dict(paths)})
            return cls("arrow")

    class ArrowRustModule:
        __version__ = "0.51.0"
        RustFeaturizer = ArrowRustFeaturizer

    monkeypatch.setattr(feature_port, "s2and_rust", ArrowRustModule)
    for filename in ("signatures.arrow", "papers.arrow", "paper_authors.arrow"):
        (tmp_path / filename).touch()
    paths = {
        "signatures": str(tmp_path / "signatures.arrow"),
        "papers": str(tmp_path / "papers.arrow"),
        "paper_authors": str(tmp_path / "paper_authors.arrow"),
    }

    with pytest.raises(MissingArrowArtifactError) as exc_info:
        feature_port.build_rust_featurizer_from_arrow_paths(
            paths,
            load_name_counts=True,
            full_scan_without_index=True,
        )
    assert exc_info.value.missing_keys == ("name_counts_index",)

    with pytest.raises(ValueError, match="does not accept name_counts_path"):
        feature_port.build_rust_featurizer_from_arrow_paths(paths, name_counts_path="name_counts.json")

    patch_tiny_name_counts_loader(monkeypatch)
    index_path, _metrics = write_name_counts_index(tmp_path / "name_counts_index")
    result = feature_port.build_rust_featurizer_from_arrow_paths(
        {**paths, "name_counts_index": index_path},
        load_name_counts=True,
        full_scan_without_index=True,
    )

    assert result.dataset_name == "arrow"
    assert calls == [
        {
            "paths": {**paths, "name_counts_index": index_path},
        }
    ]


def test_build_rust_featurizer_from_arrow_paths_requires_batch_indexes_by_default(monkeypatch, tmp_path):
    class ArrowRustFeaturizer(DummyRustFeaturizer):
        @classmethod
        def from_arrow_paths(cls, *_args, **_kwargs):
            return cls("arrow")

    class ArrowRustModule:
        __version__ = "0.51.0"
        RustFeaturizer = ArrowRustFeaturizer

    monkeypatch.setattr(feature_port, "s2and_rust", ArrowRustModule)
    paths = {
        "signatures": str(tmp_path / "signatures.arrow"),
        "papers": str(tmp_path / "papers.arrow"),
        "paper_authors": str(tmp_path / "paper_authors.arrow"),
    }
    for path in paths.values():
        Path(path).touch()

    with pytest.raises(ValueError, match="signatures_batch_index"):
        feature_port.build_rust_featurizer_from_arrow_paths(paths)

    with pytest.raises(ValueError, match="missing_papers.arrow"):
        feature_port.build_rust_featurizer_from_arrow_paths(
            {
                **paths,
                "papers": str(tmp_path / "missing_papers.arrow"),
            },
            full_scan_without_index=True,
        )

    with pytest.raises(ValueError, match="name_pairs"):
        feature_port.build_rust_featurizer_from_arrow_paths(
            {
                **paths,
                "name_pairs": str(tmp_path / "missing_name_pairs.arrow"),
            },
            full_scan_without_index=True,
        )

    index_paths = {
        "signatures_batch_index": str(tmp_path / "signatures.index"),
        "papers_batch_index": str(tmp_path / "papers.index"),
        "paper_authors_batch_index": str(tmp_path / "paper_authors.index"),
    }
    for path in index_paths.values():
        Path(path).touch()

    result = feature_port.build_rust_featurizer_from_arrow_paths({**paths, **index_paths})
    assert result.dataset_name == "arrow"


def test_concurrent_builds_for_distinct_datasets_do_not_serialize(monkeypatch):
    d1 = DummyDataset("parallel_d1", mode="train")
    d2 = DummyDataset("parallel_d2", mode="train")
    ready = threading.Event()
    build_windows: dict[str, tuple[float, float]] = {}
    window_lock = threading.Lock()

    def _build_stub(
        dataset,
        *,
        requested_build_path,
        allow_normalization_version_mismatch,
        name_counts_path=None,
        expected_normalization_version=None,
    ):
        assert allow_normalization_version_mismatch is False
        assert name_counts_path is None
        assert expected_normalization_version is None
        ready.wait(timeout=2)
        build_start = time.perf_counter()
        time.sleep(0.25)
        build_end = time.perf_counter()
        with window_lock:
            build_windows[dataset.name] = (build_start, build_end)
        return (
            DummyRustFeaturizer(dataset.name),
            requested_build_path,
            {"pre_build_seconds": 0.0, "ffi_seconds": 0.0, "post_build_seconds": 0.0},
            1,
            0.25,
        )

    monkeypatch.setattr(
        feature_port,
        "_build_rust_featurizer_strict",
        _build_stub,
    )

    errors: list[Exception] = []

    def _worker(dataset):
        try:
            feature_port._get_rust_featurizer(dataset)
        except Exception as exc:  # pragma: no cover - assertion guard
            errors.append(exc)

    t1 = threading.Thread(target=_worker, args=(d1,))
    t2 = threading.Thread(target=_worker, args=(d2,))
    t1.start()
    t2.start()
    ready.set()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert errors == []
    assert len(build_windows) == 2
    latest_start = max(window[0] for window in build_windows.values())
    earliest_end = min(window[1] for window in build_windows.values())
    assert latest_start < earliest_end


def test_concurrent_builds_for_same_dataset_share_single_inflight_build(monkeypatch):
    dataset = DummyDataset("parallel_same_dataset", mode="train")
    ready = threading.Event()
    build_calls = {"count": 0}

    def _build_stub(
        dataset_arg,
        *,
        requested_build_path,
        allow_normalization_version_mismatch,
        name_counts_path=None,
        expected_normalization_version=None,
    ):
        assert allow_normalization_version_mismatch is False
        assert name_counts_path is None
        assert expected_normalization_version is None
        build_calls["count"] += 1
        ready.wait(timeout=2)
        time.sleep(0.25)
        return (
            DummyRustFeaturizer(dataset_arg.name),
            requested_build_path,
            {"pre_build_seconds": 0.0, "ffi_seconds": 0.0, "post_build_seconds": 0.0},
            1,
            0.25,
        )

    monkeypatch.setattr(
        feature_port,
        "_build_rust_featurizer_strict",
        _build_stub,
    )

    errors: list[Exception] = []

    def _worker():
        try:
            feature_port._get_rust_featurizer(dataset)
        except Exception as exc:  # pragma: no cover - assertion guard
            errors.append(exc)

    t1 = threading.Thread(target=_worker)
    t2 = threading.Thread(target=_worker)
    t1.start()
    t2.start()
    ready.set()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert errors == []
    assert build_calls["count"] == 1


def test_increment_rust_featurizer_build_count_is_thread_safe():
    dataset = DummyDataset("build_count_threadsafe", mode="train")
    cache_key = feature_port._rust_featurizer_cache_key("from_dataset", False)  # noqa: SLF001
    with feature_port._RUST_FEATURIZER_CACHE_LOCK:
        feature_port._RUST_FEATURIZER_CACHE[dataset] = {
            cache_key: feature_port._CacheEntry(
                featurizer=DummyRustFeaturizer(dataset.name),
                build_count=cast(Any, SleepyCounter(0)),
            )
        }

    worker_count = 8
    increments_per_worker = 25
    errors: list[Exception] = []
    counts: list[int] = []
    counts_lock = threading.Lock()

    def _worker():
        try:
            for _ in range(increments_per_worker):
                next_count = feature_port._increment_rust_featurizer_build_count(dataset, cache_key)
                with counts_lock:
                    counts.append(next_count)
        except Exception as exc:  # pragma: no cover - assertion guard
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(worker_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert errors == []
    expected_count = worker_count * increments_per_worker
    assert feature_port._rust_featurizer_build_count(dataset, cache_key) == expected_count
    assert sorted(counts) == list(range(1, expected_count + 1))


def test_get_rust_featurizer_raises_after_repeated_empty_wait(monkeypatch):
    dataset = DummyDataset("empty_wait_retry_budget", mode="train")
    attempts = {"count": 0}
    runtime_context = type("RuntimeContext", (), {"operation": "test_empty_wait", "run_id": "run-empty-wait"})()
    monkeypatch.setattr(feature_port, "RUST_FEATURIZER_EMPTY_WAIT_MAX_RETRIES", 2)
    monkeypatch.setattr(feature_port, "RUST_FEATURIZER_EMPTY_WAIT_BACKOFF_SECONDS", 0.0)

    def _always_empty(_dataset, *, build_context):
        del build_context
        attempts["count"] += 1
        return None, None

    monkeypatch.setattr(feature_port, "_get_or_wait_for_cached", _always_empty)

    with pytest.raises(RuntimeError, match="empty wait state") as exc_info:
        feature_port._get_rust_featurizer(dataset, runtime_context=runtime_context)
    message = str(exc_info.value)
    assert "dataset=empty_wait_retry_budget" in message
    assert "run=run-empty-wait" in message
    assert "attempts=3" in message
    assert attempts["count"] == 3


def test_get_rust_featurizer_retries_empty_wait_then_builds(monkeypatch):
    dataset = DummyDataset("empty_wait_then_build", mode="train")
    attempts = {"count": 0}
    build_calls = {"count": 0}
    inflight = feature_port._InFlightFeaturizerBuild("from_dataset")
    expected_featurizer = DummyRustFeaturizer("built_after_empty_wait")
    monkeypatch.setattr(feature_port, "RUST_FEATURIZER_EMPTY_WAIT_MAX_RETRIES", 2)
    monkeypatch.setattr(feature_port, "RUST_FEATURIZER_EMPTY_WAIT_BACKOFF_SECONDS", 0.0)

    def _empty_then_build(_dataset, *, build_context):
        del build_context
        attempts["count"] += 1
        if attempts["count"] == 1:
            return None, None
        return None, inflight

    def _build_stub(_dataset, *, inflight_build, build_context):
        del build_context
        build_calls["count"] += 1
        assert inflight_build is inflight
        return expected_featurizer

    monkeypatch.setattr(feature_port, "_get_or_wait_for_cached", _empty_then_build)
    monkeypatch.setattr(feature_port, "_build_and_cache_rust_featurizer", _build_stub)

    featurizer = feature_port._get_rust_featurizer(dataset)
    assert featurizer is expected_featurizer
    assert attempts["count"] == 2
    assert build_calls["count"] == 1


def test_json_ingest_is_inference_only():
    dataset = DummyDataset("train_dataset", mode="train")
    dataset.signatures_path = "signatures.json"
    dataset.papers_path = "papers.json"

    feature_port._get_rust_featurizer(dataset)

    assert DummyRustFeaturizer.created == ["train_dataset"]
    assert DummyRustFeaturizer.from_json_created == []


def test_inference_without_json_paths_uses_from_dataset():
    dataset = DummyDataset("inference_no_paths", mode="inference")

    feature_port._get_rust_featurizer(dataset)

    assert DummyRustFeaturizer.created == ["inference_no_paths"]
    assert DummyRustFeaturizer.from_json_created == []


def test_json_ingest_routes_compat_contract_payload():
    dataset = DummyDataset("inference_dataset", mode="inference")
    dataset.signatures_path = "signatures.json"
    dataset.papers_path = "papers.json"
    dataset.clusters_path = "clusters.json"
    dataset.cluster_seeds_path = "cluster_seeds.json"
    dataset.specter_embeddings_path = "specter.pkl"
    dataset.compute_reference_features = True
    dataset.preprocess = False
    dataset.n_jobs = 8

    feature_port._get_rust_featurizer(dataset)

    assert DummyRustFeaturizer.created == []
    assert len(DummyRustFeaturizer.from_json_created) == 1
    args, kwargs = DummyRustFeaturizer.from_json_created[0]
    assert kwargs == {}
    assert args == (
        "signatures.json",
        "papers.json",
        "cluster_seeds.json",
        "specter.pkl",
        None,
        None,
        False,
        True,
        feature_port.CLUSTER_SEEDS_LOOKUP["require"],
        feature_port.CLUSTER_SEEDS_LOOKUP["disallow"],
        8,
        None,
        False,
    )


def test_rust_build_path_argument_forces_from_dataset_even_with_json_paths():
    dataset = DummyDataset("inference_dataset", mode="inference")
    dataset.signatures_path = "signatures.json"
    dataset.papers_path = "papers.json"

    feature_port._get_rust_featurizer(dataset, rust_build_path="from_dataset")

    assert DummyRustFeaturizer.created == ["inference_dataset"]
    assert DummyRustFeaturizer.from_json_created == []


def test_rust_build_path_argument_rebuilds_cached_different_path():
    dataset = DummyDataset("inference_dataset", mode="inference")
    dataset.signatures_path = "signatures.json"
    dataset.papers_path = "papers.json"

    feature_port._get_rust_featurizer(dataset)
    feature_port._get_rust_featurizer(dataset, rust_build_path="from_dataset")

    assert len(DummyRustFeaturizer.from_json_created) == 1
    assert DummyRustFeaturizer.created == ["inference_dataset"]


def test_json_ingest_overlay_payload_includes_only_signatures_with_name_counts():
    dataset = DummyDataset("inference_dataset", mode="inference")
    dataset.signatures_path = "signatures.json"
    dataset.papers_path = "papers.json"
    dataset.signatures = {
        "s1": type(
            "Sig",
            (),
            {
                "author_info_name_counts": NameCounts(
                    first=1.0,
                    last=2.0,
                    first_last=3.0,
                    last_first_initial=4.0,
                ),
                "extra": "full_signature",
            },
        )(),
        "s2": type("Sig", (), {"author_info_name_counts": None, "extra": "ignored"})(),
    }

    feature_port._get_rust_featurizer(dataset)

    assert len(DummyRustFeaturizer.signature_overlay_payloads) == 1
    payload = DummyRustFeaturizer.signature_overlay_payloads[0]
    assert list(payload.keys()) == ["s1"]
    payload_entry = payload["s1"]
    assert hasattr(payload_entry, "author_info_name_counts")
    assert not hasattr(payload_entry, "extra")
    assert payload_entry.author_info_name_counts.first == 1.0
    assert payload_entry.author_info_name_counts.last == 2.0
    assert payload_entry.author_info_name_counts.first_last == 3.0
    assert payload_entry.author_info_name_counts.last_first_initial == 4.0


def test_signature_name_counts_overlay_payload_surfaces_items_failure():
    class FailingSignatures:
        def __len__(self):
            return 2

        def items(self):
            raise TypeError("items failed")

    dataset = DummyDataset("overlay_items_failure", mode="inference")
    dataset.signatures = FailingSignatures()

    with pytest.raises(RuntimeError, match="iterating signatures"):
        feature_port._signature_name_counts_overlay_payload_from_dataset(dataset)


def test_json_ingest_prefers_dataset_name_counts_over_artifact():
    dataset = DummyDataset("telemetry_json_dataset", mode="inference")
    dataset.signatures_path = "signatures.json"
    dataset.papers_path = "papers.json"
    dataset.signatures = {
        "s1": type(
            "Sig",
            (),
            {
                "author_info_name_counts": NameCounts(
                    first=1.0,
                    last=2.0,
                    first_last=3.0,
                    last_first_initial=4.0,
                )
            },
        )(),
        "s2": type("Sig", (), {"author_info_name_counts": None})(),
    }

    feature_port._get_rust_featurizer(dataset, name_counts_path="name_counts.json")

    args, _kwargs = DummyRustFeaturizer.from_json_created[0]
    assert args[5] is None


def test_json_ingest_uses_name_counts_artifact_when_dataset_has_no_counts(tmp_path):
    artifact_path = tmp_path / "name_counts.json"
    artifact_path.write_text('{"normalization_version":"legacy_compat","counts":{}}', encoding="utf-8")

    dataset = DummyDataset("telemetry_json_dataset", mode="inference")
    dataset.signatures_path = "signatures.json"
    dataset.papers_path = "papers.json"
    dataset.signatures = {"s1": type("Sig", (), {"author_info_name_counts": None})()}

    feature_port._get_rust_featurizer(dataset, name_counts_path=str(artifact_path))

    args, _kwargs = DummyRustFeaturizer.from_json_created[0]
    assert args[5] == str(artifact_path)
    assert args[12] is False


def test_json_ingest_uses_env_name_counts_artifact_when_no_dataset_counts(tmp_path, monkeypatch):
    artifact_path = tmp_path / "name_counts_env.json"
    artifact_path.write_text('{"normalization_version":"legacy_compat","counts":{}}', encoding="utf-8")
    monkeypatch.setenv("S2AND_RUST_NAME_COUNTS_JSON", str(artifact_path))

    dataset = DummyDataset("telemetry_json_dataset", mode="inference")
    dataset.signatures_path = "signatures.json"
    dataset.papers_path = "papers.json"
    dataset.signatures = {"s1": type("Sig", (), {"author_info_name_counts": None})()}

    feature_port._get_rust_featurizer(dataset)

    args, _kwargs = DummyRustFeaturizer.from_json_created[0]
    assert args[5] == str(artifact_path)
    assert list(feature_port._RUST_FEATURIZER_CACHE[dataset]) == [
        feature_port._rust_featurizer_cache_key(
            "from_json_paths",
            False,
            0,
            str(artifact_path),
            feature_port.DEFAULT_NORMALIZATION_VERSION,
        )
    ]


def test_json_ingest_env_normalization_version_is_delegated_to_rust(tmp_path, monkeypatch):
    artifact_path = tmp_path / "name_counts_env.json"
    artifact_path.write_text('{"normalization_version":"canonical_v2","counts":{}}', encoding="utf-8")
    monkeypatch.setenv("S2AND_RUST_NAME_COUNTS_JSON", str(artifact_path))
    monkeypatch.setenv("S2AND_NORMALIZATION_VERSION", "canonical_v2")

    dataset = DummyDataset("telemetry_json_dataset", mode="inference")
    dataset.signatures_path = "signatures.json"
    dataset.papers_path = "papers.json"
    dataset.signatures = {"s1": type("Sig", (), {"author_info_name_counts": None})()}

    feature_port._get_rust_featurizer(dataset)

    args, _kwargs = DummyRustFeaturizer.from_json_created[0]
    assert args[5] == str(artifact_path)
    assert args[11] == "canonical_v2"


def test_json_ingest_cache_key_includes_name_counts_artifact_path(tmp_path):
    first_artifact_path = tmp_path / "name_counts_first.json"
    second_artifact_path = tmp_path / "name_counts_second.json"
    first_artifact_path.write_text('{"normalization_version":"legacy_compat","counts":{}}', encoding="utf-8")
    second_artifact_path.write_text('{"normalization_version":"legacy_compat","counts":{}}', encoding="utf-8")
    dataset = DummyDataset("telemetry_json_dataset", mode="inference")
    dataset.signatures_path = "signatures.json"
    dataset.papers_path = "papers.json"
    dataset.signatures = {"s1": type("Sig", (), {"author_info_name_counts": None})()}

    feature_port._get_rust_featurizer(dataset, name_counts_path=str(first_artifact_path))
    feature_port._get_rust_featurizer(dataset, name_counts_path=str(second_artifact_path))

    assert [created[0][5] for created in DummyRustFeaturizer.from_json_created] == [
        str(first_artifact_path),
        str(second_artifact_path),
    ]


def test_json_ingest_normalization_mismatch_allowance_is_explicit(tmp_path):
    artifact_path = tmp_path / "name_counts.json"
    artifact_path.write_text('{"normalization_version":"other","counts":{}}', encoding="utf-8")

    dataset = DummyDataset("normalization_mismatch_explicit", mode="inference")
    dataset.signatures_path = "signatures.json"
    dataset.papers_path = "papers.json"
    dataset.signatures = {"s1": type("Sig", (), {"author_info_name_counts": None})()}

    feature_port._get_rust_featurizer(
        dataset,
        allow_normalization_version_mismatch=True,
        name_counts_path=str(artifact_path),
    )

    args, _kwargs = DummyRustFeaturizer.from_json_created[0]
    assert args[5] == str(artifact_path)
    assert args[12] is True


def test_explicit_from_json_paths_requires_json_paths():
    dataset = DummyDataset("missing_json_paths", mode="train")

    with pytest.raises(RuntimeError, match="signatures_path/papers_path are missing"):
        feature_port._get_rust_featurizer(dataset, rust_build_path="from_json_paths")

    assert DummyRustFeaturizer.created == []
    assert DummyRustFeaturizer.from_json_created == []


def test_explicit_evict_and_clear_api():
    d1 = DummyDataset("d1", mode="train")
    d2 = DummyDataset("d2", mode="train")

    feature_port._get_rust_featurizer(d1)
    feature_port._get_rust_featurizer(d2)
    assert _cache_size() == 2

    assert feature_port.evict_rust_featurizer(d1) is True
    assert feature_port.evict_rust_featurizer(d1) is False
    assert _cache_size() == 1

    cleared = feature_port.clear_rust_featurizer_cache()
    assert cleared == 1
    assert _cache_size() == 0


def test_evict_rust_featurizer_clears_build_counts():
    dataset = DummyDataset("evict_build_counts", mode="train")
    cache_key = feature_port._rust_featurizer_cache_key("from_dataset", False)  # noqa: SLF001

    feature_port._get_rust_featurizer(dataset)

    assert feature_port._rust_featurizer_build_count(dataset, cache_key) == 1  # noqa: SLF001
    assert feature_port.evict_rust_featurizer(dataset) is True
    assert feature_port._rust_featurizer_build_count(dataset, cache_key) == 0  # noqa: SLF001


def test_load_s2and_rust_extension_falls_back_from_namespace_package(monkeypatch):
    class NamespaceOnly:
        pass

    class NativeExtension:
        RustFeaturizer = object()

    def fake_import_module(name: str):
        if name == "s2and_rust":
            return NamespaceOnly()
        if name == "s2and_rust._s2and_rust":
            return NativeExtension
        raise _missing_module(name)

    monkeypatch.setattr(runtime.importlib, "import_module", fake_import_module)

    loaded = runtime.load_s2and_rust_extension()
    assert loaded is NativeExtension


def test_load_s2and_rust_extension_returns_first_valid_module(monkeypatch):
    class ValidRustFeaturizer:
        @staticmethod
        def from_dataset(*args, **kwargs):
            return None

        @staticmethod
        def from_json_paths(*args, **kwargs):
            return None

        def signature_ids(self):
            return []

    class TopLevelModule:
        RustFeaturizer = ValidRustFeaturizer

    def fake_import_module(name: str):
        if name == "s2and_rust":
            return TopLevelModule
        raise _missing_module(name)

    monkeypatch.setattr(runtime.importlib, "import_module", fake_import_module)

    loaded = runtime.load_s2and_rust_extension()
    assert loaded is TopLevelModule
