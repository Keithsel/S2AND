from __future__ import annotations

import pytest

from s2and import feature_port

if not feature_port.rust_featurizer_available():
    raise pytest.skip.Exception("s2and_rust extension is unavailable", allow_module_level=True)

import numpy as np

import s2and.model as model_module
from s2and.consts import LARGE_INTEGER
from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer

s2and_rust = feature_port.s2and_rust


def _dummy_dataset(name: str) -> ANDData:
    return ANDData(
        "tests/dummy/signatures.json",
        "tests/dummy/papers.json",
        clusters="tests/dummy/clusters.json",
        name=name,
        load_name_counts=False,
        n_jobs=1,
    )


def _specter_dataset(name: str, specter_embeddings: object) -> ANDData:
    signatures = {
        "s1": {
            "signature_id": "s1",
            "paper_id": "p1",
            "author_info": {
                "first": "Ada",
                "middle": "",
                "last": "Lovelace",
                "suffix": "",
                "affiliations": [],
                "email": "",
                "position": 0,
                "block": "a lovelace",
                "source_ids": [],
            },
        },
        "s2": {
            "signature_id": "s2",
            "paper_id": "p2",
            "author_info": {
                "first": "Ada",
                "middle": "",
                "last": "Lovelace",
                "suffix": "",
                "affiliations": [],
                "email": "",
                "position": 0,
                "block": "a lovelace",
                "source_ids": [],
            },
        },
    }
    papers = {
        "p1": {
            "paper_id": "p1",
            "title": "Graph Models",
            "abstract": "",
            "venue": "",
            "journal_name": "",
            "year": 2020,
            "authors": [{"position": 0, "author_name": "Ada Lovelace"}],
            "references": [],
        },
        "p2": {
            "paper_id": "p2",
            "title": "Graph Models",
            "abstract": "",
            "venue": "",
            "journal_name": "",
            "year": 2020,
            "authors": [{"position": 0, "author_name": "Ada Lovelace"}],
            "references": [],
        },
    }
    return ANDData(
        signatures=signatures,
        papers=papers,
        name=name,
        mode="inference",
        specter_embeddings=specter_embeddings,
        load_name_counts=False,
        preprocess=True,
        name_tuples=set(),
        n_jobs=1,
    )


def _dummy_clusterer(
    *,
    cluster_model: object | None,
    use_default_constraints_as_supervision: bool = False,
) -> Clusterer:
    return Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=None,
        cluster_model=cluster_model,
        n_jobs=1,
        use_cache=False,
        use_default_constraints_as_supervision=use_default_constraints_as_supervision,
        batch_size=2,
    )


def _partial_supervision_for_upper_triangle(signatures: list[str]) -> tuple[dict[tuple[str, str], float], np.ndarray]:
    values: list[float] = []
    partial_supervision: dict[tuple[str, str], float] = {}
    next_value = 11
    for i in range(len(signatures)):
        for j in range(i + 1, len(signatures)):
            value = float(next_value) / 100.0
            partial_supervision[(signatures[i], signatures[j])] = value
            values.append(value)
            next_value += 11
    return partial_supervision, np.asarray(values, dtype=np.float64)


def test_rust_from_dataset_supports_tuple_specter_payloads():
    tuple_dataset = _specter_dataset(
        "tuple_specter_rust_dataset",
        (np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32), ["p1", "p2"]),
    )
    dict_dataset = _specter_dataset(
        "dict_specter_rust_dataset",
        {
            "p1": np.asarray([1.0, 0.0], dtype=np.float32),
            "p2": np.asarray([1.0, 0.0], dtype=np.float32),
        },
    )
    no_specter_dataset = _specter_dataset("no_specter_rust_dataset", None)

    tuple_featurizer = s2and_rust.RustFeaturizer.from_dataset(tuple_dataset, 0.0, 10000.0, 1)
    dict_featurizer = s2and_rust.RustFeaturizer.from_dataset(dict_dataset, 0.0, 10000.0, 1)
    no_specter_featurizer = s2and_rust.RustFeaturizer.from_dataset(no_specter_dataset, 0.0, 10000.0, 1)
    pairs = [("s1", "s2")]
    tuple_features = np.asarray(tuple_featurizer.featurize_pairs_matrix(pairs, None, 1, np.nan))
    dict_features = np.asarray(dict_featurizer.featurize_pairs_matrix(pairs, None, 1, np.nan))
    no_specter_features = np.asarray(no_specter_featurizer.featurize_pairs_matrix(pairs, None, 1, np.nan))

    np.testing.assert_allclose(tuple_features, dict_features, equal_nan=True)
    assert not np.allclose(tuple_features, no_specter_features, equal_nan=True)


def test_make_distance_matrices_rust_blockwise_fastcluster(monkeypatch):
    monkeypatch.setenv("S2AND_BACKEND", "rust")
    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        model_module.Clusterer,
        "distance_matrix_helper",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("legacy pair helper should not be called")),
    )

    featurize_call_sizes: list[int] = []

    def fake_many_pairs_featurize(signature_pairs, *_args, **_kwargs):
        featurize_call_sizes.append(len(signature_pairs))
        labels = np.asarray([float(pair[2]) for pair in signature_pairs], dtype=np.float64)
        features = np.zeros((len(signature_pairs), 1), dtype=np.float64)
        return features, labels, None

    def fake_predict_and_combine(
        _classifier,
        _nameless_classifier,
        _features,
        labels,
        _nameless_features,
        _batch_label,
        runtime_context=None,
        **_kwargs,
    ):
        del runtime_context, _kwargs
        return np.asarray(labels + LARGE_INTEGER, dtype=np.float64), 0.0

    monkeypatch.setattr(model_module, "many_pairs_featurize", fake_many_pairs_featurize)
    monkeypatch.setattr(model_module, "_predict_and_combine", fake_predict_and_combine)

    dataset = _dummy_dataset("dummy_rust_blockwise_fastcluster")
    clusterer = _dummy_clusterer(cluster_model=None)
    signatures = ["0", "1", "2", "3"]
    partial_supervision, expected_flat = _partial_supervision_for_upper_triangle(signatures)

    output = clusterer.make_distance_matrices(
        {"block": signatures},
        dataset,
        partial_supervision=partial_supervision,
    )
    matrix = output["block"]

    assert matrix.dtype == np.float64
    np.testing.assert_allclose(matrix, expected_flat, rtol=1e-10, atol=1e-12)
    assert featurize_call_sizes == [2, 2, 2]


def test_make_distance_matrices_rust_blockwise_square_matrix(monkeypatch):
    monkeypatch.setenv("S2AND_BACKEND", "rust")
    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        model_module.Clusterer,
        "distance_matrix_helper",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("legacy pair helper should not be called")),
    )

    featurize_call_sizes: list[int] = []

    def fake_many_pairs_featurize(signature_pairs, *_args, **_kwargs):
        featurize_call_sizes.append(len(signature_pairs))
        labels = np.asarray([float(pair[2]) for pair in signature_pairs], dtype=np.float64)
        features = np.zeros((len(signature_pairs), 1), dtype=np.float64)
        return features, labels, None

    def fake_predict_and_combine(
        _classifier,
        _nameless_classifier,
        _features,
        labels,
        _nameless_features,
        _batch_label,
        runtime_context=None,
        **_kwargs,
    ):
        del runtime_context, _kwargs
        return np.asarray(labels + LARGE_INTEGER, dtype=np.float64), 0.0

    monkeypatch.setattr(model_module, "many_pairs_featurize", fake_many_pairs_featurize)
    monkeypatch.setattr(model_module, "_predict_and_combine", fake_predict_and_combine)

    dataset = _dummy_dataset("dummy_rust_blockwise_square")
    clusterer = _dummy_clusterer(cluster_model=object())
    signatures = ["0", "1", "2", "3"]
    partial_supervision, expected_flat = _partial_supervision_for_upper_triangle(signatures)

    output = clusterer.make_distance_matrices(
        {"block": signatures},
        dataset,
        partial_supervision=partial_supervision,
    )
    matrix = output["block"]

    expected_square = np.zeros((4, 4), dtype=np.float16)
    expected_square[np.triu_indices(4, k=1)] = expected_flat.astype(np.float16)
    expected_square = expected_square + expected_square.T
    np.fill_diagonal(expected_square, 0)

    assert matrix.shape == (4, 4)
    np.testing.assert_allclose(matrix, expected_square, rtol=0, atol=0)
    assert featurize_call_sizes == [2, 2, 2]


def test_make_distance_matrices_rust_fused_upper_triangle_api(monkeypatch):
    monkeypatch.setenv("S2AND_BACKEND", "rust")
    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        model_module.Clusterer,
        "distance_matrix_helper",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("legacy pair helper should not be called")),
    )
    monkeypatch.setattr(
        model_module,
        "many_pairs_featurize",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("many_pairs_featurize should not be called")),
    )
    monkeypatch.setattr(
        model_module,
        "get_constraints_matrix_indexed_rust",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy indexed constraint API should not be called")
        ),
    )

    captured = {"constraint_calls": 0, "feature_calls": 0}

    class _FakeFusedFeaturizer:
        def signature_ids(self):
            return ["0", "1", "2", "3"]

        def get_constraints_block_upper_triangle_indexed(
            self,
            block_signature_indices,
            start_offset=0,
            max_pairs=None,
            *_args,
            **_kwargs,
        ):
            captured["constraint_calls"] += 1
            block_size = len(block_signature_indices)
            all_pairs = [(i, j) for i in range(block_size) for j in range(i + 1, block_size)]
            pair_slice = all_pairs[start_offset : start_offset + int(max_pairs or len(all_pairs))]
            left = [int(i) for i, _ in pair_slice]
            right = [int(j) for _, j in pair_slice]
            return left, right, [None] * len(pair_slice)

        def featurize_block_upper_triangle_matrix_indexed(
            self,
            block_signature_indices,
            start_offset=0,
            max_pairs=None,
            selected_indices=None,
            *_args,
            **_kwargs,
        ):
            captured["feature_calls"] += 1
            block_size = len(block_signature_indices)
            all_pairs = [(i, j) for i in range(block_size) for j in range(i + 1, block_size)]
            pair_slice = all_pairs[start_offset : start_offset + int(max_pairs or len(all_pairs))]
            out_cols = len(selected_indices) if selected_indices is not None else 39
            out = np.zeros((len(pair_slice), out_cols), dtype=np.float64)
            for row_offset in range(len(pair_slice)):
                out[row_offset, :] = float(start_offset + row_offset + 1) / 10.0
            return out

    monkeypatch.setattr(model_module, "_get_rust_featurizer", lambda *_args, **_kwargs: _FakeFusedFeaturizer())

    def fake_predict_and_combine(
        _classifier,
        _nameless_classifier,
        features,
        labels,
        _nameless_features,
        _batch_label,
        runtime_context=None,
        **_kwargs,
    ):
        del labels, runtime_context, _kwargs
        return np.asarray(features[:, 0], dtype=np.float64), 0.0

    monkeypatch.setattr(model_module, "_predict_and_combine", fake_predict_and_combine)

    dataset = _dummy_dataset("dummy_rust_blockwise_fused")
    clusterer = _dummy_clusterer(
        cluster_model=None,
        use_default_constraints_as_supervision=True,
    )
    signatures = ["0", "1", "2", "3"]
    output = clusterer.make_distance_matrices(
        {"block": signatures},
        dataset,
        partial_supervision={},
    )
    matrix = output["block"]

    np.testing.assert_allclose(matrix, np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64), rtol=0, atol=0)
    assert captured["constraint_calls"] == 3
    assert captured["feature_calls"] == 3


def test_make_distance_matrices_from_rust_featurizer_avoids_anddata_lookup(monkeypatch):
    monkeypatch.setattr(
        model_module,
        "_get_rust_featurizer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ANDData featurizer lookup is not needed")),
    )
    monkeypatch.setattr(
        model_module,
        "many_pairs_featurize",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Python pair featurization is not needed")),
    )

    captured = {"constraint_calls": 0, "feature_calls": 0}

    class _FakeFusedFeaturizer:
        def signature_ids(self):
            return ["0", "1", "2", "3"]

        def get_constraints_block_upper_triangle_indexed(
            self,
            block_signature_indices,
            start_offset=0,
            max_pairs=None,
            *_args,
            **_kwargs,
        ):
            captured["constraint_calls"] += 1
            block_size = len(block_signature_indices)
            all_pairs = [(i, j) for i in range(block_size) for j in range(i + 1, block_size)]
            pair_slice = all_pairs[start_offset : start_offset + int(max_pairs or len(all_pairs))]
            left = [int(i) for i, _ in pair_slice]
            right = [int(j) for _, j in pair_slice]
            return left, right, [None] * len(pair_slice)

        def featurize_block_upper_triangle_matrix_indexed(
            self,
            block_signature_indices,
            start_offset=0,
            max_pairs=None,
            selected_indices=None,
            *_args,
            **_kwargs,
        ):
            captured["feature_calls"] += 1
            block_size = len(block_signature_indices)
            all_pairs = [(i, j) for i in range(block_size) for j in range(i + 1, block_size)]
            pair_slice = all_pairs[start_offset : start_offset + int(max_pairs or len(all_pairs))]
            out_cols = len(selected_indices) if selected_indices is not None else 39
            out = np.zeros((len(pair_slice), out_cols), dtype=np.float64)
            for row_offset in range(len(pair_slice)):
                out[row_offset, :] = float(start_offset + row_offset + 1) / 10.0
            return out

    def fake_predict_and_combine(
        _classifier,
        _nameless_classifier,
        features,
        labels,
        _nameless_features,
        _batch_label,
        runtime_context=None,
        **_kwargs,
    ):
        del labels, runtime_context, _kwargs
        return np.asarray(features[:, 0], dtype=np.float64), 0.0

    monkeypatch.setattr(model_module, "_predict_and_combine", fake_predict_and_combine)

    clusterer = _dummy_clusterer(
        cluster_model=None,
        use_default_constraints_as_supervision=True,
    )
    output = clusterer.make_distance_matrices_from_rust_featurizer(
        {"block": ["0", "1", "2", "3"]},
        _FakeFusedFeaturizer(),
    )

    np.testing.assert_allclose(output["block"], np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), rtol=0, atol=0)
    assert captured["constraint_calls"] == 3
    assert captured["feature_calls"] == 3


def test_predict_from_arrow_paths_builds_filtered_arrow_featurizer(monkeypatch):
    captured = {}

    class _FakeFeaturizer:
        def signature_ids(self):
            return ["0", "1", "2"]

    def fake_build_from_arrow_paths(paths, **kwargs):
        captured["paths"] = paths
        captured["signature_ids"] = tuple(kwargs["signature_ids"])
        captured["load_name_counts"] = kwargs["load_name_counts"]
        return _FakeFeaturizer()

    def fake_predict_from_rust_featurizer(self, block_dict, rust_featurizer, **kwargs):
        captured["block_dict"] = block_dict
        captured["rust_featurizer"] = rust_featurizer
        captured["total_ram_bytes"] = kwargs["total_ram_bytes"]
        return {"block_0": ["0", "1", "2"]}, None

    monkeypatch.setattr(model_module, "build_rust_featurizer_from_arrow_paths", fake_build_from_arrow_paths)
    monkeypatch.setattr(Clusterer, "predict_from_rust_featurizer", fake_predict_from_rust_featurizer)

    clusterer = _dummy_clusterer(cluster_model=None)
    result, dists = clusterer.predict_from_arrow_paths(
        {"block": ["0", "1", "2"]},
        {"signatures": "signatures.arrow", "papers": "papers.arrow", "paper_authors": "paper_authors.arrow"},
        load_name_counts=True,
        total_ram_bytes=123,
    )

    assert result == {"block_0": ["0", "1", "2"]}
    assert dists is None
    assert captured["signature_ids"] == ("0", "1", "2")
    assert captured["load_name_counts"] is True
    assert captured["total_ram_bytes"] == 123


def test_predict_from_arrow_paths_rejects_reference_features(monkeypatch):
    def fail_build_from_arrow_paths(*_args, **_kwargs):
        raise AssertionError("Arrow featurizer build should fail fast before touching inputs")

    monkeypatch.setattr(model_module, "build_rust_featurizer_from_arrow_paths", fail_build_from_arrow_paths)

    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["reference_features"]),
        classifier=None,
        cluster_model=None,
        n_jobs=1,
    )
    with pytest.raises(ValueError, match="reference_features"):
        clusterer.predict_from_arrow_paths(
            {"block": ["0", "1"]},
            {"signatures": "signatures.arrow", "papers": "papers.arrow", "paper_authors": "paper_authors.arrow"},
        )


def test_predict_auto_routes_to_arrow_paths_when_dataset_advertises_them(tmp_path, monkeypatch):
    captured = {}
    dataset = _dummy_dataset("dummy_predict_auto_arrow")
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
    runtime_context = type(
        "RuntimeContext",
        (),
        {
            "operation": "cluster_predict",
            "requested_backend": "rust",
            "resolved_backend": "rust",
            "use_rust": True,
            "run_id": "test-auto-arrow-predict",
            "source": "test",
        },
    )()

    def fail_predict_helper(*_args, **_kwargs):
        raise AssertionError("predict should route to Arrow/Rust before legacy predict_helper")

    def fake_predict_from_arrow_paths(self, block_dict, paths, **kwargs):
        captured["self"] = self
        captured["block_dict"] = block_dict
        captured["paths"] = dict(paths)
        captured["runtime_context"] = kwargs["runtime_context"]
        return {"arrow": ["0", "1"]}, None

    monkeypatch.setattr(model_module, "build_runtime_context", lambda _operation: runtime_context)
    monkeypatch.setattr(Clusterer, "predict_helper", fail_predict_helper)
    monkeypatch.setattr(Clusterer, "predict_from_arrow_paths", fake_predict_from_arrow_paths)

    clusterer = _dummy_clusterer(cluster_model=None)
    result, dists = clusterer.predict({"block": ["0", "1"]}, dataset)

    assert result == {"arrow": ["0", "1"]}
    assert dists is None
    assert captured["self"] is clusterer
    assert captured["block_dict"] == {"block": ["0", "1"]}
    assert captured["paths"] == arrow_paths
    assert captured["runtime_context"] is runtime_context


def test_arrow_path_discovery_uses_original_signature_path_after_filtering(tmp_path):
    json_dataset_dir = tmp_path / "s2and_mini" / "demo"
    filtered_dir = tmp_path / "filtered"
    arrow_dataset_dir = tmp_path / "s2and_mini_arrow" / "demo"
    json_dataset_dir.mkdir(parents=True)
    filtered_dir.mkdir()
    arrow_dataset_dir.mkdir(parents=True)
    for filename in ("signatures.arrow", "papers.arrow", "paper_authors.arrow", "specter2.arrow"):
        (arrow_dataset_dir / filename).touch()
    dataset = type(
        "Dataset",
        (),
        {
            "original_signatures_path": str(json_dataset_dir / "demo_signatures.json"),
            "signatures_path": str(filtered_dir / "signatures_filtered.json"),
            "specter_embeddings_path": str(json_dataset_dir / "demo_specter2.pkl"),
        },
    )()

    resolved = model_module._resolve_dataset_arrow_paths(
        dataset,
        require_specter=True,
        require_cluster_seeds=False,
    )

    assert resolved == {
        "signatures": str(arrow_dataset_dir / "signatures.arrow"),
        "papers": str(arrow_dataset_dir / "papers.arrow"),
        "paper_authors": str(arrow_dataset_dir / "paper_authors.arrow"),
        "specter": str(arrow_dataset_dir / "specter2.arrow"),
    }
