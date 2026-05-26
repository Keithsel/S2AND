"""Compatibility/parity coverage for legacy service-shaped JSON ingest.

Production inference converts service-shaped JSON to Arrow before entering Rust.
These tests keep the legacy JSON loader honest for compatibility and parity only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from s2and.data import ANDData
from s2and.feature_port import _get_rust_featurizer
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer, _sync_rust_cluster_seeds
from s2and.runtime import build_runtime_context
from s2and.rust_calls import get_constraint_rust
from s2and.rust_lifecycle import build_rust_json_ingest_contract
from tests.helpers import equalish, import_s2and_rust

_DUMMY_DIR = Path("tests/dummy")


class _ConstantProbabilityClassifier:
    def predict_proba(self, features_2d: np.ndarray) -> np.ndarray:
        features_2d = np.asarray(features_2d, dtype=np.float64)
        merge_probability = np.full(features_2d.shape[0], 0.7, dtype=np.float64)
        return np.stack([1.0 - merge_probability, merge_probability], axis=1)


def _service_name_counts() -> dict[str, dict[str, float]]:
    return {
        "first_dict": {"abdul": 10.0, "alexander": 20.0},
        "last_dict": {"sattar": 30.0, "konovalov": 40.0},
        "first_last_dict": {"abdul sattar": 50.0, "alexander konovalov": 60.0},
        "last_first_initial_dict": {"sattar a": 70.0, "konovalov a": 80.0},
    }


def _edge_case_name_counts() -> dict[str, dict[str, float]]:
    return {
        "first_dict": {"jose": 10.0, "joanna": 11.0, "li": 12.0},
        "last_dict": {"muller": 20.0, "wang": 21.0},
        "first_last_dict": {"jose muller": 30.0, "joanna muller": 31.0, "li wang": 32.0},
        "last_first_initial_dict": {"muller j": 40.0, "wang l": 41.0},
    }


def _skip_unless_rust_from_json_paths_available() -> None:
    has_rust, rust_module = import_s2and_rust(required_method="from_json_paths")
    if not has_rust:
        raise pytest.skip.Exception(f"s2and_rust RustFeaturizer.from_json_paths is unavailable: {rust_module}")


@pytest.fixture(autouse=True)
def _rust_from_json_paths_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    _skip_unless_rust_from_json_paths_available()
    monkeypatch.setenv("S2AND_BACKEND", "rust")


def _service_specter_embeddings() -> dict[str, list[float]]:
    return {
        "53235312": [0.1, 0.2],
        "27077319": [0.3, 0.4],
        "19901392": [0.5, 0.6],
    }


def _service_cluster_seeds() -> dict[str, dict[str, str]]:
    return {
        "0": {"1": "disallow"},
        "1": {"2": "require"},
    }


def _build_service_shaped_dataset(
    *,
    specter_embeddings: dict[str, list[float]] | None = None,
) -> tuple[ANDData, dict[str, list[float]]]:
    if specter_embeddings is None:
        specter_embeddings = _service_specter_embeddings()
    dataset = ANDData(
        signatures=str(_DUMMY_DIR / "signatures.json"),
        papers=str(_DUMMY_DIR / "papers.json"),
        cluster_seeds=_service_cluster_seeds(),
        specter_embeddings=specter_embeddings,
        altered_cluster_signatures=["1"],
        name="service_json_ingest_contract",
        mode="inference",
        block_type="s2",
        n_jobs=1,
        load_name_counts=_service_name_counts(),
        name_tuples=set(),
        use_orcid_id=True,
    )
    return dataset, specter_embeddings


def _load_json_fixture(filename: str) -> dict:
    with (_DUMMY_DIR / filename).open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


def _build_dict_backed_service_dataset() -> ANDData:
    return ANDData(
        signatures=_load_json_fixture("signatures.json"),
        papers=_load_json_fixture("papers.json"),
        cluster_seeds=_service_cluster_seeds(),
        specter_embeddings=_service_specter_embeddings(),
        altered_cluster_signatures=["1"],
        name="service_dict_backed_contract",
        mode="inference",
        block_type="s2",
        n_jobs=1,
        load_name_counts=_service_name_counts(),
        name_tuples=set(),
        use_orcid_id=True,
    )


def _edge_case_payload() -> tuple[dict, dict, dict[str, dict[str, str]], dict[str, list[float]]]:
    signatures = {
        "edge-1": {
            "signature_id": "edge-1",
            "paper_id": 9101,
            "given_name": "Jose Muller",
            "sourced_author_ids": [],
            "sourced_author_source": "DBLP",
            "author_info": {
                "first": "Jose",
                "middle": None,
                "last": "Muller",
                "suffix": None,
                "position": 0,
                "email": None,
                "affiliations": ["Universite de Montreal"],
                "block": "j muller",
                "given_block": "j muller",
                "source_id_source": "DBLP",
                "source_ids": [],
            },
        },
        "edge-2": {
            "signature_id": "edge-2",
            "paper_id": 9102,
            "given_name": "Jose Muller",
            "sourced_author_ids": [],
            "sourced_author_source": "DBLP",
            "author_info": {
                "first": "Jose\u0301",
                "middle": "A.",
                "last": "Mu\u0308ller",
                "suffix": None,
                "position": 1,
                "email": None,
                "affiliations": ["Universite\u00a0de Montreal"],
                "block": "j muller",
                "given_block": "j muller",
                "source_id_source": "DBLP",
                "source_ids": [],
            },
        },
        "edge-3": {
            "signature_id": "edge-3",
            "paper_id": 9103,
            "given_name": "Li Wang",
            "sourced_author_ids": [],
            "sourced_author_source": "DBLP",
            "author_info": {
                "first": "Li",
                "middle": None,
                "last": "Wang",
                "suffix": None,
                "position": 0,
                "email": None,
                "affiliations": ["AI Lab\tNorth"],
                "block": "l wang",
                "given_block": "l wang",
                "source_id_source": "DBLP",
                "source_ids": [],
            },
        },
    }
    papers = {
        "9101": {
            "paper_id": 9101,
            "title": "TGF-\u03b2 signaling in Jose Muller labs",
            "abstract": None,
            "journal_name": None,
            "venue": "Conference\u00a0on Biology",
            "year": 2024,
            "sources": [],
            "fields_of_study": [],
            "authors": [{"position": 0, "author_name": "Jose Muller"}],
            "references": [],
        },
        "9102": {
            "paper_id": 9102,
            "title": "TGF-\u03b2 signaling in Jose\u0301 Mu\u0308ller labs",
            "abstract": "",
            "journal_name": "Journal\tName",
            "venue": "Conference on Biology",
            "year": 2024,
            "sources": [],
            "fields_of_study": [],
            "authors": [
                {"position": 0, "author_name": "Alice Example"},
                {"position": 1, "author_name": "Jose Muller"},
            ],
            "references": [9101],
        },
        "9103": {
            "paper_id": 9103,
            "title": "RNA study with extra\twhitespace and cafe accents",
            "abstract": "A small abstract with cafe and resume style words.",
            "journal_name": None,
            "venue": None,
            "year": 2023,
            "sources": [],
            "fields_of_study": [],
            "authors": [{"position": 0, "author_name": "Li Wang"}],
            "references": [],
        },
    }
    cluster_seeds = {"edge-1": {"edge-2": "require", "edge-3": "disallow"}}
    specter_embeddings = {"9101": [0.1, 0.2], "9102": [0.3, 0.4]}
    return signatures, papers, cluster_seeds, specter_embeddings


def _build_edge_case_dataset(
    *,
    tmp_path: Path | None,
) -> ANDData:
    signatures, papers, cluster_seeds, specter_embeddings = _edge_case_payload()
    signatures_arg: str | dict = signatures
    papers_arg: str | dict = papers
    if tmp_path is not None:
        signatures_path = tmp_path / "signatures.json"
        papers_path = tmp_path / "papers.json"
        signatures_path.write_text(json.dumps(signatures), encoding="utf-8")
        papers_path.write_text(json.dumps(papers), encoding="utf-8")
        signatures_arg = str(signatures_path)
        papers_arg = str(papers_path)

    return ANDData(
        signatures=signatures_arg,
        papers=papers_arg,
        cluster_seeds=cluster_seeds,
        specter_embeddings=specter_embeddings,
        altered_cluster_signatures=["edge-1"],
        name="edge_case_service_contract",
        mode="inference",
        block_type="s2",
        n_jobs=1,
        load_name_counts=_edge_case_name_counts(),
        name_tuples=set(),
        use_orcid_id=True,
    )


def _build_service_clusterer() -> Clusterer:
    return Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=_ConstantProbabilityClassifier(),
        n_jobs=1,
        use_cache=False,
        use_default_constraints_as_supervision=True,
    )


def _normalize_partition(clusters: dict[str, list[str]]) -> set[frozenset[str]]:
    return {frozenset(signature_ids) for signature_ids in clusters.values()}


def test_path_backed_inference_anddata_matches_service_lifecycle(monkeypatch):
    dataset, specter_embeddings = _build_service_shaped_dataset()

    assert dataset.signatures_path is not None
    assert dataset.papers_path is not None
    assert dataset.specter_embeddings is not None
    assert Path(dataset.signatures_path) == _DUMMY_DIR / "signatures.json"
    assert Path(dataset.papers_path) == _DUMMY_DIR / "papers.json"
    assert dataset.cluster_seeds_path is None
    assert dataset.specter_embeddings_path is None
    assert dataset.rust_lifecycle_policy.rust_build_path == "from_json_paths"
    assert dataset.rust_lifecycle_policy.skip_python_paper_preprocess is True
    assert dataset.cluster_seeds_require == {"1": 1, "2": 1}
    assert dataset.cluster_seeds_disallow == {("0", "1")}
    assert dataset.altered_cluster_signatures == ["1"]
    assert set(dataset.specter_embeddings) == set(specter_embeddings)
    assert all(paper.is_english is None for paper in dataset.papers.values())

    contract = build_rust_json_ingest_contract(
        dataset,
        name_counts_path=None,
        cluster_seed_require_value=0.0,
        cluster_seed_disallow_value=10000.0,
        num_threads=1,
    )
    assert contract.cluster_seeds_path is None
    assert contract.specter_embeddings == specter_embeddings
    assert contract.preprocess is True


def test_service_shaped_from_json_paths_feature_parity_with_dict_backed_inference(monkeypatch):
    dict_backed_dataset = _build_dict_backed_service_dataset()
    path_backed_dataset, _specter_embeddings = _build_service_shaped_dataset()
    runtime_context = build_runtime_context("service_json_ingest_feature_parity")

    dict_backed_featurizer = _get_rust_featurizer(dict_backed_dataset, runtime_context=runtime_context)
    path_backed_featurizer = _get_rust_featurizer(path_backed_dataset, runtime_context=runtime_context)

    for sig_id_1, sig_id_2 in (("0", "1"), ("0", "2"), ("1", "2")):
        dict_features = dict_backed_featurizer.featurize_pair(sig_id_1, sig_id_2)
        path_features = path_backed_featurizer.featurize_pair(sig_id_1, sig_id_2)
        assert len(dict_features) == len(path_features)
        for idx, (dict_value, path_value) in enumerate(zip(dict_features, path_features, strict=True)):
            assert equalish(dict_value, path_value), (
                f"feature mismatch pair=({sig_id_1}, {sig_id_2}) idx={idx} "
                f"dict_backed={dict_value} path_backed={path_value}"
            )


def test_service_shaped_from_json_paths_prediction_partition_parity_with_dict_backed_inference(monkeypatch):
    dict_backed_dataset = _build_dict_backed_service_dataset()
    path_backed_dataset, _specter_embeddings = _build_service_shaped_dataset()
    block_dict = {"a sattar": ["0", "1", "2"]}
    runtime_context = build_runtime_context("service_json_ingest_prediction_parity")

    dict_backed_featurizer = _get_rust_featurizer(dict_backed_dataset, runtime_context=runtime_context)
    path_backed_featurizer = _get_rust_featurizer(path_backed_dataset, runtime_context=runtime_context)

    dict_backed_clusters, _ = _build_service_clusterer().predict_from_rust_featurizer(
        block_dict,
        dict_backed_featurizer,
        cluster_model_params={"eps": 0.5},
        runtime_context=runtime_context,
        cluster_seeds_require=dict_backed_dataset.cluster_seeds_require,
        cluster_seeds_disallow=dict_backed_dataset.cluster_seeds_disallow,
    )
    path_backed_clusters, _ = _build_service_clusterer().predict_from_rust_featurizer(
        block_dict,
        path_backed_featurizer,
        cluster_model_params={"eps": 0.5},
        runtime_context=runtime_context,
        cluster_seeds_require=path_backed_dataset.cluster_seeds_require,
        cluster_seeds_disallow=path_backed_dataset.cluster_seeds_disallow,
    )

    assert _normalize_partition(path_backed_clusters) == _normalize_partition(dict_backed_clusters)


def test_service_shaped_edge_case_from_json_paths_feature_parity_with_dict_backed_inference(
    monkeypatch,
    tmp_path,
):
    dict_backed_dataset = _build_edge_case_dataset(tmp_path=None)
    path_backed_dataset = _build_edge_case_dataset(tmp_path=tmp_path)
    runtime_context = build_runtime_context("service_json_ingest_edge_feature_parity")

    dict_backed_featurizer = _get_rust_featurizer(dict_backed_dataset, runtime_context=runtime_context)
    path_backed_featurizer = _get_rust_featurizer(path_backed_dataset, runtime_context=runtime_context)
    telemetry = path_backed_featurizer.json_ingest_telemetry()
    assert telemetry is not None
    assert telemetry["counts"]["missing_specter_paper_count"] >= 1

    for sig_id_1, sig_id_2 in (("edge-1", "edge-2"), ("edge-1", "edge-3"), ("edge-2", "edge-3")):
        dict_features = dict_backed_featurizer.featurize_pair(sig_id_1, sig_id_2)
        path_features = path_backed_featurizer.featurize_pair(sig_id_1, sig_id_2)
        assert len(dict_features) == len(path_features)
        for idx, (dict_value, path_value) in enumerate(zip(dict_features, path_features, strict=True)):
            assert equalish(dict_value, path_value), (
                f"edge feature mismatch pair=({sig_id_1}, {sig_id_2}) idx={idx} "
                f"dict_backed={dict_value} path_backed={path_value}"
            )


def test_service_shaped_from_json_paths_reports_partial_specter_embeddings(monkeypatch):
    partial_specter_embeddings = {"53235312": [0.1, 0.2]}
    dataset, _specter_embeddings = _build_service_shaped_dataset(specter_embeddings=partial_specter_embeddings)
    runtime_context = build_runtime_context("service_json_ingest_partial_specter")

    rust_featurizer = _get_rust_featurizer(dataset, runtime_context=runtime_context)
    telemetry = rust_featurizer.json_ingest_telemetry()
    assert telemetry is not None
    assert telemetry["counts"]["missing_specter_paper_count"] >= 1

    feature_names = FeaturizationInfo().get_feature_names()
    specter_index = feature_names.index("specter_cosine_sim")
    features = rust_featurizer.featurize_pair("0", "1")
    assert math.isnan(float(features[specter_index]))


def test_path_backed_inference_dict_cluster_seeds_sync_to_rust_constraints(monkeypatch):
    dataset, _specter_embeddings = _build_service_shaped_dataset()
    runtime_context = build_runtime_context("service_json_ingest_contract_test")

    require_pair = ("1", "2")
    disallow_pair = ("0", "1")
    assert dataset.get_constraint(*require_pair) == 0
    assert dataset.get_constraint(*disallow_pair) == 10000

    _sync_rust_cluster_seeds(dataset, runtime_context=runtime_context)

    assert get_constraint_rust(dataset, *require_pair, runtime_context=runtime_context) == dataset.get_constraint(
        *require_pair
    )
    assert get_constraint_rust(dataset, *disallow_pair, runtime_context=runtime_context) == dataset.get_constraint(
        *disallow_pair
    )
