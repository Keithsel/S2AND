from __future__ import annotations

import importlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

import s2and.incremental_linking.feature_block_arrow as feature_block_arrow_module
from s2and.data import ANDData, NameCounts
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.feature_block import (
    RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS,
    FeatureBlockSignatureOrder,
    arrow_ipc_physical_layout,
    cluster_seed_disallows_from_arrow_paths,
    feature_block_for_signature_order,
    feature_block_from_anddata,
    feature_block_from_arrow_paths,
    feature_block_from_raw_payloads,
    feature_block_signature_order_from_raw_candidate_plan,
    feature_block_to_mini_anddata,
    raw_planner_arrow_physical_layout,
    write_arrow_ipc_table,
    write_feature_block_arrow_from_anddata,
    write_feature_block_arrow_tables,
    write_name_counts_arrow,
    write_name_counts_index,
    write_name_pairs_arrow,
    write_raw_arrow_batch_lookup_indexes,
)
from s2and.incremental_linking.features import LinkerFeatureMatrix
from s2and.incremental_linking.query_adapter import (
    build_cluster_summary,
    build_cluster_summary_from_feature_block,
    build_name_count_rarity_row_signals,
    extract_query_features,
    extract_query_features_from_feature_block,
)
from s2and.incremental_linking.retrieval import build_linker_retrieval_batch_from_raw_candidate_plan
from s2and.incremental_linking.runtime import (
    CandidateBatchPairwiseModelResult,
    LinkOrAbstainCompactResult,
    LinkOrAbstainProductionResult,
    LinkOrAbstainRetrievedCandidatesResult,
    predict_incremental_link_or_abstain_from_raw_arrow_paths,
    predict_incremental_link_or_abstain_from_raw_feature_block,
    predict_incremental_link_or_abstain_from_raw_payloads,
)


def test_feature_block_public_exports_remain_importable() -> None:
    module = importlib.import_module("s2and.incremental_linking.feature_block")
    expected_names = {
        "RAW_PLANNER_ARROW_BATCH_INDEX_KEYS",
        "RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS",
        "TemporaryArrowPaths",
        "FeatureBlock",
        "FeatureBlockPaper",
        "FeatureBlockPaperAuthor",
        "FeatureBlockSignature",
        "FeatureBlockSignatureOrder",
        "arrow_ipc_physical_layout",
        "arrow_paths_with_temporary_cluster_seeds",
        "cluster_seed_disallows_from_arrow_paths",
        "feature_block_for_signature_order",
        "feature_block_from_anddata",
        "feature_block_from_arrow_paths",
        "feature_block_from_raw_payloads",
        "feature_block_signature_order_from_raw_candidate_plan",
        "feature_block_to_mini_anddata",
        "normalize_cluster_seed_disallow_pairs",
        "raw_planner_arrow_physical_layout",
        "read_cluster_seed_disallows_arrow",
        "write_arrow_batch_lookup_index",
        "write_arrow_ipc_table",
        "write_cluster_seed_disallows_arrow",
        "write_cluster_seeds_arrow",
        "write_feature_block_arrow_from_anddata",
        "write_feature_block_arrow_tables",
        "write_name_counts_arrow",
        "write_name_counts_index",
        "write_name_pairs_arrow",
        "write_raw_arrow_batch_lookup_indexes",
    }
    missing = sorted(name for name in expected_names if not hasattr(module, name))
    assert missing == []


def _signature_payload(
    signature_id: str,
    paper_id: str,
    *,
    first: str,
    last: str,
    position: int,
    orcid: str | None = None,
) -> dict[str, object]:
    author_info: dict[str, object] = {
        "first": first,
        "middle": "",
        "last": last,
        "suffix": "",
        "affiliations": ["Analytical Engine Lab"],
        "email": "",
        "position": position,
        "block": f"{first[:1].lower()} {last.lower()}",
        "source_ids": [orcid] if orcid else [],
    }
    if orcid is not None:
        author_info["source_id_source"] = "ORCID"
    return {
        "signature_id": signature_id,
        "paper_id": paper_id,
        "author_info": author_info,
        "sourced_author_ids": [f"source-{signature_id}"],
    }


def _paper_payload(paper_id: str, *, title: str, year: int, authors: list[str]) -> dict[str, object]:
    return {
        "paper_id": paper_id,
        "title": title,
        "abstract": "",
        "venue": "Royal Society",
        "journal_name": "",
        "year": year,
        "references": [],
        "authors": [{"position": index, "author_name": name} for index, name in enumerate(authors)],
    }


def _tiny_anddata() -> ANDData:
    dataset = ANDData(
        signatures={
            "q": _signature_payload(
                "q",
                "p_q",
                first="Ada",
                last="Lovelace",
                position=0,
                orcid="0000-0000-0000-0001",
            ),
            "s1": _signature_payload("s1", "p1", first="Ada", last="Lovelace", position=0),
            "s2": _signature_payload("s2", "p2", first="Grace", last="Hopper", position=0),
        },
        papers={
            "p_q": _paper_payload("p_q", title="Notes", year=1843, authors=["Ada Lovelace", "Charles Babbage"]),
            "p1": _paper_payload("p1", title="Notes", year=1843, authors=["Ada Lovelace", "Charles Babbage"]),
            "p2": _paper_payload("p2", title="Compiler", year=1952, authors=["Grace Hopper"]),
        },
        name="tiny_feature_block",
        mode="inference",
        load_name_counts=False,
        preprocess=False,
        name_tuples=set(),
    )
    dataset.cluster_seeds_require = {"s1": "c_ada", "s2": "c_grace"}
    dataset.cluster_seeds_disallow = {("q", "s2")}
    dataset.signatures["q"] = dataset.signatures["q"]._replace(
        author_info_name_counts=NameCounts(first=10.0, last=20.0, first_last=5.0, last_first_initial=8.0)
    )
    dataset.specter_embeddings = {
        "p_q": np.asarray([1.0, 0.0], dtype=np.float32),
        "p1": np.asarray([1.0, 0.1], dtype=np.float32),
    }
    return dataset


def _raw_plan() -> dict[str, object]:
    row_count = 2
    pair_count = 3
    plan: dict[str, object] = {
        "row_count": row_count,
        "pair_count": pair_count,
        "query_signature_ids": ["q"],
        "query_views": ["full"],
        "row_query_signature_indices": np.asarray([0, 0], dtype=np.uint32),
        "left_signature_ids": ["q", "q", "q"],
        "right_signature_ids": ["s1", "s2", "s3"],
        "pair_row_indices": np.asarray([0, 1, 1], dtype=np.uint32),
        "row_component_keys": ["c_ada", "c_other"],
        "retrieval_scores": np.asarray([0.9, 0.2], dtype=np.float32),
        "retrieval_ranks": np.asarray([1, 2], dtype=np.uint16),
        "row_component_sizes": np.asarray([1, 2], dtype=np.float32),
        "row_named_signature_counts": np.asarray([1, 2], dtype=np.float32),
        "row_dominant_first_names": np.asarray(["ada", "grace"], dtype=object),
        "row_candidate_year_min": np.asarray([1843, 1952], dtype=np.int32),
        "row_candidate_year_max": np.asarray([1843, 1952], dtype=np.int32),
        "row_candidate_year_range_missing": np.asarray([0, 0], dtype=np.uint8),
        "row_query_first_tokens": np.asarray(["ada", "ada"], dtype=object),
        "row_query_years": np.asarray([1843, 1843], dtype=np.int32),
        "row_query_year_missing": np.asarray([0, 0], dtype=np.uint8),
        "row_query_has_affiliations": np.asarray([1, 1], dtype=np.float32),
        "row_query_has_coauthors": np.asarray([1, 1], dtype=np.float32),
        "row_orcid_match": np.asarray([0, 0], dtype=np.float32),
        "middle_initial_compatibility": np.asarray([1, 0], dtype=np.float32),
        "affiliation_overlap": np.asarray([1, 0], dtype=np.float32),
        "coauthor_overlap": np.asarray([1, 0], dtype=np.float32),
        "venue_overlap": np.asarray([1, 0], dtype=np.float32),
        "year_compatibility": np.asarray([1, 0], dtype=np.float32),
        "title_overlap": np.asarray([1, 0], dtype=np.float32),
        "specter_centroid_similarity": np.asarray([1, 0], dtype=np.float32),
        "specter_exemplar_similarity": np.asarray([1, 0], dtype=np.float32),
        "row_last_name_count_min_rarity": np.asarray([0.1, 0.2], dtype=np.float32),
        "row_candidate_last_name_count_min_rarity": np.asarray([0.1, 0.2], dtype=np.float32),
        "row_candidate_last_first_name_count_min_rarity": np.asarray([0.1, 0.2], dtype=np.float32),
        "row_last_first_name_count_min_rarity": np.asarray([0.1, 0.2], dtype=np.float32),
        "row_first_prefix_x_last_first_name_count_min_rarity": np.asarray([0.1, 0.0], dtype=np.float32),
        "row_candidate_cluster_max_paper_author_count": np.asarray([2, 2], dtype=np.float32),
        "row_paper_author_list_max_jaccard": np.asarray([1, 0.2], dtype=np.float32),
        "row_paper_author_list_max_containment": np.asarray([1, 0.5], dtype=np.float32),
        "row_paper_author_list_max_overlap_count": np.asarray([2, 1], dtype=np.float32),
        "row_local_author_window10_jaccard_max": np.asarray([1, 0], dtype=np.float32),
        "row_local_author_window10_overlap_count_max": np.asarray([1, 0], dtype=np.float32),
        "row_best_author_count_log_absdiff": np.asarray([0, 0], dtype=np.float32),
        "query_authors": ["ada lovelace"],
        "component_members": {"c_ada": ["s1"], "c_other": ["s2", "s3"]},
    }
    return plan


def _raw_payloads_for_plan() -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]], dict[str, str]]:
    signatures = {
        "q": _signature_payload(
            "q",
            "p_q",
            first="Ada",
            last="Lovelace",
            position=0,
            orcid="0000-0000-0000-0001",
        ),
        "s1": _signature_payload("s1", "p1", first="Ada", last="Lovelace", position=0),
        "s2": _signature_payload("s2", "p2", first="Grace", last="Hopper", position=0),
        "s3": _signature_payload("s3", "p3", first="Grace", last="Hopper", position=1),
        "unused": _signature_payload("unused", "p_unused", first="Unused", last="Person", position=0),
    }
    papers = {
        "p_q": _paper_payload("p_q", title="Notes", year=1843, authors=["Ada Lovelace", "Charles Babbage"]),
        "p1": _paper_payload("p1", title="Notes", year=1843, authors=["Ada Lovelace", "Charles Babbage"]),
        "p2": _paper_payload("p2", title="Compiler", year=1952, authors=["Grace Hopper", "Jane Doe"]),
        "p3": _paper_payload("p3", title="Compiler", year=1952, authors=["Jane Doe", "Grace Hopper"]),
        "p_unused": _paper_payload("p_unused", title="Unused", year=2000, authors=["Unused Person"]),
    }
    cluster_seeds_require = {"s1": "c_ada", "s2": "c_other", "s3": "c_other", "unused": "c_unused"}
    return signatures, papers, cluster_seeds_require


def _write_feature_block_arrow_paths(tmp_path: Path) -> dict[str, str]:
    signatures, papers, cluster_seeds_require = _raw_payloads_for_plan()
    feature_block = feature_block_from_raw_payloads(
        signatures=signatures,
        papers=papers,
        raw_candidate_plan=_raw_plan(),
        cluster_seeds_require=cluster_seeds_require,
        cluster_seeds_disallow=[("q", "s2"), ("q", "unused")],
    )
    return write_feature_block_arrow_tables(feature_block, tmp_path, include_empty_cluster_seeds=True)


def test_feature_block_from_anddata_builds_requested_mini_contract() -> None:
    feature_block = feature_block_from_anddata(
        _tiny_anddata(),
        signature_ids=["q", "s1"],
        query_signature_ids=["q"],
    )

    assert feature_block.signature_ids == ("q", "s1")
    assert feature_block.query_signature_ids == ("q",)
    assert feature_block.signature_id_to_index == {"q": 0, "s1": 1}
    assert feature_block.cluster_seeds_require == (("s1", "c_ada"),)
    assert feature_block.cluster_seeds_disallow == ()
    assert feature_block.seed_component_members == {"c_ada": ("s1",)}
    assert [paper.paper_id for paper in feature_block.papers] == ["p_q", "p1"]
    assert [(row.paper_id, row.position, row.author_name) for row in feature_block.paper_authors] == [
        ("p_q", 0, "ada lovelace"),
        ("p_q", 1, "charles babbage"),
        ("p1", 0, "ada lovelace"),
        ("p1", 1, "charles babbage"),
    ]
    assert feature_block.signatures[0].author_orcid == "0000-0000-0000-0001"
    assert feature_block.specter_paper_ids == ("p_q", "p1")
    np.testing.assert_allclose(feature_block.specter_embeddings, [[1.0, 0.0], [1.0, 0.1]])


def test_feature_block_to_arrow_tables_matches_raw_schema() -> None:
    pa = pytest.importorskip("pyarrow")

    tables = feature_block_from_anddata(
        _tiny_anddata(), signature_ids=["q", "s1"], query_signature_ids=["q"]
    ).to_arrow_tables()

    assert set(tables) == {
        "signatures",
        "papers",
        "paper_authors",
        "cluster_seeds",
        "cluster_seed_disallows",
        "specter",
    }
    assert tables["signatures"].column_names == [
        "signature_id",
        "paper_id",
        "author_first",
        "author_middle",
        "author_last",
        "author_suffix",
        "author_affiliations",
        "author_orcid",
        "author_position",
        "author_block",
        "author_email",
        "source_author_ids",
    ]
    assert tables["signatures"].schema.field("author_suffix").type == pa.string()
    assert tables["papers"].schema.field("abstract").type == pa.string()
    assert tables["papers"].schema.field("predicted_language").type == pa.string()
    assert tables["papers"].schema.field("is_reliable").type == pa.bool_()
    assert tables["cluster_seeds"].to_pydict() == {"signature_id": ["s1"], "cluster_id": ["c_ada"]}
    assert tables["cluster_seed_disallows"].to_pydict() == {"signature_id_1": [], "signature_id_2": []}


def test_feature_block_to_arrow_tables_keeps_all_null_optional_columns_typed() -> None:
    pa = pytest.importorskip("pyarrow")

    tables = feature_block_from_raw_payloads(
        signatures={
            "q": {
                "signature_id": "q",
                "paper_id": "p_q",
                "author_info": {
                    "first": "Ada",
                    "middle": None,
                    "last": "Lovelace",
                    "suffix": None,
                    "affiliations": [],
                    "position": 0,
                },
            }
        },
        papers={"p_q": _paper_payload("p_q", title="", year=1843, authors=["Ada Lovelace"])},
        raw_candidate_plan={**_raw_plan(), "left_signature_ids": ["q"], "right_signature_ids": ["q"]},
        cluster_seeds_require={},
    ).to_arrow_tables()

    assert tables["signatures"].schema.field("author_suffix").type == pa.string()
    assert tables["signatures"].schema.field("author_email").type == pa.string()
    assert tables["papers"].schema.field("predicted_language").type == pa.string()
    assert tables["papers"].schema.field("is_reliable").type == pa.bool_()
    assert tables["cluster_seeds"].schema.field("signature_id").type == pa.string()
    assert tables["cluster_seed_disallows"].schema.field("signature_id_1").type == pa.string()


def test_write_feature_block_arrow_from_anddata_skips_empty_seed_table(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")

    paths = write_feature_block_arrow_from_anddata(
        _tiny_anddata(),
        tmp_path,
        signature_ids=["q"],
        query_signature_ids=["q"],
        include_specter=False,
    )

    assert set(paths) == {"signatures", "papers", "paper_authors"}
    with pa.memory_map(paths["signatures"], "r") as source:
        signatures = pa.ipc.open_file(source).read_all()
    assert "name_count_first" not in signatures.column_names
    assert signatures.to_pydict()["signature_id"] == ["q"]


def test_feature_block_from_arrow_paths_reads_cluster_seed_disallows(tmp_path: Path) -> None:
    arrow_paths = _write_feature_block_arrow_paths(tmp_path)

    feature_block = feature_block_from_arrow_paths(arrow_paths, raw_candidate_plan=_raw_plan())

    assert feature_block.cluster_seeds_disallow == (("q", "s2"),)


def test_feature_block_from_arrow_paths_rejects_invalid_cluster_seed_disallows(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    arrow_paths = _write_feature_block_arrow_paths(tmp_path)
    invalid = pa.table(
        {
            "signature_id_1": pa.array(["q", "s2"], type=pa.string()),
            "signature_id_2": pa.array(["s2", "q"], type=pa.string()),
        }
    )
    write_arrow_ipc_table(invalid, Path(arrow_paths["cluster_seed_disallows"]))

    with pytest.raises(ValueError, match="duplicated"):
        feature_block_from_arrow_paths(arrow_paths, raw_candidate_plan=_raw_plan())


def test_cluster_seed_disallows_from_arrow_paths_rejects_missing_explicit_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing_cluster_seed_disallows.arrow"

    with pytest.raises(FileNotFoundError, match="cluster_seed_disallows"):
        cluster_seed_disallows_from_arrow_paths({"cluster_seed_disallows": str(missing_path)})


def test_feature_block_from_arrow_paths_rejects_duplicate_signature_rows(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    arrow_paths = _write_feature_block_arrow_paths(tmp_path)
    duplicate_signatures = pa.table(
        {
            "signature_id": pa.array(["q", "q"], type=pa.string()),
            "paper_id": pa.array(["p_q", "p_q"], type=pa.string()),
        }
    )
    write_arrow_ipc_table(duplicate_signatures, Path(arrow_paths["signatures"]))

    with pytest.raises(ValueError, match="duplicate signature_id"):
        feature_block_from_arrow_paths(arrow_paths, raw_candidate_plan=_raw_plan())


def test_feature_block_from_arrow_paths_rejects_missing_signature_paper(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    arrow_paths = _write_feature_block_arrow_paths(tmp_path)
    incomplete_papers = pa.table({"paper_id": pa.array(["p_q", "p1", "p2"], type=pa.string())})
    write_arrow_ipc_table(incomplete_papers, Path(arrow_paths["papers"]))

    with pytest.raises(ValueError, match="missing signature paper_ids"):
        feature_block_from_arrow_paths(arrow_paths, raw_candidate_plan=_raw_plan())


def test_feature_block_from_arrow_paths_rejects_duplicate_paper_author_positions(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    arrow_paths = _write_feature_block_arrow_paths(tmp_path)
    duplicate_authors = pa.table(
        {
            "paper_id": pa.array(["p_q", "p_q"], type=pa.string()),
            "position": pa.array([0, 0], type=pa.int64()),
            "author_name": pa.array(["Ada Lovelace", "A. Lovelace"], type=pa.string()),
        }
    )
    write_arrow_ipc_table(duplicate_authors, Path(arrow_paths["paper_authors"]))

    with pytest.raises(ValueError, match="duplicate"):
        feature_block_from_arrow_paths(arrow_paths, raw_candidate_plan=_raw_plan())


def test_feature_block_from_arrow_paths_rejects_null_paper_author_position(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    arrow_paths = _write_feature_block_arrow_paths(tmp_path)
    null_position_authors = pa.table(
        {
            "paper_id": pa.array(["p_q"], type=pa.string()),
            "position": pa.array([None], type=pa.int64()),
            "author_name": pa.array(["Ada Lovelace"], type=pa.string()),
        }
    )
    write_arrow_ipc_table(null_position_authors, Path(arrow_paths["paper_authors"]))

    with pytest.raises(ValueError, match="null position"):
        feature_block_from_arrow_paths(arrow_paths, raw_candidate_plan=_raw_plan())


def test_feature_block_from_arrow_paths_rejects_malformed_optional_scalars(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    arrow_paths = _write_feature_block_arrow_paths(tmp_path)
    malformed_papers = pa.table(
        {
            "paper_id": pa.array(["p_q", "p1", "p2", "p3"], type=pa.string()),
            "title": pa.array(["Notes", "Notes", "Compiler", "Compiler"], type=pa.string()),
            "venue": pa.array(["Royal Society", "Royal Society", "", ""], type=pa.string()),
            "journal_name": pa.array(["", "", "", ""], type=pa.string()),
            "year": pa.array(["1843", "20xx", "1952", "1952"], type=pa.string()),
        }
    )
    write_arrow_ipc_table(malformed_papers, Path(arrow_paths["papers"]))

    with pytest.raises(ValueError, match="papers.year"):
        feature_block_from_arrow_paths(arrow_paths, raw_candidate_plan=_raw_plan())


def test_feature_block_from_arrow_paths_reads_specter_when_requested(tmp_path: Path) -> None:
    signatures, papers, cluster_seeds_require = _raw_payloads_for_plan()
    feature_block = feature_block_from_raw_payloads(
        signatures=signatures,
        papers=papers,
        raw_candidate_plan=_raw_plan(),
        cluster_seeds_require=cluster_seeds_require,
        specter_embeddings={
            "p_q": np.asarray([1.0, 0.0], dtype=np.float32),
            "p1": np.asarray([0.5, 0.5], dtype=np.float32),
            "p_unused": np.asarray([0.0, 1.0], dtype=np.float32),
        },
    )
    arrow_paths = write_feature_block_arrow_tables(feature_block, tmp_path, include_empty_cluster_seeds=True)

    materialized = feature_block_from_arrow_paths(arrow_paths, raw_candidate_plan=_raw_plan(), include_specter=True)

    assert materialized.specter_paper_ids == ("p_q", "p1")
    np.testing.assert_allclose(materialized.specter_embeddings, [[1.0, 0.0], [0.5, 0.5]])


def test_write_arrow_ipc_table_writes_bounded_record_batches(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")

    table = pa.table({"signature_id": pa.array([str(index) for index in range(5)], type=pa.string())})
    path = write_arrow_ipc_table(table, tmp_path / "signatures.arrow", max_record_batch_rows=2)

    assert arrow_ipc_physical_layout(path) == {
        "row_count": 5,
        "record_batch_count": 3,
        "actual_max_batch_rows": 2,
    }


def test_raw_planner_index_rejects_unbounded_large_batch(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")

    table = pa.table({"signature_id": pa.array([str(index) for index in range(5)], type=pa.string())})
    path = write_arrow_ipc_table(table, tmp_path / "signatures.arrow")

    with pytest.raises(ValueError, match="exceeding the raw-planner limit of 2"):
        write_raw_arrow_batch_lookup_indexes(
            {"signatures": path},
            tmp_path,
            max_record_batch_rows={"signatures": 2},
        )


def test_raw_planner_index_metadata_uses_stem_qualified_sidecar(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")

    table = pa.table({"signature_id": pa.array([str(index) for index in range(5)], type=pa.string())})
    path = write_arrow_ipc_table(table, tmp_path / "signatures.arrow", max_record_batch_rows=2)
    indexed_paths, index_metrics = write_raw_arrow_batch_lookup_indexes(
        {"signatures": path},
        tmp_path,
        max_record_batch_rows={"signatures": 2},
    )
    layout = raw_planner_arrow_physical_layout(indexed_paths, max_record_batch_rows={"signatures": 2})

    assert Path(indexed_paths["signatures_batch_index"]).name == "signatures.signatures_batch_index.bin"
    assert index_metrics["signatures_batch_index"]["record_batch_count"] == 3
    assert index_metrics["signatures_batch_index"]["actual_max_batch_rows"] == 2
    assert layout["schema"] == "s2and_arrow_physical_v1"
    assert layout["tables"]["signatures"]["batch_index_present"] is True
    assert layout["tables"]["signatures"]["max_record_batch_rows"] == 2
    assert RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS["signatures"] == 16_384


def test_raw_planner_index_rejects_stale_python_reuse(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    path = write_arrow_ipc_table(
        pa.table({"signature_id": pa.array(["s1", "s2"], type=pa.string())}),
        tmp_path / "signatures.arrow",
    )
    write_raw_arrow_batch_lookup_indexes({"signatures": path}, tmp_path)
    write_arrow_ipc_table(
        pa.table({"signature_id": pa.array([f"s{index}" for index in range(200)], type=pa.string())}),
        path,
    )

    with pytest.raises(ValueError, match="stale"):
        write_raw_arrow_batch_lookup_indexes({"signatures": path}, tmp_path, overwrite=False)


def test_write_name_artifacts_arrow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pa = pytest.importorskip("pyarrow")
    import s2and.data as data_module

    monkeypatch.setattr(
        data_module,
        "_load_name_counts_cached",
        lambda: ({"ada": 3}, {"lovelace": 5}, {"ada lovelace": 2}, {"lovelace a": 7}),
    )

    counts_path, counts_metrics = write_name_counts_arrow(tmp_path)
    index_path, index_metrics = write_name_counts_index(tmp_path)
    pairs_path, pairs_metrics = write_name_pairs_arrow({("ada", "a"), ("charles", "c")}, tmp_path)

    assert counts_metrics == {
        "reused": False,
        "first_count": 1,
        "last_count": 1,
        "first_last_count": 1,
        "last_first_initial_count": 1,
        "row_count": 4,
    }
    assert index_metrics["reused"] is False
    assert index_metrics["row_count"] == 4
    assert index_metrics["first_count"] == 1
    assert pairs_metrics == {"reused": False, "row_count": 2}
    with pa.memory_map(counts_path, "r") as source:
        counts = pa.ipc.open_file(source).read_all().to_pylist()
    with pa.memory_map(pairs_path, "r") as source:
        pairs = pa.ipc.open_file(source).read_all().to_pylist()
    manifest = json.loads((Path(index_path) / "manifest.json").read_text(encoding="utf-8"))
    first_path = Path(index_path) / manifest["files"]["first"]["path"]
    header = first_path.read_bytes()[:32]
    magic, record_count, blob_offset, blob_len = struct.unpack("<8sQQQ", header)
    assert manifest["schema_version"] == "name_counts_index_v1"
    assert manifest["exact_string_verification"] is True
    assert manifest["files"]["first"]["path"].startswith("generations/")
    assert magic == b"S2NCI001"
    assert record_count == 1
    assert blob_offset == 72
    assert blob_len == 3
    assert {"kind": "first", "name": "ada", "count": 3.0} in counts
    assert pairs == [{"name_1": "ada", "name_2": "a"}, {"name_1": "charles", "name_2": "c"}]
    assert write_name_counts_arrow(tmp_path)[1] == {"reused": True}
    assert write_name_counts_index(tmp_path)[1] == {"reused": True}
    assert write_name_pairs_arrow({("ada", "a")}, tmp_path)[1] == {"reused": True}


def test_write_name_counts_index_does_not_reuse_legacy_direct_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import s2and.data as data_module

    legacy_index_dir = tmp_path / "name_counts_index"
    legacy_index_dir.mkdir()
    for filename in ("first.bin", "last.bin", "first_last.bin", "last_first_initial.bin"):
        (legacy_index_dir / filename).write_bytes(b"legacy")
    monkeypatch.setattr(
        data_module,
        "_load_name_counts_cached",
        lambda: ({"ada": 3}, {"lovelace": 5}, {"ada lovelace": 2}, {"lovelace a": 7}),
    )

    index_path, index_metrics = write_name_counts_index(tmp_path)

    assert index_metrics["reused"] is False
    manifest = json.loads((Path(index_path) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"]["first"]["path"].startswith("generations/")


def test_write_name_counts_index_failed_overwrite_keeps_previous_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import s2and.data as data_module

    monkeypatch.setattr(
        data_module,
        "_load_name_counts_cached",
        lambda: ({"ada": 3}, {"lovelace": 5}, {"ada lovelace": 2}, {"lovelace a": 7}),
    )
    index_path, _metrics = write_name_counts_index(tmp_path)
    manifest_path = Path(index_path) / "manifest.json"
    original_manifest = manifest_path.read_text(encoding="utf-8")
    original_first_path = Path(index_path) / json.loads(original_manifest)["files"]["first"]["path"]

    real_write_file = feature_block_arrow_module._write_name_count_index_file  # noqa: SLF001

    def fail_after_first_file(path: Path, kind: str, mapping: object) -> dict[str, int]:
        if kind == "last":
            raise RuntimeError("simulated index write failure")
        return real_write_file(path, kind, mapping)

    monkeypatch.setattr(feature_block_arrow_module, "_write_name_count_index_file", fail_after_first_file)
    monkeypatch.setattr(
        data_module,
        "_load_name_counts_cached",
        lambda: ({"alan": 11}, {"turing": 13}, {"alan turing": 17}, {"turing a": 19}),
    )

    with pytest.raises(RuntimeError, match="simulated index write failure"):
        write_name_counts_index(tmp_path, overwrite=True)

    assert manifest_path.read_text(encoding="utf-8") == original_manifest
    assert original_first_path.exists()


def test_feature_block_to_mini_anddata_materializes_only_requested_rows() -> None:
    feature_block = feature_block_from_anddata(
        _tiny_anddata(),
        signature_ids=["q", "s1"],
        query_signature_ids=["q"],
    )

    mini = feature_block_to_mini_anddata(
        feature_block,
        name="mini_feature_block_test",
        name_tuples=set(),
    )

    assert tuple(mini.signatures) == ("q", "s1")
    assert set(mini.papers) == {"p_q", "p1"}
    assert mini.cluster_seeds_require == {"s1": "c_ada"}
    assert mini.cluster_seeds_disallow == set()
    assert mini.signatures["q"].author_info_orcid == "0000000000000001"
    np.testing.assert_allclose(mini.specter_embeddings["p1"], [1.0, 0.1])


def test_feature_block_from_raw_payloads_uses_raw_plan_mini_order() -> None:
    signatures, papers, cluster_seeds_require = _raw_payloads_for_plan()

    feature_block = feature_block_from_raw_payloads(
        signatures=signatures,
        papers=papers,
        raw_candidate_plan=_raw_plan(),
        cluster_seeds_require=cluster_seeds_require,
        cluster_seeds_disallow=[("q", "s2"), ("q", "unused")],
        specter_embeddings={
            "p_q": np.asarray([1.0, 0.0], dtype=np.float32),
            "p1": np.asarray([1.0, 0.1], dtype=np.float32),
            "p_unused": np.asarray([0.0, 1.0], dtype=np.float32),
        },
    )

    assert feature_block.signature_ids == ("q", "s1", "s2", "s3")
    assert feature_block.query_signature_ids == ("q",)
    assert feature_block.cluster_seeds_require == (("s1", "c_ada"), ("s2", "c_other"), ("s3", "c_other"))
    assert feature_block.cluster_seeds_disallow == (("q", "s2"),)
    assert [paper.paper_id for paper in feature_block.papers] == ["p_q", "p1", "p2", "p3"]
    assert feature_block.specter_paper_ids == ("p_q", "p1")


def test_feature_block_from_raw_payloads_rejects_missing_papers() -> None:
    signatures, papers, cluster_seeds_require = _raw_payloads_for_plan()
    del papers["p3"]

    with pytest.raises(ValueError, match="missing signature paper_ids"):
        feature_block_from_raw_payloads(
            signatures=signatures,
            papers=papers,
            raw_candidate_plan=_raw_plan(),
            cluster_seeds_require=cluster_seeds_require,
        )


def test_feature_block_from_raw_payloads_rejects_malformed_optional_scalars() -> None:
    signatures, papers, cluster_seeds_require = _raw_payloads_for_plan()
    signatures["s3"]["author_info"]["position"] = "later"

    with pytest.raises(ValueError, match="signatures.author_info.position"):
        feature_block_from_raw_payloads(
            signatures=signatures,
            papers=papers,
            raw_candidate_plan=_raw_plan(),
            cluster_seeds_require=cluster_seeds_require,
        )

    signatures, papers, cluster_seeds_require = _raw_payloads_for_plan()
    papers["p1"]["is_reliable"] = "maybe"

    with pytest.raises(ValueError, match="papers.is_reliable"):
        feature_block_from_raw_payloads(
            signatures=signatures,
            papers=papers,
            raw_candidate_plan=_raw_plan(),
            cluster_seeds_require=cluster_seeds_require,
        )


def test_feature_block_from_raw_payloads_accepts_anddata_specter_tuple_payload() -> None:
    signatures, papers, cluster_seeds_require = _raw_payloads_for_plan()
    specter_matrix = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.1],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    specter_keys = ["p_q", "p1", "p_unused"]

    feature_block = feature_block_from_raw_payloads(
        signatures=signatures,
        papers=papers,
        raw_candidate_plan=_raw_plan(),
        cluster_seeds_require=cluster_seeds_require,
        specter_embeddings=(specter_matrix, specter_keys),
    )

    assert feature_block.specter_paper_ids == ("p_q", "p1")
    np.testing.assert_allclose(feature_block.specter_embeddings, [[1.0, 0.0], [1.0, 0.1]])


def test_feature_block_to_mini_anddata_preserves_abstract_presence_for_scoring() -> None:
    signatures, papers, cluster_seeds_require = _raw_payloads_for_plan()
    papers["p1"]["abstract"] = "Has Abstract"

    feature_block = feature_block_from_raw_payloads(
        signatures=signatures,
        papers=papers,
        raw_candidate_plan=_raw_plan(),
        cluster_seeds_require=cluster_seeds_require,
    )
    mini = feature_block_to_mini_anddata(
        feature_block,
        name="mini_feature_block_abstract_test",
        load_name_counts=False,
        name_tuples=set(),
    )

    assert mini.papers["p1"].has_abstract is True


def test_feature_block_for_signature_order_rejects_missing_plan_signature() -> None:
    feature_block = feature_block_from_anddata(
        _tiny_anddata(),
        signature_ids=["q", "s1"],
        query_signature_ids=["q"],
    )
    order = feature_block_signature_order_from_raw_candidate_plan(_raw_plan())

    with pytest.raises(ValueError, match="missing raw-plan signatures"):
        feature_block_for_signature_order(feature_block, order)


def test_feature_block_signature_order_from_raw_candidate_plan_queries_first() -> None:
    order = feature_block_signature_order_from_raw_candidate_plan(_raw_plan())

    assert order == FeatureBlockSignatureOrder(signature_ids=("q", "s1", "s2", "s3"), query_signature_ids=("q",))
    assert order.signature_id_to_index == {"q": 0, "s1": 1, "s2": 2, "s3": 3}


def test_raw_candidate_plan_bridge_accepts_feature_block_signature_order() -> None:
    order = feature_block_signature_order_from_raw_candidate_plan(_raw_plan())

    retrieval_batch = build_linker_retrieval_batch_from_raw_candidate_plan(
        _raw_plan(),
        feature_block_signature_order=order,
    )

    candidate_batch = retrieval_batch.candidate_batch
    assert candidate_batch.row_query_signature_indices.tolist() == [0, 0]
    assert candidate_batch.left_signature_indices.tolist() == [0, 0, 0]
    assert candidate_batch.right_signature_indices.tolist() == [1, 2, 3]
    assert candidate_batch.row_component_keys == ("c_ada", "c_other")
    np.testing.assert_allclose(retrieval_batch.row_signals["retrieval_score"], [0.9, 0.2])


def test_feature_block_query_and_summary_helpers_match_mini_anddata_row_signals() -> None:
    feature_block = feature_block_from_raw_payloads(
        signatures=_raw_payloads_for_plan()[0],
        papers=_raw_payloads_for_plan()[1],
        raw_candidate_plan=_raw_plan(),
        cluster_seeds_require=_raw_payloads_for_plan()[2],
    )
    mini = feature_block_to_mini_anddata(
        feature_block,
        name="mini_feature_block_query_parity",
        load_name_counts=False,
        name_tuples=set(),
    )
    order = feature_block_signature_order_from_raw_candidate_plan(_raw_plan())
    retrieval_batch = build_linker_retrieval_batch_from_raw_candidate_plan(
        _raw_plan(),
        feature_block_signature_order=order,
    )

    mini_query = extract_query_features(mini, "q", orcid_enabled=True)
    direct_query = extract_query_features_from_feature_block(feature_block, "q", orcid_enabled=True)
    assert direct_query.first == mini_query.first
    assert direct_query.middle == mini_query.middle
    assert direct_query.paper_author_names == mini_query.paper_author_names
    assert direct_query.local10_author_names == mini_query.local10_author_names
    assert direct_query.query_author == mini_query.query_author

    mini_summary = build_cluster_summary(
        mini,
        cluster_id="c_other",
        component_key="c_other",
        signature_ids=["s2", "s3"],
        max_exemplars=4,
        orcid_enabled=True,
    )
    direct_summary = build_cluster_summary_from_feature_block(
        feature_block,
        cluster_id="c_other",
        component_key="c_other",
        signature_ids=["s2", "s3"],
        max_exemplars=4,
        orcid_enabled=True,
    )
    assert direct_summary.max_paper_author_count == mini_summary.max_paper_author_count
    assert direct_summary.member_paper_author_names == mini_summary.member_paper_author_names
    assert direct_summary.member_local10_author_names == mini_summary.member_local10_author_names

    row_query_map = {0: "q"}
    np.testing.assert_allclose(
        build_name_count_rarity_row_signals(
            retrieval_batch,
            query_signature_id_by_index=row_query_map,
            query_by_signature_id={"q": direct_query},
            summary_by_component={
                "c_ada": build_cluster_summary_from_feature_block(
                    feature_block,
                    cluster_id="c_ada",
                    component_key="c_ada",
                    signature_ids=["s1"],
                    max_exemplars=4,
                    orcid_enabled=True,
                ),
                "c_other": direct_summary,
            },
        )["paper_author_list_max_overlap_count"],
        build_name_count_rarity_row_signals(
            retrieval_batch,
            query_signature_id_by_index=row_query_map,
            query_by_signature_id={"q": mini_query},
            summary_by_component={
                "c_ada": build_cluster_summary(
                    mini,
                    cluster_id="c_ada",
                    component_key="c_ada",
                    signature_ids=["s1"],
                    max_exemplars=4,
                    orcid_enabled=True,
                ),
                "c_other": mini_summary,
            },
        )["paper_author_list_max_overlap_count"],
    )


def test_raw_feature_block_scoring_wrapper_uses_direct_rust_feature_block_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signatures, papers, cluster_seeds_require = _raw_payloads_for_plan()
    feature_block = feature_block_from_raw_payloads(
        signatures=signatures,
        papers=papers,
        raw_candidate_plan=_raw_plan(),
        cluster_seeds_require=cluster_seeds_require,
    )
    captured: dict[str, object] = {}

    class FakeFeaturizer:
        def signature_ids(self) -> list[str]:
            return ["s3", "q", "s1", "s2"]

    def fake_build_rust_featurizer_from_feature_block(feature_block_arg: object, **_kwargs: object) -> FakeFeaturizer:
        captured["feature_block_signature_ids"] = tuple(feature_block_arg.signature_ids)
        captured["feature_block_seed_map"] = dict(feature_block_arg.cluster_seeds_require)
        captured["build_kwargs"] = dict(_kwargs)
        return FakeFeaturizer()

    def fail_get_rust_featurizer(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("raw FeatureBlock scoring must not materialize mini ANDData")

    def fake_from_retrieval(**kwargs: object) -> LinkOrAbstainProductionResult:
        retrieval_batch = kwargs["retrieval_batch"]
        captured["retrieval_left_indices"] = retrieval_batch.candidate_batch.left_signature_indices.tolist()
        captured["retrieval_right_indices"] = retrieval_batch.candidate_batch.right_signature_indices.tolist()
        captured["queries"] = kwargs["queries"]
        extra_builder = kwargs["extra_row_signal_builder"]
        extra_signals = extra_builder(retrieval_batch, {1: "q"})
        captured["extra_signal_keys"] = sorted(extra_signals)
        return LinkOrAbstainProductionResult(
            feature_matrix=LinkerFeatureMatrix(
                matrix=np.empty((2, 0), dtype=np.float32),
                feature_columns=(),
                candidate_batch=retrieval_batch.candidate_batch,
            ),
            compact_result=LinkOrAbstainCompactResult(
                probabilities=np.asarray([0.8, 0.2], dtype=np.float32),
                decisions=(),
            ),
            telemetry={"pairwise_feature_seconds": 1.25, "constraint_api_mode": "rust_index_arrays"},
            retrieval_batch=retrieval_batch,
            pairwise_model_result=CandidateBatchPairwiseModelResult(
                row_signals={},
                pairwise_stats=None,
                telemetry={},
            ),
            linked_signature_clusters={"q": "c_ada"},
        )

    monkeypatch.setattr(
        "s2and.incremental_linking.runtime.feature_port.build_rust_featurizer_from_feature_block",
        fake_build_rust_featurizer_from_feature_block,
    )
    monkeypatch.setattr("s2and.incremental_linking.runtime.feature_port._get_rust_featurizer", fail_get_rust_featurizer)
    monkeypatch.setattr(
        "s2and.incremental_linking.runtime._predict_incremental_link_or_abstain_production_from_retrieval_private",
        lambda *args, **kwargs: fake_from_retrieval(**kwargs),
    )

    result = predict_incremental_link_or_abstain_from_raw_feature_block(
        type(
            "Clusterer",
            (),
            {
                "n_jobs": 1,
                "suppress_orcid": True,
                "featurizer_info": FeaturizationInfo(features_to_use=["name_counts"]),
            },
        )(),
        type("Artifact", (), {"metadata": type("Metadata", (), {"retrieval_top_k": 25})()})(),
        feature_block=feature_block,
        raw_candidate_plan=_raw_plan(),
        name_tuples=set(),
    )

    assert captured["feature_block_signature_ids"] == ("q", "s1", "s2", "s3")
    assert captured["feature_block_seed_map"] == {"s1": "c_ada", "s2": "c_other", "s3": "c_other"}
    assert captured["build_kwargs"]["load_name_counts"] is True
    queries = captured["queries"]
    assert len(queries) == 1
    assert queries[0].orcid is None
    assert captured["retrieval_left_indices"] == [1, 1, 1]
    assert captured["retrieval_right_indices"] == [2, 3, 0]
    assert "candidate_cluster_max_paper_author_count" in captured["extra_signal_keys"]
    assert result.linked_signature_clusters == {"q": "c_ada"}
    assert result.telemetry["feature_block_signature_count"] == 4
    assert "feature_block_rust_featurizer_seconds" in result.telemetry
    assert result.telemetry["constraint_api_mode"] == "rust_index_arrays"


def test_raw_feature_block_scoring_rejects_disabled_required_name_counts() -> None:
    signatures, papers, cluster_seeds_require = _raw_payloads_for_plan()
    feature_block = feature_block_from_raw_payloads(
        signatures=signatures,
        papers=papers,
        raw_candidate_plan=_raw_plan(),
        cluster_seeds_require=cluster_seeds_require,
    )

    with pytest.raises(ValueError, match="load_name_counts=False"):
        predict_incremental_link_or_abstain_from_raw_feature_block(
            type(
                "Clusterer",
                (),
                {
                    "n_jobs": 1,
                    "featurizer_info": FeaturizationInfo(features_to_use=["name_counts"]),
                },
            )(),
            type("Artifact", (), {"metadata": type("Metadata", (), {"retrieval_top_k": 25})()})(),
            feature_block=feature_block,
            raw_candidate_plan=_raw_plan(),
            load_name_counts=False,
            name_tuples=set(),
        )


def test_raw_arrow_scoring_wrapper_uses_direct_arrow_featurizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrow_paths = _write_feature_block_arrow_paths(tmp_path)
    captured: dict[str, object] = {}

    class FakeRustModule:
        @staticmethod
        def raw_block_query_candidate_plan_arrow(
            paths_arg: dict[str, str],
            query_signature_ids: list[str],
            **kwargs: object,
        ) -> dict[str, object]:
            captured["retrieval_paths"] = paths_arg
            captured["retrieval_query_signature_ids"] = tuple(query_signature_ids)
            captured["retrieval_kwargs"] = kwargs
            return _raw_plan()

    class FakeFeaturizer:
        def signature_ids(self) -> list[str]:
            return ["q", "s1", "s2", "s3"]

    def fake_build_rust_featurizer_from_arrow_paths(paths_arg: object, **kwargs: object) -> FakeFeaturizer:
        captured["featurizer_paths"] = paths_arg
        captured["featurizer_signature_ids"] = tuple(kwargs["signature_ids"])
        return FakeFeaturizer()

    def fake_from_retrieval(**kwargs: object) -> LinkOrAbstainProductionResult:
        retrieval_batch = kwargs["retrieval_batch"]
        captured["retrieval_left_indices"] = retrieval_batch.candidate_batch.left_signature_indices.tolist()
        captured["retrieval_right_indices"] = retrieval_batch.candidate_batch.right_signature_indices.tolist()
        captured["retrieval_query_author"] = retrieval_batch.row_signals["query_author"].tolist()
        captured["extra_row_signal_builder"] = kwargs["extra_row_signal_builder"]
        captured["seed_setup"] = kwargs["seed_setup"]
        captured["queries"] = kwargs["queries"]
        return LinkOrAbstainProductionResult(
            feature_matrix=LinkerFeatureMatrix(
                matrix=np.empty((2, 0), dtype=np.float32),
                feature_columns=(),
                candidate_batch=retrieval_batch.candidate_batch,
            ),
            compact_result=LinkOrAbstainCompactResult(
                probabilities=np.asarray([0.8, 0.2], dtype=np.float32),
                decisions=(),
            ),
            telemetry={"pairwise_feature_seconds": 0.5, "constraint_api_mode": "rust_index_arrays"},
            retrieval_batch=retrieval_batch,
            pairwise_model_result=CandidateBatchPairwiseModelResult(
                row_signals={},
                pairwise_stats=None,
                telemetry={},
            ),
            linked_signature_clusters={"q": "c_ada"},
        )

    monkeypatch.setattr("s2and.incremental_linking.runtime.feature_port._require_rust_runtime", lambda: FakeRustModule)
    monkeypatch.setattr(
        "s2and.incremental_linking.runtime.feature_port.build_rust_featurizer_from_arrow_paths",
        fake_build_rust_featurizer_from_arrow_paths,
    )
    monkeypatch.setattr(
        "s2and.incremental_linking.runtime._predict_incremental_link_or_abstain_production_from_retrieval_private",
        lambda *args, **kwargs: fake_from_retrieval(**kwargs),
    )

    result = predict_incremental_link_or_abstain_from_raw_arrow_paths(
        type("Clusterer", (), {"n_jobs": 1})(),
        type("Artifact", (), {"metadata": type("Metadata", (), {"retrieval_top_k": 25})()})(),
        arrow_paths=arrow_paths,
        query_signature_ids=["q"],
        top_k=2,
        n_jobs=1,
        load_name_counts=False,
        name_tuples=set(),
    )

    assert captured["retrieval_query_signature_ids"] == ("q",)
    assert captured["retrieval_kwargs"]["top_k"] == 2
    assert captured["featurizer_signature_ids"] == ("q", "s1", "s2", "s3")
    assert captured["retrieval_left_indices"] == [0, 0, 0]
    assert captured["retrieval_right_indices"] == [1, 2, 3]
    assert captured["retrieval_query_author"] == ["ada lovelace", "ada lovelace"]
    assert captured["extra_row_signal_builder"] is None
    queries = captured["queries"]
    assert isinstance(queries, tuple)
    assert queries[0].query_author == "ada lovelace"
    assert captured["seed_setup"][0] == {"s1": "c_ada", "s2": "c_other", "s3": "c_other"}
    assert result.linked_signature_clusters == {"q": "c_ada"}
    assert result.telemetry["raw_arrow_signature_count"] == 4
    assert result.telemetry["raw_arrow_seed_signature_count"] == 3
    assert "raw_arrow_retrieval_seconds" in result.telemetry
    assert captured["retrieval_paths"] == captured["featurizer_paths"]


def test_raw_arrow_scoring_wrapper_uses_provided_rust_featurizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeFeaturizer:
        def signature_ids(self) -> list[str]:
            return ["q", "s1", "s2", "s3"]

    def fail_build_rust_featurizer_from_arrow_paths(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("prebuilt raw Arrow featurizer should be reused")

    def fake_from_retrieval(**kwargs: object) -> LinkOrAbstainProductionResult:
        retrieval_batch = kwargs["retrieval_batch"]
        captured["featurizer"] = kwargs["featurizer"]
        captured["retrieval_left_indices"] = retrieval_batch.candidate_batch.left_signature_indices.tolist()
        captured["retrieval_right_indices"] = retrieval_batch.candidate_batch.right_signature_indices.tolist()
        return LinkOrAbstainProductionResult(
            feature_matrix=LinkerFeatureMatrix(
                matrix=np.empty((2, 0), dtype=np.float32),
                feature_columns=(),
                candidate_batch=retrieval_batch.candidate_batch,
            ),
            compact_result=LinkOrAbstainCompactResult(
                probabilities=np.asarray([0.8, 0.2], dtype=np.float32),
                decisions=(),
            ),
            telemetry={"pairwise_feature_seconds": 0.5, "constraint_api_mode": "rust_index_arrays"},
            retrieval_batch=retrieval_batch,
            pairwise_model_result=CandidateBatchPairwiseModelResult(
                row_signals={},
                pairwise_stats=None,
                telemetry={},
            ),
            linked_signature_clusters={"q": "c_ada"},
        )

    fake_featurizer = FakeFeaturizer()
    monkeypatch.setattr(
        "s2and.incremental_linking.runtime.feature_port.build_rust_featurizer_from_arrow_paths",
        fail_build_rust_featurizer_from_arrow_paths,
    )
    monkeypatch.setattr(
        "s2and.incremental_linking.runtime._predict_incremental_link_or_abstain_production_from_retrieval_private",
        lambda *args, **kwargs: fake_from_retrieval(**kwargs),
    )

    result = predict_incremental_link_or_abstain_from_raw_arrow_paths(
        type("Clusterer", (), {"n_jobs": 1})(),
        type("Artifact", (), {"metadata": type("Metadata", (), {"retrieval_top_k": 25})()})(),
        arrow_paths={},
        query_signature_ids=["q"],
        raw_candidate_plan=_raw_plan(),
        rust_featurizer=fake_featurizer,
        top_k=2,
        n_jobs=1,
        load_name_counts=False,
        name_tuples=set(),
    )

    assert captured["featurizer"] is fake_featurizer
    assert captured["retrieval_left_indices"] == [0, 0, 0]
    assert captured["retrieval_right_indices"] == [1, 2, 3]
    assert result.telemetry["raw_arrow_featurizer_reused"] == 1
    assert result.telemetry["raw_arrow_featurizer_seconds"] >= 0.0


def test_raw_arrow_scoring_wrapper_rejects_mismatched_raw_plan_query_ids() -> None:
    class FakeFeaturizer:
        def signature_ids(self) -> list[str]:
            return ["q", "s1", "s2", "s3"]

    with pytest.raises(ValueError, match="must exactly match requested query_signature_ids"):
        predict_incremental_link_or_abstain_from_raw_arrow_paths(
            type("Clusterer", (), {"n_jobs": 1})(),
            type("Artifact", (), {"metadata": type("Metadata", (), {"retrieval_top_k": 25})()})(),
            arrow_paths={},
            query_signature_ids=["s1"],
            raw_candidate_plan=_raw_plan(),
            rust_featurizer=FakeFeaturizer(),
            top_k=2,
            n_jobs=1,
            load_name_counts=False,
            name_tuples=set(),
        )


def test_rust_featurizer_from_feature_block_matches_mini_anddata() -> None:
    s2and_rust = pytest.importorskip("s2and_rust")
    feature_block = feature_block_from_anddata(
        _tiny_anddata(),
        signature_ids=["q", "s1", "s2"],
        query_signature_ids=["q"],
    )
    mini = feature_block_to_mini_anddata(
        feature_block,
        name="mini_feature_block_rust_parity",
        load_name_counts=False,
        name_tuples=set(),
    )

    direct = s2and_rust.RustFeaturizer.from_feature_block(
        feature_block,
        set(),
        None,
        True,
        False,
        0.0,
        10000.0,
        1,
    )
    incumbent = s2and_rust.RustFeaturizer.from_dataset(mini, 0.0, 10000.0, 1)

    assert tuple(direct.signature_ids()) == feature_block.signature_ids
    np.testing.assert_allclose(direct.featurize_pair("q", "s1"), incumbent.featurize_pair("q", "s1"))
    assert direct.get_constraint("q", "s2") == incumbent.get_constraint("q", "s2")


def test_from_retrieval_skips_pair_id_build_when_partial_supervision_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import s2and.incremental_linking.runtime as runtime_module

    order = feature_block_signature_order_from_raw_candidate_plan(_raw_plan())
    retrieval_batch = build_linker_retrieval_batch_from_raw_candidate_plan(
        _raw_plan(),
        feature_block_signature_order=order,
    )

    class FakeFeaturizer:
        def signature_ids(self) -> list[str]:
            return list(order.signature_ids)

    def fail_candidate_pair_ids(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("pair ids should not be materialized when partial_supervision is empty")

    def fake_pairwise_model(*args: object, **_kwargs: object) -> CandidateBatchPairwiseModelResult:
        candidate_batch = args[1]
        return CandidateBatchPairwiseModelResult(
            row_signals={
                "paper_author_list_max_overlap_count": np.zeros(candidate_batch.row_count, dtype=np.float32),
            },
            pairwise_stats=None,
            telemetry={"feature_seconds": 0.0},
        )

    def fake_retrieved_candidates(*args: object, **kwargs: object) -> LinkOrAbstainRetrievedCandidatesResult:
        current_retrieval_batch = args[1]
        return LinkOrAbstainRetrievedCandidatesResult(
            feature_matrix=LinkerFeatureMatrix(
                matrix=np.empty((2, 0), dtype=np.float32),
                feature_columns=(),
                candidate_batch=current_retrieval_batch.candidate_batch,
            ),
            compact_result=LinkOrAbstainCompactResult(
                probabilities=np.asarray([0.8, 0.2], dtype=np.float32),
                decisions=(),
            ),
            telemetry={"candidate_row_count": current_retrieval_batch.candidate_batch.row_count},
        )

    monkeypatch.setattr(runtime_module, "_candidate_pair_ids", fail_candidate_pair_ids)
    monkeypatch.setattr(
        runtime_module,
        "compute_candidate_batch_pairwise_model_and_aggregate_stats",
        fake_pairwise_model,
    )
    monkeypatch.setattr(runtime_module, "_production_query_author_row_signals", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        runtime_module,
        "_predict_incremental_link_or_abstain_retrieved_candidates",
        fake_retrieved_candidates,
    )

    result = runtime_module._predict_incremental_link_or_abstain_production_from_retrieval_private(
        type(
            "Clusterer",
            (),
            {
                "n_jobs": 1,
                "use_default_constraints_as_supervision": False,
                "dont_merge_cluster_seeds": True,
                "suppress_orcid": False,
                "classifier": None,
                "featurizer_info": None,
            },
        )(),
        type("Artifact", (), {"metadata": type("Metadata", (), {"retrieval_top_k": 25})()})(),
        dataset=None,
        featurizer=FakeFeaturizer(),
        retrieval_batch=retrieval_batch,
        queries=[object()],
        query_signature_ids=["q"],
        partial_supervision=None,
        seed_setup=(
            {"s1": "c_ada", "s2": "c_other", "s3": "c_other"},
            {"c_ada": "c_ada", "c_other": "c_other"},
            {"c_ada": ["s1"], "c_other": ["s2", "s3"]},
        ),
        n_jobs=1,
        total_ram_bytes=None,
        retrieval_top_k=2,
    )

    assert result.telemetry["partial_supervision_pair_count"] == 0
    assert result.telemetry["candidate_row_count"] == 2


def test_raw_payload_scoring_wrapper_builds_feature_block_and_adds_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signatures, papers, cluster_seeds_require = _raw_payloads_for_plan()
    retrieval_batch = build_linker_retrieval_batch_from_raw_candidate_plan(
        _raw_plan(),
        feature_block_signature_order=feature_block_signature_order_from_raw_candidate_plan(_raw_plan()),
    )
    captured: dict[str, object] = {}

    def fake_score(**kwargs: object) -> LinkOrAbstainProductionResult:
        feature_block = kwargs["feature_block"]
        captured["feature_block_signature_ids"] = feature_block.signature_ids
        captured["feature_block_disallow"] = feature_block.cluster_seeds_disallow
        captured["raw_candidate_plan"] = kwargs["raw_candidate_plan"]
        return LinkOrAbstainProductionResult(
            feature_matrix=LinkerFeatureMatrix(
                matrix=np.empty((2, 0), dtype=np.float32),
                feature_columns=(),
                candidate_batch=retrieval_batch.candidate_batch,
            ),
            compact_result=LinkOrAbstainCompactResult(
                probabilities=np.asarray([0.8, 0.2], dtype=np.float32),
                decisions=(),
            ),
            telemetry={"candidate_row_count": 2},
            retrieval_batch=retrieval_batch,
            pairwise_model_result=CandidateBatchPairwiseModelResult(
                row_signals={},
                pairwise_stats=None,
                telemetry={},
            ),
            linked_signature_clusters={"q": "c_ada"},
        )

    monkeypatch.setattr(
        "s2and.incremental_linking.runtime.predict_incremental_link_or_abstain_from_raw_feature_block",
        lambda *args, **kwargs: fake_score(**kwargs),
    )

    result = predict_incremental_link_or_abstain_from_raw_payloads(
        type("Clusterer", (), {"n_jobs": 1})(),
        type("Artifact", (), {"metadata": type("Metadata", (), {"retrieval_top_k": 25})()})(),
        signatures=signatures,
        papers=papers,
        raw_candidate_plan=_raw_plan(),
        cluster_seeds_require=cluster_seeds_require,
        cluster_seeds_disallow=[("q", "s2"), ("q", "unused")],
        load_name_counts=False,
        name_tuples=set(),
    )

    assert captured["feature_block_signature_ids"] == ("q", "s1", "s2", "s3")
    assert captured["feature_block_disallow"] == (("q", "s2"),)
    assert captured["raw_candidate_plan"] is not None
    assert result.linked_signature_clusters == {"q": "c_ada"}
    assert result.telemetry["candidate_row_count"] == 2
    assert result.telemetry["feature_block_raw_payload_build_seconds"] >= 0.0
