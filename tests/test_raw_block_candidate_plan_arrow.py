from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from s2and.incremental_linking.feature_block import (
    feature_block_from_arrow_paths,
    write_name_counts_index,
    write_raw_arrow_batch_lookup_indexes,
)
from s2and.incremental_linking.retrieval import build_linker_retrieval_batch_from_raw_candidate_plan
from s2and.incremental_linking.runtime import subset_raw_candidate_plan_for_query_ids
from tests.helpers import build_cluster_summary, build_query_features

pa = pytest.importorskip("pyarrow")
s2and_rust = pytest.importorskip("s2and_rust", reason="s2and_rust is unavailable")


def _write_ipc(path: Path, table: pa.Table) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    return str(path)


def _write_ipc_batches(path: Path, table: pa.Table, *, batch_size: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            for batch in table.to_batches(max_chunksize=batch_size):
                writer.write_batch(batch)
    return str(path)


def _assert_raw_candidate_plans_equal(left: dict[str, object], right: dict[str, object]) -> None:
    assert set(left) == set(right)
    for key in sorted(set(left).difference({"telemetry"})):
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, np.ndarray) or isinstance(right_value, np.ndarray):
            left_array = np.asarray(left_value)
            right_array = np.asarray(right_value)
            if left_array.dtype.kind == "f" or right_array.dtype.kind == "f":
                np.testing.assert_allclose(left_array, right_array, rtol=1e-6, atol=1e-6, err_msg=key)
            else:
                np.testing.assert_array_equal(left_array, right_array, err_msg=key)
        else:
            assert left_value == right_value, key


def _write_tiny_name_counts_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    import s2and.data as data_module

    monkeypatch.setattr(
        data_module,
        "_load_name_counts_cached",
        lambda: (
            {"alice": 10.0, "bob": 30.0},
            {"wang": 20.0, "jones": 40.0},
            {"alice wang": 5.0, "bob jones": 6.0},
            {"wang a": 8.0, "jones b": 9.0},
        ),
    )
    index_path, _metrics = write_name_counts_index(tmp_path)
    return index_path


def _base_arrow_paths(tmp_path: Path) -> dict[str, str]:
    signatures = pa.table(
        {
            "signature_id": pa.array(["q1", "s1", "s2"], type=pa.string()),
            "paper_id": pa.array(["p_q", "p1", "p2"], type=pa.string()),
            "author_first": pa.array(["Alice", "Alice", "Bob"], type=pa.string()),
            "author_middle": pa.array(["", "", ""], type=pa.string()),
            "author_last": pa.array(["Wang", "Wang", "Jones"], type=pa.string()),
            "author_suffix": pa.array(["", "", ""], type=pa.string()),
            "author_affiliations": pa.array(
                [["AI Lab"], ["AI Lab"], ["Other Lab"]],
                type=pa.list_(pa.string()),
            ),
            "author_orcid": pa.array([None, None, None], type=pa.string()),
            "author_position": pa.array([0, 0, 0], type=pa.int64()),
        }
    )
    papers = pa.table(
        {
            "paper_id": pa.array(["p_q", "p1", "p2"], type=pa.string()),
            "title": pa.array(["Graph Models", "Graph Models", "Different Topic"], type=pa.string()),
            "venue": pa.array(["NeurIPS", "NeurIPS", "ICML"], type=pa.string()),
            "journal_name": pa.array(["", "", ""], type=pa.string()),
            "year": pa.array([2020, 2020, 2010], type=pa.int64()),
        }
    )
    paper_authors = pa.table(
        {
            "paper_id": pa.array(["p_q", "p_q", "p1", "p1", "p2", "p2"], type=pa.string()),
            "position": pa.array([0, 1, 0, 1, 0, 1], type=pa.int64()),
            "author_name": pa.array(
                ["Alice Wang", "Ann Smith", "Alice Wang", "Ann Smith", "Bob Jones", "Carl Doe"],
                type=pa.string(),
            ),
        }
    )
    cluster_seeds = pa.table(
        {
            "signature_id": pa.array(["s1", "s2"], type=pa.string()),
            "cluster_id": pa.array(["c_match", "c_other"], type=pa.string()),
        }
    )
    cluster_seed_disallows = pa.table(
        {
            "signature_id_1": pa.array(["q1"], type=pa.string()),
            "signature_id_2": pa.array(["s2"], type=pa.string()),
        }
    )
    return {
        "signatures": _write_ipc(tmp_path / "signatures.arrow", signatures),
        "papers": _write_ipc(tmp_path / "papers.arrow", papers),
        "paper_authors": _write_ipc(tmp_path / "paper_authors.arrow", paper_authors),
        "cluster_seeds": _write_ipc(tmp_path / "cluster_seeds.arrow", cluster_seeds),
        "cluster_seed_disallows": _write_ipc(tmp_path / "cluster_seed_disallows.arrow", cluster_seed_disallows),
    }


def _assert_retrieval_plan_equal(raw_plan: dict[str, object], direct_plan: dict[str, object]) -> None:
    assert raw_plan["row_component_keys"] == direct_plan["row_component_keys"]
    assert int(raw_plan["row_count"]) == int(direct_plan["row_count"])
    np.testing.assert_array_equal(raw_plan["retrieval_ranks"], direct_plan["retrieval_ranks"])
    np.testing.assert_allclose(raw_plan["retrieval_scores"], direct_plan["retrieval_scores"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        raw_plan["middle_initial_compatibility"],
        direct_plan["middle_initial_compatibility"],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(raw_plan["coauthor_overlap"], direct_plan["coauthor_overlap"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        raw_plan["specter_centroid_similarity"],
        direct_plan["specter_centroid_similarity"],
        rtol=1e-6,
        atol=1e-6,
    )


def test_raw_arrow_candidate_plan_matches_existing_rust_retriever(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "raw_block_query_candidate_plan_arrow"):
        pytest.skip("raw_block_query_candidate_plan_arrow is unavailable")
    paths = _base_arrow_paths(tmp_path)

    raw_plan = s2and_rust.raw_block_query_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )

    query = build_query_features(
        first="alice",
        coauthor_blocks=frozenset({"a smith"}),
        affiliation_terms=frozenset({"ai"}),
        venue_terms=frozenset({"neurips"}),
        title_terms=frozenset({"graph", "models"}),
        year=2020,
        has_coauthors=True,
        has_affiliations=True,
        has_full_first=True,
    )
    summaries = [
        build_cluster_summary(
            component_key="c_match",
            size=1,
            first_name_counts=Counter({"alice": 1}),
            coauthor_counts=Counter({"a smith": 1}),
            affiliation_counts=Counter({"ai": 1}),
            venue_counts=Counter({"neurips": 1}),
            title_counts=Counter({"graph": 1, "models": 1}),
            year_min=2020,
            year_max=2020,
            year_mean=2020.0,
        ),
        build_cluster_summary(
            component_key="c_other",
            size=1,
            first_name_counts=Counter({"bob": 1}),
            coauthor_counts=Counter({"c doe": 1}),
            affiliation_counts=Counter(),
            venue_counts=Counter({"icml": 1}),
            title_counts=Counter({"different": 1, "topic": 1}),
            year_min=2010,
            year_max=2010,
            year_mean=2010.0,
        ),
    ]
    retriever = s2and_rust.RustHybridCentroidRetriever(summaries, include_exemplars=False)
    direct_plan = retriever.top_k_hybrid_centroid_pair_plan(
        [query],
        np.asarray([0], dtype=np.uint32),
        {"c_match": np.asarray([1], dtype=np.uint32), "c_other": np.asarray([2], dtype=np.uint32)},
        2,
        1,
    )

    _assert_retrieval_plan_equal(raw_plan, direct_plan)
    assert raw_plan["left_signature_ids"] == ["q1", "q1"]
    assert raw_plan["right_signature_ids"] == ["s1", "s2"]
    assert raw_plan["query_views"] == ["full"]
    assert raw_plan["telemetry"]["signature_count"] == 3


def test_raw_arrow_candidate_plan_batch_indexes_match_full_scan_and_bound_rows(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "raw_block_query_candidate_plan_arrow"):
        pytest.skip("raw_block_query_candidate_plan_arrow is unavailable")
    irrelevant_count = 24
    signature_ids = ["q1", "s1", "s2"] + [f"junk_sig_{index}" for index in range(irrelevant_count)]
    paper_ids = ["p_q", "p1", "p2"] + [f"junk_paper_{index}" for index in range(irrelevant_count)]
    signatures = pa.table(
        {
            "signature_id": pa.array(signature_ids, type=pa.string()),
            "paper_id": pa.array(paper_ids, type=pa.string()),
            "author_first": pa.array(["Alice", "Alice", "Bob"] + ["Noise"] * irrelevant_count, type=pa.string()),
            "author_middle": pa.array(["", "", ""] + [""] * irrelevant_count, type=pa.string()),
            "author_last": pa.array(["Wang", "Wang", "Jones"] + ["Ignored"] * irrelevant_count, type=pa.string()),
            "author_suffix": pa.array(["", "", ""] + [""] * irrelevant_count, type=pa.string()),
            "author_affiliations": pa.array(
                [["AI Lab"], ["AI Lab"], ["Other Lab"]] + [[] for _ in range(irrelevant_count)],
                type=pa.list_(pa.string()),
            ),
            "author_orcid": pa.array([None] * len(signature_ids), type=pa.string()),
            "author_position": pa.array([0] * len(signature_ids), type=pa.int64()),
        }
    )
    papers = pa.table(
        {
            "paper_id": pa.array(paper_ids, type=pa.string()),
            "title": pa.array(["Graph Models", "Graph Models", "Different Topic"] + ["Noise"] * irrelevant_count),
            "venue": pa.array(["NeurIPS", "NeurIPS", "ICML"] + [""] * irrelevant_count),
            "journal_name": pa.array(["", "", ""] + [""] * irrelevant_count),
            "year": pa.array([2020, 2020, 2010] + [1990] * irrelevant_count, type=pa.int64()),
        }
    )
    paper_authors = pa.table(
        {
            "paper_id": pa.array(paper_ids, type=pa.string()),
            "position": pa.array([0] * len(paper_ids), type=pa.int64()),
            "author_name": pa.array(["Alice Wang", "Alice Wang", "Bob Jones"] + ["Noise"] * irrelevant_count),
        }
    )
    cluster_seeds = pa.table(
        {
            "signature_id": pa.array(["s1", "s2"], type=pa.string()),
            "cluster_id": pa.array(["c_match", "c_other"], type=pa.string()),
        }
    )
    specter = pa.table(
        {
            "paper_id": pa.array(paper_ids, type=pa.string()),
            "embedding": pa.FixedSizeListArray.from_arrays(
                pa.array(
                    [1.0, 0.0, 1.0, 0.0, 0.0, 1.0] + [0.0, 0.0] * irrelevant_count,
                    type=pa.float32(),
                ),
                2,
            ),
        }
    )
    batch_size = 1
    paths = {
        "signatures": _write_ipc_batches(tmp_path / "signatures.arrow", signatures, batch_size=batch_size),
        "papers": _write_ipc_batches(tmp_path / "papers.arrow", papers, batch_size=batch_size),
        "paper_authors": _write_ipc_batches(
            tmp_path / "paper_authors.arrow",
            paper_authors,
            batch_size=batch_size,
        ),
        "cluster_seeds": _write_ipc(tmp_path / "cluster_seeds.arrow", cluster_seeds),
        "specter": _write_ipc_batches(tmp_path / "specter.arrow", specter, batch_size=batch_size),
    }
    indexed_paths, index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)

    full_scan_plan = s2and_rust.raw_block_query_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )
    indexed_plan = s2and_rust.raw_block_query_candidate_plan_arrow(
        indexed_paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )

    _assert_raw_candidate_plans_equal(indexed_plan, full_scan_plan)
    telemetry = indexed_plan["telemetry"]
    assert telemetry["indexed_arrow_candidate_plan"] is True
    assert full_scan_plan["telemetry"]["signature_count"] == 3 + irrelevant_count
    assert telemetry["signature_count"] == 3
    assert telemetry["paper_count"] == 3
    assert telemetry["paper_author_paper_count"] == 3
    assert telemetry["specter_count"] == 3
    assert telemetry["signature_rows_scanned"] == 3
    assert telemetry["paper_rows_scanned"] == 3
    assert telemetry["paper_author_rows_scanned"] == 3
    assert telemetry["specter_rows_scanned"] == 3
    assert index_metrics["signatures_batch_index"]["record_count"] == len(signature_ids)


def test_raw_arrow_candidate_plan_rejects_stale_batch_index(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "raw_block_query_candidate_plan_arrow"):
        pytest.skip("raw_block_query_candidate_plan_arrow is unavailable")
    paths = _base_arrow_paths(tmp_path)
    indexed_paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)
    with Path(paths["signatures"]).open("ab") as outfile:
        outfile.write(b"\0")

    with pytest.raises(ValueError, match="stale"):
        s2and_rust.raw_block_query_candidate_plan_arrow(
            indexed_paths,
            ["q1"],
            top_k=2,
            query_view="full",
            orcid_enabled=False,
            num_threads=1,
        )


def test_raw_arrow_candidate_plan_orcid_override_returns_all_matches(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "raw_block_query_candidate_plan_arrow"):
        pytest.skip("raw_block_query_candidate_plan_arrow is unavailable")
    signatures = pa.table(
        {
            "signature_id": pa.array(["q1", "s_good", "s_middle", "s_year", "s_none"], type=pa.string()),
            "paper_id": pa.array(["p_q", "p_good", "p_middle", "p_year", "p_none"], type=pa.string()),
            "author_first": pa.array(["Alice", "Alice", "Alice", "Alice", "Alice"], type=pa.string()),
            "author_middle": pa.array(["Q", "Q", "Z", "Q", "Q"], type=pa.string()),
            "author_last": pa.array(["Wang", "Wang", "Wang", "Wang", "Wang"], type=pa.string()),
            "author_suffix": pa.array(["", "", "", "", ""], type=pa.string()),
            "author_affiliations": pa.array([[], [], [], [], []], type=pa.list_(pa.string())),
            "author_orcid": pa.array(
                ["0000-0001", "0000-0001", "0000-0001", "0000-0001", None],
                type=pa.string(),
            ),
            "author_position": pa.array([0, 0, 0, 0, 0], type=pa.int64()),
        }
    )
    papers = pa.table(
        {
            "paper_id": pa.array(["p_q", "p_good", "p_middle", "p_year", "p_none"], type=pa.string()),
            "title": pa.array(["", "", "", "", ""], type=pa.string()),
            "venue": pa.array(["", "", "", "", ""], type=pa.string()),
            "journal_name": pa.array(["", "", "", "", ""], type=pa.string()),
            "year": pa.array([2024, 2024, 2024, 1900, 2024], type=pa.int64()),
        }
    )
    paper_authors = pa.table(
        {
            "paper_id": pa.array(["p_q", "p_good", "p_middle", "p_year", "p_none"], type=pa.string()),
            "position": pa.array([0, 0, 0, 0, 0], type=pa.int64()),
            "author_name": pa.array(["Alice Wang"] * 5, type=pa.string()),
        }
    )
    cluster_seeds = pa.table(
        {
            "signature_id": pa.array(["s_good", "s_middle", "s_year", "s_none"], type=pa.string()),
            "cluster_id": pa.array(["c_good", "c_middle", "c_year", "c_none"], type=pa.string()),
        }
    )
    paths = {
        "signatures": _write_ipc(tmp_path / "signatures.arrow", signatures),
        "papers": _write_ipc(tmp_path / "papers.arrow", papers),
        "paper_authors": _write_ipc(tmp_path / "paper_authors.arrow", paper_authors),
        "cluster_seeds": _write_ipc(tmp_path / "cluster_seeds.arrow", cluster_seeds),
    }

    plan = s2and_rust.raw_block_query_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=1,
        query_view="full",
        orcid_enabled=True,
        num_threads=1,
    )

    assert set(plan["row_component_keys"]) == {"c_good", "c_middle", "c_year"}
    assert "c_none" not in plan["row_component_keys"]
    assert plan["row_orcid_match"].tolist() == [1, 1, 1]


def test_raw_arrow_candidate_plan_matches_multi_query_auto_views_and_specter(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "raw_block_query_candidate_plan_arrow"):
        pytest.skip("raw_block_query_candidate_plan_arrow is unavailable")
    signatures = pa.table(
        {
            "signature_id": pa.array(["q_full", "q_initial", "s_full", "s_initial", "s_other"], type=pa.string()),
            "paper_id": pa.array(["p_qf", "p_qi", "p_full", "p_initial", "p_other"], type=pa.string()),
            "author_first": pa.array(["Alice", "A", "Alice", "A", "Carol"], type=pa.string()),
            "author_middle": pa.array(["B", "", "B", "", ""], type=pa.string()),
            "author_last": pa.array(["Wang", "Li", "Wang", "Li", "Jones"], type=pa.string()),
            "author_suffix": pa.array(["", "", "", "", ""], type=pa.string()),
            "author_affiliations": pa.array(
                [["AI Lab"], ["Robotics Center"], ["AI Lab"], ["Robotics Center"], []],
                type=pa.list_(pa.string()),
            ),
            "author_orcid": pa.array([None, None, None, None, None], type=pa.string()),
            "author_position": pa.array([0, 0, 0, 0, 0], type=pa.int64()),
        }
    )
    papers = pa.table(
        {
            "paper_id": pa.array(["p_qf", "p_qi", "p_full", "p_initial", "p_other"], type=pa.string()),
            "title": pa.array(
                ["Graph Models", "Robot Planning", "Graph Models", "Robot Planning", ""],
                type=pa.string(),
            ),
            "venue": pa.array(["NeurIPS", "RSS", "NeurIPS", "RSS", ""], type=pa.string()),
            "journal_name": pa.array(["", "", "", "", ""], type=pa.string()),
            "year": pa.array([2020, 2022, 2020, 2022, None], type=pa.int64()),
        }
    )
    paper_authors = pa.table(
        {
            "paper_id": pa.array(
                ["p_qf", "p_qf", "p_qi", "p_qi", "p_full", "p_full", "p_initial", "p_initial", "p_other"],
                type=pa.string(),
            ),
            "position": pa.array([0, 1, 0, 1, 0, 1, 0, 1, 0], type=pa.int64()),
            "author_name": pa.array(
                [
                    "Alice Wang",
                    "Ann Smith",
                    "A Li",
                    "Ben Stone",
                    "Alice Wang",
                    "Ann Smith",
                    "A Li",
                    "Ben Stone",
                    "Carol Jones",
                ],
                type=pa.string(),
            ),
        }
    )
    cluster_seeds = pa.table(
        {
            "signature_id": pa.array(["s_full", "s_initial", "s_other"], type=pa.string()),
            "cluster_id": pa.array(["c_full", "c_initial", "c_other"], type=pa.string()),
        }
    )
    specter = pa.table(
        {
            "paper_id": pa.array(["p_qf", "p_qi", "p_full", "p_initial", "p_other"], type=pa.string()),
            "embedding": pa.FixedSizeListArray.from_arrays(
                pa.array(
                    [
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                        0.2,
                        0.2,
                    ],
                    type=pa.float32(),
                ),
                2,
            ),
        }
    )
    paths = {
        "signatures": _write_ipc(tmp_path / "signatures.arrow", signatures),
        "papers": _write_ipc(tmp_path / "papers.arrow", papers),
        "paper_authors": _write_ipc(tmp_path / "paper_authors.arrow", paper_authors),
        "cluster_seeds": _write_ipc(tmp_path / "cluster_seeds.arrow", cluster_seeds),
        "specter": _write_ipc(tmp_path / "specter.arrow", specter),
    }

    raw_plan = s2and_rust.raw_block_query_candidate_plan_arrow(
        paths,
        ["q_full", "q_initial"],
        top_k=2,
        query_view="auto",
        orcid_enabled=False,
        num_threads=1,
    )

    queries = [
        build_query_features(
            first="alice",
            middle_initials=frozenset({"b"}),
            coauthor_blocks=frozenset({"a smith"}),
            affiliation_terms=frozenset({"ai"}),
            venue_terms=frozenset({"neurips"}),
            title_terms=frozenset({"graph", "models"}),
            year=2020,
            specter=np.asarray([1.0, 0.0], dtype=np.float32),
            has_coauthors=True,
            has_affiliations=True,
            has_full_first=True,
            has_middle=True,
        ),
        build_query_features(
            first="a",
            coauthor_blocks=frozenset({"b stone"}),
            affiliation_terms=frozenset({"robotics"}),
            venue_terms=frozenset({"rss"}),
            title_terms=frozenset({"robot", "planning"}),
            year=2022,
            specter=np.asarray([0.0, 1.0], dtype=np.float32),
            has_coauthors=True,
            has_affiliations=True,
            has_full_first=False,
        ),
    ]
    summaries = [
        build_cluster_summary(
            component_key="c_full",
            first_name_counts=Counter({"alice": 1}),
            middle_initial_counts=Counter({"b": 1}),
            coauthor_counts=Counter({"a smith": 1}),
            affiliation_counts=Counter({"ai": 1}),
            venue_counts=Counter({"neurips": 1}),
            title_counts=Counter({"graph": 1, "models": 1}),
            year_min=2020,
            year_max=2020,
            year_mean=2020.0,
            specter_centroid=np.asarray([1.0, 0.0], dtype=np.float32),
            exemplar_vectors=[np.asarray([1.0, 0.0], dtype=np.float32)],
        ),
        build_cluster_summary(
            component_key="c_initial",
            coauthor_counts=Counter({"b stone": 1}),
            affiliation_counts=Counter({"robotics": 1}),
            venue_counts=Counter({"rss": 1}),
            title_counts=Counter({"robot": 1, "planning": 1}),
            year_min=2022,
            year_max=2022,
            year_mean=2022.0,
            specter_centroid=np.asarray([0.0, 1.0], dtype=np.float32),
            exemplar_vectors=[np.asarray([0.0, 1.0], dtype=np.float32)],
        ),
        build_cluster_summary(
            component_key="c_other",
            first_name_counts=Counter({"carol": 1}),
            specter_centroid=np.asarray([0.2, 0.2], dtype=np.float32),
            exemplar_vectors=[np.asarray([0.2, 0.2], dtype=np.float32)],
        ),
    ]
    retriever = s2and_rust.RustHybridCentroidRetriever(summaries, include_exemplars=True)
    direct_plan = retriever.top_k_hybrid_centroid_pair_plan(
        queries,
        np.asarray([0, 1], dtype=np.uint32),
        {
            "c_full": np.asarray([2], dtype=np.uint32),
            "c_initial": np.asarray([3], dtype=np.uint32),
            "c_other": np.asarray([4], dtype=np.uint32),
        },
        2,
        1,
    )

    _assert_retrieval_plan_equal(raw_plan, direct_plan)
    assert raw_plan["query_views"] == ["full", "initial_only"]
    assert raw_plan["left_signature_ids"] == ["q_full", "q_full", "q_initial", "q_initial"]
    assert raw_plan["right_signature_ids"] == ["s_full", "s_other", "s_initial", "s_other"]

    subset_plan = subset_raw_candidate_plan_for_query_ids(raw_plan, ["q_initial"], zero_plan_timings=True)
    single_query_plan = s2and_rust.raw_block_query_candidate_plan_arrow(
        paths,
        ["q_initial"],
        top_k=2,
        query_view="auto",
        orcid_enabled=False,
        num_threads=1,
    )
    for key in (
        "query_signature_ids",
        "query_views",
        "query_authors",
        "row_component_keys",
        "left_signature_ids",
        "right_signature_ids",
    ):
        assert subset_plan[key] == single_query_plan[key]
    for key in (
        "row_query_signature_indices",
        "retrieval_scores",
        "retrieval_ranks",
        "left_signature_indices",
        "right_signature_indices",
        "pair_row_indices",
        "row_orcid_match",
        "specter_centroid_similarity",
    ):
        np.testing.assert_array_equal(subset_plan[key], single_query_plan[key])
    assert subset_plan["telemetry"]["query_signature_count"] == 1
    assert subset_plan["telemetry"]["signature_count"] == 0
    assert subset_plan["telemetry"]["seed_signature_count"] == raw_plan["telemetry"]["seed_signature_count"]
    assert subset_plan["telemetry"]["cluster_count"] == raw_plan["telemetry"]["cluster_count"]
    assert subset_plan["telemetry"]["timings"]["total_secs"] == 0.0
    assert subset_plan["telemetry"]["window_plan_reused"] == 1
    assert "window_query_signature_count" not in subset_plan["telemetry"]


def test_raw_arrow_candidate_plan_excludes_query_seed_and_handles_missing_metadata(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "raw_block_query_candidate_plan_arrow"):
        pytest.skip("raw_block_query_candidate_plan_arrow is unavailable")
    signatures = pa.table(
        {
            "signature_id": pa.array(["q1", "s1", "s2"], type=pa.string()),
            "paper_id": pa.array(["p_q", "p1", "p2"], type=pa.string()),
            "author_first": pa.array(["Alice", "Alice", "Bob"], type=pa.string()),
            "author_middle": pa.array([None, None, None], type=pa.string()),
            "author_last": pa.array(["Wang", "Wang", "Jones"], type=pa.string()),
            "author_suffix": pa.array([None, None, None], type=pa.string()),
            "author_affiliations": pa.array([None, None, []], type=pa.list_(pa.string())),
            "author_orcid": pa.array([None, None, None], type=pa.string()),
            "author_position": pa.array([None, None, None], type=pa.int64()),
        }
    )
    papers = pa.table(
        {
            "paper_id": pa.array(["p_q", "p1", "p2"], type=pa.string()),
            "title": pa.array([None, None, None], type=pa.string()),
            "venue": pa.array([None, None, None], type=pa.string()),
            "journal_name": pa.array([None, None, None], type=pa.string()),
            "year": pa.array([None, None, None], type=pa.int64()),
        }
    )
    paper_authors = pa.table(
        {
            "paper_id": pa.array(["p_q", "p1", "p2"], type=pa.string()),
            "position": pa.array([0, 0, 0], type=pa.int64()),
            "author_name": pa.array(["Alice Wang", "Alice Wang", "Bob Jones"], type=pa.string()),
        }
    )
    cluster_seeds = pa.table(
        {
            "signature_id": pa.array(["q1", "s1", "s2"], type=pa.string()),
            "cluster_id": pa.array(["c_self", "c_self", "c_other"], type=pa.string()),
        }
    )
    paths = {
        "signatures": _write_ipc(tmp_path / "signatures.arrow", signatures),
        "papers": _write_ipc(tmp_path / "papers.arrow", papers),
        "paper_authors": _write_ipc(tmp_path / "paper_authors.arrow", paper_authors),
        "cluster_seeds": _write_ipc(tmp_path / "cluster_seeds.arrow", cluster_seeds),
    }

    plan = s2and_rust.raw_block_query_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=2,
        query_view="auto",
        orcid_enabled=False,
        num_threads=1,
    )

    assert plan["telemetry"]["excluded_query_seed_count"] == 1
    assert plan["component_members"]["c_self"] == ["s1"]
    assert "q1" not in plan["right_signature_ids"]
    assert plan["query_views"] == ["full"]


def test_raw_arrow_candidate_plan_bridge_maps_signature_ids_to_linker_indices(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "raw_block_query_candidate_plan_arrow"):
        pytest.skip("raw_block_query_candidate_plan_arrow is unavailable")
    paths = _base_arrow_paths(tmp_path)

    raw_plan = s2and_rust.raw_block_query_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )
    retrieval_batch = build_linker_retrieval_batch_from_raw_candidate_plan(
        raw_plan,
        signature_id_to_index={"q1": 7, "s1": 11, "s2": 13},
    )

    candidate_batch = retrieval_batch.candidate_batch
    assert candidate_batch.row_query_signature_indices.tolist() == [7, 7]
    assert candidate_batch.left_signature_indices.tolist() == [7, 7]
    assert candidate_batch.right_signature_indices.tolist() == [11, 13]
    assert candidate_batch.pair_row_indices.tolist() == [0, 1]
    assert candidate_batch.row_component_keys == ("c_match", "c_other")
    assert retrieval_batch.row_signals["query_view"].tolist() == ["full", "full"]
    np.testing.assert_array_equal(
        retrieval_batch.row_signals["retrieval_score"],
        candidate_batch.retrieval_scores,
    )
    assert "candidate_cluster_max_paper_author_count" in retrieval_batch.row_signals


def test_raw_arrow_candidate_plan_rejects_name_counts_arrow_without_index(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "raw_block_query_candidate_plan_arrow"):
        pytest.skip("raw_block_query_candidate_plan_arrow is unavailable")
    paths = _base_arrow_paths(tmp_path)
    paths["name_counts"] = _write_ipc(
        tmp_path / "name_counts.arrow",
        pa.table(
            {
                "kind": pa.array(
                    [
                        "first",
                        "last",
                        "first_last",
                        "last_first_initial",
                        "first",
                        "last",
                        "first_last",
                        "last_first_initial",
                    ],
                    type=pa.string(),
                ),
                "name": pa.array(
                    ["alice", "wang", "alice wang", "wang a", "bob", "jones", "bob jones", "jones b"],
                    type=pa.string(),
                ),
                "count": pa.array([10.0, 20.0, 5.0, 8.0, 30.0, 40.0, 6.0, 9.0], type=pa.float64()),
            }
        ),
    )

    with pytest.raises(ValueError, match="requires name_counts_index"):
        s2and_rust.raw_block_query_candidate_plan_arrow(
            paths,
            ["q1"],
            top_k=2,
            query_view="full",
            orcid_enabled=False,
            num_threads=1,
        )


def test_raw_arrow_candidate_plan_emits_native_row_signals_from_name_counts_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(s2and_rust, "raw_block_query_candidate_plan_arrow"):
        pytest.skip("raw_block_query_candidate_plan_arrow is unavailable")
    paths = _base_arrow_paths(tmp_path)
    paths["name_counts_index"] = _write_tiny_name_counts_index(tmp_path / "index", monkeypatch)

    raw_plan = s2and_rust.raw_block_query_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )

    np.testing.assert_allclose(
        raw_plan["row_last_name_count_min_rarity"],
        np.asarray([1.0 / np.sqrt(20.0), 1.0 / np.sqrt(20.0)], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        raw_plan["row_last_first_name_count_min_rarity"],
        np.asarray([1.0 / np.sqrt(5.0), 1.0 / np.sqrt(5.0)], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        raw_plan["row_candidate_last_name_count_min_rarity"],
        np.asarray([1.0 / np.sqrt(20.0), 1.0 / np.sqrt(40.0)], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        raw_plan["row_candidate_last_first_name_count_min_rarity"],
        np.asarray([1.0 / np.sqrt(5.0), 1.0 / np.sqrt(6.0)], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        raw_plan["row_first_prefix_x_last_first_name_count_min_rarity"],
        np.asarray([1.0 / np.sqrt(5.0), 0.0], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_rust_featurizer_from_arrow_paths_matches_feature_block(tmp_path: Path) -> None:
    if not hasattr(s2and_rust.RustFeaturizer, "from_arrow_paths"):
        pytest.skip("RustFeaturizer.from_arrow_paths is unavailable")
    paths = _base_arrow_paths(tmp_path)
    raw_plan = s2and_rust.raw_block_query_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )
    feature_block = feature_block_from_arrow_paths(paths, raw_candidate_plan=raw_plan)

    direct = s2and_rust.RustFeaturizer.from_arrow_paths(
        paths,
        list(feature_block.signature_ids),
        set(),
        None,
        True,
        False,
        0.0,
        10000.0,
        1,
    )
    incumbent = s2and_rust.RustFeaturizer.from_feature_block(
        feature_block,
        set(),
        None,
        True,
        False,
        0.0,
        10000.0,
        1,
    )
    pairs = [("q1", "s1"), ("q1", "s2")]

    assert tuple(direct.signature_ids()) == feature_block.signature_ids
    assert feature_block.cluster_seeds_disallow == (("q1", "s2"),)
    np.testing.assert_allclose(
        direct.featurize_pairs_matrix(pairs, None, 1, np.nan),
        incumbent.featurize_pairs_matrix(pairs, None, 1, np.nan),
        equal_nan=True,
    )
    assert direct.get_constraint("s1", "s2") == incumbent.get_constraint("s1", "s2")
    assert direct.get_constraint("q1", "s2") == incumbent.get_constraint("q1", "s2") == 10000.0


def test_rust_featurizer_from_arrow_paths_uses_name_counts_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(s2and_rust.RustFeaturizer, "from_arrow_paths"):
        pytest.skip("RustFeaturizer.from_arrow_paths is unavailable")
    index_paths = _base_arrow_paths(tmp_path / "index")
    index_paths["name_counts_index"] = _write_tiny_name_counts_index(tmp_path / "index_artifact", monkeypatch)
    arrow_paths = _base_arrow_paths(tmp_path / "arrow")
    arrow_paths["name_counts"] = _write_ipc(
        tmp_path / "arrow" / "name_counts.arrow",
        pa.table(
            {
                "kind": pa.array(
                    [
                        "first",
                        "last",
                        "first_last",
                        "last_first_initial",
                        "first",
                        "last",
                        "first_last",
                        "last_first_initial",
                    ],
                    type=pa.string(),
                ),
                "name": pa.array(
                    ["alice", "wang", "alice wang", "wang a", "bob", "jones", "bob jones", "jones b"],
                    type=pa.string(),
                ),
                "count": pa.array([10.0, 20.0, 5.0, 8.0, 30.0, 40.0, 6.0, 9.0], type=pa.float64()),
            }
        ),
    )
    arrow_paths["name_counts_index"] = index_paths["name_counts_index"]
    signature_ids = ["q1", "s1", "s2"]
    pairs = [("q1", "s1"), ("q1", "s2")]

    from_index = s2and_rust.RustFeaturizer.from_arrow_paths(
        index_paths,
        signature_ids,
        set(),
        None,
        True,
        False,
        0.0,
        10000.0,
        1,
    )
    from_arrow = s2and_rust.RustFeaturizer.from_arrow_paths(
        arrow_paths,
        signature_ids,
        set(),
        None,
        True,
        False,
        0.0,
        10000.0,
        1,
    )

    np.testing.assert_allclose(
        from_index.featurize_pairs_matrix(pairs, None, 1, np.nan),
        from_arrow.featurize_pairs_matrix(pairs, None, 1, np.nan),
        equal_nan=True,
    )

    arrow_only_paths = dict(arrow_paths)
    del arrow_only_paths["name_counts_index"]
    with pytest.raises(ValueError, match="requires name_counts_index"):
        s2and_rust.RustFeaturizer.from_arrow_paths(
            arrow_only_paths,
            signature_ids,
            set(),
            None,
            True,
            False,
            0.0,
            10000.0,
            1,
        )


def test_rust_featurizer_from_arrow_paths_uses_arrow_name_pairs(tmp_path: Path) -> None:
    if not hasattr(s2and_rust.RustFeaturizer, "from_arrow_paths"):
        pytest.skip("RustFeaturizer.from_arrow_paths is unavailable")
    signatures = pa.table(
        {
            "signature_id": pa.array(["q1", "s1"], type=pa.string()),
            "paper_id": pa.array(["p_q", "p1"], type=pa.string()),
            "author_first": pa.array(["Qi-Xin", "Qadir"], type=pa.string()),
            "author_middle": pa.array(["", ""], type=pa.string()),
            "author_last": pa.array(["Ou Yang", "Ou Yang"], type=pa.string()),
            "author_suffix": pa.array([None, None], type=pa.string()),
            "author_affiliations": pa.array([[], []], type=pa.list_(pa.string())),
            "author_orcid": pa.array([None, None], type=pa.string()),
            "author_position": pa.array([0, 0], type=pa.int64()),
        }
    )
    papers = pa.table(
        {
            "paper_id": pa.array(["p_q", "p1"], type=pa.string()),
            "title": pa.array(["", ""], type=pa.string()),
            "venue": pa.array(["", ""], type=pa.string()),
            "journal_name": pa.array(["", ""], type=pa.string()),
            "year": pa.array([2020, 2020], type=pa.int64()),
        }
    )
    paper_authors = pa.table(
        {
            "paper_id": pa.array(["p_q", "p1"], type=pa.string()),
            "position": pa.array([0, 0], type=pa.int64()),
            "author_name": pa.array(["Qi-Xin Ou Yang", "Qadir Ou Yang"], type=pa.string()),
        }
    )
    paths = {
        "signatures": _write_ipc(tmp_path / "signatures.arrow", signatures),
        "papers": _write_ipc(tmp_path / "papers.arrow", papers),
        "paper_authors": _write_ipc(tmp_path / "paper_authors.arrow", paper_authors),
        "name_pairs": _write_ipc(
            tmp_path / "name_pairs.arrow",
            pa.table(
                {
                    "name_1": pa.array(["qi xin"], type=pa.string()),
                    "name_2": pa.array(["qadir"], type=pa.string()),
                }
            ),
        ),
    }

    from_pairs_arrow = s2and_rust.RustFeaturizer.from_arrow_paths(
        paths,
        ["q1", "s1"],
        None,
        None,
        True,
        False,
        0.0,
        10000.0,
        1,
    )
    from_python_set = s2and_rust.RustFeaturizer.from_arrow_paths(
        {key: value for key, value in paths.items() if key != "name_pairs"},
        ["q1", "s1"],
        {("qi xin", "qadir")},
        None,
        True,
        False,
        0.0,
        10000.0,
        1,
    )

    assert from_pairs_arrow.get_constraint("q1", "s1") == from_python_set.get_constraint("q1", "s1")
    assert from_pairs_arrow.get_constraint("q1", "s1") is None
