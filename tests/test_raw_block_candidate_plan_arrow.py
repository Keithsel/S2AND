from __future__ import annotations

import json
import os
import struct
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from s2and.incremental_linking.feature_block import (
    feature_block_signature_order_from_raw_candidate_plan,
    write_arrow_batch_lookup_index,
    write_name_counts_index,
    write_raw_arrow_batch_lookup_indexes,
)
from s2and.incremental_linking.retrieval import (
    RAW_CANDIDATE_PLAN_PAIR_KEYS,
    RAW_CANDIDATE_PLAN_ROW_KEYS,
    RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS,
    RAW_CANDIDATE_PLAN_SCHEMA_VERSION,
    build_linker_retrieval_batch_from_raw_candidate_plan,
)
from s2and.incremental_linking.runtime import (
    _merge_raw_arrow_planner_build_telemetry,
    _raw_candidate_plan_seed_setup,
    subset_raw_candidate_plan_for_query_ids,
)
from tests.helpers import build_cluster_summary, build_query_features

pa = pytest.importorskip("pyarrow")
s2and_rust = pytest.importorskip("s2and_rust", reason="s2and_rust is unavailable")
_MISSING_RUST_RAW_APIS = [name for name in ("RawBlockQueryCandidatePlanner",) if not hasattr(s2and_rust, name)]
_RUST_FEATURIZER = getattr(s2and_rust, "RustFeaturizer", None)
if _RUST_FEATURIZER is None:
    _MISSING_RUST_RAW_APIS.append("RustFeaturizer")
elif not hasattr(_RUST_FEATURIZER, "from_arrow_paths"):
    _MISSING_RUST_RAW_APIS.append("RustFeaturizer.from_arrow_paths")
if _MISSING_RUST_RAW_APIS:
    pytest.fail(f"s2and_rust is missing required raw Arrow APIs: {_MISSING_RUST_RAW_APIS}")

_FNV64_OFFSET = 14695981039346656037
_FNV64_PRIME = 1099511628211
_ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT = struct.Struct("<8sQQQQQ")
_ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT = struct.Struct("<QII")
_NAME_COUNTS_INDEX_HEADER_LEN = 32
_NAME_COUNTS_INDEX_RECORD_LEN = 40


def _minimal_raw_candidate_plan(**overrides: Any) -> dict[str, Any]:
    query_signature_ids = list(overrides.pop("query_signature_ids", ["q0"]))
    row_count = int(overrides.pop("row_count", 0))
    pair_count = int(overrides.pop("pair_count", 0))
    plan: dict[str, Any] = {
        "schema_version": RAW_CANDIDATE_PLAN_SCHEMA_VERSION,
        "query_signature_ids": query_signature_ids,
        "query_views": ["full"] * len(query_signature_ids),
        "query_authors": ["Alice"] * len(query_signature_ids),
        "row_count": row_count,
        "pair_count": pair_count,
        "row_query_signature_indices": np.zeros(row_count, dtype=np.uint32),
        "row_component_keys": [f"c{index}" for index in range(row_count)],
        "retrieval_scores": np.zeros(row_count, dtype=np.float32),
        "retrieval_ranks": np.arange(1, row_count + 1, dtype=np.uint16),
        "pair_row_indices": np.zeros(pair_count, dtype=np.uint32),
        "left_signature_indices": np.zeros(pair_count, dtype=np.uint32),
        "right_signature_indices": np.zeros(pair_count, dtype=np.uint32),
        "seed_signature_ids": [],
        "component_members": {},
        "telemetry": {},
    }
    for raw_key, _signal_key, dtype in RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS:
        if dtype is object:
            plan[raw_key] = np.asarray([""] * row_count, dtype=object)
        else:
            plan[raw_key] = np.zeros(row_count, dtype=dtype)
    plan.update(overrides)
    return plan


def test_subset_raw_candidate_plan_fast_path_rejects_out_of_range_query_index() -> None:
    raw_plan = _minimal_raw_candidate_plan(
        query_signature_ids=["q0", "q1", "q2"],
        query_views=["full", "full", "full"],
        query_authors=["Alice", "Bob", "Carol"],
        row_count=1,
        pair_count=1,
        row_query_signature_indices=np.asarray([1], dtype=np.uint32),
        row_component_keys=["c1"],
        pair_row_indices=np.asarray([0], dtype=np.uint32),
        left_signature_indices=np.asarray([0], dtype=np.uint32),
        right_signature_indices=np.asarray([3], dtype=np.uint32),
        seed_signature_ids=["s1"],
        component_members={"c1": ["s1"]},
    )

    with pytest.raises(ValueError, match="outside the selected contiguous query range"):
        subset_raw_candidate_plan_for_query_ids(raw_plan, ["q1"])


def test_subset_raw_candidate_plan_rejects_duplicate_query_ids() -> None:
    raw_plan = _minimal_raw_candidate_plan(
        query_signature_ids=["q0", "q0"],
        query_views=["full", "full"],
        query_authors=["Alice", "Alice"],
    )

    with pytest.raises(ValueError, match="query_signature_ids must be unique"):
        subset_raw_candidate_plan_for_query_ids(raw_plan, ["q0"])

    raw_plan["query_signature_ids"] = ["q0", "q1"]
    with pytest.raises(ValueError, match="requested query_signature_ids must be unique"):
        subset_raw_candidate_plan_for_query_ids(raw_plan, ["q0", "q0"])


def test_raw_candidate_plan_seed_setup_rejects_duplicate_seed_signature() -> None:
    raw_plan = {"component_members": {"c1": ["s1", "s2"], "c2": ["s1"]}}

    with pytest.raises(ValueError, match="assigns signature_id 's1' to multiple components"):
        _raw_candidate_plan_seed_setup(raw_plan)


def _fnv64_bytes(value: bytes) -> int:
    digest = _FNV64_OFFSET
    for byte in value:
        digest ^= byte
        digest = (digest * _FNV64_PRIME) & 0xFFFFFFFFFFFFFFFF
    return digest


def _append_batch_index_record(index_path: str, *, key: str, batch_index: int) -> None:
    path = Path(index_path)
    raw = path.read_bytes()
    magic, record_count, source_size, source_mtime_ns, key_column_hash, source_fingerprint = (
        _ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT.unpack_from(raw, 0)
    )
    offset = _ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT.size
    record_size = _ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT.size
    records = [
        _ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT.unpack_from(raw, offset + index * record_size)
        for index in range(record_count)
    ]
    records.append((_fnv64_bytes(key.encode("utf-8")), int(batch_index), 0))
    records.sort()
    payload = bytearray(
        _ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT.pack(
            magic,
            len(records),
            source_size,
            source_mtime_ns,
            key_column_hash,
            source_fingerprint,
        )
    )
    for record in records:
        payload.extend(_ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT.pack(*record))
    path.write_bytes(payload)


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


def _assert_raw_candidate_plans_equal(left: dict[str, Any], right: dict[str, Any]) -> None:
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


def _swap_first_two_name_count_records(index_root: str | Path, kind: str) -> None:
    index_path = Path(index_root)
    manifest = json.loads((index_path / "manifest.json").read_text(encoding="utf-8"))
    record_path = Path(manifest["files"][kind]["path"])
    if not record_path.is_absolute():
        record_path = index_path / record_path
    payload = bytearray(record_path.read_bytes())
    first_start = _NAME_COUNTS_INDEX_HEADER_LEN
    second_start = first_start + _NAME_COUNTS_INDEX_RECORD_LEN
    third_start = second_start + _NAME_COUNTS_INDEX_RECORD_LEN
    first_record = bytes(payload[first_start:second_start])
    second_record = bytes(payload[second_start:third_start])
    payload[first_start:second_start] = second_record
    payload[second_start:third_start] = first_record
    record_path.write_bytes(payload)


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
    return {
        "signatures": _write_ipc(tmp_path / "signatures.arrow", signatures),
        "papers": _write_ipc(tmp_path / "papers.arrow", papers),
        "paper_authors": _write_ipc(tmp_path / "paper_authors.arrow", paper_authors),
        "cluster_seeds": _write_ipc(tmp_path / "cluster_seeds.arrow", cluster_seeds),
    }


def _assert_retrieval_plan_equal(raw_plan: dict[str, Any], direct_plan: dict[str, Any]) -> None:
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


def _raw_candidate_plan_arrow(
    paths: dict[str, str],
    query_signature_ids: list[str],
    *,
    top_k: int = 25,
    query_view: str = "auto",
    orcid_enabled: bool = True,
    num_threads: int | None = None,
    max_exemplars: int = 4,
    include_pair_signature_ids: bool = True,
    include_component_members: bool = True,
    full_scan_without_index: bool = True,
) -> dict[str, Any]:
    planner = s2and_rust.RawBlockQueryCandidatePlanner(
        paths,
        list(query_signature_ids),
        top_k=top_k,
        query_view=query_view,
        orcid_enabled=orcid_enabled,
        num_threads=num_threads,
        max_exemplars=max_exemplars,
        include_pair_signature_ids=include_pair_signature_ids,
        include_component_members=include_component_members,
        full_scan_without_index=full_scan_without_index,
    )
    plan = planner.plan(
        list(query_signature_ids),
        top_k=top_k,
        query_view=query_view,
        include_pair_signature_ids=include_pair_signature_ids,
        include_component_members=include_component_members,
    )
    _merge_raw_arrow_planner_build_telemetry(plan, planner.build_telemetry())
    return plan


def _raw_plan_for_base_paths(paths: dict[str, str]) -> dict[str, Any]:
    return _raw_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )


def test_raw_arrow_candidate_plan_matches_existing_rust_retriever(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)

    raw_plan = _raw_candidate_plan_arrow(
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


def test_raw_arrow_candidate_planner_matches_one_shot_plan(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)

    one_shot = _raw_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
        include_pair_signature_ids=True,
        include_component_members=True,
        full_scan_without_index=True,
    )
    planner = s2and_rust.RawBlockQueryCandidatePlanner(
        paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
        include_pair_signature_ids=True,
        include_component_members=True,
        full_scan_without_index=True,
    )
    planned = planner.plan(
        ["q1"],
        top_k=2,
        query_view="full",
        include_pair_signature_ids=True,
        include_component_members=True,
    )

    _assert_raw_candidate_plans_equal(planned, one_shot)
    assert planner.build_telemetry()["query_signature_count"] == 1
    assert planner.build_telemetry()["planner_seed_state"] == 1
    assert planned["telemetry"]["planner_seed_state_reused"] == 1
    assert planned["telemetry"]["timings"]["read_cluster_seeds_secs"] == 0.0
    assert planned["telemetry"]["timings"]["read_name_counts_secs"] == 0.0


def test_raw_arrow_candidate_planner_filters_batch_query_seed_overlap(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)
    paths["cluster_seeds"] = _write_ipc(
        tmp_path / "cluster_seeds_with_query.arrow",
        pa.table(
            {
                "signature_id": pa.array(["q1", "s1", "s2"], type=pa.string()),
                "cluster_id": pa.array(["c_query", "c_match", "c_other"], type=pa.string()),
            }
        ),
    )

    one_shot = _raw_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
        include_pair_signature_ids=True,
        include_component_members=True,
        full_scan_without_index=True,
    )
    planner = s2and_rust.RawBlockQueryCandidatePlanner(
        paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
        include_pair_signature_ids=True,
        include_component_members=True,
        full_scan_without_index=True,
    )
    planned = planner.plan(
        ["q1"],
        top_k=2,
        query_view="full",
        include_pair_signature_ids=True,
        include_component_members=True,
    )

    _assert_raw_candidate_plans_equal(planned, one_shot)
    assert "q1" not in planned["component_members"].get("c_query", [])


def test_raw_arrow_candidate_planner_rejects_multi_query_seed_overlap(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)
    planner = s2and_rust.RawBlockQueryCandidatePlanner(
        paths,
        ["q1", "s1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
        include_pair_signature_ids=True,
        include_component_members=False,
        full_scan_without_index=True,
    )

    with pytest.raises(ValueError, match="singleton query windows"):
        planner.plan(
            ["q1", "s1"],
            top_k=2,
            query_view="full",
            include_pair_signature_ids=True,
            include_component_members=False,
        )

    planned = planner.plan(
        ["q1"],
        top_k=2,
        query_view="full",
        include_pair_signature_ids=True,
        include_component_members=False,
    )
    assert planned["row_component_keys"] == ["c_match", "c_other"]
    assert planned["right_signature_ids"] == ["s1", "s2"]


def test_raw_arrow_candidate_planner_requires_indexes_without_explicit_full_scan(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)

    with pytest.raises(ValueError, match="batch lookup index"):
        s2and_rust.RawBlockQueryCandidatePlanner(
            paths,
            ["q1"],
            top_k=2,
            query_view="full",
            orcid_enabled=False,
            num_threads=1,
        )


def test_raw_arrow_candidate_plan_filters_cluster_seed_disallows(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)
    paths["cluster_seed_disallows"] = _write_ipc(
        tmp_path / "cluster_seed_disallows.arrow",
        pa.table(
            {
                "signature_id_1": pa.array(["q1"], type=pa.string()),
                "signature_id_2": pa.array(["s2"], type=pa.string()),
            }
        ),
    )

    raw_plan = _raw_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )

    assert raw_plan["row_component_keys"] == ["c_match"]
    assert raw_plan["left_signature_ids"] == ["q1"]
    assert raw_plan["right_signature_ids"] == ["s1"]
    assert raw_plan["telemetry"]["cluster_seed_disallow_pair_count"] == 1
    assert raw_plan["telemetry"]["cluster_seed_disallowed_candidate_count"] == 1


def test_raw_arrow_candidate_plan_rejects_disallow_with_unknown_seed_endpoint(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)
    paths["cluster_seed_disallows"] = _write_ipc(
        tmp_path / "cluster_seed_disallows.arrow",
        pa.table(
            {
                "signature_id_1": pa.array(["q1"], type=pa.string()),
                "signature_id_2": pa.array(["missing_seed"], type=pa.string()),
            }
        ),
    )

    with pytest.raises(ValueError, match="unknown seed endpoint"):
        _raw_candidate_plan_arrow(
            paths,
            ["q1"],
            top_k=2,
            query_view="full",
            orcid_enabled=False,
            num_threads=1,
        )


def test_raw_arrow_candidate_plan_keeps_zero_specter_vectors(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)
    paths["specter"] = _write_ipc(
        tmp_path / "specter.arrow",
        pa.table(
            {
                "paper_id": pa.array(["p_q", "p1", "p2"], type=pa.string()),
                "embedding": pa.FixedSizeListArray.from_arrays(
                    pa.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0], type=pa.float32()),
                    2,
                ),
            }
        ),
    )

    raw_plan = _raw_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )

    assert raw_plan["telemetry"]["specter_count"] == 3
    assert np.isfinite(np.asarray(raw_plan["specter_centroid_similarity"], dtype=np.float32)).all()


def test_raw_arrow_candidate_plan_rejects_hidden_query_view(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)

    with pytest.raises(ValueError, match="Unknown query view"):
        _raw_candidate_plan_arrow(
            paths,
            ["q1"],
            top_k=2,
            query_view="initial_only_no_specter",
            orcid_enabled=False,
            num_threads=1,
        )


def test_raw_arrow_candidate_plan_rejects_duplicate_query_signature_ids(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)

    with pytest.raises(ValueError, match="query_signature_ids must be unique"):
        _raw_candidate_plan_arrow(
            paths,
            ["q1", "q1"],
            top_k=2,
            query_view="full",
            orcid_enabled=False,
            num_threads=1,
        )


def test_raw_arrow_candidate_plan_batch_indexes_match_full_scan_and_bound_rows(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
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

    full_scan_plan = _raw_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )
    indexed_plan = _raw_candidate_plan_arrow(
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
    assert full_scan_plan["telemetry"]["indexed_arrow_candidate_plan"] is False
    assert full_scan_plan["telemetry"]["signature_count"] == 3
    assert full_scan_plan["telemetry"]["paper_count"] == 3
    assert full_scan_plan["telemetry"]["paper_author_paper_count"] == 3
    assert full_scan_plan["telemetry"]["specter_count"] == 3
    full_scan_rows = len(signature_ids) * 2
    assert full_scan_plan["telemetry"]["signature_rows_scanned"] == full_scan_rows
    assert full_scan_plan["telemetry"]["paper_rows_scanned"] == full_scan_rows
    assert full_scan_plan["telemetry"]["paper_author_rows_scanned"] == full_scan_rows
    assert full_scan_plan["telemetry"]["specter_rows_scanned"] == full_scan_rows
    assert telemetry["signature_count"] == 3
    assert telemetry["paper_count"] == 3
    assert telemetry["paper_author_paper_count"] == 3
    assert telemetry["specter_count"] == 3
    assert telemetry["signature_rows_scanned"] == 3
    assert telemetry["paper_rows_scanned"] == 3
    assert telemetry["paper_author_rows_scanned"] == 3
    assert telemetry["specter_rows_scanned"] == 3
    timings = telemetry["timings"]
    assert isinstance(timings["drop_secs"], float)
    assert timings["drop_secs"] >= 0.0
    assert isinstance(timings["wall_secs"], float)
    assert timings["wall_secs"] >= timings["drop_secs"]
    assert index_metrics["signatures_batch_index"]["record_count"] == len(signature_ids)


def test_raw_arrow_candidate_plan_extra_hash_selected_batch_is_exact_filtered(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    signatures = pa.table(
        {
            "signature_id": pa.array(["q1", "s1", "s2", "bad"], type=pa.string()),
            "paper_id": pa.array(["p_q", "p1", "p2", None], type=pa.string()),
            "author_first": pa.array(["Alice", "Alice", "Bob", "Bad"], type=pa.string()),
            "author_middle": pa.array(["", "", "", ""], type=pa.string()),
            "author_last": pa.array(["Wang", "Wang", "Jones", "Row"], type=pa.string()),
            "author_suffix": pa.array(["", "", "", ""], type=pa.string()),
            "author_affiliations": pa.array([["AI Lab"], ["AI Lab"], ["Other Lab"], []], type=pa.list_(pa.string())),
            "author_orcid": pa.array([None, None, None, None], type=pa.string()),
            "author_position": pa.array([0, 0, 0, 0], type=pa.int64()),
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
            "paper_id": pa.array(["p_q", "p_q", "p1", "p2"], type=pa.string()),
            "position": pa.array([0, 1, 0, 0], type=pa.int64()),
            "author_name": pa.array(["Alice Wang", "Ann Smith", "Alice Wang", "Bob Jones"], type=pa.string()),
        }
    )
    cluster_seeds = pa.table(
        {
            "signature_id": pa.array(["s1", "s2"], type=pa.string()),
            "cluster_id": pa.array(["c_match", "c_other"], type=pa.string()),
        }
    )
    paths = {
        "signatures": _write_ipc_batches(tmp_path / "signatures.arrow", signatures, batch_size=1),
        "papers": _write_ipc_batches(tmp_path / "papers.arrow", papers, batch_size=1),
        "paper_authors": _write_ipc_batches(tmp_path / "paper_authors.arrow", paper_authors, batch_size=1),
        "cluster_seeds": _write_ipc(tmp_path / "cluster_seeds.arrow", cluster_seeds),
    }
    indexed_paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)
    _append_batch_index_record(indexed_paths["signatures_batch_index"], key="q1", batch_index=3)

    plan = _raw_candidate_plan_arrow(
        indexed_paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )

    assert plan["right_signature_ids"] == ["s1", "s2"]
    assert plan["telemetry"]["signature_count"] == 3
    assert plan["telemetry"]["signature_rows_scanned"] == 4


def test_rust_featurizer_from_arrow_paths_empty_indexed_keep_set_skips_stale_validation(tmp_path: Path) -> None:
    if not hasattr(s2and_rust.RustFeaturizer, "from_arrow_paths"):
        raise pytest.skip.Exception("RustFeaturizer.from_arrow_paths is unavailable")
    paths = _base_arrow_paths(tmp_path)
    indexed_paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)
    for key in ("signatures", "papers", "paper_authors"):
        with Path(paths[key]).open("ab") as outfile:
            outfile.write(b"\0")

    featurizer = s2and_rust.RustFeaturizer.from_arrow_paths(
        indexed_paths,
        [],
        set(),
        True,
        False,
        0.0,
        10000.0,
        1,
    )

    assert tuple(featurizer.signature_ids()) == ()


def test_raw_arrow_candidate_plan_rejects_stale_batch_index(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)
    indexed_paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)
    with Path(paths["signatures"]).open("ab") as outfile:
        outfile.write(b"\0")

    with pytest.raises(ValueError, match="stale"):
        _raw_candidate_plan_arrow(
            indexed_paths,
            ["q1"],
            top_k=2,
            query_view="full",
            orcid_enabled=False,
            num_threads=1,
        )


def test_arrow_batch_lookup_index_reuses_current_format_after_mtime_only_change(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)
    indexed_paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)
    signatures_path = Path(paths["signatures"])
    stat = signatures_path.stat()
    os.utime(signatures_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    _index_path, metrics = write_arrow_batch_lookup_index(
        signatures_path,
        indexed_paths["signatures_batch_index"],
        key_column="signature_id",
        table_name="signatures",
        overwrite=False,
    )

    assert metrics["reused"] is True


def test_arrow_batch_lookup_index_rejects_wrong_key_column_reuse(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)
    bad_index_path, _metrics = write_arrow_batch_lookup_index(
        paths["signatures"],
        tmp_path / "signatures.bad_key_batch_index.bin",
        key_column="paper_id",
        table_name="signatures",
        overwrite=True,
    )
    indexed_paths = dict(paths)
    indexed_paths["signatures_batch_index"] = bad_index_path

    with pytest.raises(ValueError, match="different key column"):
        _raw_candidate_plan_arrow(
            indexed_paths,
            ["q1"],
            top_k=2,
            query_view="full",
            orcid_enabled=False,
            num_threads=1,
        )


def test_arrow_batch_lookup_index_rejects_same_size_same_mtime_sampled_source_change(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)
    filler_count = 30_000
    filler_ids = [f"x{index:013d}" for index in range(filler_count)]
    paths["signatures"] = _write_ipc_batches(
        tmp_path / "signatures.arrow",
        pa.table(
            {
                "signature_id": pa.array(["q1", "s1", "s2", *filler_ids], type=pa.string()),
                "paper_id": pa.array(
                    ["p_q", "p1", "p2", *[f"p_x{index}" for index in range(filler_count)]], type=pa.string()
                ),
                "author_first": pa.array(["Alice", "Alice", "Bob", *(["Filler"] * filler_count)], type=pa.string()),
                "author_middle": pa.array(["", "", "", *([""] * filler_count)], type=pa.string()),
                "author_last": pa.array(["Wang", "Wang", "Jones", *(["Person"] * filler_count)], type=pa.string()),
                "author_suffix": pa.array(["", "", "", *([""] * filler_count)], type=pa.string()),
                "author_affiliations": pa.array(
                    [["AI Lab"], ["AI Lab"], ["Other Lab"], *([[]] * filler_count)],
                    type=pa.list_(pa.string()),
                ),
                "author_orcid": pa.array([None, None, None, *([None] * filler_count)], type=pa.string()),
                "author_position": pa.array([0, 0, 0, *([0] * filler_count)], type=pa.int64()),
            }
        ),
        batch_size=1000,
    )
    indexed_paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)
    signatures_path = Path(paths["signatures"])
    stat = signatures_path.stat()
    payload = signatures_path.read_bytes()
    old_value = b"q1"
    new_value = b"qX"
    rewrite_offset = payload.index(old_value)
    assert len(old_value) == len(new_value)
    assert rewrite_offset < 65_536
    signatures_path.write_bytes(payload[:rewrite_offset] + new_value + payload[rewrite_offset + len(old_value) :])
    os.utime(signatures_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    with pytest.raises(ValueError, match="stale"):
        _raw_candidate_plan_arrow(
            indexed_paths,
            ["q1"],
            top_k=2,
            query_view="full",
            orcid_enabled=False,
            num_threads=1,
        )


def test_raw_arrow_candidate_plan_rejects_duplicate_signature_ids(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)
    duplicate_signatures = pa.table(
        {
            "signature_id": pa.array(["q1", "q1", "s1", "s2"], type=pa.string()),
            "paper_id": pa.array(["p_q", "p_q", "p1", "p2"], type=pa.string()),
            "author_first": pa.array(["Alice", "Alice", "Alice", "Bob"], type=pa.string()),
            "author_middle": pa.array(["", "", "", ""], type=pa.string()),
            "author_last": pa.array(["Wang", "Wang", "Wang", "Jones"], type=pa.string()),
            "author_suffix": pa.array(["", "", "", ""], type=pa.string()),
            "author_affiliations": pa.array(
                [["AI Lab"], ["AI Lab"], ["AI Lab"], ["Other Lab"]],
                type=pa.list_(pa.string()),
            ),
            "author_orcid": pa.array([None, None, None, None], type=pa.string()),
            "author_position": pa.array([0, 0, 0, 0], type=pa.int64()),
        }
    )
    _write_ipc(Path(paths["signatures"]), duplicate_signatures)

    with pytest.raises(ValueError, match="duplicate signature_id"):
        _raw_plan_for_base_paths(paths)


def test_raw_arrow_candidate_plan_rejects_integer_signature_id_column(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)
    integer_id_signatures = pa.table(
        {
            "signature_id": pa.array([1, 2, 3], type=pa.int64()),
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
    _write_ipc(Path(paths["signatures"]), integer_id_signatures)

    with pytest.raises(TypeError, match="signature_id must be a string column"):
        _raw_plan_for_base_paths(paths)


def test_raw_arrow_candidate_plan_rejects_duplicate_paper_ids(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)
    duplicate_papers = pa.table(
        {
            "paper_id": pa.array(["p_q", "p_q", "p1", "p2"], type=pa.string()),
            "title": pa.array(["Graph Models", "Graph Models", "Graph Models", "Different Topic"], type=pa.string()),
            "venue": pa.array(["NeurIPS", "NeurIPS", "NeurIPS", "ICML"], type=pa.string()),
            "journal_name": pa.array(["", "", "", ""], type=pa.string()),
            "year": pa.array([2020, 2020, 2020, 2010], type=pa.int64()),
        }
    )
    _write_ipc(Path(paths["papers"]), duplicate_papers)

    with pytest.raises(ValueError, match="duplicate paper_id"):
        _raw_plan_for_base_paths(paths)


def test_raw_arrow_candidate_plan_rejects_integer_predicted_language_column(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)
    malformed_papers = pa.table(
        {
            "paper_id": pa.array(["p_q", "p1", "p2"], type=pa.string()),
            "title": pa.array(["Graph Models", "Graph Models", "Different Topic"], type=pa.string()),
            "venue": pa.array(["NeurIPS", "NeurIPS", "ICML"], type=pa.string()),
            "journal_name": pa.array(["", "", ""], type=pa.string()),
            "year": pa.array([2020, 2020, 2010], type=pa.int64()),
            "predicted_language": pa.array([1, 1, 2], type=pa.int64()),
        }
    )
    _write_ipc(Path(paths["papers"]), malformed_papers)

    with pytest.raises(TypeError, match="predicted_language must be a string column"):
        _raw_plan_for_base_paths(paths)


def test_raw_arrow_candidate_plan_rejects_duplicate_paper_author_positions(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)
    duplicate_authors = pa.table(
        {
            "paper_id": pa.array(["p_q", "p_q", "p1", "p2"], type=pa.string()),
            "position": pa.array([0, 0, 0, 0], type=pa.int64()),
            "author_name": pa.array(["Alice Wang", "A. Wang", "Alice Wang", "Bob Jones"], type=pa.string()),
        }
    )
    _write_ipc(Path(paths["paper_authors"]), duplicate_authors)

    with pytest.raises(ValueError, match="duplicate"):
        _raw_plan_for_base_paths(paths)


def test_raw_arrow_candidate_plan_rejects_null_paper_author_name(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)
    null_author_name = pa.table(
        {
            "paper_id": pa.array(["p_q", "p1", "p2"], type=pa.string()),
            "position": pa.array([0, 0, 0], type=pa.int64()),
            "author_name": pa.array(["Alice Wang", None, "Bob Jones"], type=pa.string()),
        }
    )
    _write_ipc(Path(paths["paper_authors"]), null_author_name)

    with pytest.raises(ValueError, match="author_name is null"):
        _raw_plan_for_base_paths(paths)


def test_raw_arrow_candidate_plan_rejects_string_paper_author_position(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)
    string_position = pa.table(
        {
            "paper_id": pa.array(["p_q", "p1", "p2"], type=pa.string()),
            "position": pa.array(["0", "0", "0"], type=pa.string()),
            "author_name": pa.array(["Alice Wang", "Alice Wang", "Bob Jones"], type=pa.string()),
        }
    )
    _write_ipc(Path(paths["paper_authors"]), string_position)

    with pytest.raises(TypeError, match="position must be an int64 column"):
        _raw_plan_for_base_paths(paths)


def test_raw_arrow_candidate_plan_rejects_invalid_cluster_seed_rows(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)
    idempotent_duplicate_seeds = pa.table(
        {
            "signature_id": pa.array(["s1", "s1", "s2"], type=pa.string()),
            "cluster_id": pa.array(["c_match", "c_match", "c_other"], type=pa.string()),
        }
    )
    _write_ipc(Path(paths["cluster_seeds"]), idempotent_duplicate_seeds)
    with pytest.raises(ValueError, match="duplicate signature_id"):
        _raw_plan_for_base_paths(paths)

    duplicate_seeds = pa.table(
        {
            "signature_id": pa.array(["s1", "s1", "s2"], type=pa.string()),
            "cluster_id": pa.array(["c_match", "c_other", "c_other"], type=pa.string()),
        }
    )
    _write_ipc(Path(paths["cluster_seeds"]), duplicate_seeds)
    with pytest.raises(ValueError, match="duplicate signature_id"):
        _raw_plan_for_base_paths(paths)

    integer_seed_ids = pa.table(
        {
            "signature_id": pa.array([1, 2], type=pa.int64()),
            "cluster_id": pa.array(["c_match", "c_other"], type=pa.string()),
        }
    )
    _write_ipc(Path(paths["cluster_seeds"]), integer_seed_ids)
    with pytest.raises(TypeError, match="signature_id must be a string column"):
        _raw_plan_for_base_paths(paths)

    empty_cluster_id = pa.table(
        {
            "signature_id": pa.array(["s1", "s2"], type=pa.string()),
            "cluster_id": pa.array(["", "c_other"], type=pa.string()),
        }
    )
    _write_ipc(Path(paths["cluster_seeds"]), empty_cluster_id)
    with pytest.raises(ValueError, match="empty cluster_id"):
        _raw_plan_for_base_paths(paths)


def test_raw_arrow_candidate_plan_rejects_integer_is_reliable_column(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)
    malformed_papers = pa.table(
        {
            "paper_id": pa.array(["p_q", "p1", "p2"], type=pa.string()),
            "title": pa.array(["Graph Models", "Graph Models", "Different Topic"], type=pa.string()),
            "venue": pa.array(["NeurIPS", "NeurIPS", "ICML"], type=pa.string()),
            "journal_name": pa.array(["", "", ""], type=pa.string()),
            "year": pa.array([2020, 2020, 2010], type=pa.int64()),
            "is_reliable": pa.array([1, 1, 0], type=pa.int64()),
        }
    )
    _write_ipc(Path(paths["papers"]), malformed_papers)

    with pytest.raises(TypeError, match="is_reliable must be a boolean column"):
        _raw_plan_for_base_paths(paths)


def test_raw_arrow_candidate_plan_rejects_duplicate_cluster_seed_disallows(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)
    duplicate_disallows = pa.table(
        {
            "signature_id_1": pa.array(["s1", "s2"], type=pa.string()),
            "signature_id_2": pa.array(["s2", "s1"], type=pa.string()),
        }
    )
    paths["cluster_seed_disallows"] = _write_ipc(tmp_path / "cluster_seed_disallows.arrow", duplicate_disallows)

    with pytest.raises(ValueError, match="duplicate pair"):
        _raw_plan_for_base_paths(paths)


def test_raw_arrow_candidate_plan_rejects_empty_cluster_seed_disallow_endpoint(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)
    empty_endpoint = pa.table(
        {
            "signature_id_1": pa.array(["s1"], type=pa.string()),
            "signature_id_2": pa.array([""], type=pa.string()),
        }
    )
    paths["cluster_seed_disallows"] = _write_ipc(tmp_path / "cluster_seed_disallows.arrow", empty_endpoint)

    with pytest.raises(ValueError, match="empty signature_id"):
        _raw_plan_for_base_paths(paths)


def test_raw_arrow_candidate_plan_rejects_duplicate_specter_paper_ids(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)
    specter = pa.table(
        {
            "paper_id": pa.array(["p_q", "p_q", "p1", "p2"], type=pa.string()),
            "embedding": pa.FixedSizeListArray.from_arrays(
                pa.array([1.0, 0.0, 1.0, 0.0, 0.8, 0.2, 0.0, 1.0], type=pa.float32()),
                2,
            ),
        }
    )
    paths["specter"] = _write_ipc(tmp_path / "specter.arrow", specter)

    with pytest.raises(ValueError, match="duplicate paper_id"):
        _raw_plan_for_base_paths(paths)


def test_raw_arrow_candidate_plan_rejects_null_specter_embedding(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)
    specter = pa.table(
        {
            "paper_id": pa.array(["p_q", "p1", "p2"], type=pa.string()),
            "embedding": pa.array(
                [[1.0, 0.0], None, [0.0, 1.0]],
                type=pa.list_(pa.float32(), 2),
            ),
        }
    )
    paths["specter"] = _write_ipc(tmp_path / "specter.arrow", specter)

    with pytest.raises(ValueError, match="null embedding"):
        _raw_plan_for_base_paths(paths)


def test_rust_featurizer_from_arrow_paths_deduplicates_unsorted_requested_ids(tmp_path: Path) -> None:
    if not hasattr(s2and_rust.RustFeaturizer, "from_arrow_paths"):
        raise pytest.skip.Exception("RustFeaturizer.from_arrow_paths is unavailable")
    paths = _base_arrow_paths(tmp_path)

    with pytest.raises(ValueError, match="filtered full scan"):
        s2and_rust.RustFeaturizer.from_arrow_paths(
            paths,
            ["q1", "s1"],
            set(),
            True,
            False,
            0.0,
            10000.0,
            1,
        )

    featurizer = s2and_rust.RustFeaturizer.from_arrow_paths(
        paths,
        ["q1", "s1", "q1", "s2", "s1"],
        set(),
        True,
        False,
        0.0,
        10000.0,
        1,
        True,
    )

    assert tuple(featurizer.signature_ids()) == ("q1", "s1", "s2")


def test_rust_featurizer_from_arrow_paths_uses_batch_indexes(tmp_path: Path) -> None:
    if not hasattr(s2and_rust.RustFeaturizer, "from_arrow_paths"):
        raise pytest.skip.Exception("RustFeaturizer.from_arrow_paths is unavailable")
    paths = _base_arrow_paths(tmp_path)
    indexed_paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)

    full_scan = s2and_rust.RustFeaturizer.from_arrow_paths(
        paths,
        ["q1", "s1"],
        set(),
        True,
        False,
        0.0,
        10000.0,
        1,
        True,
    )
    indexed = s2and_rust.RustFeaturizer.from_arrow_paths(
        indexed_paths,
        ["q1", "s1"],
        set(),
        True,
        False,
        0.0,
        10000.0,
        1,
    )

    assert tuple(indexed.signature_ids()) == ("q1", "s1")
    np.testing.assert_allclose(
        indexed.featurize_pairs_matrix([("q1", "s1")], None, 1, np.nan),
        full_scan.featurize_pairs_matrix([("q1", "s1")], None, 1, np.nan),
        rtol=1e-6,
        atol=1e-6,
    )

    with Path(paths["signatures"]).open("ab") as outfile:
        outfile.write(b"\0")
    with pytest.raises(ValueError, match="stale"):
        s2and_rust.RustFeaturizer.from_arrow_paths(
            indexed_paths,
            ["q1", "s1"],
            set(),
            True,
            False,
            0.0,
            10000.0,
            1,
        )


def test_raw_arrow_candidate_plan_orcid_override_returns_all_matches(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
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
                [
                    "https://orcid.org/0000\u20100002\u20101825\u20100097",
                    "0000\u20110002\u20111825\u20110097",
                    "ORCID: 0000000218250097",
                    "0000-0002-1825-0097",
                    None,
                ],
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

    plan = _raw_candidate_plan_arrow(
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


def test_raw_arrow_candidate_plan_orcid_override_is_exempt_from_seed_disallows(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    signatures = pa.table(
        {
            "signature_id": pa.array(["q1", "s_good", "s_other"], type=pa.string()),
            "paper_id": pa.array(["p_q", "p_good", "p_other"], type=pa.string()),
            "author_first": pa.array(["Alice", "Alice", "Alice"], type=pa.string()),
            "author_middle": pa.array(["", "", ""], type=pa.string()),
            "author_last": pa.array(["Wang", "Wang", "Wang"], type=pa.string()),
            "author_suffix": pa.array(["", "", ""], type=pa.string()),
            "author_affiliations": pa.array([[], [], []], type=pa.list_(pa.string())),
            "author_orcid": pa.array(
                ["0000-0002-1825-0097", "0000-0002-1825-0097", "0000-0002-1825-0097"],
                type=pa.string(),
            ),
            "author_position": pa.array([0, 0, 0], type=pa.int64()),
        }
    )
    papers = pa.table(
        {
            "paper_id": pa.array(["p_q", "p_good", "p_other"], type=pa.string()),
            "title": pa.array(["", "", ""], type=pa.string()),
            "venue": pa.array(["", "", ""], type=pa.string()),
            "journal_name": pa.array(["", "", ""], type=pa.string()),
            "year": pa.array([2024, 2024, 2024], type=pa.int64()),
        }
    )
    paper_authors = pa.table(
        {
            "paper_id": pa.array(["p_q", "p_good", "p_other"], type=pa.string()),
            "position": pa.array([0, 0, 0], type=pa.int64()),
            "author_name": pa.array(["Alice Wang"] * 3, type=pa.string()),
        }
    )
    cluster_seeds = pa.table(
        {
            "signature_id": pa.array(["s_good", "s_other"], type=pa.string()),
            "cluster_id": pa.array(["c_good", "c_other"], type=pa.string()),
        }
    )
    disallows = pa.table(
        {
            "signature_id_1": pa.array(["q1"], type=pa.string()),
            "signature_id_2": pa.array(["s_good"], type=pa.string()),
        }
    )
    paths = {
        "signatures": _write_ipc(tmp_path / "signatures.arrow", signatures),
        "papers": _write_ipc(tmp_path / "papers.arrow", papers),
        "paper_authors": _write_ipc(tmp_path / "paper_authors.arrow", paper_authors),
        "cluster_seeds": _write_ipc(tmp_path / "cluster_seeds.arrow", cluster_seeds),
        "cluster_seed_disallows": _write_ipc(tmp_path / "cluster_seed_disallows.arrow", disallows),
    }

    plan = _raw_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=1,
        query_view="full",
        orcid_enabled=True,
        num_threads=1,
    )

    assert set(plan["row_component_keys"]) == {"c_good", "c_other"}
    assert set(plan["left_signature_ids"]) == {"q1"}
    assert set(plan["right_signature_ids"]) == {"s_good", "s_other"}
    assert plan["row_orcid_match"].tolist() == [1, 1]
    assert plan["telemetry"]["cluster_seed_disallowed_candidate_count"] == 1


def test_raw_arrow_candidate_plan_missing_query_position_has_no_coauthor_overlap(
    tmp_path: Path,
) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    signatures = pa.table(
        {
            "signature_id": pa.array(["q1", "s_self", "s_real"], type=pa.string()),
            "paper_id": pa.array(["p_q", "p_self", "p_real"], type=pa.string()),
            "author_first": pa.array(["Alice", "Alice", "Alice"], type=pa.string()),
            "author_middle": pa.array(["", "", ""], type=pa.string()),
            "author_last": pa.array(["Wang", "Wang", "Wang"], type=pa.string()),
            "author_suffix": pa.array(["", "", ""], type=pa.string()),
            "author_affiliations": pa.array([[], [], []], type=pa.list_(pa.string())),
            "author_orcid": pa.array([None, None, None], type=pa.string()),
            "author_position": pa.array([None, 0, 0], type=pa.int64()),
        }
    )
    papers = pa.table(
        {
            "paper_id": pa.array(["p_q", "p_self", "p_real"], type=pa.string()),
            "title": pa.array(["", "", ""], type=pa.string()),
            "venue": pa.array(["", "", ""], type=pa.string()),
            "journal_name": pa.array(["", "", ""], type=pa.string()),
            "year": pa.array([2024, 2024, 2024], type=pa.int64()),
        }
    )
    paper_authors = pa.table(
        {
            "paper_id": pa.array(["p_q", "p_q", "p_self", "p_self", "p_real", "p_real"], type=pa.string()),
            "position": pa.array([0, 1, 0, 1, 0, 1], type=pa.int64()),
            "author_name": pa.array(
                ["Alice Wang", "Ann Smith", "Alice Wang", "Alice Wang", "Alice Wang", "Ann Smith"],
                type=pa.string(),
            ),
        }
    )
    cluster_seeds = pa.table(
        {
            "signature_id": pa.array(["s_self", "s_real"], type=pa.string()),
            "cluster_id": pa.array(["c_self", "c_real"], type=pa.string()),
        }
    )
    paths = {
        "signatures": _write_ipc(tmp_path / "signatures.arrow", signatures),
        "papers": _write_ipc(tmp_path / "papers.arrow", papers),
        "paper_authors": _write_ipc(tmp_path / "paper_authors.arrow", paper_authors),
        "cluster_seeds": _write_ipc(tmp_path / "cluster_seeds.arrow", cluster_seeds),
    }

    plan = _raw_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )

    overlap_by_component = dict(zip(plan["row_component_keys"], plan["coauthor_overlap"], strict=True))
    assert overlap_by_component["c_self"] == 0.0
    assert overlap_by_component["c_real"] == 0.0


def test_raw_arrow_candidate_plan_matches_multi_query_auto_views_and_specter(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
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

    raw_plan = _raw_candidate_plan_arrow(
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
    single_query_plan = _raw_candidate_plan_arrow(
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
    for key in RAW_CANDIDATE_PLAN_ROW_KEYS:
        assert key in subset_plan
        assert len(subset_plan[key]) == int(subset_plan["row_count"]), key
    for key in RAW_CANDIDATE_PLAN_PAIR_KEYS:
        assert key in subset_plan
        assert len(subset_plan[key]) == int(subset_plan["pair_count"]), key
    assert subset_plan["telemetry"]["query_signature_count"] == 1
    assert subset_plan["telemetry"]["signature_count"] == 0
    assert subset_plan["telemetry"]["seed_signature_count"] == raw_plan["telemetry"]["seed_signature_count"]
    assert subset_plan["telemetry"]["cluster_count"] == raw_plan["telemetry"]["cluster_count"]
    assert subset_plan["telemetry"]["timings"]["total_secs"] == 0.0
    assert subset_plan["telemetry"]["window_plan_reused"] == 1
    assert "window_query_signature_count" not in subset_plan["telemetry"]


def test_raw_arrow_candidate_plan_excludes_query_seed_and_handles_missing_metadata(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
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

    plan = _raw_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=2,
        query_view="auto",
        orcid_enabled=False,
        num_threads=1,
    )

    assert plan["telemetry"]["excluded_query_seed_count"] == 1
    assert plan["component_members"]["c_self"] == ["s1"]
    assert plan["seed_signature_ids"] == []
    assert plan["telemetry"]["payload_seed_signature_count"] == 0
    assert "seed_component_keys" not in plan
    assert "q1" not in plan["right_signature_ids"]
    assert set(plan["right_signature_ids"]) == {"s1", "s2"}
    assert plan["query_views"] == ["full"]
    np.testing.assert_array_equal(plan["row_query_has_coauthors"], np.zeros(int(plan["row_count"]), dtype=np.uint8))
    np.testing.assert_allclose(plan["coauthor_overlap"], np.zeros(int(plan["row_count"]), dtype=np.float32))

    narrow_plan = _raw_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=1,
        query_view="auto",
        orcid_enabled=False,
        num_threads=1,
    )

    assert "c_self" in narrow_plan["component_members"]
    assert narrow_plan["component_members"]["c_other"] == ["s2"]


def test_raw_arrow_candidate_plan_rejects_null_paper_author_position(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)
    paper_authors = pa.table(
        {
            "paper_id": pa.array(["p_q"], type=pa.string()),
            "position": pa.array([None], type=pa.int64()),
            "author_name": pa.array(["Alice Wang"], type=pa.string()),
        }
    )
    paths["paper_authors"] = _write_ipc(tmp_path / "paper_authors.arrow", paper_authors)

    with pytest.raises(ValueError, match="position is null"):
        _raw_candidate_plan_arrow(
            paths,
            ["q1"],
            top_k=2,
            query_view="full",
            orcid_enabled=False,
            num_threads=1,
        )


def test_raw_arrow_candidate_plan_rejects_null_string_list_elements(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)
    signatures = pa.table(
        {
            "signature_id": pa.array(["q1", "s1", "s2"], type=pa.string()),
            "paper_id": pa.array(["p_q", "p1", "p2"], type=pa.string()),
            "author_first": pa.array(["Alice", "Alice", "Bob"], type=pa.string()),
            "author_middle": pa.array(["", "", ""], type=pa.string()),
            "author_last": pa.array(["Wang", "Wang", "Jones"], type=pa.string()),
            "author_suffix": pa.array(["", "", ""], type=pa.string()),
            "author_affiliations": pa.array([[None], ["AI Lab"], ["Other Lab"]], type=pa.list_(pa.string())),
            "author_orcid": pa.array([None, None, None], type=pa.string()),
            "author_position": pa.array([0, 0, 0], type=pa.int64()),
        }
    )
    paths["signatures"] = _write_ipc(tmp_path / "signatures_with_null_list_element.arrow", signatures)

    with pytest.raises(ValueError, match="author_affiliations cannot contain null list elements"):
        _raw_candidate_plan_arrow(
            paths,
            ["q1"],
            top_k=2,
            query_view="full",
            orcid_enabled=False,
            num_threads=1,
        )


def test_raw_arrow_candidate_plan_rejects_nonempty_null_list_child(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)
    signatures = pa.table(
        {
            "signature_id": pa.array(["q1", "s1", "s2"], type=pa.string()),
            "paper_id": pa.array(["p_q", "p1", "p2"], type=pa.string()),
            "author_first": pa.array(["Alice", "Alice", "Bob"], type=pa.string()),
            "author_middle": pa.array(["", "", ""], type=pa.string()),
            "author_last": pa.array(["Wang", "Wang", "Jones"], type=pa.string()),
            "author_suffix": pa.array(["", "", ""], type=pa.string()),
            "author_affiliations": pa.array([[None], [], []], type=pa.list_(pa.null())),
            "author_orcid": pa.array([None, None, None], type=pa.string()),
            "author_position": pa.array([0, 0, 0], type=pa.int64()),
        }
    )
    paths["signatures"] = _write_ipc(tmp_path / "signatures_with_null_child.arrow", signatures)

    with pytest.raises(ValueError, match="author_affiliations cannot contain null list elements"):
        _raw_candidate_plan_arrow(
            paths,
            ["q1"],
            top_k=2,
            query_view="full",
            orcid_enabled=False,
            num_threads=1,
        )


def test_raw_arrow_candidate_plan_bridge_maps_signature_ids_to_linker_indices(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)

    raw_plan = _raw_candidate_plan_arrow(
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
    assert cast(Any, candidate_batch.row_query_signature_indices).tolist() == [7, 7]
    assert candidate_batch.left_signature_indices.tolist() == [7, 7]
    assert candidate_batch.right_signature_indices.tolist() == [11, 13]
    assert candidate_batch.pair_row_indices.tolist() == [0, 1]
    assert candidate_batch.row_component_keys == ("c_match", "c_other")
    assert retrieval_batch.row_signals["query_view"].tolist() == ["full", "full"]
    np.testing.assert_array_equal(
        retrieval_batch.row_signals["retrieval_score"],
        cast(Any, candidate_batch.retrieval_scores),
    )
    assert "candidate_cluster_max_paper_author_count" in retrieval_batch.row_signals


def test_raw_arrow_labeled_candidate_plan_scores_frozen_rows_without_cluster_seeds(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "raw_arrow_labeled_candidate_plan"):
        raise pytest.skip.Exception("raw_arrow_labeled_candidate_plan is unavailable")
    paths = _base_arrow_paths(tmp_path)
    paths.pop("cluster_seeds")

    raw_plan = s2and_rust.raw_arrow_labeled_candidate_plan(
        paths,
        ["q1", "q1"],
        ["full", "full"],
        ["q1-full", "q1-full"],
        ["c_other", "c_match"],
        np.asarray([1, 2], dtype=np.uint16),
        {"c_match": ["s1"], "c_other": ["s2"]},
        component_scope="block-local",
        orcid_enabled=False,
        num_threads=1,
        full_scan_without_index=True,
    )

    assert raw_plan["schema_version"] == "raw_arrow_labeled_candidate_plan_v1"
    assert raw_plan["row_component_keys"] == ["c_other", "c_match"]
    assert raw_plan["left_signature_ids"] == ["q1", "q1"]
    assert raw_plan["right_signature_ids"] == ["s2", "s1"]
    np.testing.assert_array_equal(raw_plan["pair_row_indices"], np.asarray([0, 1], dtype=np.uint32))
    assert raw_plan["retrieval_ranks"].tolist() == [2, 1]
    assert raw_plan["retrieval_scores"][1] > raw_plan["retrieval_scores"][0]
    assert raw_plan["row_query_views"] == ["full", "full"]
    assert "row_candidate_cluster_max_paper_author_count" in raw_plan
    assert raw_plan["telemetry"]["component_scope"] == "block-local"


def test_raw_arrow_labeled_candidate_plan_rejects_compact_pair_payload(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "raw_arrow_labeled_candidate_plan"):
        raise pytest.skip.Exception("raw_arrow_labeled_candidate_plan is unavailable")
    paths = _base_arrow_paths(tmp_path)
    paths.pop("cluster_seeds")

    with pytest.raises(ValueError, match="requires include_pair_signature_ids=True"):
        s2and_rust.raw_arrow_labeled_candidate_plan(
            paths,
            ["q1"],
            ["full"],
            ["q1-full"],
            ["c_match"],
            np.asarray([1], dtype=np.uint16),
            {"c_match": ["s1"]},
            component_scope="block-local",
            orcid_enabled=False,
            num_threads=1,
            include_pair_signature_ids=False,
            full_scan_without_index=True,
        )


def test_raw_arrow_labeled_candidate_plan_scores_use_all_components_for_global_df(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "raw_arrow_labeled_candidate_plan"):
        raise pytest.skip.Exception("raw_arrow_labeled_candidate_plan is unavailable")
    paths = _base_arrow_paths(tmp_path)
    paths.pop("cluster_seeds")
    component_members = {"c_match": ["s1"], "c_other": ["s2"]}

    one_row = s2and_rust.raw_arrow_labeled_candidate_plan(
        paths,
        ["q1"],
        ["full"],
        ["q1-full"],
        ["c_match"],
        np.asarray([1], dtype=np.uint16),
        component_members,
        component_scope="block-local",
        orcid_enabled=False,
        num_threads=1,
        full_scan_without_index=True,
    )
    two_rows = s2and_rust.raw_arrow_labeled_candidate_plan(
        paths,
        ["q1", "q1"],
        ["full", "full"],
        ["q1-full", "q1-full"],
        ["c_match", "c_other"],
        np.asarray([1, 2], dtype=np.uint16),
        component_members,
        component_scope="block-local",
        orcid_enabled=False,
        num_threads=1,
        full_scan_without_index=True,
    )

    assert one_row["telemetry"]["component_count"] == 2
    assert two_rows["telemetry"]["component_count"] == 2
    np.testing.assert_allclose(one_row["retrieval_scores"][0], two_rows["retrieval_scores"][0], rtol=1e-6, atol=1e-6)


def test_raw_arrow_labeled_candidate_plan_initial_view_keeps_full_first_token(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "raw_arrow_labeled_candidate_plan"):
        raise pytest.skip.Exception("raw_arrow_labeled_candidate_plan is unavailable")
    paths = _base_arrow_paths(tmp_path)
    paths.pop("cluster_seeds")

    raw_plan = s2and_rust.raw_arrow_labeled_candidate_plan(
        paths,
        ["q1"],
        ["initial_only"],
        ["q1-initial"],
        ["c_match"],
        np.asarray([1], dtype=np.uint16),
        {"c_match": ["s1"]},
        component_scope="block-local",
        orcid_enabled=False,
        num_threads=1,
        full_scan_without_index=True,
    )

    assert raw_plan["row_query_views"] == ["initial_only"]
    assert raw_plan["row_query_first_tokens"] == ["alice"]


def test_raw_arrow_candidate_plan_initial_view_keeps_full_first_token(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)

    raw_plan = _raw_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=2,
        query_view="initial_only",
        orcid_enabled=False,
        num_threads=1,
        full_scan_without_index=True,
    )

    assert raw_plan["query_views"] == ["initial_only"]
    assert raw_plan["row_query_first_tokens"] == ["alice", "alice"]


def test_raw_arrow_labeled_candidate_plan_applies_block_local_members(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "raw_arrow_labeled_candidate_plan"):
        raise pytest.skip.Exception("raw_arrow_labeled_candidate_plan is unavailable")
    paths = _base_arrow_paths(tmp_path)
    paths.pop("cluster_seeds")
    signatures = pa.table(
        {
            "signature_id": pa.array(["q1", "s1", "s2"], type=pa.string()),
            "paper_id": pa.array(["p_q", "p1", "p2"], type=pa.string()),
            "author_first": pa.array(["Alice", "Alice", "Bob"], type=pa.string()),
            "author_middle": pa.array(["", "", ""], type=pa.string()),
            "author_last": pa.array(["Wang", "Wang", "Jones"], type=pa.string()),
            "author_suffix": pa.array(["", "", ""], type=pa.string()),
            "author_affiliations": pa.array([["AI Lab"], ["AI Lab"], ["Other Lab"]], type=pa.list_(pa.string())),
            "author_orcid": pa.array([None, None, None], type=pa.string()),
            "author_position": pa.array([0, 0, 0], type=pa.int64()),
            "author_block": pa.array(["block-a", "block-a", "block-b"], type=pa.string()),
        }
    )
    paths["signatures"] = _write_ipc(tmp_path / "signatures_with_blocks.arrow", signatures)

    raw_plan = s2and_rust.raw_arrow_labeled_candidate_plan(
        paths,
        ["q1"],
        ["full"],
        ["q1-full"],
        ["block-a::c"],
        np.asarray([1], dtype=np.uint16),
        {"block-a::c": ["q1", "s1", "s2"]},
        component_scope="block-local",
        orcid_enabled=False,
        num_threads=1,
        full_scan_without_index=True,
    )

    assert raw_plan["left_signature_ids"] == ["q1"]
    assert raw_plan["right_signature_ids"] == ["s1"]
    assert raw_plan["row_component_sizes"].tolist() == [1]


def test_raw_arrow_candidate_plan_bridge_maps_compact_numeric_indices(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)

    raw_plan = _raw_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=1,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
        include_pair_signature_ids=False,
        include_component_members=False,
    )
    assert "left_signature_ids" not in raw_plan
    assert "right_signature_ids" not in raw_plan
    assert "component_members" not in raw_plan
    assert raw_plan["seed_signature_ids"] == ["s1"]
    assert raw_plan["telemetry"]["seed_signature_count"] == 2
    assert raw_plan["telemetry"]["payload_seed_signature_count"] == 1

    signature_order = feature_block_signature_order_from_raw_candidate_plan(raw_plan)
    retrieval_batch = build_linker_retrieval_batch_from_raw_candidate_plan(
        raw_plan,
        feature_block_signature_order=signature_order,
    )

    candidate_batch = retrieval_batch.candidate_batch
    assert signature_order.signature_ids == ("q1", "s1")
    assert cast(Any, candidate_batch.row_query_signature_indices).tolist() == [0]
    assert candidate_batch.left_signature_indices.tolist() == [0]
    assert candidate_batch.right_signature_indices.tolist() == [1]
    assert candidate_batch.pair_row_indices.tolist() == [0]
    assert candidate_batch.row_component_keys == ("c_match",)


def test_raw_arrow_candidate_plan_rejects_name_counts_arrow_without_index(tmp_path: Path) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
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
        _raw_candidate_plan_arrow(
            paths,
            ["q1"],
            top_k=2,
            query_view="full",
            orcid_enabled=False,
            num_threads=1,
        )


def test_raw_arrow_candidate_plan_rejects_name_counts_index_dir_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)
    paths["name_counts"] = _write_ipc(
        tmp_path / "name_counts.arrow",
        pa.table(
            {
                "kind": pa.array(["first"], type=pa.string()),
                "name": pa.array(["alice"], type=pa.string()),
                "count": pa.array([10.0], type=pa.float64()),
            }
        ),
    )
    paths["name_counts_index_dir"] = _write_tiny_name_counts_index(tmp_path / "index", monkeypatch)

    with pytest.raises(ValueError, match="requires name_counts_index"):
        _raw_candidate_plan_arrow(
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
    if not hasattr(s2and_rust, "RawBlockQueryCandidatePlanner"):
        raise pytest.skip.Exception("RawBlockQueryCandidatePlanner is unavailable")
    paths = _base_arrow_paths(tmp_path)
    paths["name_counts_index"] = _write_tiny_name_counts_index(tmp_path / "index", monkeypatch)

    raw_plan = _raw_candidate_plan_arrow(
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


def test_rust_featurizer_from_arrow_paths_applies_cluster_seed_disallows(tmp_path: Path) -> None:
    if not hasattr(s2and_rust.RustFeaturizer, "from_arrow_paths"):
        raise pytest.skip.Exception("RustFeaturizer.from_arrow_paths is unavailable")
    paths = _base_arrow_paths(tmp_path)
    raw_plan = _raw_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )
    paths["cluster_seed_disallows"] = _write_ipc(
        tmp_path / "cluster_seed_disallows.arrow",
        pa.table(
            {
                "signature_id_1": pa.array(["q1"], type=pa.string()),
                "signature_id_2": pa.array(["s2"], type=pa.string()),
            }
        ),
    )
    signature_order = feature_block_signature_order_from_raw_candidate_plan(raw_plan)

    direct = s2and_rust.RustFeaturizer.from_arrow_paths(
        paths,
        list(signature_order.signature_ids),
        set(),
        True,
        False,
        0.0,
        10000.0,
        1,
        True,
    )
    pairs = [("q1", "s1"), ("q1", "s2")]

    assert tuple(direct.signature_ids()) == signature_order.signature_ids
    assert direct.featurize_pairs_matrix(pairs, None, 1, np.nan).shape == (2, 39)
    assert direct.get_constraint("q1", "s1") is None
    assert direct.get_constraint("q1", "s2") == 10000.0


def test_rust_featurizer_missing_name_counts_presence_is_consistent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(s2and_rust.RustFeaturizer, "from_arrow_paths"):
        raise pytest.skip.Exception("RustFeaturizer.from_arrow_paths is unavailable")
    paths = _base_arrow_paths(tmp_path)
    signature_ids = ["q1", "s1", "s2"]

    from_arrow = s2and_rust.RustFeaturizer.from_arrow_paths(
        paths,
        signature_ids,
        set(),
        True,
        False,
        0.0,
        10000.0,
        1,
        True,
    )

    assert from_arrow.signature_name_counts_present() == [("q1", False), ("s1", False), ("s2", False)]

    paths_with_index = dict(paths)
    paths_with_index["name_counts_index"] = _write_tiny_name_counts_index(tmp_path / "index_artifact", monkeypatch)
    with_name_counts = s2and_rust.RustFeaturizer.from_arrow_paths(
        paths_with_index,
        signature_ids,
        set(),
        True,
        False,
        0.0,
        10000.0,
        1,
        True,
    )
    assert with_name_counts.signature_name_counts_present() == [("q1", True), ("s1", True), ("s2", True)]


def test_rust_featurizer_from_arrow_paths_uses_name_counts_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(s2and_rust.RustFeaturizer, "from_arrow_paths"):
        raise pytest.skip.Exception("RustFeaturizer.from_arrow_paths is unavailable")
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
        True,
        False,
        0.0,
        10000.0,
        1,
        True,
    )
    from_arrow = s2and_rust.RustFeaturizer.from_arrow_paths(
        arrow_paths,
        signature_ids,
        set(),
        True,
        False,
        0.0,
        10000.0,
        1,
        True,
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
            True,
            False,
            0.0,
            10000.0,
            1,
            True,
        )

    alias_only_paths = dict(arrow_paths)
    alias_only_paths["name_counts_index_dir"] = alias_only_paths.pop("name_counts_index")
    with pytest.raises(ValueError, match="requires name_counts_index"):
        s2and_rust.RustFeaturizer.from_arrow_paths(
            alias_only_paths,
            signature_ids,
            set(),
            True,
            False,
            0.0,
            10000.0,
            1,
            True,
        )


def test_rust_featurizer_rejects_unsorted_name_counts_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(s2and_rust.RustFeaturizer, "from_arrow_paths"):
        raise pytest.skip.Exception("RustFeaturizer.from_arrow_paths is unavailable")
    paths = _base_arrow_paths(tmp_path)
    paths["name_counts_index"] = _write_tiny_name_counts_index(tmp_path / "index_artifact", monkeypatch)
    _swap_first_two_name_count_records(paths["name_counts_index"], "first")

    with pytest.raises(ValueError, match="not sorted"):
        s2and_rust.RustFeaturizer.from_arrow_paths(
            paths,
            ["q1", "s1", "s2"],
            set(),
            True,
            False,
            0.0,
            10000.0,
            1,
            True,
        )


def test_rust_featurizer_from_arrow_paths_uses_arrow_name_pairs(tmp_path: Path) -> None:
    if not hasattr(s2and_rust.RustFeaturizer, "from_arrow_paths"):
        raise pytest.skip.Exception("RustFeaturizer.from_arrow_paths is unavailable")
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
        True,
        False,
        0.0,
        10000.0,
        1,
        True,
    )
    from_python_set = s2and_rust.RustFeaturizer.from_arrow_paths(
        {key: value for key, value in paths.items() if key != "name_pairs"},
        ["q1", "s1"],
        {("qi xin", "qadir")},
        True,
        False,
        0.0,
        10000.0,
        1,
        True,
    )

    assert from_pairs_arrow.get_constraint("q1", "s1") == from_python_set.get_constraint("q1", "s1")
    assert from_pairs_arrow.get_constraint("q1", "s1") is None


def test_rust_featurizer_from_arrow_paths_rejects_integer_name_pairs(tmp_path: Path) -> None:
    if not hasattr(s2and_rust.RustFeaturizer, "from_arrow_paths"):
        raise pytest.skip.Exception("RustFeaturizer.from_arrow_paths is unavailable")
    paths = _base_arrow_paths(tmp_path)
    paths["name_pairs"] = _write_ipc(
        tmp_path / "name_pairs.arrow",
        pa.table(
            {
                "name_1": pa.array([1], type=pa.int64()),
                "name_2": pa.array(["alice"], type=pa.string()),
            }
        ),
    )

    with pytest.raises(TypeError, match="name_pairs.name_1 must be a string column"):
        s2and_rust.RustFeaturizer.from_arrow_paths(
            paths,
            ["q1", "s1"],
            None,
            True,
            False,
            0.0,
            10000.0,
            1,
            True,
        )
