from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

import s2and.subblocking as subblocking
from s2and.data import Signature


def _write_ipc(path, table: pa.Table) -> str:
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    return str(path)


def _signature(signature_id: str, *, first: str, middle: str | None = None, orcid: str | None = None) -> Signature:
    return Signature(
        author_info_first=first,
        author_info_first_normalized_without_apostrophe=first,
        author_info_middle=middle,
        author_info_middle_normalized_without_apostrophe=middle or "",
        author_info_last_normalized="wang",
        author_info_last="Wang",
        author_info_suffix_normalized=None,
        author_info_suffix=None,
        author_info_first_normalized=first,
        author_info_coauthors=None,
        author_info_coauthor_blocks=None,
        author_info_full_name=None,
        author_info_affiliations=[],
        author_info_affiliations_n_grams=None,
        author_info_coauthor_n_grams=None,
        author_info_email=None,
        author_info_orcid=orcid,
        author_info_name_counts=None,
        author_info_position=0,
        author_info_block="h wang",
        author_info_given_block=None,
        author_info_estimated_gender=None,
        author_info_estimated_ethnicity=None,
        paper_id=int(signature_id[1:]) if signature_id[1:].isdigit() else 0,
        sourced_author_source=None,
        sourced_author_ids=[],
        author_id=None,
        signature_id=signature_id,
    )


def test_make_subblocks_uses_specter_for_oversized_single_letter_block(monkeypatch):
    dataset = SimpleNamespace(
        signatures={
            "s1": _signature("s1", first="h", middle=""),
            "s2": _signature("s2", first="h", middle=""),
            "s3": _signature("s3", first="h", middle=""),
            "s4": _signature("s4", first="h", middle=""),
        },
        random_seed=0,
    )

    monkeypatch.setattr(
        subblocking,
        "cluster_with_specter",
        lambda signature_ids, anddata, target_subblock_size, compute_block_fn=None: {
            "0": list(signature_ids[:2]),
            "1": list(signature_ids[2:]),
        },
    )

    subblocks, _telemetry = subblocking.make_subblocks_with_telemetry(
        ["s1", "s2", "s3", "s4"],
        dataset,
        maximum_size=2,
        first_k_letter_counts_sorted={},
    )

    assert sorted(sorted(signature_ids) for signature_ids in subblocks.values()) == [["s1", "s2"], ["s3", "s4"]]


def test_make_subblocks_skips_specter_when_single_letter_block_is_in_budget(monkeypatch):
    dataset = SimpleNamespace(
        signatures={
            "s1": _signature("s1", first="h", middle=""),
            "s2": _signature("s2", first="h", middle=""),
        },
        random_seed=0,
    )

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("cluster_with_specter should not be called when the dead-end block is already in budget")

    monkeypatch.setattr(subblocking, "cluster_with_specter", _fail_if_called)

    subblocks, _telemetry = subblocking.make_subblocks_with_telemetry(
        ["s1", "s2"],
        dataset,
        maximum_size=2,
        first_k_letter_counts_sorted={},
    )

    assert sorted(sorted(signature_ids) for signature_ids in subblocks.values()) == [["s1", "s2"]]


def test_make_subblocks_with_telemetry_uses_python_implementation(monkeypatch):
    dataset = SimpleNamespace(
        signatures={
            "s1": _signature("s1", first="anna", middle=""),
            "s2": _signature("s2", first="anna", middle=""),
        },
        random_seed=0,
    )
    observed = {"python_subdivide_called": False}

    def fake_subdivide_helper(names, sig_ids, maximum_size, starting_k=2):
        del names, maximum_size, starting_k
        observed["python_subdivide_called"] = True
        return {"an": np.array(list(sig_ids))}, {}

    monkeypatch.setattr(subblocking, "subdivide_helper", fake_subdivide_helper)

    subblocks, telemetry = subblocking.make_subblocks_with_telemetry(
        ["s1", "s2"],
        dataset,
        maximum_size=3,
        first_k_letter_counts_sorted={},
    )

    assert observed["python_subdivide_called"] is True
    assert subblocks == {"an": ["s1", "s2"]}
    assert telemetry["input_signature_count"] == 2


def test_subdivide_helper_accepts_prefix_exactly_at_capacity() -> None:
    names = np.array(["anna", "anna", "bill"])
    signature_ids = np.array(["s1", "s2", "s3"])

    output, dead_ends = subblocking.subdivide_helper(names, signature_ids, maximum_size=2, starting_k=2)

    assert dead_ends == {}
    assert {key: sorted(values.tolist()) for key, values in output.items()} == {
        "an": ["s1", "s2"],
        "bi": ["s3"],
    }


def test_normalize_orcid_for_subblocking_matches_rust_arrow_canonical_form() -> None:
    assert (
        subblocking.normalize_orcid_for_subblocking(" https://orcid.org/0000-0002-1825-0097 ") == "0000-0002-1825-0097"
    )
    assert subblocking.normalize_orcid_for_subblocking("0000\u20110002\u20111825\u20110097") == "0000-0002-1825-0097"
    assert subblocking.normalize_orcid_for_subblocking("ORCID: 000000021825009x") == "0000-0002-1825-009X"
    assert subblocking.normalize_orcid_for_subblocking("https://example.org/0000-0002-1825-0097") == (
        "0000-0002-1825-0097"
    )
    assert subblocking.normalize_orcid_for_subblocking("0000-0002-1825-009 https://example.org") is None
    assert subblocking.normalize_orcid_for_subblocking("0000-0002-1825") is None
    assert subblocking.normalize_orcid_for_subblocking("   ") is None


def test_signature_name_parts_for_subblocking_recomputes_deferred_normalized_fields() -> None:
    signature = SimpleNamespace(
        author_info_first="Arif\u2010ullah",
        author_info_middle=None,
        author_info_first_normalized_without_apostrophe=None,
        author_info_middle_normalized_without_apostrophe=None,
    )

    assert subblocking.signature_name_parts_for_subblocking(signature) == ("arif ullah", "")


def test_coauthor_blocks_from_full_arrow_normalizes_author_names(tmp_path) -> None:
    paper_authors = pa.table(
        {
            "paper_id": pa.array(["p1", "p1", "p1", "p1"], type=pa.string()),
            "position": pa.array([0, 1, 2, 3], type=pa.int64()),
            "author_name": pa.array(
                ["Ali Khan", "Waseeq-Ur-Rehamn", "Mariarosaria D'Alfonso", "Maciej G\u00f3rski"],
                type=pa.string(),
            ),
        }
    )

    out = subblocking._coauthor_blocks_by_paper_from_full_arrow(  # noqa: SLF001
        {"paper_authors": _write_ipc(tmp_path / "paper_authors.arrow", paper_authors)},
        ["p1"],
        {},
    )

    assert out == {
        "p1": [
            (0, "a khan"),
            (1, "w rehamn"),
            (2, "m alfonso"),
            (3, "m gorski"),
        ]
    }


def test_coauthor_blocks_from_rowwise_arrow_normalizes_author_names(tmp_path) -> None:
    paper_authors = pa.table(
        {
            "paper_id": pa.array(["p1", "p1"], type=pa.string()),
            "position": pa.array([0, 1], type=pa.int64()),
            "author_name": pa.array(["O'Connor", "Maciej Górski"], type=pa.string()),
        }
    )

    out = subblocking._coauthor_blocks_by_paper_from_arrow(  # noqa: SLF001
        {"paper_authors": _write_ipc(tmp_path / "paper_authors.arrow", paper_authors)},
        ["p1"],
        load_metrics={},
    )

    assert out == {"p1": [(0, "o connor"), (1, "m gorski")]}


def test_subblock_merge_candidate_metadata_preserves_middle_values_with_equals() -> None:
    assert subblocking._subblock_merge_candidate_metadata("a|middle=b=c", 2) == (2, "a", "b=c", "b=c", "b=c")


def test_make_subblocks_merges_normalized_orcid_components_when_enabled(monkeypatch):
    dataset = SimpleNamespace(
        signatures={
            "s1": _signature("s1", first="aa", middle="", orcid="https://orcid.org/0000-0002-1825-0097"),
            "s2": _signature("s2", first="bb", middle="", orcid="ORCID: 0000000218250097"),
            "s3": _signature("s3", first="aa", middle="", orcid="   "),
            "s4": _signature("s4", first="cc", middle="", orcid="   "),
        },
        random_seed=0,
    )

    call_count = {"value": 0}

    def fake_subdivide_helper(names, sig_ids, maximum_size, starting_k=2):
        del names, sig_ids, maximum_size, starting_k
        call_count["value"] += 1
        if call_count["value"] == 1:
            return {
                "a": np.array(["s1", "s3"]),
                "b": np.array(["s2"]),
                "c": np.array(["s4"]),
            }, {}
        raise AssertionError("Unexpected extra call to subdivide_helper")

    monkeypatch.setattr(subblocking, "subdivide_helper", fake_subdivide_helper)

    subblocks, telemetry = subblocking.make_subblocks_with_telemetry(
        ["s1", "s2", "s3", "s4"],
        dataset,
        maximum_size=3,
        first_k_letter_counts_sorted={},
    )

    assert telemetry["orcid_subblocking_enabled"] is True
    assert sorted(sorted(signature_ids) for signature_ids in subblocks.values()) == [["s1", "s2", "s3"], ["s4"]]


def test_make_subblocks_leaves_orcid_components_split_when_disabled(monkeypatch):
    dataset = SimpleNamespace(
        signatures={
            "s1": _signature("s1", first="aa", middle="", orcid="https://orcid.org/0000-0002-1825-0097"),
            "s2": _signature("s2", first="bb", middle="", orcid="ORCID: 0000000218250097"),
            "s3": _signature("s3", first="cc", middle=""),
        },
        random_seed=0,
    )

    call_count = {"value": 0}

    def fake_subdivide_helper(names, sig_ids, maximum_size, starting_k=2):
        del names, sig_ids, maximum_size, starting_k
        call_count["value"] += 1
        if call_count["value"] == 1:
            return {
                "a": np.array(["s1"]),
                "b": np.array(["s2"]),
                "c": np.array(["s3"]),
            }, {}
        raise AssertionError("Unexpected extra call to subdivide_helper")

    monkeypatch.setattr(subblocking, "subdivide_helper", fake_subdivide_helper)

    subblocks, telemetry = subblocking.make_subblocks_with_telemetry(
        ["s1", "s2", "s3"],
        dataset,
        maximum_size=3,
        first_k_letter_counts_sorted={},
        use_orcid_subblocking=False,
    )

    assert telemetry["orcid_subblocking_enabled"] is False
    assert sorted(sorted(signature_ids) for signature_ids in subblocks.values()) == [["s1"], ["s2"], ["s3"]]


def test_make_subblocks_does_not_merge_orcid_components_past_capacity(monkeypatch):
    dataset = SimpleNamespace(
        signatures={
            "s1": _signature("s1", first="aa", middle="", orcid="0000-0000-0000-0001"),
            "s2": _signature("s2", first="aa", middle="", orcid="0000-0000-0000-0002"),
            "s3": _signature("s3", first="bb", middle="", orcid="0000-0000-0000-0001"),
            "s4": _signature("s4", first="bb", middle=""),
            "s5": _signature("s5", first="cc", middle="", orcid="0000-0000-0000-0002"),
            "s6": _signature("s6", first="cc", middle=""),
        },
        random_seed=0,
    )

    call_count = {"value": 0}

    def fake_subdivide_helper(names, sig_ids, maximum_size, starting_k=2):
        del names, sig_ids, maximum_size, starting_k
        call_count["value"] += 1
        if call_count["value"] == 1:
            return {
                "a": np.array(["s1", "s2"]),
                "b": np.array(["s3", "s4"]),
                "c": np.array(["s5", "s6"]),
            }, {}
        raise AssertionError("Unexpected extra call to subdivide_helper")

    def fail_if_specter_called(*_args, **_kwargs):
        raise AssertionError("cluster_with_specter should not be called in this regression test")

    monkeypatch.setattr(subblocking, "subdivide_helper", fake_subdivide_helper)
    monkeypatch.setattr(subblocking, "cluster_with_specter", fail_if_specter_called)

    subblocks, telemetry = subblocking.make_subblocks_with_telemetry(
        ["s1", "s2", "s3", "s4", "s5", "s6"],
        dataset,
        maximum_size=2,
        first_k_letter_counts_sorted={},
    )

    assert sorted(sorted(signature_ids) for signature_ids in subblocks.values()) == [
        ["s1", "s2"],
        ["s3", "s4"],
        ["s5", "s6"],
    ]
    assert telemetry["orcid_merge_skipped_due_to_capacity_count"] == 2
    assert telemetry["orcid_merge_skipped_due_to_capacity_signature_count"] == 4


def _require_rust_arrow_subblocking():
    rust_module = pytest.importorskip("s2and_rust")
    if not hasattr(rust_module, "make_subblocks_with_telemetry_arrow"):
        raise pytest.skip.Exception("s2and_rust.make_subblocks_with_telemetry_arrow is unavailable")
    return rust_module


def _write_signatures_arrow(path, rows: list[tuple[str, str, str, str | None]]) -> None:
    pa = pytest.importorskip("pyarrow")
    ipc = pytest.importorskip("pyarrow.ipc")
    table = pa.table(
        {
            "signature_id": pa.array([row[0] for row in rows], type=pa.string()),
            "paper_id": pa.array([f"p_{row[0]}" for row in rows], type=pa.string()),
            "author_first": pa.array([row[1] for row in rows], type=pa.string()),
            "author_middle": pa.array([row[2] for row in rows], type=pa.string()),
            "author_last": pa.array(["wang"] * len(rows), type=pa.string()),
            "author_suffix": pa.array([""] * len(rows), type=pa.string()),
            "author_affiliations": pa.array([[] for _row in rows], type=pa.list_(pa.string())),
            "author_orcid": pa.array([row[3] for row in rows], type=pa.string()),
            "author_position": pa.array([0] * len(rows), type=pa.int64()),
        }
    )
    with path.open("wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def test_rust_arrow_make_subblocks_matches_python_fallback_path(tmp_path):
    _require_rust_arrow_subblocking()
    signatures_path = tmp_path / "signatures.arrow"
    _write_signatures_arrow(
        signatures_path,
        [
            ("s1", "hui", "", None),
            ("s2", "hui", "", None),
            ("s3", "hui", "", None),
            ("s4", "hui", "", None),
        ],
    )
    dataset = SimpleNamespace(
        signatures={
            "s1": _signature("s1", first="hui", middle=""),
            "s2": _signature("s2", first="hui", middle=""),
            "s3": _signature("s3", first="hui", middle=""),
            "s4": _signature("s4", first="hui", middle=""),
        },
        random_seed=0,
    )

    def fallback(signature_ids, anddata, target_subblock_size, compute_block_fn=None):
        del anddata, target_subblock_size, compute_block_fn
        ids = list(signature_ids)
        return {"0": ids[:2], "1": ids[2:]}

    python_subblocks, python_telemetry = subblocking.make_subblocks_with_telemetry(
        ["s1", "s2", "s3", "s4"],
        dataset,
        maximum_size=2,
        first_k_letter_counts_sorted={},
        specter_cluster_fn=fallback,
    )
    rust_subblocks, rust_telemetry = subblocking._make_subblocks_with_telemetry_arrow_rust(
        {"signatures": str(signatures_path)},
        ["s1", "s2", "s3", "s4"],
        dataset,
        maximum_size=2,
        first_k_letter_counts_sorted={},
        specter_cluster_fn=fallback,
        full_scan_without_index=True,
    )

    assert rust_subblocks == python_subblocks
    assert rust_telemetry == python_telemetry


def test_rust_arrow_make_subblocks_matches_python_orcid_repair(tmp_path):
    _require_rust_arrow_subblocking()
    signatures_path = tmp_path / "signatures.arrow"
    _write_signatures_arrow(
        signatures_path,
        [
            ("s1", "aa", "", "https://orcid.org/0000-0002-1825-0097"),
            ("s2", "bb", "", "ORCID: 0000000218250097"),
            ("s3", "aa", "", "   "),
            ("s4", "cc", "", "   "),
        ],
    )
    dataset = SimpleNamespace(
        signatures={
            "s1": _signature("s1", first="aa", middle="", orcid="https://orcid.org/0000-0002-1825-0097"),
            "s2": _signature("s2", first="bb", middle="", orcid="ORCID: 0000000218250097"),
            "s3": _signature("s3", first="aa", middle="", orcid="   "),
            "s4": _signature("s4", first="cc", middle="", orcid="   "),
        },
        random_seed=0,
    )

    python_subblocks, python_telemetry = subblocking.make_subblocks_with_telemetry(
        ["s1", "s2", "s3", "s4"],
        dataset,
        maximum_size=3,
        first_k_letter_counts_sorted={},
    )
    rust_subblocks, rust_telemetry = subblocking._make_subblocks_with_telemetry_arrow_rust(
        {"signatures": str(signatures_path)},
        ["s1", "s2", "s3", "s4"],
        dataset,
        maximum_size=3,
        first_k_letter_counts_sorted={},
        full_scan_without_index=True,
    )

    assert sorted(sorted(signature_ids) for signature_ids in rust_subblocks.values()) == sorted(
        sorted(signature_ids) for signature_ids in python_subblocks.values()
    )
    assert rust_telemetry == python_telemetry
