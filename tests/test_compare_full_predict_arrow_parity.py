from __future__ import annotations

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")

from scripts.verification.compare_full_predict_arrow_parity import (  # noqa: E402
    _assert_exact,
    _numeric_report,
    _write_raw_planner_indexes_and_layout,
)


def test_assert_exact_rejects_constraint_index_mismatch_with_equal_values() -> None:
    report = {
        "distance_comparison": {},
        "feature_constraint_comparison": {
            "feature_matrix": {
                "allclose_equal_nan": True,
                "nan_mismatch_count": 0,
            },
            "constraints": {
                "left_indices_equal": False,
                "right_indices_equal": True,
                "values_equal": True,
            },
        },
        "clusters_exact_match": True,
    }

    with pytest.raises(AssertionError, match="constraint index mismatch"):
        _assert_exact(report)


def test_numeric_report_uses_configured_nan_mismatch_policy() -> None:
    left = np.asarray([1.0, np.nan])
    right = np.asarray([1.0, 2.0])

    assert _numeric_report(left, right, treat_nan_as_mismatch=True)["nan_mismatch_count"] == 1
    assert _numeric_report(left, right, treat_nan_as_mismatch=False)["nan_mismatch_count"] == 0


def _write_ipc(path, table) -> str:
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    return str(path)


def test_parity_arrow_writer_adds_current_raw_planner_indexes(tmp_path) -> None:
    arrow_paths = {
        "signatures": _write_ipc(
            tmp_path / "signatures.arrow",
            pa.table({"signature_id": pa.array(["s1"], type=pa.string())}),
        ),
        "papers": _write_ipc(
            tmp_path / "papers.arrow",
            pa.table({"paper_id": pa.array(["p1"], type=pa.string())}),
        ),
        "paper_authors": _write_ipc(
            tmp_path / "paper_authors.arrow",
            pa.table({"paper_id": pa.array(["p1"], type=pa.string())}),
        ),
        "specter": _write_ipc(
            tmp_path / "specter.arrow",
            pa.table({"paper_id": pa.array(["p1"], type=pa.string())}),
        ),
    }

    indexed_paths, index_metrics, physical_layout = _write_raw_planner_indexes_and_layout(arrow_paths, tmp_path)

    assert indexed_paths["signatures_batch_index"].endswith("signatures.signatures_batch_index.bin")
    assert index_metrics["signatures_batch_index"]["schema_version"] == "arrow_batch_lookup_index"
    assert index_metrics["signatures_batch_index"]["magic"] == "S2ABI001"
    assert physical_layout["schema"] == "s2and_arrow_physical_v1"
