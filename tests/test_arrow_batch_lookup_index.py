from __future__ import annotations

from pathlib import Path

import pytest

from s2and.incremental_linking.feature_block import (
    read_arrow_batch_lookup_index_batch_indices,
    validate_arrow_batch_lookup_index,
    write_arrow_batch_lookup_index,
    write_arrow_ipc_table,
)


def test_raw_planner_index_rejects_same_size_unsampled_middle_rewrite_python(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")

    signature_ids = [f"key{index:013d}" for index in range(30_000)]
    path = write_arrow_ipc_table(
        pa.table(
            {
                "signature_id": pa.array(signature_ids, type=pa.string()),
                "payload": pa.array(["x" * 8] * len(signature_ids), type=pa.string()),
            }
        ),
        tmp_path / "signatures.arrow",
        max_record_batch_rows=1000,
    )
    index_path = tmp_path / "signatures.signatures_batch_index.bin"
    write_arrow_batch_lookup_index(path, index_path, key_column="signature_id", table_name="signatures")
    payload = Path(path).read_bytes()
    old_value = signature_ids[len(signature_ids) // 2].encode()
    new_value = b"new0000000000000"
    rewrite_offset = payload.index(old_value)
    assert len(old_value) == len(new_value)
    assert rewrite_offset > 65_536
    assert rewrite_offset < len(payload) - 65_536
    Path(path).write_bytes(payload[:rewrite_offset] + new_value + payload[rewrite_offset + len(old_value) :])

    with pytest.raises(ValueError, match="stale"):
        write_arrow_batch_lookup_index(
            path,
            index_path,
            key_column="signature_id",
            table_name="signatures",
            overwrite=False,
        )
    with pytest.raises(ValueError, match="stale"):
        validate_arrow_batch_lookup_index(path, index_path, key_column="signature_id")
    with pytest.raises(ValueError, match="stale"):
        read_arrow_batch_lookup_index_batch_indices(
            path,
            index_path,
            key_column="signature_id",
            values=[signature_ids[0]],
        )
