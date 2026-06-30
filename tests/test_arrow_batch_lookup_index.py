from __future__ import annotations

import os
from pathlib import Path

import pytest

import s2and.incremental_linking.feature_block_arrow as feature_block_arrow_module
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


def test_raw_planner_index_rejects_source_changed_while_building(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pa = pytest.importorskip("pyarrow")

    path = write_arrow_ipc_table(
        pa.table({"signature_id": pa.array(["s1", "s2"], type=pa.string())}),
        tmp_path / "signatures.arrow",
        max_record_batch_rows=1,
    )
    real_reader = feature_block_arrow_module._read_arrow_batch_lookup_records  # noqa: SLF001

    def mutating_reader(*args, **kwargs):
        result = real_reader(*args, **kwargs)
        arrow_path = Path(args[0])
        stat = arrow_path.stat()
        os.utime(arrow_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        return result

    monkeypatch.setattr(feature_block_arrow_module, "_read_arrow_batch_lookup_records", mutating_reader)

    with pytest.raises(ValueError, match="changed while building batch lookup index"):
        write_arrow_batch_lookup_index(
            path,
            tmp_path / "signatures.signatures_batch_index.bin",
            key_column="signature_id",
            table_name="signatures",
        )


def test_raw_planner_index_rejects_source_changed_while_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pa = pytest.importorskip("pyarrow")

    path = write_arrow_ipc_table(
        pa.table({"signature_id": pa.array(["s1", "s2"], type=pa.string())}),
        tmp_path / "signatures.arrow",
        max_record_batch_rows=1,
    )
    index_path = tmp_path / "signatures.signatures_batch_index.bin"
    write_arrow_batch_lookup_index(path, index_path, key_column="signature_id", table_name="signatures")
    real_fingerprint_once = feature_block_arrow_module._source_file_fingerprint_once  # noqa: SLF001

    def mutating_fingerprint_once(path_arg: Path, *, source_size: int) -> int:
        fingerprint = real_fingerprint_once(path_arg, source_size=source_size)
        stat = path_arg.stat()
        os.utime(path_arg, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        return fingerprint

    monkeypatch.setattr(feature_block_arrow_module, "_source_file_fingerprint_once", mutating_fingerprint_once)

    with pytest.raises(ValueError, match="changed while reading batch lookup index"):
        read_arrow_batch_lookup_index_batch_indices(
            path,
            index_path,
            key_column="signature_id",
            values=["s1"],
        )


def test_request_time_batch_lookup_does_not_fingerprint_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pa = pytest.importorskip("pyarrow")

    path = write_arrow_ipc_table(
        pa.table({"signature_id": pa.array(["s1", "s2"], type=pa.string())}),
        tmp_path / "signatures.arrow",
        max_record_batch_rows=1,
    )
    index_path = tmp_path / "signatures.signatures_batch_index.bin"
    write_arrow_batch_lookup_index(path, index_path, key_column="signature_id", table_name="signatures")

    def fail_fingerprint_once(path_arg: Path, *, source_size: int) -> int:
        raise AssertionError(f"request-time lookup should not fingerprint {path_arg} size={source_size}")

    monkeypatch.setattr(feature_block_arrow_module, "_source_file_fingerprint_once", fail_fingerprint_once)

    assert feature_block_arrow_module.read_arrow_batch_lookup_index_batch_indices_for_request(
        path,
        index_path,
        key_column="signature_id",
        values=["s1"],
    ) == {0}
