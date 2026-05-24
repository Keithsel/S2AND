"""FeatureBlock Arrow IPC, sidecar, and artifact IO helpers."""

from __future__ import annotations

import json
import shutil
import struct
import tempfile
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np

from s2and.incremental_linking.feature_block_contract import (
    FeatureBlock,
    FeatureBlockPaper,
    FeatureBlockPaperAuthor,
    FeatureBlockSignature,
    _feature_block_specter_from_mapping,
    _optional_bool,
    _optional_int,
    _optional_str,
    _strict_string_tuple,
    feature_block_signature_order_from_raw_candidate_plan,
    filter_cluster_seed_disallows_for_signature_subset,
    normalize_cluster_seed_disallow_pairs,
)

NAME_COUNTS_INDEX_SCHEMA_VERSION = "name_counts_index_v1"
ARROW_PHYSICAL_LAYOUT_SCHEMA_VERSION = "s2and_arrow_physical_v1"
ARROW_BATCH_LOOKUP_INDEX_SCHEMA_VERSION = "arrow_batch_lookup_index"
_NAME_COUNTS_INDEX_MAGIC = b"S2NCI001"
_ARROW_BATCH_LOOKUP_INDEX_MAGIC = b"S2ABI001"
_NAME_COUNTS_INDEX_HASH_DOMAIN = b"s2and-name-counts-index-v1\x00"
_ARROW_BATCH_LOOKUP_INDEX_SOURCE_HASH_DOMAIN = b"s2and-arrow-batch-lookup-index-source\x00"
_ARROW_BATCH_LOOKUP_INDEX_SOURCE_SAMPLE_BYTES = 65_536
_NAME_COUNTS_INDEX_HEADER_STRUCT = struct.Struct("<8sQQQ")
_NAME_COUNTS_INDEX_RECORD_STRUCT = struct.Struct("<QQQIId")
_ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT = struct.Struct("<8sQQQQQ")
_ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT = struct.Struct("<QII")
_FNV64_OFFSET = 14695981039346656037
_FNV64_PRIME = 1099511628211

RAW_PLANNER_ARROW_KEY_COLUMNS: dict[str, str] = {
    "signatures": "signature_id",
    "papers": "paper_id",
    "paper_authors": "paper_id",
    "specter": "paper_id",
}
RAW_PLANNER_ARROW_BATCH_INDEX_KEYS: dict[str, str] = {
    "signatures": "signatures_batch_index",
    "papers": "papers_batch_index",
    "paper_authors": "paper_authors_batch_index",
    "specter": "specter_batch_index",
}
RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS: dict[str, int] = {
    "signatures": 16_384,
    "papers": 16_384,
    "paper_authors": 16_384,
    "specter": 2_048,
}


def write_cluster_seeds_arrow(path: Path, cluster_seeds_require: Mapping[Any, Any]) -> None:
    """Write the canonical Arrow cluster-seed table."""

    import pyarrow as pa

    path.parent.mkdir(parents=True, exist_ok=True)
    items: list[tuple[str, str]] = []
    seen_signature_ids: set[str] = set()
    for signature_id, cluster_id in cluster_seeds_require.items():
        signature_key = str(signature_id)
        cluster_key = str(cluster_id)
        if not signature_key:
            raise ValueError("cluster seeds Arrow cannot contain empty signature_id values")
        if not cluster_key:
            raise ValueError(f"cluster seeds Arrow cannot contain empty cluster_id values: {signature_key!r}")
        if signature_key in seen_signature_ids:
            raise ValueError(f"cluster seeds Arrow contains duplicate signature_id: {signature_key!r}")
        seen_signature_ids.add(signature_key)
        items.append((signature_key, cluster_key))
    table = pa.table(
        {
            "signature_id": pa.array([signature_id for signature_id, _cluster_id in items], type=pa.string()),
            "cluster_id": pa.array([cluster_id for _signature_id, cluster_id in items], type=pa.string()),
        }
    )
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def read_cluster_seeds_arrow(path: Path) -> dict[str, str]:
    """Read and validate a canonical Arrow cluster-seed table."""

    import pyarrow as pa

    with pa.memory_map(str(path), "r") as source:
        table = pa.ipc.open_file(source).read_all()
    missing_columns = sorted({"signature_id", "cluster_id"} - set(table.column_names))
    if missing_columns:
        raise ValueError(f"cluster seeds Arrow is missing required columns: {missing_columns}")
    rows: dict[str, str] = {}
    for index in range(table.num_rows):
        signature_value = table["signature_id"][index].as_py()
        cluster_value = table["cluster_id"][index].as_py()
        if signature_value is None or cluster_value is None:
            raise ValueError("cluster seeds Arrow cannot contain null signature_id or cluster_id values")
        signature_id = str(signature_value)
        cluster_id = str(cluster_value)
        if not signature_id:
            raise ValueError("cluster seeds Arrow cannot contain empty signature_id values")
        if not cluster_id:
            raise ValueError(f"cluster seeds Arrow cannot contain empty cluster_id values: {signature_id!r}")
        existing_cluster_id = rows.get(signature_id)
        if existing_cluster_id is not None:
            if existing_cluster_id == cluster_id:
                continue
            raise ValueError(
                f"cluster seeds Arrow assigns signature_id {signature_id!r} to multiple clusters: "
                f"{existing_cluster_id!r} and {cluster_id!r}"
            )
        rows[signature_id] = cluster_id
    return rows


def write_cluster_seed_disallows_arrow(path: Path, pairs: Iterable[tuple[Any, Any]]) -> None:
    """Write the canonical Arrow cluster-seed disallow table."""

    import pyarrow as pa

    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_cluster_seed_disallow_pairs(pairs)
    table = pa.table(
        {
            "signature_id_1": pa.array([left for left, _right in normalized], type=pa.string()),
            "signature_id_2": pa.array([right for _left, right in normalized], type=pa.string()),
        }
    )
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def read_cluster_seed_disallows_arrow(path: Path) -> tuple[tuple[str, str], ...]:
    """Read and validate a canonical Arrow cluster-seed disallow table."""

    import pyarrow as pa

    with pa.memory_map(str(path), "r") as source:
        table = pa.ipc.open_file(source).read_all()
    missing_columns = sorted({"signature_id_1", "signature_id_2"} - set(table.column_names))
    if missing_columns:
        raise ValueError(f"cluster seed disallows Arrow is missing required columns: {missing_columns}")
    rows = []
    for left, right in zip(
        table["signature_id_1"].to_pylist(),
        table["signature_id_2"].to_pylist(),
        strict=True,
    ):
        if left is None or right is None:
            raise ValueError("cluster seed disallows Arrow cannot contain null signature ids")
        rows.append((str(left), str(right)))
    return normalize_cluster_seed_disallow_pairs(rows)


def cluster_seed_disallows_path_from_arrow_paths(arrow_paths: Mapping[str, Any] | None) -> Path | None:
    """Return the explicit cluster-seed disallow sidecar path, if configured."""

    if arrow_paths is None:
        return None
    path_value = arrow_paths.get("cluster_seed_disallows")
    if path_value is None:
        return None
    path = Path(str(path_value))
    if not path.exists():
        raise FileNotFoundError(f"Arrow path cluster_seed_disallows={path} does not exist")
    return path


def cluster_seed_disallows_from_arrow_paths(arrow_paths: Mapping[str, Any] | None) -> set[tuple[str, str]]:
    """Read explicit Arrow cluster-seed disallows, failing on stale configured paths."""

    path = cluster_seed_disallows_path_from_arrow_paths(arrow_paths)
    if path is None:
        return set()
    return set(read_cluster_seed_disallows_arrow(path))


def _stringified_arrow_paths(paths: Mapping[Any, Any], *, omit_none: bool = False) -> dict[str, str]:
    stringified: dict[str, str] = {}
    for key, value in paths.items():
        if value is None:
            if omit_none:
                continue
            raise ValueError(f"Arrow path for {key!r} is None")
        stringified[str(key)] = str(value)
    return stringified


@contextmanager
def temporary_arrow_paths_with_cluster_seeds(
    arrow_paths: Mapping[str, Any],
    cluster_seeds_require: Mapping[Any, Any],
    *,
    prefix: str,
    reuse_existing_cluster_seeds_when_empty: bool = False,
    cluster_seeds_disallow: Iterable[tuple[Any, Any]] | None = None,
) -> Iterator[dict[str, str]]:
    """Yield Arrow paths with request-scoped cluster-seed sidecars."""

    paths = _stringified_arrow_paths(arrow_paths)
    if (
        reuse_existing_cluster_seeds_when_empty
        and not cluster_seeds_require
        and cluster_seeds_disallow is None
        and paths.get("cluster_seeds") is not None
        and Path(paths["cluster_seeds"]).exists()
    ):
        yield paths
        return

    with tempfile.TemporaryDirectory(prefix=prefix) as tmpdir:
        tmpdir_path = Path(tmpdir)
        reusable_cluster_seeds_path = (
            reuse_existing_cluster_seeds_when_empty
            and not cluster_seeds_require
            and paths.get("cluster_seeds") is not None
            and Path(paths["cluster_seeds"]).exists()
        )
        if not reusable_cluster_seeds_path:
            cluster_seed_path = tmpdir_path / "cluster_seeds.arrow"
            write_cluster_seeds_arrow(cluster_seed_path, cluster_seeds_require)
            paths["cluster_seeds"] = str(cluster_seed_path)
        if cluster_seeds_disallow is not None:
            disallow_path = tmpdir_path / "cluster_seed_disallows.arrow"
            write_cluster_seed_disallows_arrow(disallow_path, cluster_seeds_disallow)
            paths["cluster_seed_disallows"] = str(disallow_path)
        yield paths


def _record_batch_limit_for_table(
    table_name: str,
    max_record_batch_rows: Mapping[str, int] | int | None,
) -> int | None:
    if max_record_batch_rows is None:
        return None
    if isinstance(max_record_batch_rows, Mapping):
        raw_limit = cast(Mapping[str, int], max_record_batch_rows).get(table_name)
        if raw_limit is None:
            return None
    else:
        raw_limit = max_record_batch_rows
    limit = int(raw_limit)
    if limit <= 0:
        raise ValueError(f"max_record_batch_rows must be positive for {table_name!r}: {limit}")
    return limit


def write_arrow_ipc_table(
    table: Any,
    path: str | Path,
    *,
    max_record_batch_rows: int | None = None,
) -> str:
    """Write one Arrow IPC file-format table and return its path."""

    import pyarrow as pa

    batch_limit = None if max_record_batch_rows is None else int(max_record_batch_rows)
    if batch_limit is not None and batch_limit <= 0:
        raise ValueError(f"max_record_batch_rows must be positive: {max_record_batch_rows}")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(output_path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table, max_chunksize=batch_limit)
    return str(output_path)


def arrow_ipc_physical_layout(path: str | Path) -> dict[str, int]:
    """Return row and record-batch layout metrics for one Arrow IPC file."""

    import pyarrow as pa

    row_count = 0
    max_batch_rows = 0
    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_file(source)
        record_batch_count = int(reader.num_record_batches)
        for batch_index in range(record_batch_count):
            batch_rows = int(reader.get_batch(batch_index).num_rows)
            row_count += batch_rows
            max_batch_rows = max(max_batch_rows, batch_rows)
    return {
        "row_count": row_count,
        "record_batch_count": record_batch_count,
        "actual_max_batch_rows": max_batch_rows,
    }


def _raise_if_record_batch_limit_exceeded(
    *,
    arrow_path: str | Path,
    table_name: str,
    batch_index: int,
    batch_rows: int,
    max_record_batch_rows: int | None,
) -> None:
    if max_record_batch_rows is None or batch_rows <= max_record_batch_rows:
        return
    batch_label = f"record batch {batch_index}" if batch_index >= 0 else "at least one record batch"
    raise ValueError(
        f"{table_name}.arrow has {batch_label} with {batch_rows} rows, "
        f"exceeding the raw-planner limit of {max_record_batch_rows}: {arrow_path!s}. "
        "Rewrite the Arrow IPC file with bounded record batches before building lookup indexes."
    )


def _read_arrow_batch_lookup_index_header(index_path: Path) -> dict[str, int | str]:
    with index_path.open("rb") as infile:
        header = infile.read(_ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT.size)
        if len(header) != _ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT.size:
            raise ValueError(f"Arrow batch lookup index is truncated: {index_path!s}")
    magic = header[:8]
    if magic != _ARROW_BATCH_LOOKUP_INDEX_MAGIC:
        raise ValueError(f"Arrow batch lookup index has invalid magic bytes: {index_path!s}")
    (
        _magic,
        record_count,
        source_size,
        source_mtime_ns,
        key_column_hash,
        source_fingerprint,
    ) = _ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT.unpack(header)
    return {
        "magic": magic.decode("ascii"),
        "record_count": int(record_count),
        "source_size": int(source_size),
        "source_mtime_ns": int(source_mtime_ns),
        "key_column_hash": int(key_column_hash),
        "source_fingerprint": int(source_fingerprint),
    }


def write_arrow_batch_lookup_index(
    arrow_path: str | Path,
    index_path: str | Path,
    *,
    key_column: str,
    table_name: str = "arrow",
    max_record_batch_rows: int | None = None,
    overwrite: bool = True,
) -> tuple[str, dict[str, int | str | bool]]:
    """Write a Rust-readable key-hash to Arrow record-batch lookup index."""

    import pyarrow as pa

    output_path = Path(index_path)
    if output_path.exists() and not overwrite:
        arrow_path_obj = Path(arrow_path)
        index_header = _read_arrow_batch_lookup_index_header(output_path)
        source_stat = arrow_path_obj.stat()
        key_column_hash = _fnv64_bytes(str(key_column).encode("utf-8"))
        if (
            int(index_header["source_size"]) != source_stat.st_size
            or int(index_header["key_column_hash"]) != key_column_hash
            or int(index_header["source_fingerprint"]) != _source_file_sample_fingerprint(arrow_path_obj)
        ):
            raise ValueError(
                f"Arrow batch lookup index is stale for {arrow_path_obj!s}: {output_path!s}. "
                "Rebuild it with overwrite=True."
            )
        layout = arrow_ipc_physical_layout(arrow_path)
        _raise_if_record_batch_limit_exceeded(
            arrow_path=arrow_path,
            table_name=table_name,
            batch_index=-1,
            batch_rows=layout["actual_max_batch_rows"],
            max_record_batch_rows=max_record_batch_rows,
        )
        return str(output_path), {
            "reused": True,
            "schema_version": ARROW_BATCH_LOOKUP_INDEX_SCHEMA_VERSION,
            **layout,
            "magic": str(index_header["magic"]),
            "record_count": int(index_header["record_count"]),
            "key_column_hash": int(index_header["key_column_hash"]),
            "source_fingerprint": int(index_header["source_fingerprint"]),
            "max_record_batch_rows": int(max_record_batch_rows or 0),
        }

    records: list[tuple[int, int]] = []
    row_count = 0
    max_batch_rows = 0
    arrow_path_obj = Path(arrow_path)
    with pa.memory_map(str(arrow_path), "r") as source:
        reader = pa.ipc.open_file(source)
        record_batch_count = int(reader.num_record_batches)
        key_column_index = reader.schema.get_field_index(key_column)
        if key_column_index < 0:
            raise KeyError(f"Arrow IPC file {arrow_path!s} is missing key column {key_column!r}")
        for batch_index in range(record_batch_count):
            batch = reader.get_batch(batch_index)
            batch_rows = int(batch.num_rows)
            max_batch_rows = max(max_batch_rows, batch_rows)
            _raise_if_record_batch_limit_exceeded(
                arrow_path=arrow_path,
                table_name=table_name,
                batch_index=batch_index,
                batch_rows=batch_rows,
                max_record_batch_rows=max_record_batch_rows,
            )
            keys = batch.column(key_column_index).to_pylist()
            row_count += len(keys)
            if any(key is None for key in keys):
                raise ValueError(
                    f"Arrow IPC file {arrow_path!s} contains null values in key column {key_column!r} "
                    f"for batch {batch_index}"
                )
            records.extend((_fnv64_bytes(str(key).encode("utf-8")), batch_index) for key in keys)

    records.sort()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_stat = arrow_path_obj.stat()
    key_column_hash = _fnv64_bytes(str(key_column).encode("utf-8"))
    source_fingerprint = _source_file_sample_fingerprint(arrow_path_obj)
    with output_path.open("wb") as outfile:
        outfile.write(
            _ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT.pack(
                _ARROW_BATCH_LOOKUP_INDEX_MAGIC,
                len(records),
                source_stat.st_size,
                source_stat.st_mtime_ns,
                key_column_hash,
                source_fingerprint,
            )
        )
        for key_hash, batch_index in records:
            outfile.write(_ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT.pack(key_hash, batch_index, 0))
    return str(output_path), {
        "reused": False,
        "schema_version": ARROW_BATCH_LOOKUP_INDEX_SCHEMA_VERSION,
        "magic": _ARROW_BATCH_LOOKUP_INDEX_MAGIC.decode("ascii"),
        "row_count": row_count,
        "record_count": len(records),
        "key_column_hash": key_column_hash,
        "source_fingerprint": source_fingerprint,
        "record_batch_count": record_batch_count,
        "actual_max_batch_rows": max_batch_rows,
        "max_record_batch_rows": int(max_record_batch_rows or 0),
    }


def write_raw_arrow_batch_lookup_indexes(
    paths: Mapping[str, Any],
    output_dir: str | Path | None = None,
    *,
    max_record_batch_rows: Mapping[str, int] | int | None = RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS,
    overwrite: bool = True,
) -> tuple[dict[str, str], dict[str, dict[str, int | str | bool]]]:
    """Write optional batch lookup indexes for raw Arrow planner inputs."""

    output_path = Path(output_dir) if output_dir is not None else None
    indexed_paths = _stringified_arrow_paths(paths, omit_none=True)
    metrics: dict[str, dict[str, int | str | bool]] = {}
    for arrow_key, key_column in RAW_PLANNER_ARROW_KEY_COLUMNS.items():
        arrow_value = paths.get(arrow_key)
        if arrow_value is None:
            continue
        index_key = RAW_PLANNER_ARROW_BATCH_INDEX_KEYS[arrow_key]
        batch_limit = _record_batch_limit_for_table(arrow_key, max_record_batch_rows)
        arrow_file_path = Path(str(arrow_value))
        index_file_path = (
            output_path / f"{arrow_file_path.stem}.{index_key}.bin"
            if output_path is not None
            else arrow_file_path.with_name(f"{arrow_file_path.stem}.{index_key}.bin")
        )
        index_file, index_metrics = write_arrow_batch_lookup_index(
            arrow_file_path,
            index_file_path,
            key_column=key_column,
            table_name=arrow_key,
            max_record_batch_rows=batch_limit,
            overwrite=overwrite,
        )
        indexed_paths[index_key] = index_file
        metrics[index_key] = index_metrics
    return indexed_paths, metrics


def raw_planner_arrow_physical_layout(
    paths: Mapping[str, Any],
    *,
    max_record_batch_rows: Mapping[str, int] | int | None = RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS,
) -> dict[str, Any]:
    """Build manifest-ready physical-layout metadata for raw-planner Arrow inputs."""

    tables: dict[str, dict[str, int | str | bool]] = {}
    for table_name in RAW_PLANNER_ARROW_KEY_COLUMNS:
        path_value = paths.get(table_name)
        if path_value is None:
            continue
        batch_limit = _record_batch_limit_for_table(table_name, max_record_batch_rows)
        layout = arrow_ipc_physical_layout(path_value)
        _raise_if_record_batch_limit_exceeded(
            arrow_path=path_value,
            table_name=table_name,
            batch_index=-1,
            batch_rows=layout["actual_max_batch_rows"],
            max_record_batch_rows=batch_limit,
        )
        index_key = RAW_PLANNER_ARROW_BATCH_INDEX_KEYS[table_name]
        index_path = paths.get(index_key)
        tables[table_name] = {
            "key": RAW_PLANNER_ARROW_KEY_COLUMNS[table_name],
            "max_record_batch_rows": int(batch_limit or 0),
            "batch_index_path_key": index_key,
            "batch_index_present": bool(index_path),
            **layout,
        }
    return {
        "schema": ARROW_PHYSICAL_LAYOUT_SCHEMA_VERSION,
        "optimized_for": "incremental_raw_candidate_planning",
        "tables": tables,
    }


def write_feature_block_arrow_tables(
    feature_block: FeatureBlock,
    output_dir: str | Path,
    *,
    include_empty_cluster_seeds: bool = False,
    max_record_batch_rows: Mapping[str, int] | int | None = RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS,
    overwrite: bool = True,
) -> dict[str, str]:
    """Write `FeatureBlock` Arrow IPC tables and return paths keyed by table name."""

    output_path = Path(output_dir)
    tables = feature_block.to_arrow_tables()
    paths: dict[str, str] = {}
    for name, table in tables.items():
        if (
            name in {"cluster_seeds", "cluster_seed_disallows"}
            and table.num_rows == 0
            and not include_empty_cluster_seeds
        ):
            continue
        path = output_path / f"{name}.arrow"
        if overwrite or not path.exists():
            write_arrow_ipc_table(
                table,
                path,
                max_record_batch_rows=_record_batch_limit_for_table(name, max_record_batch_rows),
            )
        paths[name] = str(path)
    return paths


def write_name_counts_arrow(output_dir: str | Path, *, overwrite: bool = False) -> tuple[str, dict[str, int | bool]]:
    """Write the global name-count lookup as a Rust-readable Arrow IPC table."""

    import pyarrow as pa

    from s2and.data import _load_name_counts_cached

    output_path = Path(output_dir) / "name_counts.arrow"
    if output_path.exists() and not overwrite:
        return str(output_path), {"reused": True}

    first_dict, last_dict, first_last_dict, last_first_initial_dict = _load_name_counts_cached()
    kinds: list[str] = []
    names: list[str] = []
    counts: list[float] = []
    metrics: dict[str, int | bool] = {"reused": False}
    for kind, mapping in (
        ("first", first_dict),
        ("last", last_dict),
        ("first_last", first_last_dict),
        ("last_first_initial", last_first_initial_dict),
    ):
        metrics[f"{kind}_count"] = len(mapping)
        for name, count in mapping.items():
            kinds.append(kind)
            names.append(str(name))
            counts.append(float(count))

    table = pa.table(
        {
            "kind": pa.array(kinds, type=pa.string()),
            "name": pa.array(names, type=pa.string()),
            "count": pa.array(counts, type=pa.float64()),
        }
    )
    write_arrow_ipc_table(table, output_path)
    metrics["row_count"] = table.num_rows
    return str(output_path), metrics


def _fnv64_update(digest: int, value: bytes) -> int:
    for byte in value:
        digest ^= byte
        digest = (digest * _FNV64_PRIME) & 0xFFFFFFFFFFFFFFFF
    return digest


def _fnv64_bytes(value: bytes) -> int:
    return _fnv64_update(_FNV64_OFFSET, value)


def _source_file_sample_fingerprint(path: Path) -> int:
    source_size = path.stat().st_size
    digest = _fnv64_bytes(_ARROW_BATCH_LOOKUP_INDEX_SOURCE_HASH_DOMAIN)
    digest = _fnv64_update(digest, int(source_size).to_bytes(8, "little", signed=False))
    with path.open("rb") as infile:
        digest = _fnv64_update(
            digest,
            infile.read(min(_ARROW_BATCH_LOOKUP_INDEX_SOURCE_SAMPLE_BYTES, source_size)),
        )
        if source_size > _ARROW_BATCH_LOOKUP_INDEX_SOURCE_SAMPLE_BYTES:
            suffix_start = max(
                _ARROW_BATCH_LOOKUP_INDEX_SOURCE_SAMPLE_BYTES,
                source_size - _ARROW_BATCH_LOOKUP_INDEX_SOURCE_SAMPLE_BYTES,
            )
            infile.seek(suffix_start)
            digest = _fnv64_update(digest, infile.read(source_size - suffix_start))
    return digest


def _name_counts_index_hashes(kind: str, name_bytes: bytes) -> tuple[int, int]:
    return (
        _fnv64_bytes(name_bytes),
        _fnv64_bytes(_NAME_COUNTS_INDEX_HASH_DOMAIN + kind.encode("utf-8") + b"\x00" + name_bytes),
    )


def _write_name_count_index_file(path: Path, kind: str, mapping: Mapping[Any, Any]) -> dict[str, int]:
    records: list[tuple[int, int, bytes, float]] = []
    for raw_name, raw_count in mapping.items():
        name_bytes = str(raw_name).encode("utf-8")
        hash_1, hash_2 = _name_counts_index_hashes(kind, name_bytes)
        records.append((hash_1, hash_2, name_bytes, float(raw_count)))
    records.sort(key=lambda item: (item[0], item[1], item[2]))

    blob = bytearray()
    packed_records = bytearray()
    for hash_1, hash_2, name_bytes, count in records:
        name_offset = len(blob)
        blob.extend(name_bytes)
        packed_records.extend(
            _NAME_COUNTS_INDEX_RECORD_STRUCT.pack(
                hash_1,
                hash_2,
                name_offset,
                len(name_bytes),
                0,
                count,
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    blob_offset = _NAME_COUNTS_INDEX_HEADER_STRUCT.size + len(packed_records)
    with path.open("wb") as output:
        output.write(
            _NAME_COUNTS_INDEX_HEADER_STRUCT.pack(
                _NAME_COUNTS_INDEX_MAGIC,
                len(records),
                blob_offset,
                len(blob),
            )
        )
        output.write(packed_records)
        output.write(blob)
    return {"record_count": len(records), "byte_count": path.stat().st_size}


def _name_counts_index_manifest_paths(index_dir: Path) -> dict[str, Path] | None:
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise TypeError(f"name-count index manifest must contain an object: {manifest_path}")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError(f"name-count index manifest is missing files: {manifest_path}")
    resolved: dict[str, Path] = {}
    for kind in ("first", "last", "first_last", "last_first_initial"):
        entry = files.get(kind)
        if not isinstance(entry, Mapping) or entry.get("path") is None:
            raise ValueError(f"name-count index manifest is missing files.{kind}.path: {manifest_path}")
        raw_path = Path(str(entry["path"]))
        resolved[kind] = raw_path if raw_path.is_absolute() else index_dir / raw_path
    return resolved


def _name_counts_index_complete(index_dir: Path) -> bool:
    manifest_paths = _name_counts_index_manifest_paths(index_dir)
    return manifest_paths is not None and all(path.exists() for path in manifest_paths.values())


def _remove_stale_name_counts_generations(index_dir: Path, current_generation_name: str) -> None:
    generations_dir = index_dir / "generations"
    if not generations_dir.exists():
        return
    for child in generations_dir.iterdir():
        if child.name == current_generation_name or child.name.startswith(".") or not child.is_dir():
            continue
        shutil.rmtree(child)


def write_name_counts_index(output_dir: str | Path, *, overwrite: bool = False) -> tuple[str, dict[str, int | bool]]:
    """Write the global name-count lookup as exact-verified sorted binary indexes."""

    from s2and.data import _load_name_counts_cached

    index_dir = Path(output_dir) / "name_counts_index"
    manifest_path = index_dir / "manifest.json"
    if not overwrite and _name_counts_index_complete(index_dir):
        return str(index_dir), {"reused": True}

    first_dict, last_dict, first_last_dict, last_first_initial_dict = _load_name_counts_cached()
    metrics: dict[str, int | bool] = {"reused": False}
    total_records = 0
    total_bytes = 0
    manifest_files: dict[str, dict[str, int | str]] = {}
    generations_dir = index_dir / "generations"
    generations_dir.mkdir(parents=True, exist_ok=True)
    generation_name = f"gen-{uuid.uuid4().hex}"
    tmp_generation_dir = Path(tempfile.mkdtemp(prefix=f".{generation_name}.", dir=str(generations_dir)))
    generation_dir = generations_dir / generation_name
    tmp_manifest_path = index_dir / f".manifest.{generation_name}.json"
    manifest_published = False
    try:
        for kind, mapping in (
            ("first", first_dict),
            ("last", last_dict),
            ("first_last", first_last_dict),
            ("last_first_initial", last_first_initial_dict),
        ):
            filename = f"{kind}.bin"
            tmp_file = tmp_generation_dir / filename
            file_metrics = _write_name_count_index_file(tmp_file, kind, mapping)
            record_count = file_metrics["record_count"]
            byte_count = file_metrics["byte_count"]
            metrics[f"{kind}_count"] = record_count
            metrics[f"{kind}_bytes"] = byte_count
            total_records += record_count
            total_bytes += byte_count
            manifest_files[kind] = {
                "path": f"generations/{generation_name}/{filename}",
                "record_count": record_count,
                "byte_count": byte_count,
            }

        manifest = {
            "schema_version": NAME_COUNTS_INDEX_SCHEMA_VERSION,
            "magic": _NAME_COUNTS_INDEX_MAGIC.decode("ascii"),
            "record_layout": "hash1:u64,hash2:u64,name_offset:u64,name_len:u32,reserved:u32,count:f64",
            "sort_order": "hash1,hash2,utf8_name_bytes",
            "hash": "fnv1a64(name_bytes), fnv1a64(domain + kind + NUL + name_bytes)",
            "exact_string_verification": True,
            "files": manifest_files,
        }
        tmp_generation_dir.rename(generation_dir)
        tmp_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for entry in manifest_files.values():
            path = index_dir / str(entry["path"])
            if not path.exists():
                raise FileNotFoundError(f"name-count index generation is incomplete: {path}")
        tmp_manifest_path.replace(manifest_path)
        manifest_published = True
        _remove_stale_name_counts_generations(index_dir, generation_name)
    finally:
        if tmp_manifest_path.exists():
            tmp_manifest_path.unlink()
        if tmp_generation_dir.exists():
            shutil.rmtree(tmp_generation_dir)
        if generation_dir.exists() and not manifest_published:
            shutil.rmtree(generation_dir)
    metrics["row_count"] = total_records
    metrics["byte_count"] = total_bytes
    return str(index_dir), metrics


def _resolve_name_pairs(name_tuples: set[tuple[str, str]] | str | None) -> set[tuple[str, str]]:
    from s2and.data import _load_name_tuples_from_file

    if name_tuples == "filtered":
        return _load_name_tuples_from_file("s2and_name_tuples_filtered.txt")
    if name_tuples is None:
        return _load_name_tuples_from_file("s2and_name_tuples.txt")
    if isinstance(name_tuples, set):
        return name_tuples
    raise ValueError("name_tuples must be None, 'filtered', or a set of (first_a, first_b) tuples")


def write_name_pairs_arrow(
    name_tuples: set[tuple[str, str]] | str | None,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[str, dict[str, int | bool]]:
    """Write name-alias pairs as a Rust-readable Arrow IPC table."""

    import pyarrow as pa

    output_path = Path(output_dir) / "name_pairs.arrow"
    if output_path.exists() and not overwrite:
        return str(output_path), {"reused": True}

    pairs = sorted((str(left), str(right)) for left, right in _resolve_name_pairs(name_tuples))
    table = pa.table(
        {
            "name_1": pa.array([left for left, _right in pairs], type=pa.string()),
            "name_2": pa.array([right for _left, right in pairs], type=pa.string()),
        }
    )
    write_arrow_ipc_table(table, output_path)
    return str(output_path), {"reused": False, "row_count": table.num_rows}


def _arrow_rows_by_unique_key(
    rows: Iterable[Mapping[str, Any]],
    *,
    table_name: str,
    key_column: str,
) -> dict[str, Mapping[str, Any]]:
    rows_by_key: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        key_value = row.get(key_column)
        if key_value is None:
            raise ValueError(f"{table_name} Arrow cannot contain null {key_column} values")
        key = str(key_value)
        if not key:
            raise ValueError(f"{table_name} Arrow cannot contain empty {key_column} values")
        if key in rows_by_key:
            raise ValueError(f"{table_name} Arrow contains duplicate {key_column}: {key!r}")
        rows_by_key[key] = row
    return rows_by_key


def feature_block_from_arrow_paths(
    paths: Mapping[str, Any],
    *,
    raw_candidate_plan: Mapping[str, Any],
    include_specter: bool = False,
) -> FeatureBlock:
    """Build a mini signal-only `FeatureBlock` from Arrow IPC inputs."""

    pa = __import__("pyarrow")
    pc = __import__("pyarrow.compute").compute

    signature_order = feature_block_signature_order_from_raw_candidate_plan(raw_candidate_plan)
    selected_signature_ids = tuple(signature_order.signature_ids)
    selected_signature_id_set = set(selected_signature_ids)

    signatures_table = _read_arrow_ipc_table(pa, paths["signatures"])
    signatures_table = _filter_arrow_table_by_values(pa, pc, signatures_table, "signature_id", selected_signature_ids)
    signatures_by_id = _arrow_rows_by_unique_key(
        signatures_table.to_pylist(),
        table_name="signatures",
        key_column="signature_id",
    )
    missing_signatures = [
        signature_id for signature_id in selected_signature_ids if signature_id not in signatures_by_id
    ]
    if missing_signatures:
        raise ValueError(f"Arrow signatures are missing raw-plan signature ids: {missing_signatures[:10]}")
    signature_rows = tuple(
        _feature_block_signature_from_arrow_row(signature_id, signatures_by_id[signature_id])
        for signature_id in selected_signature_ids
    )

    paper_ids = tuple(dict.fromkeys(row.paper_id for row in signature_rows))
    papers_table = _read_arrow_ipc_table(pa, paths["papers"])
    papers_table = _filter_arrow_table_by_values(pa, pc, papers_table, "paper_id", paper_ids)
    papers_by_id = _arrow_rows_by_unique_key(
        papers_table.to_pylist(),
        table_name="papers",
        key_column="paper_id",
    )
    missing_paper_ids = [paper_id for paper_id in paper_ids if paper_id not in papers_by_id]
    if missing_paper_ids:
        raise ValueError(f"Arrow papers are missing signature paper_ids: {missing_paper_ids[:10]}")
    paper_rows = tuple(_feature_block_paper_from_arrow_row(paper_id, papers_by_id[paper_id]) for paper_id in paper_ids)

    paper_author_rows: tuple[FeatureBlockPaperAuthor, ...] = ()
    paper_authors_path = paths.get("paper_authors")
    if paper_authors_path is not None:
        paper_authors_table = _read_arrow_ipc_table(pa, paper_authors_path)
        paper_authors_table = _filter_arrow_table_by_values(pa, pc, paper_authors_table, "paper_id", paper_ids)
        paper_author_row_list: list[FeatureBlockPaperAuthor] = []
        seen_paper_author_positions: set[tuple[str, int]] = set()
        for row in paper_authors_table.to_pylist():
            paper_id_value = row.get("paper_id")
            if paper_id_value is None:
                raise ValueError("paper_authors Arrow cannot contain null paper_id values")
            paper_id = str(paper_id_value)
            if not paper_id:
                raise ValueError("paper_authors Arrow cannot contain empty paper_id values")
            if row.get("position") is None:
                raise ValueError("paper_authors Arrow cannot contain null position values")
            position = int(row["position"])
            key = (paper_id, position)
            if key in seen_paper_author_positions:
                raise ValueError(f"paper_authors Arrow contains duplicate (paper_id, position): {key!r}")
            seen_paper_author_positions.add(key)
            paper_author_row_list.append(
                FeatureBlockPaperAuthor(
                    paper_id=paper_id,
                    position=position,
                    author_name=str(row.get("author_name") or ""),
                )
            )
        paper_author_rows = tuple(paper_author_row_list)

    component_members = raw_candidate_plan.get("component_members")
    if not isinstance(component_members, Mapping):
        raise ValueError("raw candidate plan must include component_members")
    require_pairs = tuple(
        (str(signature_id), str(component_key))
        for component_key, members in component_members.items()
        for signature_id in members
        if str(signature_id) in selected_signature_id_set
    )
    disallow_pairs: tuple[tuple[str, str], ...] = ()
    disallow_path = paths.get("cluster_seed_disallows")
    if disallow_path is not None:
        disallow_table = _read_arrow_ipc_table(pa, disallow_path)
        missing_disallow_columns = sorted({"signature_id_1", "signature_id_2"}.difference(disallow_table.column_names))
        if missing_disallow_columns:
            raise ValueError(f"cluster seed disallows Arrow is missing required columns: {missing_disallow_columns}")
        disallow_rows: list[tuple[str, str]] = []
        for left, right in zip(
            disallow_table["signature_id_1"].to_pylist(),
            disallow_table["signature_id_2"].to_pylist(),
            strict=True,
        ):
            if left is None or right is None:
                raise ValueError("cluster seed disallows Arrow cannot contain null signature ids")
            disallow_rows.append((str(left), str(right)))
        disallow_pairs = filter_cluster_seed_disallows_for_signature_subset(
            disallow_rows,
            selected_signature_id_set,
        )

    specter_paper_ids: tuple[str, ...] = ()
    specter_embeddings: np.ndarray | None = None
    specter_path = paths.get("specter")
    if include_specter and specter_path is not None:
        specter_table = _read_arrow_ipc_table(pa, specter_path)
        specter_table = _filter_arrow_table_by_values(pa, pc, specter_table, "paper_id", paper_ids)
        specter_by_id = _arrow_rows_by_unique_key(
            specter_table.to_pylist(),
            table_name="specter",
            key_column="paper_id",
        )
        specter_payload: dict[str, Any] = {}
        for paper_id in paper_ids:
            row = specter_by_id.get(paper_id)
            if row is None:
                continue
            embedding = row.get("embedding")
            if embedding is None:
                raise ValueError("specter Arrow cannot contain null embedding values")
            specter_payload[paper_id] = embedding
        specter_paper_ids, specter_embeddings = _feature_block_specter_from_mapping(paper_ids, specter_payload)
    return FeatureBlock(
        signatures=signature_rows,
        papers=paper_rows,
        paper_authors=paper_author_rows,
        cluster_seeds_require=require_pairs,
        cluster_seeds_disallow=disallow_pairs,
        query_signature_ids=signature_order.query_signature_ids,
        specter_paper_ids=specter_paper_ids,
        specter_embeddings=specter_embeddings,
    )


def _read_arrow_ipc_table(pa: Any, path: Any) -> Any:
    with pa.memory_map(str(path), "r") as source:
        return pa.ipc.open_file(source).read_all()


def _filter_arrow_table_by_values(pa: Any, pc: Any, table: Any, column: str, values: Sequence[str]) -> Any:
    value_list = [str(value) for value in values]
    if not value_list:
        return table.slice(0, 0)
    mask = pc.is_in(table[column], value_set=pa.array(value_list, type=table[column].type))
    return table.filter(mask)


def _feature_block_signature_from_arrow_row(signature_id: str, row: Mapping[str, Any]) -> FeatureBlockSignature:
    return FeatureBlockSignature(
        signature_id=str(row.get("signature_id", signature_id)),
        paper_id=str(row["paper_id"]),
        author_first=_optional_str(row.get("author_first")),
        author_middle=_optional_str(row.get("author_middle")),
        author_last=_optional_str(row.get("author_last")),
        author_suffix=_optional_str(row.get("author_suffix")),
        author_affiliations=_strict_string_tuple(
            row.get("author_affiliations"),
            field_name="signatures.author_affiliations",
        ),
        author_orcid=_optional_str(row.get("author_orcid")),
        author_position=_optional_int(row.get("author_position"), field_name="signatures.author_position"),
        author_block=_optional_str(row.get("author_block")),
        author_email=_optional_str(row.get("author_email")),
        source_author_ids=_strict_string_tuple(
            row.get("source_author_ids"),
            field_name="signatures.source_author_ids",
            skip_none=True,
        ),
    )


def _feature_block_paper_from_arrow_row(paper_id: str, row: Mapping[str, Any]) -> FeatureBlockPaper:
    return FeatureBlockPaper(
        paper_id=str(row.get("paper_id", paper_id)),
        title=_optional_str(row.get("title")),
        abstract=_optional_str(row.get("abstract")),
        venue=_optional_str(row.get("venue")),
        journal_name=_optional_str(row.get("journal_name")),
        year=_optional_int(row.get("year"), field_name="papers.year"),
        predicted_language=_optional_str(row.get("predicted_language")),
        is_reliable=_optional_bool(row.get("is_reliable"), field_name="papers.is_reliable"),
    )
