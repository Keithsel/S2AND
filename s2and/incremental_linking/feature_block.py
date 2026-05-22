"""Narrow inference-block contract for incremental linking.

`ANDData` remains the broad reference object. This module defines the smaller
shape that fast inference paths should target before entering Rust.
"""

from __future__ import annotations

import json
import struct
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FEATURE_BLOCK_SCHEMA_VERSION = "feature_block_v2"
FEATURE_BLOCK_ARROW_MANIFEST_SCHEMA_VERSION = "feature_block_arrow_v2"

NAME_COUNTS_INDEX_SCHEMA_VERSION = "name_counts_index_v1"
ARROW_PHYSICAL_LAYOUT_SCHEMA_VERSION = "s2and_arrow_physical_v1"
ARROW_BATCH_LOOKUP_INDEX_SCHEMA_VERSION = "arrow_batch_lookup_index_v1"
_NAME_COUNTS_INDEX_MAGIC = b"S2NCI001"
_ARROW_BATCH_LOOKUP_INDEX_MAGIC = b"S2ABI001"
_NAME_COUNTS_INDEX_HASH_DOMAIN = b"s2and-name-counts-index-v1\x00"
_NAME_COUNTS_INDEX_HEADER_STRUCT = struct.Struct("<8sQQQ")
_NAME_COUNTS_INDEX_RECORD_STRUCT = struct.Struct("<QQQIId")
_ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT = struct.Struct("<8sQQQ")
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


@dataclass
class TemporaryArrowPaths:
    """Arrow path bundle with optional request-scoped temp resources."""

    paths: dict[str, str]
    _tmpdir: tempfile.TemporaryDirectory[str] | None = field(default=None, repr=False)

    def close(self) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def __del__(self) -> None:
        self.close()


def normalize_cluster_seed_disallow_pairs(
    pairs: Iterable[tuple[Any, Any]],
    *,
    valid_signature_ids: Iterable[str] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return canonical undirected disallow pairs after schema validation."""

    valid_signature_id_set = None if valid_signature_ids is None else {str(value) for value in valid_signature_ids}
    normalized: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for left, right in pairs:
        left_id = str(left)
        right_id = str(right)
        if not left_id or not right_id:
            raise ValueError("cluster seed disallow pairs cannot contain empty signature ids")
        if left_id == right_id:
            raise ValueError(f"cluster seed disallow pair cannot be a self-pair: {left_id!r}")
        if valid_signature_id_set is not None:
            missing = sorted({left_id, right_id}.difference(valid_signature_id_set))
            if missing:
                raise ValueError(f"cluster seed disallow pair contains signatures missing from FeatureBlock: {missing}")
        pair = tuple(sorted((left_id, right_id)))
        if pair in seen:
            raise ValueError(f"cluster seed disallow pair is duplicated: {pair}")
        seen.add(pair)
        normalized.append(pair)
    return tuple(normalized)


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


def arrow_paths_with_temporary_cluster_seeds(
    arrow_paths: Mapping[str, Any],
    cluster_seeds_require: Mapping[Any, Any],
    *,
    prefix: str,
    reuse_existing_cluster_seeds_when_empty: bool = False,
) -> TemporaryArrowPaths:
    """Return Arrow paths with a request-scoped cluster-seeds sidecar."""

    paths = {str(key): str(value) for key, value in arrow_paths.items()}
    if (
        reuse_existing_cluster_seeds_when_empty
        and not cluster_seeds_require
        and paths.get("cluster_seeds") is not None
        and Path(paths["cluster_seeds"]).exists()
    ):
        return TemporaryArrowPaths(paths=paths)

    tmpdir = tempfile.TemporaryDirectory(prefix=prefix)
    output_path = Path(tmpdir.name) / "cluster_seeds.arrow"
    write_cluster_seeds_arrow(output_path, cluster_seeds_require)
    paths["cluster_seeds"] = str(output_path)
    return TemporaryArrowPaths(paths=paths, _tmpdir=tmpdir)


@dataclass(frozen=True)
class FeatureBlockSignature:
    """One signature row in the inference contract."""

    signature_id: str
    paper_id: str
    author_first: str | None
    author_middle: str | None
    author_last: str | None
    author_suffix: str | None
    author_affiliations: tuple[str, ...]
    author_orcid: str | None
    author_position: int | None
    author_block: str | None = None
    author_email: str | None = None
    source_author_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "signature_id", str(self.signature_id))
        object.__setattr__(self, "paper_id", str(self.paper_id))
        object.__setattr__(self, "author_affiliations", tuple(str(value) for value in self.author_affiliations))
        object.__setattr__(self, "source_author_ids", tuple(str(value) for value in self.source_author_ids))
        if not self.signature_id:
            raise ValueError("FeatureBlockSignature.signature_id must be non-empty")
        if not self.paper_id:
            raise ValueError(f"FeatureBlockSignature.paper_id must be non-empty for {self.signature_id!r}")
        if self.author_position is not None:
            object.__setattr__(self, "author_position", int(self.author_position))


@dataclass(frozen=True)
class FeatureBlockPaper:
    """One paper row in the inference contract."""

    paper_id: str
    title: str | None
    abstract: str | None
    venue: str | None
    journal_name: str | None
    year: int | None
    predicted_language: str | None = None
    is_reliable: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "paper_id", str(self.paper_id))
        if not self.paper_id:
            raise ValueError("FeatureBlockPaper.paper_id must be non-empty")
        if self.year is not None:
            object.__setattr__(self, "year", int(self.year))
        if self.is_reliable is not None:
            object.__setattr__(self, "is_reliable", bool(self.is_reliable))


@dataclass(frozen=True)
class FeatureBlockPaperAuthor:
    """One paper-author child row in the inference contract."""

    paper_id: str
    position: int
    author_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "paper_id", str(self.paper_id))
        object.__setattr__(self, "position", int(self.position))
        object.__setattr__(self, "author_name", str(self.author_name))
        if not self.paper_id:
            raise ValueError("FeatureBlockPaperAuthor.paper_id must be non-empty")


@dataclass(frozen=True)
class FeatureBlockSignatureOrder:
    """Deterministic mini-block signature order for numeric linker arrays."""

    signature_ids: tuple[str, ...]
    query_signature_ids: tuple[str, ...] = ()
    schema_version: str = FEATURE_BLOCK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        signature_ids = tuple(str(value) for value in self.signature_ids)
        query_signature_ids = tuple(str(value) for value in self.query_signature_ids)
        if len(set(signature_ids)) != len(signature_ids):
            raise ValueError("FeatureBlockSignatureOrder.signature_ids must be unique")
        missing_queries = sorted(set(query_signature_ids) - set(signature_ids))
        if missing_queries:
            raise ValueError(f"query_signature_ids are missing from signature_ids: {missing_queries}")
        object.__setattr__(self, "signature_ids", signature_ids)
        object.__setattr__(self, "query_signature_ids", query_signature_ids)

    @property
    def signature_id_to_index(self) -> dict[str, int]:
        """Return this order as the numeric linker index map."""

        return {signature_id: index for index, signature_id in enumerate(self.signature_ids)}


@dataclass(frozen=True)
class FeatureBlock:
    """Typed inference inputs, smaller than full `ANDData`."""

    signatures: tuple[FeatureBlockSignature, ...]
    papers: tuple[FeatureBlockPaper, ...] = ()
    paper_authors: tuple[FeatureBlockPaperAuthor, ...] = ()
    cluster_seeds_require: tuple[tuple[str, str], ...] = ()
    cluster_seeds_disallow: tuple[tuple[str, str], ...] = ()
    query_signature_ids: tuple[str, ...] = ()
    specter_paper_ids: tuple[str, ...] = ()
    specter_embeddings: np.ndarray | None = None
    schema_version: str = FEATURE_BLOCK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        signature_ids = self.signature_ids
        if len(set(signature_ids)) != len(signature_ids):
            raise ValueError("FeatureBlock signatures must have unique signature_id values")
        paper_ids = tuple(paper.paper_id for paper in self.papers)
        if len(set(paper_ids)) != len(paper_ids):
            raise ValueError("FeatureBlock papers must have unique paper_id values")

        signature_id_set = set(signature_ids)
        query_signature_ids = tuple(str(value) for value in self.query_signature_ids)
        missing_queries = sorted(set(query_signature_ids) - signature_id_set)
        if missing_queries:
            raise ValueError(f"query_signature_ids are missing from FeatureBlock signatures: {missing_queries}")
        object.__setattr__(self, "query_signature_ids", query_signature_ids)

        require_pair_list: list[tuple[str, str]] = []
        seen_require_signature_ids: set[str] = set()
        for signature_id, component_id in self.cluster_seeds_require:
            signature_key = str(signature_id)
            component_key = str(component_id)
            if not signature_key:
                raise ValueError("cluster_seeds_require cannot contain empty signature_id values")
            if not component_key:
                raise ValueError(f"cluster_seeds_require cannot contain empty component_id values: {signature_key!r}")
            if signature_key in seen_require_signature_ids:
                raise ValueError(f"cluster_seeds_require contains duplicate signature_id: {signature_key!r}")
            seen_require_signature_ids.add(signature_key)
            require_pair_list.append((signature_key, component_key))
        require_pairs = tuple(require_pair_list)
        missing_require = sorted(
            signature_id for signature_id, _component_id in require_pairs if signature_id not in signature_id_set
        )
        if missing_require:
            raise ValueError(f"cluster_seeds_require contains signatures missing from FeatureBlock: {missing_require}")
        object.__setattr__(self, "cluster_seeds_require", require_pairs)

        disallow_pairs = normalize_cluster_seed_disallow_pairs(
            self.cluster_seeds_disallow,
            valid_signature_ids=signature_id_set,
        )
        object.__setattr__(self, "cluster_seeds_disallow", disallow_pairs)

        specter_paper_ids = tuple(str(value) for value in self.specter_paper_ids)
        if self.specter_embeddings is None:
            if specter_paper_ids:
                raise ValueError("specter_paper_ids requires specter_embeddings")
        else:
            embeddings = np.asarray(self.specter_embeddings, dtype=np.float32)
            if embeddings.ndim != 2:
                raise ValueError(f"specter_embeddings must be 2D, got shape={embeddings.shape}")
            if embeddings.shape[0] != len(specter_paper_ids):
                raise ValueError(
                    "specter_embeddings row count must match specter_paper_ids: "
                    f"{embeddings.shape[0]} != {len(specter_paper_ids)}"
                )
            object.__setattr__(self, "specter_embeddings", np.ascontiguousarray(embeddings, dtype=np.float32))
        object.__setattr__(self, "specter_paper_ids", specter_paper_ids)

    @property
    def signature_ids(self) -> tuple[str, ...]:
        """Return signature ids in this block's deterministic order."""

        return tuple(signature.signature_id for signature in self.signatures)

    @property
    def signature_order(self) -> FeatureBlockSignatureOrder:
        """Return the signature order used by numeric linker arrays."""

        return FeatureBlockSignatureOrder(
            signature_ids=self.signature_ids,
            query_signature_ids=self.query_signature_ids,
        )

    @property
    def signature_id_to_index(self) -> dict[str, int]:
        """Return a signature-id to row-index map."""

        return self.signature_order.signature_id_to_index

    @property
    def seed_component_members(self) -> dict[str, tuple[str, ...]]:
        """Return require-seed members grouped by component id."""

        members: dict[str, list[str]] = {}
        for signature_id, component_id in self.cluster_seeds_require:
            members.setdefault(component_id, []).append(signature_id)
        return {component_id: tuple(signature_ids) for component_id, signature_ids in members.items()}

    def to_arrow_tables(self) -> dict[str, Any]:
        """Return Arrow tables for the current Rust raw-candidate schema."""

        import pyarrow as pa

        tables: dict[str, Any] = {
            "signatures": pa.table(
                {
                    "signature_id": pa.array([row.signature_id for row in self.signatures], type=pa.string()),
                    "paper_id": pa.array([row.paper_id for row in self.signatures], type=pa.string()),
                    "author_first": pa.array([row.author_first for row in self.signatures], type=pa.string()),
                    "author_middle": pa.array([row.author_middle for row in self.signatures], type=pa.string()),
                    "author_last": pa.array([row.author_last for row in self.signatures], type=pa.string()),
                    "author_suffix": pa.array([row.author_suffix for row in self.signatures], type=pa.string()),
                    "author_affiliations": pa.array(
                        [list(row.author_affiliations) for row in self.signatures],
                        type=pa.list_(pa.string()),
                    ),
                    "author_orcid": pa.array([row.author_orcid for row in self.signatures], type=pa.string()),
                    "author_position": pa.array([row.author_position for row in self.signatures], type=pa.int64()),
                    "author_block": pa.array([row.author_block for row in self.signatures], type=pa.string()),
                    "author_email": pa.array([row.author_email for row in self.signatures], type=pa.string()),
                    "source_author_ids": pa.array(
                        [list(row.source_author_ids) for row in self.signatures],
                        type=pa.list_(pa.string()),
                    ),
                }
            ),
            "papers": pa.table(
                {
                    "paper_id": pa.array([row.paper_id for row in self.papers], type=pa.string()),
                    "title": pa.array([row.title for row in self.papers], type=pa.string()),
                    "abstract": pa.array([row.abstract for row in self.papers], type=pa.string()),
                    "venue": pa.array([row.venue for row in self.papers], type=pa.string()),
                    "journal_name": pa.array([row.journal_name for row in self.papers], type=pa.string()),
                    "year": pa.array([row.year for row in self.papers], type=pa.int64()),
                    "predicted_language": pa.array(
                        [row.predicted_language for row in self.papers],
                        type=pa.string(),
                    ),
                    "is_reliable": pa.array([row.is_reliable for row in self.papers], type=pa.bool_()),
                }
            ),
            "paper_authors": pa.table(
                {
                    "paper_id": pa.array([row.paper_id for row in self.paper_authors], type=pa.string()),
                    "position": pa.array([row.position for row in self.paper_authors], type=pa.int64()),
                    "author_name": pa.array([row.author_name for row in self.paper_authors], type=pa.string()),
                }
            ),
            "cluster_seeds": pa.table(
                {
                    "signature_id": pa.array(
                        [signature_id for signature_id, _component_id in self.cluster_seeds_require],
                        type=pa.string(),
                    ),
                    "cluster_id": pa.array(
                        [component_id for _signature_id, component_id in self.cluster_seeds_require],
                        type=pa.string(),
                    ),
                }
            ),
            "cluster_seed_disallows": pa.table(
                {
                    "signature_id_1": pa.array(
                        [left for left, _right in self.cluster_seeds_disallow],
                        type=pa.string(),
                    ),
                    "signature_id_2": pa.array(
                        [right for _left, right in self.cluster_seeds_disallow],
                        type=pa.string(),
                    ),
                }
            ),
        }
        if self.specter_embeddings is not None:
            flat = pa.array(np.ravel(self.specter_embeddings), type=pa.float32())
            tables["specter"] = pa.table(
                {
                    "paper_id": list(self.specter_paper_ids),
                    "embedding": pa.FixedSizeListArray.from_arrays(flat, int(self.specter_embeddings.shape[1])),
                }
            )
        return tables


def _record_batch_limit_for_table(
    table_name: str,
    max_record_batch_rows: Mapping[str, int] | int | None,
) -> int | None:
    if max_record_batch_rows is None:
        return None
    if isinstance(max_record_batch_rows, Mapping):
        raw_limit = max_record_batch_rows.get(table_name)
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
    magic, record_count, source_size, source_mtime_ns = _ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT.unpack(header)
    if magic != _ARROW_BATCH_LOOKUP_INDEX_MAGIC:
        raise ValueError(f"Arrow batch lookup index has invalid magic bytes: {index_path!s}")
    return {
        "magic": magic.decode("ascii"),
        "record_count": int(record_count),
        "source_size": int(source_size),
        "source_mtime_ns": int(source_mtime_ns),
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
        if (
            int(index_header["source_size"]) != source_stat.st_size
            or int(index_header["source_mtime_ns"]) != source_stat.st_mtime_ns
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
            **index_header,
            **layout,
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
            records.extend((_fnv64_bytes(str(key).encode("utf-8")), batch_index) for key in keys if key is not None)

    records.sort()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_stat = arrow_path_obj.stat()
    with output_path.open("wb") as outfile:
        outfile.write(
            _ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT.pack(
                _ARROW_BATCH_LOOKUP_INDEX_MAGIC,
                len(records),
                source_stat.st_size,
                source_stat.st_mtime_ns,
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
    indexed_paths = {str(key): str(value) for key, value in paths.items()}
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


def write_feature_block_arrow_from_anddata(
    dataset: Any,
    output_dir: str | Path,
    *,
    signature_ids: Sequence[Any] | None = None,
    query_signature_ids: Sequence[Any] = (),
    cluster_seeds_require: Mapping[Any, Any] | None = None,
    include_specter: bool = True,
    include_empty_cluster_seeds: bool = False,
    max_record_batch_rows: Mapping[str, int] | int | None = RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS,
    overwrite: bool = True,
) -> dict[str, str]:
    """Build a `FeatureBlock` from `ANDData` and write complete Arrow IPC tables."""

    feature_block = feature_block_from_anddata(
        dataset,
        signature_ids=signature_ids,
        query_signature_ids=query_signature_ids,
        cluster_seeds_require=cluster_seeds_require,
        include_specter=include_specter,
    )
    return write_feature_block_arrow_tables(
        feature_block,
        output_dir,
        include_empty_cluster_seeds=include_empty_cluster_seeds,
        max_record_batch_rows=max_record_batch_rows,
        overwrite=overwrite,
    )


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


def _fnv64_bytes(value: bytes) -> int:
    digest = _FNV64_OFFSET
    for byte in value:
        digest ^= byte
        digest = (digest * _FNV64_PRIME) & 0xFFFFFFFFFFFFFFFF
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


def write_name_counts_index(output_dir: str | Path, *, overwrite: bool = False) -> tuple[str, dict[str, int | bool]]:
    """Write the global name-count lookup as exact-verified sorted binary indexes."""

    from s2and.data import _load_name_counts_cached

    index_dir = Path(output_dir) / "name_counts_index"
    files = {
        "first": index_dir / "first.bin",
        "last": index_dir / "last.bin",
        "first_last": index_dir / "first_last.bin",
        "last_first_initial": index_dir / "last_first_initial.bin",
    }
    manifest_path = index_dir / "manifest.json"
    if manifest_path.exists() and all(path.exists() for path in files.values()) and not overwrite:
        return str(index_dir), {"reused": True}

    first_dict, last_dict, first_last_dict, last_first_initial_dict = _load_name_counts_cached()
    metrics: dict[str, int | bool] = {"reused": False}
    total_records = 0
    total_bytes = 0
    manifest_files: dict[str, dict[str, int | str]] = {}
    for kind, mapping in (
        ("first", first_dict),
        ("last", last_dict),
        ("first_last", first_last_dict),
        ("last_first_initial", last_first_initial_dict),
    ):
        file_metrics = _write_name_count_index_file(files[kind], kind, mapping)
        record_count = file_metrics["record_count"]
        byte_count = file_metrics["byte_count"]
        metrics[f"{kind}_count"] = record_count
        metrics[f"{kind}_bytes"] = byte_count
        total_records += record_count
        total_bytes += byte_count
        manifest_files[kind] = {
            "path": files[kind].name,
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
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def feature_block_signature_order_from_raw_candidate_plan(plan: Mapping[str, Any]) -> FeatureBlockSignatureOrder:
    """Build a deterministic mini-block signature order from a raw candidate plan."""

    query_signature_ids = tuple(str(value) for value in _required_plan_sequence(plan, "query_signature_ids"))
    ordered_ids: list[str] = []
    seen: set[str] = set()
    for value in (
        *query_signature_ids,
        *_required_plan_sequence(plan, "left_signature_ids"),
        *_required_plan_sequence(plan, "right_signature_ids"),
    ):
        signature_id = str(value)
        if signature_id in seen:
            continue
        seen.add(signature_id)
        ordered_ids.append(signature_id)
    return FeatureBlockSignatureOrder(signature_ids=tuple(ordered_ids), query_signature_ids=query_signature_ids)


def feature_block_from_anddata(
    dataset: Any,
    *,
    signature_ids: Sequence[Any] | None = None,
    query_signature_ids: Sequence[Any] = (),
    cluster_seeds_require: Mapping[Any, Any] | None = None,
    include_specter: bool = True,
) -> FeatureBlock:
    """Build a `FeatureBlock` view from an existing `ANDData`-like object."""

    resolved_signature_ids = tuple(
        str(value) for value in (dataset.signatures.keys() if signature_ids is None else signature_ids)
    )
    signatures = tuple(
        _feature_block_signature_from_anddata(dataset, signature_id) for signature_id in resolved_signature_ids
    )
    papers = _feature_block_papers_from_anddata(dataset, signatures)
    paper_authors = _feature_block_paper_authors_from_papers(
        papers_by_id={paper.paper_id: paper for paper in papers}, dataset=dataset
    )

    signature_id_set = set(resolved_signature_ids)
    source_cluster_seeds = dict(
        getattr(dataset, "cluster_seeds_require", {}) if cluster_seeds_require is None else cluster_seeds_require
    )
    require_pairs = tuple(
        (str(signature_id), str(component_id))
        for signature_id, component_id in source_cluster_seeds.items()
        if str(signature_id) in signature_id_set
    )
    disallow_pairs = tuple(
        (str(left), str(right))
        for left, right in getattr(dataset, "cluster_seeds_disallow", set())
        if str(left) in signature_id_set and str(right) in signature_id_set
    )
    specter_paper_ids, specter_embeddings = _feature_block_specter_from_anddata(
        dataset,
        papers,
        include_specter=include_specter,
    )
    return FeatureBlock(
        signatures=signatures,
        papers=papers,
        paper_authors=paper_authors,
        cluster_seeds_require=require_pairs,
        cluster_seeds_disallow=disallow_pairs,
        query_signature_ids=tuple(str(value) for value in query_signature_ids),
        specter_paper_ids=specter_paper_ids,
        specter_embeddings=specter_embeddings,
    )


def feature_block_from_raw_payloads(
    *,
    signatures: Mapping[str, Mapping[str, Any]],
    papers: Mapping[str, Mapping[str, Any]],
    raw_candidate_plan: Mapping[str, Any],
    cluster_seeds_require: Mapping[Any, Any],
    cluster_seeds_disallow: Iterable[tuple[Any, Any]] | None = None,
    specter_embeddings: Mapping[Any, Any] | tuple[Any, Any] | None = None,
) -> FeatureBlock:
    """Build a mini `FeatureBlock` directly from raw JSON-shaped payloads."""

    signature_order = feature_block_signature_order_from_raw_candidate_plan(raw_candidate_plan)
    selected_signature_ids = signature_order.signature_ids
    signature_rows = tuple(
        _feature_block_signature_from_raw_payload(signature_id, signatures[str(signature_id)])
        for signature_id in selected_signature_ids
    )
    paper_ids = tuple(dict.fromkeys(row.paper_id for row in signature_rows))
    paper_rows = tuple(
        _feature_block_paper_from_raw_payload(paper_id, papers[str(paper_id)])
        for paper_id in paper_ids
        if str(paper_id) in papers
    )
    paper_author_rows = tuple(
        row
        for paper_id in paper_ids
        if str(paper_id) in papers
        for row in _feature_block_paper_authors_from_raw_payload(paper_id, papers[str(paper_id)])
    )
    selected_signature_id_set = set(selected_signature_ids)
    require_pairs = tuple(
        (str(signature_id), str(component_id))
        for signature_id, component_id in cluster_seeds_require.items()
        if str(signature_id) in selected_signature_id_set
    )
    disallow_pairs = tuple(
        (str(left), str(right))
        for left, right in (cluster_seeds_disallow or ())
        if str(left) in selected_signature_id_set and str(right) in selected_signature_id_set
    )
    specter_paper_ids, specter_matrix = _feature_block_specter_from_mapping(
        paper_ids,
        specter_embeddings,
    )
    return FeatureBlock(
        signatures=signature_rows,
        papers=paper_rows,
        paper_authors=paper_author_rows,
        cluster_seeds_require=require_pairs,
        cluster_seeds_disallow=disallow_pairs,
        query_signature_ids=signature_order.query_signature_ids,
        specter_paper_ids=specter_paper_ids,
        specter_embeddings=specter_matrix,
    )


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

    del include_specter
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
            position = 0 if row.get("position") is None else int(row["position"])
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

    seed_signature_ids = raw_candidate_plan.get("seed_signature_ids")
    seed_component_keys = raw_candidate_plan.get("seed_component_keys")
    if seed_signature_ids is not None or seed_component_keys is not None:
        if seed_signature_ids is None or seed_component_keys is None:
            raise ValueError("raw candidate plan seed_signature_ids and seed_component_keys must both be present")
        if len(seed_signature_ids) != len(seed_component_keys):
            raise ValueError(
                "raw candidate plan seed_signature_ids and seed_component_keys must have equal length: "
                f"{len(seed_signature_ids)} != {len(seed_component_keys)}"
            )
        require_pairs = tuple(
            (str(signature_id), str(component_key))
            for signature_id, component_key in zip(seed_signature_ids, seed_component_keys, strict=True)
            if str(signature_id) in selected_signature_id_set
        )
    else:
        component_members = raw_candidate_plan.get("component_members", {})
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
            left_id = str(left)
            right_id = str(right)
            if left_id in selected_signature_id_set and right_id in selected_signature_id_set:
                disallow_rows.append((left_id, right_id))
        disallow_pairs = tuple(disallow_rows)
    return FeatureBlock(
        signatures=signature_rows,
        papers=paper_rows,
        paper_authors=paper_author_rows,
        cluster_seeds_require=require_pairs,
        cluster_seeds_disallow=disallow_pairs,
        query_signature_ids=signature_order.query_signature_ids,
        specter_paper_ids=(),
        specter_embeddings=None,
    )


def feature_block_for_signature_order(
    feature_block: FeatureBlock,
    signature_order: FeatureBlockSignatureOrder,
) -> FeatureBlock:
    """Return a `FeatureBlock` subset ordered for numeric raw-plan scoring."""

    signatures_by_id = {row.signature_id: row for row in feature_block.signatures}
    missing = [signature_id for signature_id in signature_order.signature_ids if signature_id not in signatures_by_id]
    if missing:
        raise ValueError(f"FeatureBlock is missing raw-plan signatures: {missing}")

    signatures = tuple(signatures_by_id[signature_id] for signature_id in signature_order.signature_ids)
    paper_ids = set(row.paper_id for row in signatures)
    papers = tuple(row for row in feature_block.papers if row.paper_id in paper_ids)
    paper_id_set = set(row.paper_id for row in papers)
    paper_authors = tuple(row for row in feature_block.paper_authors if row.paper_id in paper_id_set)
    signature_id_set = set(signature_order.signature_ids)
    require_pairs = tuple(
        (signature_id, component_id)
        for signature_id, component_id in feature_block.cluster_seeds_require
        if signature_id in signature_id_set
    )
    disallow_pairs = tuple(
        (left, right)
        for left, right in feature_block.cluster_seeds_disallow
        if left in signature_id_set and right in signature_id_set
    )
    specter_paper_ids: list[str] = []
    specter_rows: list[np.ndarray] = []
    if feature_block.specter_embeddings is not None:
        specter_by_paper_id = {
            paper_id: np.asarray(feature_block.specter_embeddings[index], dtype=np.float32)
            for index, paper_id in enumerate(feature_block.specter_paper_ids)
        }
        for paper_id in feature_block.specter_paper_ids:
            if paper_id in paper_id_set and paper_id in specter_by_paper_id:
                specter_paper_ids.append(paper_id)
                specter_rows.append(specter_by_paper_id[paper_id])
    return FeatureBlock(
        signatures=signatures,
        papers=papers,
        paper_authors=paper_authors,
        cluster_seeds_require=require_pairs,
        cluster_seeds_disallow=disallow_pairs,
        query_signature_ids=signature_order.query_signature_ids,
        specter_paper_ids=tuple(specter_paper_ids),
        specter_embeddings=None if not specter_rows else np.vstack(specter_rows).astype(np.float32),
    )


def feature_block_to_mini_anddata(
    feature_block: FeatureBlock,
    *,
    name: str = "feature_block_mini",
    n_jobs: int = 1,
    preprocess: bool = True,
    name_tuples: set[tuple[str, str]] | str | None = "filtered",
    load_name_counts: bool | dict[str, Any] = False,
) -> Any:
    """Materialize a small compatibility `ANDData` from a `FeatureBlock`.

    This is a bridge for scoring wrappers that still call existing pairwise and
    constraint code. It is intended for query plus retrieved candidate members,
    not for full-block Arrow-to-Python materialization.
    """

    from s2and.data import ANDData

    dataset = ANDData(
        signatures=_feature_block_signatures_payload(feature_block),
        papers=_feature_block_papers_payload(feature_block),
        name=name,
        mode="inference",
        specter_embeddings=_feature_block_specter_payload(feature_block),
        load_name_counts=load_name_counts,
        n_jobs=n_jobs,
        preprocess=preprocess,
        name_tuples=name_tuples,
        use_orcid_id=True,
        use_sinonym_overwrite=False,
    )
    dataset.cluster_seeds_require = dict(feature_block.cluster_seeds_require)
    dataset.cluster_seeds_disallow = set(feature_block.cluster_seeds_disallow)

    return dataset


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
        author_affiliations=tuple(str(value) for value in (row.get("author_affiliations") or ())),
        author_orcid=_optional_str(row.get("author_orcid")),
        author_position=_optional_int(row.get("author_position")),
        author_block=_optional_str(row.get("author_block")),
        author_email=_optional_str(row.get("author_email")),
        source_author_ids=tuple(str(value) for value in (row.get("source_author_ids") or ()) if value is not None),
    )


def _feature_block_paper_from_arrow_row(paper_id: str, row: Mapping[str, Any]) -> FeatureBlockPaper:
    return FeatureBlockPaper(
        paper_id=str(row.get("paper_id", paper_id)),
        title=_optional_str(row.get("title")),
        abstract=_optional_str(row.get("abstract")),
        venue=_optional_str(row.get("venue")),
        journal_name=_optional_str(row.get("journal_name")),
        year=_optional_int(row.get("year")),
        predicted_language=_optional_str(row.get("predicted_language")),
        is_reliable=_optional_bool(row.get("is_reliable")),
    )


def _required_plan_sequence(plan: Mapping[str, Any], key: str) -> tuple[Any, ...]:
    if key not in plan:
        raise KeyError(f"raw candidate plan is missing required key: {key}")
    value = plan[key]
    if isinstance(value, np.ndarray):
        return tuple(value.tolist())
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return tuple(value)
    raise TypeError(f"raw candidate plan key {key!r} must be a sequence")


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return None


def _source_author_ids_payload(values: Sequence[str]) -> list[str]:
    return [str(value) for value in values]


def _raw_author_info(raw_signature: Mapping[str, Any]) -> Mapping[str, Any]:
    value = raw_signature.get("author_info", {})
    if not isinstance(value, Mapping):
        raise TypeError("raw signature author_info must be a mapping")
    return value


def _raw_orcid(author_info: Mapping[str, Any]) -> str | None:
    source = author_info.get("source_id_source")
    source_ids = author_info.get("source_ids") or []
    if source == "ORCID" and source_ids:
        return _optional_str(source_ids[0])
    return _optional_str(author_info.get("orcid"))


def _feature_block_signature_from_raw_payload(
    signature_id: str,
    raw_signature: Mapping[str, Any],
) -> FeatureBlockSignature:
    author_info = _raw_author_info(raw_signature)
    return FeatureBlockSignature(
        signature_id=str(raw_signature.get("signature_id", signature_id)),
        paper_id=str(raw_signature["paper_id"]),
        author_first=_optional_str(author_info.get("first")),
        author_middle=_optional_str(author_info.get("middle")),
        author_last=_optional_str(author_info.get("last")),
        author_suffix=_optional_str(author_info.get("suffix")),
        author_affiliations=tuple(str(value) for value in (author_info.get("affiliations") or ())),
        author_orcid=_raw_orcid(author_info),
        author_position=_optional_int(author_info.get("position")),
        author_block=_optional_str(author_info.get("block")),
        author_email=_optional_str(author_info.get("email")),
        source_author_ids=tuple(str(value) for value in (raw_signature.get("sourced_author_ids") or ())),
    )


def _feature_block_paper_from_raw_payload(
    paper_id: str,
    raw_paper: Mapping[str, Any],
) -> FeatureBlockPaper:
    return FeatureBlockPaper(
        paper_id=str(raw_paper.get("paper_id", paper_id)),
        title=_optional_str(raw_paper.get("title")),
        abstract=_optional_str(raw_paper.get("abstract")),
        venue=_optional_str(raw_paper.get("venue")),
        journal_name=_optional_str(raw_paper.get("journal_name")),
        year=_optional_int(raw_paper.get("year")),
        predicted_language=_optional_str(raw_paper.get("predicted_language")),
        is_reliable=_optional_bool(raw_paper.get("is_reliable")),
    )


def _feature_block_paper_authors_from_raw_payload(
    paper_id: str,
    raw_paper: Mapping[str, Any],
) -> tuple[FeatureBlockPaperAuthor, ...]:
    rows: list[FeatureBlockPaperAuthor] = []
    for index, author in enumerate(raw_paper.get("authors") or ()):
        if not isinstance(author, Mapping):
            continue
        position = _optional_int(author.get("position"))
        rows.append(
            FeatureBlockPaperAuthor(
                paper_id=str(paper_id),
                position=index if position is None else position,
                author_name=str(author.get("author_name") or author.get("name") or ""),
            )
        )
    return tuple(rows)


def _feature_block_signatures_payload(feature_block: FeatureBlock) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for row in feature_block.signatures:
        source_ids = [row.author_orcid] if row.author_orcid else []
        author_info: dict[str, Any] = {
            "first": row.author_first or "",
            "middle": row.author_middle or "",
            "last": row.author_last or "",
            "suffix": row.author_suffix or "",
            "affiliations": list(row.author_affiliations),
            "email": row.author_email or "",
            "source_ids": source_ids,
            "position": 0 if row.author_position is None else int(row.author_position),
            "block": row.author_block or "",
        }
        if row.author_orcid:
            author_info["source_id_source"] = "ORCID"
        payload[row.signature_id] = {
            "signature_id": row.signature_id,
            "paper_id": row.paper_id,
            "author_info": author_info,
            "sourced_author_ids": _source_author_ids_payload(row.source_author_ids),
        }
    return payload


def _feature_block_papers_payload(feature_block: FeatureBlock) -> dict[str, dict[str, Any]]:
    authors_by_paper_id: dict[str, list[dict[str, Any]]] = {}
    for row in feature_block.paper_authors:
        authors_by_paper_id.setdefault(row.paper_id, []).append(
            {
                "position": int(row.position),
                "author_name": row.author_name,
            }
        )
    return {
        row.paper_id: {
            "paper_id": row.paper_id,
            "title": row.title or "",
            "abstract": row.abstract or "",
            "venue": row.venue or "",
            "journal_name": row.journal_name or "",
            "year": row.year,
            "predicted_language": row.predicted_language,
            "is_reliable": row.is_reliable,
            "references": [],
            "authors": sorted(authors_by_paper_id.get(row.paper_id, []), key=lambda item: int(item["position"])),
        }
        for row in feature_block.papers
    }


def _feature_block_specter_payload(feature_block: FeatureBlock) -> dict[str, np.ndarray] | None:
    if feature_block.specter_embeddings is None:
        return None
    return {
        paper_id: np.ascontiguousarray(feature_block.specter_embeddings[index], dtype=np.float32)
        for index, paper_id in enumerate(feature_block.specter_paper_ids)
    }


def _feature_block_specter_from_mapping(
    paper_ids: Sequence[str],
    specter_embeddings: Mapping[Any, Any] | tuple[Any, Any] | None,
) -> tuple[tuple[str, ...], np.ndarray | None]:
    if not specter_embeddings:
        return (), None
    if isinstance(specter_embeddings, tuple):
        if len(specter_embeddings) != 2:
            raise ValueError("SPECTER tuple payload must be (matrix, paper_ids)")
        matrix, keys = specter_embeddings
        matrix_array = np.asarray(matrix, dtype=np.float32)
        if matrix_array.ndim != 2:
            raise ValueError(f"SPECTER tuple matrix must be 2D, got shape={matrix_array.shape}")
        key_list = list(keys)
        if len(key_list) != int(matrix_array.shape[0]):
            raise ValueError(
                "SPECTER tuple key count must match matrix rows: " f"keys={len(key_list)}, rows={matrix_array.shape[0]}"
            )
        specter_mapping: Mapping[Any, Any] = {
            str(key): np.ascontiguousarray(matrix_array[index], dtype=np.float32) for index, key in enumerate(key_list)
        }
    else:
        specter_mapping = specter_embeddings
    selected_paper_ids: list[str] = []
    vectors: list[np.ndarray] = []
    expected_dim: int | None = None
    for paper_id in paper_ids:
        vector = specter_mapping.get(str(paper_id))
        if vector is None:
            vector = specter_mapping.get(paper_id)
        if vector is None:
            continue
        array = np.asarray(vector, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError(f"SPECTER vector for paper_id={paper_id!r} must be 1D, got shape={array.shape}")
        if expected_dim is None:
            expected_dim = int(array.shape[0])
        elif int(array.shape[0]) != expected_dim:
            raise ValueError(
                "SPECTER vectors in a FeatureBlock must have equal dimensions: "
                f"expected {expected_dim}, got {array.shape[0]} for paper_id={paper_id!r}"
            )
        selected_paper_ids.append(str(paper_id))
        vectors.append(array)
    if not vectors:
        return (), None
    return tuple(selected_paper_ids), np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)


def _feature_block_signature_from_anddata(dataset: Any, signature_id: str) -> FeatureBlockSignature:
    signature = dataset.signatures[str(signature_id)]
    return FeatureBlockSignature(
        signature_id=str(signature_id),
        paper_id=str(signature.paper_id),
        author_first=_optional_str(getattr(signature, "author_info_first", None)),
        author_middle=_optional_str(getattr(signature, "author_info_middle", None)),
        author_last=_optional_str(getattr(signature, "author_info_last", None)),
        author_suffix=_optional_str(getattr(signature, "author_info_suffix", None)),
        author_affiliations=tuple(str(value) for value in (getattr(signature, "author_info_affiliations", None) or ())),
        author_orcid=_optional_str(getattr(signature, "author_info_orcid", None)),
        author_position=_optional_int(getattr(signature, "author_info_position", None)),
        author_block=_optional_str(getattr(signature, "author_info_block", None)),
        author_email=_optional_str(getattr(signature, "author_info_email", None)),
        source_author_ids=tuple(str(value) for value in (getattr(signature, "sourced_author_ids", None) or ())),
    )


def _feature_block_papers_from_anddata(
    dataset: Any,
    signatures: Sequence[FeatureBlockSignature],
) -> tuple[FeatureBlockPaper, ...]:
    papers: list[FeatureBlockPaper] = []
    seen: set[str] = set()
    for signature in signatures:
        paper_id = str(signature.paper_id)
        if paper_id in seen:
            continue
        paper = getattr(dataset, "papers", {}).get(paper_id)
        if paper is None:
            continue
        seen.add(paper_id)
        papers.append(
            FeatureBlockPaper(
                paper_id=paper_id,
                title=_optional_str(getattr(paper, "title", None)),
                abstract="Has Abstract" if bool(getattr(paper, "has_abstract", False)) else "",
                venue=_optional_str(getattr(paper, "venue", None)),
                journal_name=_optional_str(getattr(paper, "journal_name", None)),
                year=_optional_int(getattr(paper, "year", None)),
                predicted_language=_optional_str(getattr(paper, "predicted_language", None)),
                is_reliable=_optional_bool(getattr(paper, "is_reliable", None)),
            )
        )
    return tuple(papers)


def _feature_block_paper_authors_from_papers(
    *,
    papers_by_id: Mapping[str, FeatureBlockPaper],
    dataset: Any,
) -> tuple[FeatureBlockPaperAuthor, ...]:
    rows: list[FeatureBlockPaperAuthor] = []
    for paper_id in papers_by_id:
        paper = getattr(dataset, "papers", {}).get(str(paper_id))
        if paper is None:
            continue
        for index, author in enumerate(getattr(paper, "authors", None) or ()):
            position = _optional_int(getattr(author, "position", index))
            rows.append(
                FeatureBlockPaperAuthor(
                    paper_id=paper_id,
                    position=index if position is None else position,
                    author_name=str(getattr(author, "author_name", "") or ""),
                )
            )
    return tuple(rows)


def _feature_block_specter_from_anddata(
    dataset: Any,
    papers: Sequence[FeatureBlockPaper],
    *,
    include_specter: bool,
) -> tuple[tuple[str, ...], np.ndarray | None]:
    if not include_specter:
        return (), None
    specter = getattr(dataset, "specter_embeddings", None)
    if not specter:
        return (), None
    paper_ids: list[str] = []
    vectors: list[np.ndarray] = []
    expected_dim: int | None = None
    for paper in papers:
        vector = specter.get(str(paper.paper_id))
        if vector is None:
            continue
        array = np.asarray(vector, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError(f"SPECTER vector for paper_id={paper.paper_id!r} must be 1D, got shape={array.shape}")
        if expected_dim is None:
            expected_dim = int(array.shape[0])
        elif int(array.shape[0]) != expected_dim:
            raise ValueError(
                "SPECTER vectors in a FeatureBlock must have equal dimensions: "
                f"expected {expected_dim}, got {array.shape[0]} for paper_id={paper.paper_id!r}"
            )
        paper_ids.append(str(paper.paper_id))
        vectors.append(array)
    if not vectors:
        return (), None
    return tuple(paper_ids), np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)
