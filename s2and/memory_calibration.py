from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PhaseASample:
    accumulator_entries_peak: int
    chunk_features_peak_bytes: int
    phase_a_pair_buffer_peak_bytes: int
    phase_a_fixed_overhead_bytes: int
    observed_peak_delta_bytes: int


@dataclass(frozen=True)
class RustBatchSample:
    total_rows: int
    predicted_features_matrix_bytes: int
    predicted_labels_bytes: int
    predicted_chunk_bytes: int
    predicted_fixed_overhead_bytes: int
    observed_peak_delta_bytes: int


def iter_memory_telemetry_records(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Yield structured memory telemetry records from JSONL input."""

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid memory telemetry JSONL at line {line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Invalid memory telemetry JSONL at line {line_number}: expected an object")
        yield record


def _int_field(record: Mapping[str, Any], key: str) -> int:
    value = record[key]
    if isinstance(value, bool):
        raise TypeError(f"{key} must be an integer, not bool")
    return int(value)


def _optional_int_field(record: Mapping[str, Any], key: str, default: int) -> int:
    if key not in record:
        return default
    return _int_field(record, key)


def extract_phase_a_sample(record: Mapping[str, Any]) -> PhaseASample | None:
    if record.get("stage") != "phase_a_seed_distances":
        return None
    try:
        accumulator_entries_peak = (
            _int_field(record, "accumulator_entries_peak_sample")
            if "accumulator_entries_peak_sample" in record
            else _int_field(record, "accumulator_entries_peak")
        )
        return PhaseASample(
            accumulator_entries_peak=accumulator_entries_peak,
            chunk_features_peak_bytes=_int_field(record, "chunk_features_peak_bytes"),
            phase_a_pair_buffer_peak_bytes=_optional_int_field(record, "phase_a_pair_buffer_peak_bytes", 0),
            phase_a_fixed_overhead_bytes=_int_field(record, "phase_a_fixed_overhead_bytes"),
            observed_peak_delta_bytes=_int_field(record, "observed_peak_delta_bytes"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid phase_a_seed_distances memory telemetry record") from exc


def extract_rust_batch_sample(record: Mapping[str, Any]) -> RustBatchSample | None:
    if record.get("stage") != "pair_featurization_rust_batch":
        return None
    try:
        return RustBatchSample(
            total_rows=_int_field(record, "total_rows"),
            predicted_features_matrix_bytes=_int_field(record, "predicted_features_matrix_bytes"),
            predicted_labels_bytes=_int_field(record, "predicted_labels_bytes"),
            predicted_chunk_bytes=_int_field(record, "predicted_chunk_bytes"),
            predicted_fixed_overhead_bytes=_int_field(record, "predicted_fixed_overhead_bytes"),
            observed_peak_delta_bytes=_int_field(record, "observed_peak_delta_bytes"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid pair_featurization_rust_batch memory telemetry record") from exc


def effective_accumulator_entry_bytes(sample: PhaseASample) -> float | None:
    if sample.accumulator_entries_peak <= 0:
        return None
    residual_bytes = (
        int(sample.observed_peak_delta_bytes)
        - int(sample.chunk_features_peak_bytes)
        - int(sample.phase_a_pair_buffer_peak_bytes)
        - int(sample.phase_a_fixed_overhead_bytes)
    )
    if residual_bytes <= 0:
        return None
    return float(residual_bytes) / float(sample.accumulator_entries_peak)


def effective_rust_batch_persistent_row_overhead_bytes(sample: RustBatchSample) -> float | None:
    if sample.total_rows <= 0:
        return None
    residual_bytes = (
        int(sample.observed_peak_delta_bytes)
        - int(sample.predicted_features_matrix_bytes)
        - int(sample.predicted_labels_bytes)
        - int(sample.predicted_chunk_bytes)
        - int(sample.predicted_fixed_overhead_bytes)
    )
    if residual_bytes <= 0:
        return None
    return float(residual_bytes) / float(sample.total_rows)


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= p <= 100:
        raise ValueError("percentile p must be in [0, 100]")

    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (float(p) / 100.0) * float(len(sorted_values) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
