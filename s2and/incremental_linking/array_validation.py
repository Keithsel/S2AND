"""Array boundary validation shared by incremental-linking Rust bridges."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import Any

import numpy as np

UINT32_MAX = int(np.iinfo(np.uint32).max)
UINT16_MAX = int(np.iinfo(np.uint16).max)


def as_uint32_1d(name: str, values: Sequence[Any] | np.ndarray) -> np.ndarray:
    """Return a contiguous uint32 array without allowing wraparound casts."""

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape={array.shape}")
    if array.size == 0:
        return np.zeros(0, dtype=np.uint32)
    if np.issubdtype(array.dtype, np.integer):
        if np.any(array < 0) or np.any(array > UINT32_MAX):
            raise ValueError(f"{name} values must be in uint32 range [0, {UINT32_MAX}]")
        return np.ascontiguousarray(array, dtype=np.uint32)
    if array.dtype != object:
        raise ValueError(f"{name} values must be integers in uint32 range [0, {UINT32_MAX}]")

    checked = np.empty(array.size, dtype=np.uint32)
    for offset, value in enumerate(array.tolist()):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name}[{offset}] must be an integer in uint32 range [0, {UINT32_MAX}]")
        integer_value = int(value)
        if integer_value < 0 or integer_value > UINT32_MAX:
            raise ValueError(f"{name}[{offset}]={integer_value} is outside uint32 range [0, {UINT32_MAX}]")
        checked[offset] = integer_value
    return checked


def as_uint16_1d(name: str, values: Sequence[Any] | np.ndarray) -> np.ndarray:
    """Return a contiguous uint16 array without allowing wraparound casts."""

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape={array.shape}")
    if array.size == 0:
        return np.zeros(0, dtype=np.uint16)
    if np.issubdtype(array.dtype, np.integer):
        if np.any(array < 0) or np.any(array > UINT16_MAX):
            raise ValueError(f"{name} values must be in uint16 range [0, {UINT16_MAX}]")
        return np.ascontiguousarray(array, dtype=np.uint16)
    if array.dtype != object:
        raise ValueError(f"{name} values must be integers in uint16 range [0, {UINT16_MAX}]")

    checked = np.empty(array.size, dtype=np.uint16)
    for offset, value in enumerate(array.tolist()):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name}[{offset}] must be an integer in uint16 range [0, {UINT16_MAX}]")
        integer_value = int(value)
        if integer_value < 0 or integer_value > UINT16_MAX:
            raise ValueError(f"{name}[{offset}]={integer_value} is outside uint16 range [0, {UINT16_MAX}]")
        checked[offset] = integer_value
    return checked
