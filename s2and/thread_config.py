"""Thread-count normalization helpers."""

from __future__ import annotations

import os
from typing import Any


def resolve_n_jobs(n_jobs: Any, *, default: int = 1) -> int:
    """Return a positive worker count, honoring sklearn negative `n_jobs` semantics.

    `n_jobs=-1` means all available CPUs; `-2` means all but one, and so on. A
    value of `0` is not valid in sklearn, but legacy S2AND callers used it as a
    single-thread fallback, so preserve that behavior.
    """

    value = default if n_jobs is None else n_jobs
    parsed = int(value)
    if parsed == 0:
        return 1
    if parsed < 0:
        cpu_count = os.cpu_count() or 1
        return max(1, cpu_count + 1 + parsed)
    return parsed
