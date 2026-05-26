from __future__ import annotations

from pathlib import Path

import pytest

from scripts._rust_suite import largest_block_cmd


def test_run_single_arrow_rejects_python_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires --backend rust"):
        largest_block_cmd._run_single(
            backend="python",
            dataset_name="qian",
            block_key="a smith",
            n_jobs=1,
            profile_output_path=str(tmp_path / "profile.txt"),
            input_format="arrow",
        )
