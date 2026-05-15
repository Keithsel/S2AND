from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_inventors_scripts_help_without_ijson() -> None:
    """Optional ijson dependency should not be required for argparse help."""

    repo_root = Path(__file__).resolve().parents[1]
    scripts = [
        repo_root / "scripts" / "make_inventors_s2and_subset.py",
        repo_root / "scripts" / "make_inventors_split_and_histograms.py",
    ]

    for script in scripts:
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout.lower()
