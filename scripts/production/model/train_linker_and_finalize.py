"""Train the linker half and finalize a native production model bundle."""

# ruff: noqa: E402,I001

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.production.model.linker_train_calibrate_eval import main  # noqa: E402


if __name__ == "__main__":
    main()
