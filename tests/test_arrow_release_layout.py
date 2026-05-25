from __future__ import annotations

import json
from pathlib import Path


def _touch_json(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({}), encoding="utf-8")


def _touch_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _validate_required_release_files(release_root: Path, dataset_name: str) -> None:
    required_paths = [
        release_root / "manifest.json",
        release_root / "LICENSE.txt",
        release_root / "lid.176.bin",
        release_root / "production_model_v1.21" / "manifest.json",
        release_root / "name_counts_index" / "manifest.json",
        release_root / dataset_name / "manifest.json",
        release_root / dataset_name / "signatures.arrow",
        release_root / dataset_name / "papers.arrow",
        release_root / dataset_name / "paper_authors.arrow",
        release_root / dataset_name / "specter.arrow",
        release_root / dataset_name / "signatures.signatures_batch_index.bin",
    ]

    missing_paths = [path.relative_to(release_root) for path in required_paths if not path.exists()]
    assert missing_paths == []


def test_docs_work_plan_arrow_release_layout_required_files(tmp_path: Path) -> None:
    release_root = tmp_path / "release"
    dataset_name = "s2and_mini"

    for manifest_path in (
        release_root / "manifest.json",
        release_root / "production_model_v1.21" / "manifest.json",
        release_root / "name_counts_index" / "manifest.json",
        release_root / dataset_name / "manifest.json",
    ):
        _touch_json(manifest_path)

    for file_path in (
        release_root / "LICENSE.txt",
        release_root / "lid.176.bin",
        release_root / dataset_name / "signatures.arrow",
        release_root / dataset_name / "papers.arrow",
        release_root / dataset_name / "paper_authors.arrow",
        release_root / dataset_name / "specter.arrow",
        release_root / dataset_name / "signatures.signatures_batch_index.bin",
    ):
        _touch_file(file_path)

    _validate_required_release_files(release_root, dataset_name)
