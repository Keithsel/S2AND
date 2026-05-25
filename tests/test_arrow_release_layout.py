from __future__ import annotations

import json
from pathlib import Path


def _touch_json(path: Path, payload: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({} if payload is None else payload), encoding="utf-8")


def _touch_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _validate_required_release_files(release_root: Path, dataset_name: str) -> None:
    dataset_manifest_path = release_root / dataset_name / "manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    manifest_paths = dataset_manifest.get("paths", {}) if isinstance(dataset_manifest, dict) else {}
    embedding_path = manifest_paths.get("specter") or manifest_paths.get("specter2") or "specter.arrow"
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
        release_root / dataset_name / str(embedding_path),
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
    ):
        _touch_json(manifest_path)
    _touch_json(
        release_root / dataset_name / "manifest.json",
        {
            "paths": {
                "signatures": "signatures.arrow",
                "papers": "papers.arrow",
                "paper_authors": "paper_authors.arrow",
                "specter2": "specter2.arrow",
            }
        },
    )

    for file_path in (
        release_root / "LICENSE.txt",
        release_root / "lid.176.bin",
        release_root / dataset_name / "signatures.arrow",
        release_root / dataset_name / "papers.arrow",
        release_root / dataset_name / "paper_authors.arrow",
        release_root / dataset_name / "specter2.arrow",
        release_root / dataset_name / "signatures.signatures_batch_index.bin",
    ):
        _touch_file(file_path)

    _validate_required_release_files(release_root, dataset_name)
