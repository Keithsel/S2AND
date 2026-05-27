from __future__ import annotations

import json
from pathlib import Path

import pytest

from s2and.arrow_inputs import (
    MissingArrowArtifactError,
    normalize_arrow_paths,
    require_arrow_artifacts,
    require_filtered_arrow_batch_indexes,
    validate_arrow_prediction_artifacts,
)
from s2and.incremental_linking.feature_block import write_name_counts_index
from tests.helpers import patch_tiny_name_counts_loader


def _touch_paths(tmp_path: Path, keys: tuple[str, ...], *, suffix: str = ".arrow") -> dict[str, str]:
    paths = {}
    for key in keys:
        path = tmp_path / f"{key}{suffix}"
        path.touch()
        paths[key] = str(path)
    return paths


def test_require_arrow_artifacts_reports_missing_keys_and_files(tmp_path: Path) -> None:
    signatures_path = tmp_path / "signatures.arrow"
    signatures_path.touch()
    missing_index_path = tmp_path / "signatures.signatures_batch_index.bin"

    with pytest.raises(MissingArrowArtifactError) as exc_info:
        require_arrow_artifacts(
            {
                "signatures": signatures_path,
                "signatures_batch_index": missing_index_path,
            },
            required_keys=("signatures", "signatures_batch_index", "papers"),
            context="test context",
            producer_hint="test hint",
        )

    error = exc_info.value
    assert error.context == "test context"
    assert error.required_keys == ("signatures", "signatures_batch_index", "papers")
    assert error.missing_keys == ("papers",)
    assert error.missing_files == {"signatures_batch_index": str(missing_index_path)}
    assert "test hint" in str(error)


def test_validate_arrow_prediction_artifacts_requires_filtered_read_indexes(tmp_path: Path) -> None:
    paths = _touch_paths(tmp_path, ("signatures", "papers", "paper_authors", "specter"))

    with pytest.raises(MissingArrowArtifactError) as exc_info:
        validate_arrow_prediction_artifacts(
            paths,
            require_specter=True,
            require_name_counts_index=False,
            require_batch_indexes=True,
        )

    assert exc_info.value.missing_keys == (
        "paper_authors_batch_index",
        "papers_batch_index",
        "signatures_batch_index",
        "specter_batch_index",
    )


def test_validate_arrow_prediction_artifacts_rejects_missing_declared_seed_sidecar(tmp_path: Path) -> None:
    paths = _touch_paths(tmp_path, ("signatures", "papers", "paper_authors"))
    seed_path = tmp_path / "missing_cluster_seeds.arrow"
    paths["cluster_seeds"] = str(seed_path)

    with pytest.raises(MissingArrowArtifactError) as exc_info:
        validate_arrow_prediction_artifacts(
            paths,
            require_specter=False,
            require_name_counts_index=False,
        )

    assert exc_info.value.missing_files == {"cluster_seeds": str(seed_path)}


def test_validate_arrow_prediction_artifacts_rejects_wrong_path_kinds(tmp_path: Path) -> None:
    signatures_dir = tmp_path / "signatures.arrow"
    signatures_dir.mkdir()
    papers_path = tmp_path / "papers.arrow"
    paper_authors_path = tmp_path / "paper_authors.arrow"
    name_counts_file = tmp_path / "name_counts_index"
    papers_path.touch()
    paper_authors_path.touch()
    name_counts_file.write_text("not a directory", encoding="utf-8")

    with pytest.raises(MissingArrowArtifactError) as exc_info:
        validate_arrow_prediction_artifacts(
            {
                "signatures": str(signatures_dir),
                "papers": str(papers_path),
                "paper_authors": str(paper_authors_path),
                "name_counts_index": str(name_counts_file),
            },
            require_specter=False,
            require_name_counts_index=True,
        )

    assert exc_info.value.missing_files == {
        "name_counts_index": f"{name_counts_file} (expected directory)",
        "signatures": f"{signatures_dir} (expected file)",
    }


def test_validate_arrow_prediction_artifacts_requires_manifest_backed_name_counts_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _touch_paths(tmp_path, ("signatures", "papers", "paper_authors"))
    empty_index_dir = tmp_path / "empty_name_counts_index"
    empty_index_dir.mkdir()

    with pytest.raises(MissingArrowArtifactError) as exc_info:
        validate_arrow_prediction_artifacts(
            {**paths, "name_counts_index": str(empty_index_dir)},
            require_specter=False,
            require_name_counts_index=True,
        )

    assert exc_info.value.missing_files["name_counts_index"].endswith("manifest.json (missing manifest.json)")

    patch_tiny_name_counts_loader(monkeypatch)
    valid_index_dir, _metrics = write_name_counts_index(tmp_path / "valid_index")
    assert (
        validate_arrow_prediction_artifacts(
            {**paths, "name_counts_index": valid_index_dir},
            require_specter=False,
            require_name_counts_index=True,
        )["name_counts_index"]
        == valid_index_dir
    )

    manifest_path = Path(valid_index_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "unexpected"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(MissingArrowArtifactError) as bad_schema_exc:
        validate_arrow_prediction_artifacts(
            {**paths, "name_counts_index": valid_index_dir},
            require_specter=False,
            require_name_counts_index=True,
        )
    assert "schema_version" in bad_schema_exc.value.missing_files["name_counts_index"]


def test_validate_arrow_prediction_artifacts_ignores_unused_specter_path(tmp_path: Path) -> None:
    paths = _touch_paths(tmp_path, ("signatures", "papers", "paper_authors", "specter"))
    paths.update(
        _touch_paths(
            tmp_path, ("signatures_batch_index", "papers_batch_index", "paper_authors_batch_index"), suffix=".bin"
        )
    )

    normalized = validate_arrow_prediction_artifacts(
        paths,
        require_specter=False,
        require_name_counts_index=False,
        require_batch_indexes=True,
    )

    assert "specter" not in normalized
    assert "specter_batch_index" not in normalized


def test_validate_arrow_prediction_artifacts_accepts_selected_specter2_alias(tmp_path: Path) -> None:
    paths = _touch_paths(tmp_path, ("signatures", "papers", "paper_authors", "specter2"))
    paths.update(
        _touch_paths(
            tmp_path,
            ("signatures_batch_index", "papers_batch_index", "paper_authors_batch_index", "specter2_batch_index"),
            suffix=".bin",
        )
    )

    normalized = validate_arrow_prediction_artifacts(
        paths,
        require_specter=True,
        require_name_counts_index=False,
        require_batch_indexes=True,
    )

    assert normalized["specter"] == paths["specter2"]
    assert normalized["specter_batch_index"] == paths["specter2_batch_index"]


def test_validate_arrow_prediction_artifacts_clears_invalid_legacy_specter_after_alias(tmp_path: Path) -> None:
    paths = _touch_paths(tmp_path, ("signatures", "papers", "paper_authors", "specter2"))
    paths.update(
        _touch_paths(
            tmp_path,
            ("signatures_batch_index", "papers_batch_index", "paper_authors_batch_index", "specter2_batch_index"),
            suffix=".bin",
        )
    )
    paths["specter"] = None  # type: ignore[assignment]
    paths["specter_batch_index"] = None  # type: ignore[assignment]

    normalized = validate_arrow_prediction_artifacts(
        paths,
        require_specter=True,
        require_name_counts_index=False,
        require_batch_indexes=True,
    )

    assert normalized["specter"] == paths["specter2"]
    assert normalized["specter_batch_index"] == paths["specter2_batch_index"]


def test_require_filtered_arrow_batch_indexes_ignores_specter_index_without_specter(tmp_path: Path) -> None:
    paths = {}
    for key in ("signatures", "papers", "paper_authors"):
        path = tmp_path / f"{key}.arrow"
        path.touch()
        paths[key] = str(path)
        index_path = tmp_path / f"{key}.{key}_batch_index.bin"
        index_path.touch()
        paths[f"{key}_batch_index"] = str(index_path)

    require_filtered_arrow_batch_indexes(paths)


def test_normalize_arrow_paths_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="is None"):
        normalize_arrow_paths({"signatures": None})
    with pytest.raises(ValueError, match="is empty"):
        normalize_arrow_paths({"signatures": " "})
    with pytest.raises(ValueError, match="current directory"):
        normalize_arrow_paths({"signatures": "."})
