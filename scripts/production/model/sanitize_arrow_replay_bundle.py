"""Sanitize linker replay metadata for the Arrow-only public release.

The Arrow replay release ships runtime tables under ``datasets/<dataset>/``
plus typed offline ``components/``, ``labels/``, and ``splits/`` artifacts. It
does not ship legacy raw JSON, pickle embeddings, or precomputed feature rows.
This tool rewrites copied legacy bundle metadata so it reflects that contract.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ARROW_REPLAY_BUNDLE_SCHEMA = "s2and_linker_replay_arrow_bundle_v1"
ARROW_REPLAY_SOURCE_PROVENANCE_SCHEMA = "s2and_linker_replay_arrow_source_provenance_v1"
REQUIRED_ARROW_ASSET_KEYS = ("candidate_members", "featureless_rows", "splits")
OMITTED_LEGACY_ASSET_KEYS = ("raw_metadata", "embeddings", "corrected_feature_rows")
OMITTED_LEGACY_ASSETS = ("raw/*.json", "embeddings/*.pkl", "features_corrected/*.parquet")
LEGACY_CLASSIC_ROW_PATH_KEYS = (
    "train_path",
    "classic_gate_source_path",
    "s2and_eval_path",
    "hwang_eval_path",
    "s_lee_eval_path",
    "s_park_eval_path",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace_json(path: Path, payload: Any) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            encoding="utf-8",
            delete=False,
        ) as temp_file:
            temp_file.write(encoded)
            temp_path = Path(temp_file.name)
        temp_path.replace(path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def _require_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    return dict(value)


def _dataset_names_from_source_manifest(source_manifest: Mapping[str, Any]) -> list[str]:
    names: set[str] = set()
    for key in ("components", "raw", "embeddings"):
        entries = source_manifest.get(key, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, Mapping) and entry.get("dataset") is not None:
                names.add(str(entry["dataset"]))
    return sorted(names)


def _legacy_source_counts(source_manifest: Mapping[str, Any]) -> dict[str, Any]:
    raw_counts: dict[str, dict[str, int]] = {}
    for entry in source_manifest.get("raw", []):
        if not isinstance(entry, Mapping) or entry.get("dataset") is None:
            continue
        raw_counts[str(entry["dataset"])] = {
            "papers": int(entry.get("papers", 0)),
            "signatures": int(entry.get("signatures", 0)),
        }

    embedding_counts: dict[str, dict[str, int | str]] = {}
    for entry in source_manifest.get("embeddings", []):
        if not isinstance(entry, Mapping) or entry.get("dataset") is None:
            continue
        embedding_counts[str(entry["dataset"])] = {
            "embedding_count": int(entry.get("embedding_count", 0)),
            "embedding_dim": int(entry.get("embedding_dim", 0)),
            "missing_embedding_count": int(entry.get("missing_embedding_count", 0)),
            "paper_count": int(entry.get("paper_count", 0)),
            "source_kind": str(entry.get("source_kind", "")),
        }

    return {
        "embeddings": embedding_counts,
        "raw": raw_counts,
    }


def _sanitized_models(models: Any) -> dict[str, Any]:
    out = _require_mapping(models or {}, context="bundle models")
    classic = out.get("classic")
    if not isinstance(classic, Mapping):
        return out
    sanitized_classic = {
        str(key): value
        for key, value in classic.items()
        if str(key) not in LEGACY_CLASSIC_ROW_PATH_KEYS and str(key) != "extra_eval_paths"
    }
    out["classic"] = sanitized_classic
    return out


def sanitized_bundle_payload(payload: Mapping[str, Any], *, bundle_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return Arrow-only ``bundle.json`` payload and a summary report."""

    out = dict(payload)
    assets = _require_mapping(out.get("assets"), context="bundle assets")
    missing_assets = [key for key in REQUIRED_ARROW_ASSET_KEYS if key not in assets]
    if missing_assets:
        raise ValueError(f"Arrow replay bundle is missing required assets: {missing_assets}")
    removed_assets = sorted(key for key in OMITTED_LEGACY_ASSET_KEYS if key in assets)
    out["assets"] = {key: assets[key] for key in assets if key not in OMITTED_LEGACY_ASSET_KEYS}
    out["models"] = _sanitized_models(out.get("models"))
    out["bundle_name"] = bundle_name
    out["schema"] = ARROW_REPLAY_BUNDLE_SCHEMA
    out["runtime_contract"] = {
        "arrow_dataset_root": "datasets",
        "arrow_tables": ("signatures.arrow", "papers.arrow", "paper_authors.arrow", "specter2.arrow"),
        "typed_offline_assets": ("components/*.parquet", "labels/*.parquet", "splits/*"),
        "omitted_legacy_assets": OMITTED_LEGACY_ASSETS,
        "feature_materialization": "recomputed by --feature-mode arrow-rust into the run output directory",
    }
    out["notes"] = (
        "Arrow-only linker replay bundle. Runtime papers, signatures, paper authors, "
        "and SPECTER2 rows are stored under datasets/<dataset>/ as Arrow IPC files. "
        "Candidate members, labels, and splits remain typed offline artifacts. "
        "Legacy raw JSON, pickle embeddings, and precomputed feature rows are not "
        "part of this bundle; promoted feature tables are regenerated for the "
        "selected pairwise model during replay."
    )
    return out, {
        "bundle_name": bundle_name,
        "removed_bundle_asset_keys": removed_assets,
        "removed_classic_model_path_keys": list(LEGACY_CLASSIC_ROW_PATH_KEYS) + ["extra_eval_paths"],
        "retained_bundle_asset_keys": sorted(out["assets"]),
    }


def sanitized_source_provenance_payload(
    payload: Mapping[str, Any],
    *,
    bundle_name: str,
    legacy_source_bundle_name: str | None,
    legacy_source_url: str | None,
) -> dict[str, Any]:
    """Return source provenance without paths to omitted physical assets."""

    source_manifest = dict(payload)
    provenance: dict[str, Any] = {
        "schema": ARROW_REPLAY_SOURCE_PROVENANCE_SCHEMA,
        "bundle_name": bundle_name,
        "datasets": _dataset_names_from_source_manifest(source_manifest),
        "runtime_contract": {
            "arrow_dataset_root": "datasets",
            "typed_offline_assets": ("components/*.parquet", "labels/*.parquet", "splits/*"),
            "omitted_legacy_assets": OMITTED_LEGACY_ASSETS,
        },
        "components": source_manifest.get("components", []),
        "label_tables": source_manifest.get("table_summaries", []),
        "needed_component_counts": source_manifest.get("needed_component_counts", {}),
        "needed_query_signature_counts": source_manifest.get("needed_query_signature_counts", {}),
        "needed_signature_counts": source_manifest.get("needed_signature_counts", {}),
        "validation": source_manifest.get("validation", {}),
        "legacy_source_counts": _legacy_source_counts(source_manifest),
    }
    if legacy_source_bundle_name:
        provenance["legacy_source_bundle_name"] = legacy_source_bundle_name
    if legacy_source_url:
        provenance["legacy_source_url"] = legacy_source_url
    return provenance


def sanitize_arrow_replay_bundle(
    bundle_root: Path,
    *,
    write: bool,
    legacy_source_bundle_name: str | None = None,
    legacy_source_url: str | None = None,
) -> dict[str, Any]:
    """Sanitize ``bundle.json`` and optional source provenance for one bundle."""

    bundle_root = bundle_root.resolve()
    bundle_path = bundle_root / "bundle.json"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing bundle.json: {bundle_path}")
    bundle_payload, report = sanitized_bundle_payload(_read_json(bundle_path), bundle_name=bundle_root.name)
    if write:
        _replace_json(bundle_path, bundle_payload)

    source_manifest_path = bundle_root / "source_bundle_manifest.json"
    if source_manifest_path.exists():
        source_provenance = sanitized_source_provenance_payload(
            _require_mapping(_read_json(source_manifest_path), context="source bundle manifest"),
            bundle_name=bundle_root.name,
            legacy_source_bundle_name=legacy_source_bundle_name,
            legacy_source_url=legacy_source_url,
        )
        report["source_provenance_path"] = str(source_manifest_path)
        report["source_provenance_schema"] = ARROW_REPLAY_SOURCE_PROVENANCE_SCHEMA
        if write:
            _replace_json(source_manifest_path, source_provenance)
    report["mode"] = "write" if write else "dry-run"
    report["bundle_root"] = str(bundle_root)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_root", type=Path)
    parser.add_argument("--write", action="store_true", help="Rewrite metadata files in place.")
    parser.add_argument("--legacy-source-bundle-name", default=None)
    parser.add_argument("--legacy-source-url", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = sanitize_arrow_replay_bundle(
        args.bundle_root,
        write=bool(args.write),
        legacy_source_bundle_name=args.legacy_source_bundle_name,
        legacy_source_url=args.legacy_source_url,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
