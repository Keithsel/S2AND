from __future__ import annotations

import json
from pathlib import Path

from s2and.incremental_linking.feature_block import (
    cleanup_stale_name_counts_generations,
    write_name_counts_index,
)


def _write_generation(index_dir: Path, generation_name: str) -> None:
    generation_dir = index_dir / "generations" / generation_name
    generation_dir.mkdir(parents=True)
    for filename in ("first.bin", "last.bin", "first_last.bin", "last_first_initial.bin"):
        (generation_dir / filename).write_bytes(b"index")
    (generation_dir / ".published").write_text("", encoding="utf-8")


def test_write_name_counts_index_does_not_delete_previous_published_generation(tmp_path, monkeypatch) -> None:
    import s2and.data as data_module

    monkeypatch.setattr(
        data_module,
        "_load_name_counts_cached",
        lambda: ({"ada": 1}, {"lovelace": 1}, {"ada lovelace": 1}, {"lovelace a": 1}),
    )

    index_path, _first_metrics = write_name_counts_index(tmp_path, overwrite=True)
    _index_path, _second_metrics = write_name_counts_index(tmp_path, overwrite=True)

    generations = [path for path in (Path(index_path) / "generations").iterdir() if path.is_dir()]
    assert len(generations) == 2
    assert all((path / ".published").exists() for path in generations)


def test_cleanup_stale_name_counts_generations_keeps_manifest_generation(tmp_path: Path) -> None:
    index_dir = tmp_path / "name_counts_index"
    _write_generation(index_dir, "gen-old")
    _write_generation(index_dir, "gen-current")
    manifest = {
        "files": {
            "first": {"path": "generations/gen-current/first.bin"},
            "last": {"path": "generations/gen-current/last.bin"},
            "first_last": {"path": "generations/gen-current/first_last.bin"},
            "last_first_initial": {"path": "generations/gen-current/last_first_initial.bin"},
        }
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    metrics = cleanup_stale_name_counts_generations(index_dir)

    assert metrics == {"removed_generation_count": 1}
    assert not (index_dir / "generations" / "gen-old").exists()
    assert (index_dir / "generations" / "gen-current").exists()
