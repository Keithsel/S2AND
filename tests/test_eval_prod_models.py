from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import scripts.eval_prod_models as eval_prod_models

_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec"


def _is_lfs_pointer(path: Path) -> bool:
    if not path.is_file():
        return False
    return path.read_bytes()[: len(_LFS_POINTER_PREFIX)] == _LFS_POINTER_PREFIX


def _skip_if_missing_or_lfs_pointer(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        if os.environ.get("CI"):
            raise pytest.fail.Exception(f"missing LFS-backed artifact(s) in CI: {missing}")
        raise pytest.skip.Exception(f"missing LFS-backed artifact(s): {missing}")
    pointers = [str(path) for path in paths if _is_lfs_pointer(path)]
    if pointers:
        if os.environ.get("CI"):
            raise pytest.fail.Exception(f"Git LFS artifact(s) not materialized in CI: {pointers}")
        raise pytest.skip.Exception(f"Git LFS artifact(s) not materialized: {pointers}")


def test_first_missing_arrow_dataset_error_reports_failing_pair(monkeypatch) -> None:
    def fake_resolve(_arrow_root: str, dataset_name: str, specter_suffix: str) -> dict[str, str]:
        if dataset_name == "second" and specter_suffix == "_specter2.pkl":
            raise FileNotFoundError("missing specter2.arrow")
        return {}

    monkeypatch.setattr(eval_prod_models, "resolve_arrow_dataset_paths", fake_resolve)

    error = eval_prod_models.first_missing_arrow_dataset_error(
        "arrow-root",
        ["first", "second"],
        ["_specter.pickle", "_specter2.pkl"],
    )

    assert error is not None
    assert "dataset='second'" in str(error)
    assert "specter_suffix='_specter2.pkl'" in str(error)


def test_empty_optional_dataset_and_specter_lists_fall_back_to_defaults() -> None:
    assert eval_prod_models._resolve_requested_datasets(["pubmed", "qian"], [], "mini") == ["pubmed", "qian"]
    assert eval_prod_models._resolve_requested_specter_suffixes(["s1", "s2"], []) == ["s1", "s2"]


def test_read_arrow_s2_blocks_reads_columns_without_row_dicts(tmp_path: Path) -> None:
    import pyarrow as pa

    table = pa.table(
        {
            "signature_id": pa.array(["s1", "s2", "s3"], type=pa.string()),
            "author_block": pa.array(["a smith", "a smith", "b jones"], type=pa.string()),
        }
    )
    path = tmp_path / "signatures.arrow"
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)

    assert eval_prod_models.read_arrow_s2_blocks(str(path)) == {
        "a smith": ["s1", "s2"],
        "b jones": ["s3"],
    }


@pytest.mark.parametrize("block_count", [1, 2, 4])
def test_split_blocks_like_anddata_rejects_tiny_smoke_datasets_like_anddata(block_count: int) -> None:
    blocks = {f"b{index}": [f"s{index}"] for index in range(block_count)}

    with pytest.raises(ValueError):
        eval_prod_models.split_blocks_like_anddata(blocks, random_seed=1)


def test_split_blocks_like_anddata_preserves_input_order_for_legacy_eval_split() -> None:
    block_items = [(f"b{index:02d}", [f"s{index}"]) for index in range(20)]
    forward = dict(block_items)
    reverse = dict(reversed(block_items))

    forward_split = eval_prod_models.split_blocks_like_anddata(forward, random_seed=1)
    reverse_split = eval_prod_models.split_blocks_like_anddata(reverse, random_seed=1)

    assert [set(split) for split in forward_split] != [set(split) for split in reverse_split]


def _read_minimal_incremental_signatures(signatures_path: Path) -> dict[str, Any]:
    import pyarrow as pa

    with pa.memory_map(str(signatures_path), "r") as source:
        table = pa.ipc.open_file(source).read_all()
    signatures: dict[str, Any] = {}
    for row in table.to_pylist():
        signature_id = str(row["signature_id"])
        signatures[signature_id] = SimpleNamespace(
            signature_id=signature_id,
            paper_id=str(row["paper_id"]),
            author_info_first=row["author_first"],
            author_info_first_normalized_without_apostrophe=row["author_first"],
            author_info_last=row["author_last"],
            author_info_orcid=row["author_orcid"],
        )
    return signatures


class _ArrowIncrementalFixtureDataset:
    def __init__(
        self,
        arrow_paths: Mapping[str, str],
        signatures: dict[str, Any],
        cluster_seeds_path: Path,
    ) -> None:
        self.arrow_paths = {str(key): str(value) for key, value in arrow_paths.items() if key != "clusters"}
        self.arrow_paths["cluster_seeds"] = str(cluster_seeds_path)
        self.signatures = signatures
        self.cluster_seeds_require: dict[str, str] = {}
        self.cluster_seeds_disallow: set[tuple[str, str]] = set()
        self.altered_cluster_signatures: list[str] = []
        self.name_tuples = "filtered"
        self.max_seed_cluster_id = 0
        self.name = "pubmed_specter2_arrow_incremental_fixture"
        self.name_counts_last_first_initial_semantics: str | None = None

    def set_name_counts_last_first_initial_semantics(self, semantics: str) -> None:
        self.name_counts_last_first_initial_semantics = semantics


def test_resolve_arrow_dataset_paths_includes_name_counts_index_from_manifest(tmp_path: Path) -> None:
    dataset_root = tmp_path / "arrow" / "dummy"
    dataset_root.mkdir(parents=True)
    name_counts_index = tmp_path / "name_counts_index"
    name_counts_index.mkdir()
    for filename in (
        "signatures.arrow",
        "papers.arrow",
        "paper_authors.arrow",
        "specter.arrow",
        "dummy_clusters.json",
    ):
        (dataset_root / filename).touch()
    (dataset_root / "manifest.json").write_text(
        json.dumps({"paths": {"name_counts_index": str(name_counts_index)}}),
        encoding="utf-8",
    )

    resolved = eval_prod_models.resolve_arrow_dataset_paths(str(tmp_path / "arrow"), "dummy", "_specter.pickle")

    assert resolved["name_counts_index"] == str(name_counts_index)


def test_resolve_arrow_dataset_paths_requires_eval_name_counts_index(tmp_path: Path) -> None:
    dataset_root = tmp_path / "arrow" / "dummy"
    dataset_root.mkdir(parents=True)
    for filename in (
        "signatures.arrow",
        "papers.arrow",
        "paper_authors.arrow",
        "specter.arrow",
        "dummy_clusters.json",
    ):
        (dataset_root / filename).touch()

    with pytest.raises(FileNotFoundError, match="Missing Arrow name_counts_index"):
        eval_prod_models.resolve_arrow_dataset_paths(str(tmp_path / "arrow"), "dummy", "_specter.pickle")


def test_resolve_arrow_dataset_paths_rejects_bad_manifest_name_counts_index(tmp_path: Path) -> None:
    dataset_root = tmp_path / "arrow" / "dummy"
    dataset_root.mkdir(parents=True)
    (tmp_path / "arrow" / "name_counts_index").mkdir()
    for filename in (
        "signatures.arrow",
        "papers.arrow",
        "paper_authors.arrow",
        "specter.arrow",
        "dummy_clusters.json",
    ):
        (dataset_root / filename).touch()
    (dataset_root / "manifest.json").write_text(
        json.dumps({"paths": {"name_counts_index": "missing/name_counts_index"}}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="names missing name_counts_index"):
        eval_prod_models.resolve_arrow_dataset_paths(str(tmp_path / "arrow"), "dummy", "_specter.pickle")


def test_cluster_eval_arrow_passes_name_counts_index_and_batch_indexes(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeClusterer:
        def predict_from_arrow_paths(self, block_dict, arrow_paths, **kwargs):
            captured["block_dict"] = dict(block_dict)
            captured["arrow_paths"] = dict(arrow_paths)
            captured["kwargs"] = dict(kwargs)
            return {"pred": ["s1"]}, None

    monkeypatch.setattr(eval_prod_models, "read_arrow_s2_blocks", lambda _path: {"block": ["s1"]})
    monkeypatch.setattr(
        eval_prod_models,
        "split_blocks_like_anddata",
        lambda blocks, *, random_seed: ({}, {}, dict(blocks)),
    )
    monkeypatch.setattr(eval_prod_models, "read_signature_to_cluster_id", lambda _path: {"s1": "truth"})

    arrow_paths = {
        "signatures": "signatures.arrow",
        "papers": "papers.arrow",
        "paper_authors": "paper_authors.arrow",
        "specter": "specter.arrow",
        "clusters": "clusters.json",
        "name_counts_index": "name_counts_index",
        "signatures_batch_index": "signatures.signatures_batch_index.bin",
    }
    eval_prod_models.cluster_eval_arrow(
        arrow_paths,
        SimpleNamespace(predict_from_arrow_paths=FakeClusterer().predict_from_arrow_paths),
        random_seed=42,
        n_jobs=1,
    )

    assert captured["block_dict"] == {"block": ["s1"]}
    assert captured["kwargs"]["load_name_counts"] is True
    assert captured["arrow_paths"]["name_counts_index"] == "name_counts_index"
    assert captured["arrow_paths"]["signatures_batch_index"] == "signatures.signatures_batch_index.bin"
    assert "clusters" not in captured["arrow_paths"]


def test_construct_cluster_to_signatures_reports_missing_assignments() -> None:
    with pytest.raises(ValueError, match="missing cluster assignments"):
        eval_prod_models.construct_cluster_to_signatures({"s1": "c1"}, {"block": ["s1", "s2"]})


@pytest.mark.requires_lfs
def test_pubmed_specter2_arrow_fixture_matches_production_eval() -> None:
    pytest.importorskip("s2and_rust")

    from s2and.production_model import load_production_model

    fixture_root = Path("tests/fixtures/arrow/pubmed_specter2")
    fixture_dataset = fixture_root / "pubmed"
    production_model = Path("s2and/data/production_model_v1.21")
    _skip_if_missing_or_lfs_pointer(
        [
            fixture_dataset / "signatures.arrow",
            fixture_dataset / "papers.arrow",
            fixture_dataset / "paper_authors.arrow",
            fixture_dataset / "specter2.arrow",
            fixture_dataset / "signatures.signatures_batch_index.bin",
            fixture_dataset / "papers.papers_batch_index.bin",
            fixture_dataset / "paper_authors.paper_authors_batch_index.bin",
            fixture_dataset / "specter2.specter_batch_index.bin",
            fixture_dataset / "name_counts_index/generations/pubmed-specter2/first.bin",
            fixture_dataset / "name_counts_index/generations/pubmed-specter2/last.bin",
            fixture_dataset / "name_counts_index/generations/pubmed-specter2/first_last.bin",
            fixture_dataset / "name_counts_index/generations/pubmed-specter2/last_first_initial.bin",
            production_model / "manifest.json",
            production_model / "clusterer.json",
            production_model / "pairwise/main.lgb",
            production_model / "pairwise/nameless.lgb",
            production_model / "pairwise/metadata.json",
        ]
    )

    arrow_paths = eval_prod_models.resolve_arrow_dataset_paths(str(fixture_root), "pubmed", "_specter2.pkl")
    assert Path(arrow_paths["specter"]).name == "specter2.arrow"
    assert Path(arrow_paths["name_counts_index"]).resolve() == (fixture_dataset / "name_counts_index").resolve()

    clusterer = load_production_model(str(production_model))
    clusterer.use_cache = False
    clusterer.n_jobs = 4
    cluster_metrics, _ = eval_prod_models.cluster_eval_arrow(
        arrow_paths,
        clusterer,
        random_seed=42,
        n_jobs=4,
    )

    assert cluster_metrics["B3 (P, R, F1)"] == pytest.approx((1.0, 0.892, 0.943), abs=5e-4)


@pytest.mark.requires_lfs
def test_pubmed_specter2_arrow_fixture_incremental_smoke_matches_expected_b3(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    s2and_rust = pytest.importorskip("s2and_rust")
    if not hasattr(s2and_rust, "raw_block_query_candidate_plan_arrow"):
        raise pytest.skip.Exception("raw Arrow incremental candidate planning is unavailable")

    from s2and.eval import b3_precision_recall_fscore
    from s2and.incremental_linking.feature_block import write_cluster_seeds_arrow
    from s2and.production_model import load_production_model

    monkeypatch.setenv("S2AND_BACKEND", "rust")
    fixture_root = Path("tests/fixtures/arrow/pubmed_specter2")
    fixture_dataset = fixture_root / "pubmed"
    production_model = Path("s2and/data/production_model_v1.21")
    _skip_if_missing_or_lfs_pointer(
        [
            fixture_dataset / "signatures.arrow",
            fixture_dataset / "papers.arrow",
            fixture_dataset / "paper_authors.arrow",
            fixture_dataset / "specter2.arrow",
            fixture_dataset / "signatures.signatures_batch_index.bin",
            fixture_dataset / "papers.papers_batch_index.bin",
            fixture_dataset / "paper_authors.paper_authors_batch_index.bin",
            fixture_dataset / "specter2.specter_batch_index.bin",
            fixture_dataset / "name_counts_index/generations/pubmed-specter2/first.bin",
            fixture_dataset / "name_counts_index/generations/pubmed-specter2/last.bin",
            fixture_dataset / "name_counts_index/generations/pubmed-specter2/first_last.bin",
            fixture_dataset / "name_counts_index/generations/pubmed-specter2/last_first_initial.bin",
            production_model / "manifest.json",
            production_model / "clusterer.json",
            production_model / "pairwise/main.lgb",
            production_model / "pairwise/nameless.lgb",
            production_model / "pairwise/metadata.json",
            production_model / "incremental_linker/booster.lgb",
            production_model / "incremental_linker/metadata.json",
        ]
    )

    arrow_paths = eval_prod_models.resolve_arrow_dataset_paths(str(fixture_root), "pubmed", "_specter2.pkl")
    signatures = _read_minimal_incremental_signatures(fixture_dataset / "signatures.arrow")
    _train_block_dict, _val_block_dict, test_block_dict = eval_prod_models.split_blocks_like_anddata(
        eval_prod_models.read_arrow_s2_blocks(arrow_paths["signatures"]),
        random_seed=42,
    )
    signature_to_cluster_id = eval_prod_models.read_signature_to_cluster_id(arrow_paths["clusters"])
    cluster_to_signatures = eval_prod_models.construct_cluster_to_signatures(signature_to_cluster_id, test_block_dict)

    clusterer = load_production_model(str(production_model))
    clusterer.use_cache = False
    clusterer.n_jobs = 4
    predicted_clusters: dict[str, list[str]] = {}
    total_query_count = 0
    total_candidate_row_count = 0

    for block_index, (block_key, block_signatures) in enumerate(sorted(test_block_dict.items())):
        seed_signature_to_cluster: dict[str, str] = {}
        seen_cluster_ids: set[str] = set()
        for signature_id in block_signatures:
            cluster_id = signature_to_cluster_id[signature_id]
            if cluster_id in seen_cluster_ids:
                continue
            seed_signature_to_cluster[signature_id] = cluster_id
            seen_cluster_ids.add(cluster_id)

        cluster_seeds_path = tmp_path / f"cluster_seeds_{block_index}.arrow"
        write_cluster_seeds_arrow(cluster_seeds_path, seed_signature_to_cluster)
        dataset = _ArrowIncrementalFixtureDataset(arrow_paths, signatures, cluster_seeds_path)
        result = cast(
            dict[str, Any],
            clusterer.predict_incremental(
                list(block_signatures),
                cast(Any, dataset),
                prevent_new_incompatibilities=False,
                batching_threshold=None,
                total_ram_bytes=1_000_000_000_000,
            ),
        )
        telemetry = cast(Mapping[str, Any], result["incremental_linker_telemetry"])
        query_count = len(block_signatures) - len(seed_signature_to_cluster)
        assert result["incremental_linker_query_view"] == "raw_arrow"
        assert telemetry["arrow_promoted_incremental"] == 1
        assert telemetry["seed_setup_cluster_seeds_source"] == "arrow"
        assert telemetry["seed_arrow_reused_source"] == 1
        assert telemetry["query_count"] == query_count
        total_query_count += int(telemetry["query_count"])
        total_candidate_row_count += int(telemetry["candidate_row_count"])

        block_signature_set = set(block_signatures)
        for cluster_id, members in cast(Mapping[str, Sequence[str]], result["clusters"]).items():
            kept_members = [str(member) for member in members if str(member) in block_signature_set]
            if kept_members:
                predicted_clusters[f"{block_index}:{block_key}:{cluster_id}"] = kept_members

    cluster_metrics = b3_precision_recall_fscore(cluster_to_signatures, predicted_clusters)
    assert total_query_count == 127
    assert total_candidate_row_count > 0
    assert cluster_metrics[:3] == pytest.approx((1.0, 0.816, 0.899), abs=5e-4)
