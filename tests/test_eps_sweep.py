from __future__ import annotations

import pickle
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from s2and import text as s2and_text
from scripts.eps_sweep import sweep_eps_on_linking_gold


def test_load_gold_drops_unlabeled_singleton_orcid_rows(tmp_path) -> None:
    gold_path = tmp_path / "gold.parquet"
    pd.DataFrame(
        [
            {
                "dataset": "unit",
                "table_name": "train.parquet",
                "split": "train",
                "source_key": "train",
                "supervision_type": "unlabeled_singleton_orcid",
                "query_signature_id": "q1",
                "member_signature_id": "m1",
                "query_view": "full",
                "label": 0,
                "weight_pair": 1.0,
                "weight_query_balanced": 1.0,
                "weight_query_label_balanced": 1.0,
                "weight_query_class_balanced": 1.0,
            },
            {
                "dataset": "unit",
                "table_name": "train.parquet",
                "split": "train",
                "source_key": "train",
                "supervision_type": "positive_repeat_orcid",
                "query_signature_id": "q2",
                "member_signature_id": "m2",
                "query_view": "full",
                "label": 1,
                "weight_pair": 1.0,
                "weight_query_balanced": 1.0,
                "weight_query_label_balanced": 1.0,
                "weight_query_class_balanced": 1.0,
            },
        ]
    ).to_parquet(gold_path, index=False)

    loaded = sweep_eps_on_linking_gold._load_gold(gold_path)

    assert loaded["query_signature_id"].tolist() == ["q2"]
    assert loaded["supervision_type"].tolist() == ["positive_repeat_orcid"]


def test_eps_sweep_runtime_environment_disables_fasttext(monkeypatch) -> None:
    previous_enabled = s2and_text.fasttext_loading_enabled()
    s2and_text.set_fasttext_loading_enabled(True)
    monkeypatch.setenv("S2AND_SKIP_FASTTEXT", "0")

    try:
        sweep_eps_on_linking_gold._configure_runtime_environment(cast(Any, SimpleNamespace(backend="python", n_jobs=2)))

        assert s2and_text.fasttext_loading_enabled() is False
        assert sweep_eps_on_linking_gold.os.environ["S2AND_BACKEND"] == "python"
        assert sweep_eps_on_linking_gold.os.environ["OMP_NUM_THREADS"] == "2"
        assert sweep_eps_on_linking_gold.os.environ["RAYON_NUM_THREADS"] == "2"
        assert sweep_eps_on_linking_gold.os.environ["S2AND_SKIP_FASTTEXT"] == "1"
    finally:
        s2and_text.set_fasttext_loading_enabled(previous_enabled)


def test_ensure_distance_caches_skips_singleton_without_compute_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        sweep_eps_on_linking_gold,
        "_build_arrow_featurizer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("singleton block should not build featurizer")),
    )
    args = SimpleNamespace(
        arrow_root=tmp_path / "arrow",
        batching_threshold=10,
        compute_missing_dists=False,
        dataset="dummy",
        model_path=tmp_path / "model.pkl",
        overwrite_dists=False,
        pair_chunk_size=3,
        suppress_orcid_constraints=False,
        use_orcid_subblocking=False,
    )
    clusterer = SimpleNamespace(batch_size=99)

    rows = sweep_eps_on_linking_gold._ensure_distance_caches(
        cast(Any, args),
        clusterer,
        {"singleton": ["s1"]},
        tmp_path / "cache",
        {"signatures": "signatures.arrow"},
    )

    assert rows[0]["block_key"] == "singleton"
    assert rows[0]["pair_count"] == 0
    assert rows[0]["computed"] is False
    assert clusterer.batch_size == 99


def test_distance_cache_metadata_rejects_overwritten_model_path(tmp_path) -> None:
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"first model")
    args = SimpleNamespace(
        arrow_root=tmp_path / "arrow",
        batching_threshold=10,
        dataset="dummy",
        model_path=model_path,
        pair_chunk_size=3,
        suppress_orcid_constraints=False,
        use_orcid_subblocking=False,
    )
    metadata = sweep_eps_on_linking_gold._cache_metadata(
        cast(Any, args),
        "block",
        ["s1", "s2"],
        "arrow-digest",
    )
    cache_path = tmp_path / "cache.pkl"
    with cache_path.open("wb") as outfile:
        pickle.dump({"metadata": metadata, "dist": [0.25]}, outfile)

    model_path.write_bytes(b"second model with different contents")
    expected_metadata = sweep_eps_on_linking_gold._cache_metadata(
        cast(Any, args),
        "block",
        ["s1", "s2"],
        "arrow-digest",
    )

    with pytest.raises(ValueError, match="model_"):
        sweep_eps_on_linking_gold._load_cached_distance(cache_path, expected_metadata)
