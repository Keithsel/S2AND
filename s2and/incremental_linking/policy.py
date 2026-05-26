"""Small shared policy helpers for incremental linking orchestration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from s2and.arrow_inputs import require_name_counts_index_artifact
from s2and.incremental_linking.feature_block import normalize_cluster_seed_disallow_pairs


def clusterer_uses_name_count_features(clusterer: Any) -> bool:
    """Return whether the clusterer requires global name-count features."""

    for attr_name in ("featurizer_info", "nameless_featurizer_info"):
        featurizer_info = getattr(clusterer, attr_name, None)
        features_to_use = getattr(featurizer_info, "features_to_use", ())
        if "name_counts" in features_to_use:
            return True
    return False


def clusterer_uses_embedding_features(clusterer: Any) -> bool:
    """Return whether the clusterer requires SPECTER embedding features."""

    for attr_name in ("featurizer_info", "nameless_featurizer_info"):
        featurizer_info = getattr(clusterer, attr_name, None)
        features_to_use = getattr(featurizer_info, "features_to_use", ())
        if "embedding_similarity" in features_to_use:
            return True
    return False


def existing_name_counts_index_path(paths: Mapping[str, Any]) -> str | None:
    """Return the configured name-count index path when it exists."""

    path_value = paths.get("name_counts_index")
    if path_value is not None:
        return require_name_counts_index_artifact(
            path_value,
            context="Arrow name-count index",
            producer_hint="pass a manifest-backed name_counts_index directory",
        )
    return None


def arrow_paths_have_name_counts_index(paths: Mapping[str, Any]) -> bool:
    """Return whether Arrow paths include an existing name-count index."""

    return existing_name_counts_index_path(paths) is not None


def require_arrow_name_counts_index_for_clusterer(
    clusterer: Any,
    arrow_paths: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """Raise when a name-count model is used without an Arrow index sidecar."""

    if clusterer_uses_name_count_features(clusterer) and not arrow_paths_have_name_counts_index(arrow_paths):
        raise ValueError(
            f"{context} with selected name_counts features requires name_counts_index. "
            "Pass the S2AND name-count index directory in arrow_paths['name_counts_index']."
        )


def resolve_load_name_counts_policy(
    clusterer: Any,
    load_name_counts: bool | None | dict[str, Any],
    *,
    context: str,
) -> bool:
    """Return the effective name-count load policy for raw scoring."""

    if isinstance(load_name_counts, dict):
        raise ValueError(f"{context} accepts load_name_counts as a bool or None, not a dict")
    clusterer_requires_name_counts = clusterer_uses_name_count_features(clusterer)
    if load_name_counts is False and clusterer_requires_name_counts:
        raise ValueError(
            f"{context} cannot run with load_name_counts=False when the clusterer selects name_counts features"
        )
    if load_name_counts is None:
        return clusterer_requires_name_counts
    return bool(load_name_counts)


def dataset_cluster_seed_disallows(dataset: Any) -> set[tuple[str, str]]:
    """Return normalized disallow constraints stored on a request dataset."""

    return set(normalize_cluster_seed_disallow_pairs(getattr(dataset, "cluster_seeds_disallow", set()) or set()))


def request_cluster_seed_disallow_parts(
    dataset: Any,
    arrow_disallows: Iterable[tuple[Any, Any]],
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], set[tuple[str, str]]]:
    """Return request, dataset, and Arrow disallow sets with one normalization policy."""

    dataset_disallows = dataset_cluster_seed_disallows(dataset)
    arrow_disallow_set = set(normalize_cluster_seed_disallow_pairs(arrow_disallows))
    request_disallows = set(arrow_disallow_set)
    request_disallows.update(dataset_disallows)
    return request_disallows, dataset_disallows, arrow_disallow_set
