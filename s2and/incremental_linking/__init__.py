"""Private incremental-linking runtime helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "IncrementalLinkingArtifact": ("s2and.incremental_linking.artifact", "IncrementalLinkingArtifact"),
    "IncrementalLinkingArtifactMetadata": (
        "s2and.incremental_linking.artifact",
        "IncrementalLinkingArtifactMetadata",
    ),
    "LinkerCandidateBatch": ("s2and.incremental_linking.linker_pairwise", "LinkerCandidateBatch"),
    "LinkerFeatureMatrix": ("s2and.incremental_linking.features", "LinkerFeatureMatrix"),
    "LinkerRetrievalBatch": ("s2and.incremental_linking.retrieval", "LinkerRetrievalBatch"),
    "LinkOrAbstainCompactResult": ("s2and.incremental_linking.runtime", "LinkOrAbstainCompactResult"),
    "LinkOrAbstainDecision": ("s2and.incremental_linking.runtime", "LinkOrAbstainDecision"),
    "LinkOrAbstainProductionResult": ("s2and.incremental_linking.runtime", "LinkOrAbstainProductionResult"),
    "LinkOrAbstainRetrievedCandidatesResult": (
        "s2and.incremental_linking.runtime",
        "LinkOrAbstainRetrievedCandidatesResult",
    ),
    "assemble_linker_feature_matrix": ("s2and.incremental_linking.features", "assemble_linker_feature_matrix"),
    "build_linker_retrieval_batch_from_raw_candidate_plan": (
        "s2and.incremental_linking.retrieval",
        "build_linker_retrieval_batch_from_raw_candidate_plan",
    ),
    "build_linker_retrieval_batch_rust": (
        "s2and.incremental_linking.retrieval",
        "build_linker_retrieval_batch_rust",
    ),
    "build_promoted_non_pairwise_row_features": (
        "s2and.incremental_linking.row_features",
        "build_promoted_non_pairwise_row_features",
    ),
    "load_incremental_linking_artifact": ("s2and.incremental_linking.artifact", "load_incremental_linking_artifact"),
    "naturalize_incremental_clusters": ("s2and.incremental_linking.runtime", "naturalize_incremental_clusters"),
    "predict_incremental_link_or_abstain_from_raw_arrow_paths": (
        "s2and.incremental_linking.runtime",
        "predict_incremental_link_or_abstain_from_raw_arrow_paths",
    ),
    "promoted_linker_feature_columns": ("s2and.incremental_linking.features", "promoted_linker_feature_columns"),
    "save_incremental_linking_artifact": ("s2and.incremental_linking.artifact", "save_incremental_linking_artifact"),
    "signature_id_to_index_map": ("s2and.incremental_linking.runtime", "signature_id_to_index_map"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
