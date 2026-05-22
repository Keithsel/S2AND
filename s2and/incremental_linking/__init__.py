"""Private incremental-linking runtime helpers."""

from s2and.incremental_linking.artifact import (
    IncrementalLinkingArtifact,
    IncrementalLinkingArtifactMetadata,
    load_incremental_linking_artifact,
    save_incremental_linking_artifact,
)
from s2and.incremental_linking.features import (
    LinkerFeatureMatrix,
    assemble_linker_feature_matrix,
    promoted_linker_feature_columns,
)
from s2and.incremental_linking.linker_pairwise import LinkerCandidateBatch
from s2and.incremental_linking.retrieval import (
    LinkerRetrievalBatch,
    build_linker_retrieval_batch_from_raw_candidate_plan,
    build_linker_retrieval_batch_rust,
)
from s2and.incremental_linking.row_features import build_promoted_non_pairwise_row_features
from s2and.incremental_linking.runtime import (
    LinkOrAbstainCompactResult,
    LinkOrAbstainDecision,
    LinkOrAbstainProductionResult,
    LinkOrAbstainRetrievedCandidatesResult,
    naturalize_incremental_clusters,
    predict_incremental_link_or_abstain_from_raw_arrow_paths,
    signature_id_to_index_map,
)

__all__ = [
    "IncrementalLinkingArtifact",
    "IncrementalLinkingArtifactMetadata",
    "LinkerCandidateBatch",
    "LinkerFeatureMatrix",
    "LinkerRetrievalBatch",
    "LinkOrAbstainCompactResult",
    "LinkOrAbstainDecision",
    "LinkOrAbstainProductionResult",
    "LinkOrAbstainRetrievedCandidatesResult",
    "assemble_linker_feature_matrix",
    "build_linker_retrieval_batch_from_raw_candidate_plan",
    "build_linker_retrieval_batch_rust",
    "build_promoted_non_pairwise_row_features",
    "load_incremental_linking_artifact",
    "naturalize_incremental_clusters",
    "predict_incremental_link_or_abstain_from_raw_arrow_paths",
    "promoted_linker_feature_columns",
    "save_incremental_linking_artifact",
    "signature_id_to_index_map",
]
