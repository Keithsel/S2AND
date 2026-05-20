"""Private incremental-linking runtime helpers."""

from s2and.incremental_linking.artifact import (
    IncrementalLinkingArtifact,
    IncrementalLinkingArtifactMetadata,
    load_incremental_linking_artifact,
    save_incremental_linking_artifact,
)
from s2and.incremental_linking.feature_block import (
    FeatureBlock,
    FeatureBlockPaper,
    FeatureBlockPaperAuthor,
    FeatureBlockSignature,
    FeatureBlockSignatureOrder,
    feature_block_for_signature_order,
    feature_block_from_anddata,
    feature_block_from_arrow_paths,
    feature_block_from_raw_payloads,
    feature_block_signature_order_from_raw_candidate_plan,
    feature_block_to_mini_anddata,
    write_feature_block_arrow_from_anddata,
    write_feature_block_arrow_tables,
    write_name_counts_arrow,
    write_name_counts_index,
    write_name_pairs_arrow,
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
    predict_incremental_link_or_abstain_from_raw_feature_block,
    predict_incremental_link_or_abstain_from_raw_payloads,
    signature_id_to_index_map,
)

__all__ = [
    "IncrementalLinkingArtifact",
    "IncrementalLinkingArtifactMetadata",
    "FeatureBlock",
    "FeatureBlockPaper",
    "FeatureBlockPaperAuthor",
    "FeatureBlockSignature",
    "FeatureBlockSignatureOrder",
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
    "feature_block_for_signature_order",
    "feature_block_from_anddata",
    "feature_block_from_arrow_paths",
    "feature_block_from_raw_payloads",
    "feature_block_signature_order_from_raw_candidate_plan",
    "feature_block_to_mini_anddata",
    "load_incremental_linking_artifact",
    "naturalize_incremental_clusters",
    "predict_incremental_link_or_abstain_from_raw_arrow_paths",
    "predict_incremental_link_or_abstain_from_raw_feature_block",
    "predict_incremental_link_or_abstain_from_raw_payloads",
    "promoted_linker_feature_columns",
    "save_incremental_linking_artifact",
    "signature_id_to_index_map",
    "write_feature_block_arrow_from_anddata",
    "write_feature_block_arrow_tables",
    "write_name_counts_arrow",
    "write_name_counts_index",
    "write_name_pairs_arrow",
]
