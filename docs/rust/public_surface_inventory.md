# Rust Public Surface Inventory

Status date: 2026-05-25

This inventory records the current Python-visible `s2and_rust` surface before
module splitting or API deletion. It is intentionally about ownership and
cleanup risk, not a user-facing API promise.

## Module Exports

| Export | Owner / caller | Status |
|---|---|---|
| `RustFeaturizer` | `s2and/feature_port.py`, `s2and/rust_calls.py`, production Arrow paths, parity tests | Core class. Production Rust inference should enter through `from_arrow_paths`; non-Arrow constructors are compatibility/training/parity. |
| `RustHybridCentroidRetriever` | `s2and/incremental_linking/retrieval.py`, raw Arrow planners, training query-support code | Core retrieval class. Production runtime should prefer `top_k_hybrid_centroid_pair_plan(...)`. |
| `RustNameCompatibleSubblockSelector` | Internal helper used by `RustHybridCentroidRetriever.top_k_hybrid_centroid_pair_plan(...)`; direct Python use is test-only | Keep until pair-plan subblock filtering no longer needs it. |
| `RawBlockQueryCandidatePlanner` | `s2and/incremental_linking/production.py`, `s2and/incremental_linking/runtime.py` | Canonical production raw Arrow planner. |
| `raw_arrow_labeled_candidate_plan(...)` | `scripts/production/model/linker_train_calibrate_eval.py` | Training/materialization replay surface, not request-time inference. |
| `promoted_linker_non_pairwise_features(...)` | `s2and/incremental_linking/row_features.py` | Production promoted-linker row feature builder. |
| `make_subblocks_with_telemetry_arrow(...)` | `s2and/subblocking.py` | Arrow subblocking helper used by large-block prediction. |
| `signature_ngrams_batch(...)` | `s2and/feature_port.py` and `s2and/data.py` preprocessing | Training/eval preprocessing accelerator, not Arrow production inference. |
| `get_build_info(...)` | `scripts/_rust_suite/common.py`, capability tests | Diagnostics and ABI metadata. |

## `RustFeaturizer`

| Method | Owner / caller | Status |
|---|---|---|
| `from_arrow_paths(...)` | `feature_port.build_rust_featurizer_from_arrow_paths(...)`; full predict, subblocked predict, raw Arrow scoring | Production Arrow constructor. Filtered reads require batch indexes by default; explicit full-scan opt-in is compatibility/testing only. |
| `from_dataset(...)` | `feature_port.build_rust_featurizer(...)`, `_get_rust_featurizer(...)`; training/eval, parity, compatibility | Keep callable but do not present as production inference. |
| `from_json_paths(...)` | `feature_port.build_rust_featurizer(...)`; JSON compatibility scripts/tests | Compatibility and benchmark surface. |
| `json_ingest_telemetry(...)` | JSON ingest validation and service-JSON tests | Compatibility telemetry. |
| `update_cluster_seeds(...)` and `update_signature_name_counts(...)` | cache/seed update helpers in `feature_port.py` and tests | Compatibility/training lifecycle helpers. |
| `signature_ids(...)` | pairwise matrix wrappers, promoted incremental runtime, parity scripts | Shared index-order contract; keep. |
| `signature_rule_metadata(...)`, `signature_name_counts_present(...)`, `cluster_seeds_require(...)` | parity/debug tests and state restoration checks | Debug/parity metadata; not production routing. |
| `get_constraint(...)` | `s2and/model.py`, `s2and/rust_calls.py`, tests | Single-pair `ANDData` Rust constraint helper used by compatibility/full-predict plumbing. |
| `get_constraints_matrix_indexed(...)` | `model.py`, `rust_calls.py`, parity tests | Maintained indexed constraint API. |
| `get_constraints_block_upper_triangle_indexed(...)` | `model.py`, Arrow parity script | Maintained blockwise constraint API. |
| `linker_pair_index_arrays_constraint_labels(...)` | promoted linker training/materialization and runtime tests | Maintained promoted incremental constraint-label API. |
| `linker_pair_distance_accumulators(...)` | promoted incremental runtime and tests | Maintained promoted incremental aggregate API. |
| `featurize_pair(...)` | parity/debug tests and compatibility wrappers | Keep as debug/parity helper only. |
| `featurize_pairs(...)` | legacy row-by-row fallback in `s2and/featurizer.py` | Keep until Python featurizer no longer needs row-by-row fallback. |
| `featurize_pairs_matrix(...)` | pairwise compatibility, parity, and Arrow parity script | Matrix API retained while callers still pass string pairs. |
| `featurize_pairs_matrix_indexed(...)` | `s2and/featurizer.py`, capability probes | Preferred pairwise matrix API for indexed callers. |
| `linker_pair_index_arrays_and_aggregate_stats(...)` | `s2and/incremental_linking/linker_pairwise.py` | Canonical promoted linker pair-feature plus aggregate API. |
| `linker_pair_index_arrays_and_aggregate_stats(..., emit_matrix=False)` | `s2and/incremental_linking/linker_pairwise.py`, capability probes | Canonical aggregate-only mode; preserves the no-matrix fast path without a second PyO3 method. |
| `featurize_block_upper_triangle_matrix_indexed(...)` | blockwise full predict | Maintained blockwise feature API. |
| `save(...)` / `load(...)` | lifecycle/debug persistence; `load(...)` is used by counter-data measurement scripts | Compatibility/debug persistence. |

## Retrieval Classes

| Method | Owner / caller | Status |
|---|---|---|
| `RustHybridCentroidRetriever.__new__(...)` | raw Arrow planners, training query support, tests | Maintained constructor. |
| `top_k_hybrid_centroid_pair_plan(...)` | `s2and/incremental_linking/retrieval.py`, raw Arrow planners | Canonical runtime retrieval output. |
| `top_k_experimental_weighted_hybrid_centroid_subset(...)` | `s2and/incremental_linking_training/query_support.py`, tests | Training/query-support scoring surface. |
| `RustNameCompatibleSubblockSelector.select(...)` | tests only; internal Rust helper trio used by pair-plan | Keep while pair-plan subblock filtering depends on the selector internals. |
| `RawBlockQueryCandidatePlanner.__new__(...)`, `build_telemetry(...)`, `plan(...)` | `s2and/incremental_linking/production.py`, `s2and/incremental_linking/runtime.py`; tests | Canonical reusable production raw Arrow planner. |

## Python Wrapper Ownership

| Wrapper | Owner / caller | Status |
|---|---|---|
| `feature_port.build_rust_featurizer_from_arrow_paths(...)` | strict full predict, subblocked predict, raw Arrow scoring | Production constructor wrapper. |
| `feature_port.build_rust_featurizer(...)`, `_get_rust_featurizer(...)`, `warm_rust_featurizer(...)` | `ANDData` training/eval, compatibility, parity, legacy scripts | Compatibility/training dispatcher. |
| `rust_calls.get_constraints_matrix_indexed_rust(...)` and `get_constraints_block_upper_triangle_indexed_rust(...)` | full predict and parity | Maintained constraint wrappers. |
| `rust_calls.build_linker_pair_features_and_aggregate_stats_arrays_rust(...)` | promoted incremental pairwise scoring | Maintained canonical array wrapper. |
| `rust_calls.build_linker_pair_aggregate_stats_arrays_rust(...)` | promoted incremental aggregate-only path | Thin Python wrapper over `linker_pair_index_arrays_and_aggregate_stats(..., emit_matrix=False)`. |
| `runtime.detect_rust_runtime_capabilities(...)` markers | backend selection and tests | Update markers before deleting any method they probe. |

## Cleanup Notes

- Keep `RawBlockQueryCandidatePlanner` as the canonical raw Arrow planning
  API; it owns reusable seed state and strict indexed-read defaults.
- Do not delete `RustNameCompatibleSubblockSelector` internals; the pair-plan
  route still uses them for retrieval subblock filtering.
- Status 2026-05-25: `RustHybridCentroidRetriever.summary_count(...)` was
  removed after a repo-local no-caller scan.
- Status 2026-05-25:
  `linker_pair_features_and_aggregate_stats_indexed(...)` and its Python
  wrapper were removed after the repo-local callers moved to the canonical
  index-array API.
- Status 2026-05-25: aggregate-only remains a runtime mode, but the separate
  `linker_pair_index_arrays_aggregate_stats(...)` PyO3 method was folded into
  `linker_pair_index_arrays_and_aggregate_stats(..., emit_matrix=False)`.
- Status 2026-05-25: the string-pair `get_constraints_matrix(...)` PyO3 method
  and `rust_calls.get_constraints_matrix_rust(...)` wrapper were removed after
  parity tests moved to indexed constraint matrices.
- Status 2026-05-25: direct retriever debug APIs
  `top_k_hybrid_centroid(...)` and `chooser_feature_rows_subset(...)` were
  removed after capability probes and tests moved to the canonical pair-plan
  route.
- Status 2026-05-25: the one-shot
  `raw_block_query_candidate_plan_arrow(...)` PyO3 wrapper was removed after
  runtime callers moved to `RawBlockQueryCandidatePlanner`.
- Status 2026-05-25: `RustFeaturizer.from_feature_block(...)`,
  `feature_port.build_rust_featurizer_from_feature_block(...)`, and raw
  payload scoring wrappers were removed after a repo-local no-caller scan.
  Lower-level Python `FeatureBlock` builders remain only for fixture,
  compatibility-conversion, and parity-helper tests; production Rust scoring
  uses Arrow request tables.
- Status 2026-05-25: Arrow string columns are strict at the Rust boundary.
  ID, text/language, and alias columns must be Arrow string types; integer
  coercion is not accepted. Declared alias override paths (`name_pairs` or
  `name_tuples`) are preflighted like other optional sidecars.
- Status 2026-05-25: Arrow graph subblocking uses raw-planner batch lookup
  indexes for filtered evidence reads and no longer exposes the unused Python
  full-table graph loader.
