# Work Plan (Rust/Platform Backlog)

Status date: 2026-05-20

This doc is not the active giant-block experiment log. The current Rust/platform
direction is driven by the raw Arrow candidate-plan evidence in
`docs/rust/raw_block_query_candidate_plan.md`.

This file tracks Rust/platform items that are still open and worth sequencing.
It intentionally excludes items that are already done.

Start here:
- Threading and preprocessing defaults: `docs/threading.md`
- Rust runtime contract and verification commands: `docs/rust/runtime.md`
- Rust gate commands: `docs/rust/baselines.md`
- Artifact divergence and migration plan: `docs/rust/artifact_divergence.md`
- Raw Arrow candidate-plan evidence: `docs/rust/raw_block_query_candidate_plan.md`
- Environment variables: `docs/environment.md`
- Stage-wise memory telemetry: `docs/stage_memory_estimates.md`

## Partial

### Narrow inference contract / FeatureBlock (Ask-first)

Status:
- Started and now the main architectural boundary for the next Rust work.
- Code boundary: `s2and/incremental_linking/feature_block.py`.
- Initial focused tests: `tests/test_feature_block.py`.
- Public raw scoring wrappers now build a mini `FeatureBlock` from the raw
  candidate plan or raw payloads and run the existing Rust pairwise-feature and
  constraint-label kernels through `RustFeaturizer.from_feature_block(...)`:
  `predict_incremental_link_or_abstain_from_raw_feature_block(...)` and
  `predict_incremental_link_or_abstain_from_raw_payloads(...)`.
- Direct Arrow scoring is now wired for raw incremental requests:
  `RustFeaturizer.from_arrow_paths(...)` builds the filtered scoring featurizer
  from Arrow IPC, and
  `predict_incremental_link_or_abstain_from_raw_arrow_paths(...)` runs
  retrieval, scoring, constraints, pairwise aggregation, and link/abstain
  without constructing full-block `ANDData`.
- Full-block `predict` now has the same direct Rust/Arrow entry point:
  `Clusterer.predict_from_arrow_paths(...)` builds a filtered Rust featurizer
  from Arrow IPC and reuses the existing blockwise Rust feature, constraint, and
  clustering logic without `ANDData`.
- The complete-Arrow full-predict parity harness now compares the full
  39-feature matrix, upper-triangle constraint labels, distance matrices, and
  final clusters between incumbent `ANDData` and direct Arrow/Rust paths.
- Evidence files:
  `scratch/baseline/a_silva_single_query/arrow_full_specter/raw_candidate_plan_compare_link_abstain_mini_feature_block_20260520.json`
  and
  `scratch/baseline/h_wang_single_query/arrow_full_specter/raw_candidate_plan_compare_link_abstain_mini_feature_block_20260520.json`.
  Both show exact normalized decisions, linked clusters, candidate-row counts,
  pair counts, probabilities, and 53-feature matrices.
- Current direct-Arrow profile evidence:
  - h_wang raw incremental, release build, no global name-count load:
    `scratch/baseline/h_wang_single_query/arrow_full_specter/direct_raw_arrow_wrapper_release_no_counts_20260520.json`
    reports 15.02s predict time after model load, including 11.93s raw Arrow
    retrieval/summary build, 1.26s filtered Arrow scoring-featurizer build, and
    the same linked cluster as the exact raw FeatureBlock comparison.
  - The same h_wang wrapper with current fixture global name-count loading:
    `direct_raw_arrow_wrapper_release_load_counts_20260520.json` reports
    60.06s predict time, with 45.97s in cold name-count loading/build.
  - a_silva full predict, 1000 signatures / 499,500 pairs, release build, direct
    Arrow/no global name-count load:
    `scratch/baseline/a_silva_single_query/arrow_full_specter/direct_predict_arrow_release_subset1000_no_counts_20260520.json`
    reports 1.31s for direct Arrow predict after model load. The incumbent
    full-scope profile for the same pair count spent 50.86s in `ANDData`
    construction and 6.19s in predict.
- Latest raw Arrow scoring update:
  - Rust now emits the remaining promoted row signals needed by the raw
    link/abstain scorer: name-count rarity, candidate max paper-author count,
    paper-author-list overlap, local author-window overlap, and author-count
    delta. The public raw Arrow wrapper no longer builds a Python signal
    `FeatureBlock`; `raw_arrow_signal_seconds` is ~0.0002s on h_wang.
  - The hot path should use embedded per-signature name-count columns in
    `signatures.arrow`. A full h_wang embedded-count bundle was generated at
    `scratch/baseline/h_wang_single_query/arrow_full_specter_embedded_counts`
    in 115.2s. With embedded counts and the default packaged filtered
    `name_tuples` text aliases,
    `direct_raw_arrow_wrapper_native_signals_embedded_counts_20260520.json`
    reports 17.41s predict time, 16.04s raw retrieval/summary, 1.34s filtered
    Arrow featurizer build, and the same linked cluster.
  - Do not read the global `name_counts.arrow` lookup per request unless a
    block lacks embedded counts. On the old h_wang fixture without embedded
    counts, the 1.4GB global Arrow lookup adds 25.73s in raw retrieval and
    35.51s in featurizer build:
    `direct_raw_arrow_wrapper_native_signals_external_name_artifacts_20260520.json`.
  - A sorted exact-verified `name_counts_index/` sidecar now covers the case
    where counts cannot be embedded. On the same old h_wang fixture,
    `direct_raw_arrow_wrapper_native_signals_name_counts_index_20260520.json`
    reports 17.04s predict time; name-count setup falls to 0.028s, and the
    filtered scoring-featurizer build falls to 1.27s. The one-time index build
    wrote 35,419,433 rows / 1.86GB in 478.2s.
  - h_wang full predict, 1000 signatures / 499,500 pairs, embedded counts and
    the default packaged filtered `name_tuples` aliases:
    `direct_predict_embedded_counts_subset1000_20260520.json`
    reports 2.23s for direct Arrow predict after model load.
  - h_wang full predict, 1000 signatures / 499,500 pairs, old signatures table
    plus `name_counts_index/`:
    `direct_predict_name_counts_index_subset1000_20260520.json` reports 2.31s
    for direct Arrow predict after model load.

Decision:
- Do not port all of `ANDData` to Rust. `ANDData` is the broad Python
  compatibility/reference object for loading, training/eval splits, pair
  sampling, Sinonym mutation, legacy artifact behavior, cluster bookkeeping,
  and parity tests.
- Define a smaller inference contract, tentatively `FeatureBlock`, containing
  only the inputs needed for retrieval, constraints, pair features, and
  link/abstain scoring.
- Feed the same Rust inference core from two adapters:
  - `ANDData -> FeatureBlock`, for incumbent-path parity and existing callers.
  - Arrow/raw request payload -> `FeatureBlock`, for fast production inference
    without constructing full-block `ANDData`.

What remains:
- Regenerate durable production Arrow artifacts with the complete inference
  schema when adopting this path operationally. The bounded scratch artifacts now
  include abstracts/abstract presence, optional author email/block/source ids,
  paper language/reliability, SPECTER, name counts, and seed tables when seed
  semantics are requested. Do not bundle name-pair aliases per dataset; use the
  default packaged filtered alias file unless a non-default experiment explicitly
  overrides it.
- Keep the tracked complete-Arrow fixture parity harness as a stable manual/CI
  gate. Current scratch coverage is exact on 1000-signature a_silva and h_wang
  blocks, with SPECTER, without SPECTER, and with cluster seeds; the tracked
  50-signature embedded-count gate is also exact across features, constraints,
  distances, and clusters.
- Further optimization should target raw Arrow read/summary construction and
  reusable component summaries. Pairwise/model scoring and the former Python
  row-signal bridge are no longer material for the single-query raw wrapper.
- Keep `ANDData` as the oracle for semantics until each field group has parity
  evidence.

When to consider it:
- Before adding more Rust API surface. Without this boundary, the raw path can
  accidentally grow into a second implementation of `ANDData`.

Verification bar:
- Tiny fixture round-trips first.
- Exact candidate ids, pair ids, feature matrices, probabilities, and
  normalized final decisions for bounded h_wang and non-h_wang checks.
- Stage telemetry that keeps `ANDData` construction, `FeatureBlock`
  construction, Rust feature prep, retrieval, constraints, and scoring separate.

### Artifact format unification (Ask-first)

Status:
- Partially implemented. The target is no longer "serialize existing Python
  dict artifacts faster." The target is typed inference inputs consumed
  directly by Rust, with compatibility adapters for legacy artifacts.

What remains:
- `name_counts.arrow` is now a Rust-native Arrow lookup input for the direct
  Arrow path, with long-form columns `kind`, `name`, and `count`. A sorted
  exact-verified `name_counts_index/` sidecar is now the request-time fallback
  when per-signature counts are not embedded. The full `name_counts_rust.json`
  path remains a compatibility input. Runtime bundles should prefer embedded
  per-signature count columns, use the index when embedding is unavailable, and
  keep the global Arrow table for artifact generation or parity; cold-reading
  the 1.4GB global lookup in a request is too slow.
- Name aliases should remain a single shared runtime default, not a per-dataset
  Arrow artifact. The direct Arrow path still accepts `name_pairs` / `name_tuples`
  path keys for experiments, but production mini/full Arrow bundles should rely
  on the packaged filtered `s2and_name_tuples_filtered.txt` default. A
  `zbmath` build microbenchmark measured the filtered text fallback at ~22ms
  over no aliases and faster than a `name_pairs.arrow` override, so this does
  not need a name-count-style mmap/index sidecar.
- SPECTER is now first-class in the Arrow path and tuple-form SPECTER payloads
  are handled by the incumbent Rust `ANDData` path. A future artifact choice
  can still compare Arrow fixed-size-list against Safetensors.
- The `FeatureBlock` paper schema now preserves `predicted_language` and
  `is_reliable`, because those are feature-bearing fields for exact scoring.

Latest verification:
- Tracked 50-signature a_silva embedded-count full-predict parity:
  `scratch/baseline/a_silva_single_query/tracked_gate_subset50_embedded_counts_20260520/compare_full_predict_arrow_parity_subset50_embedded_counts_20260520.json`
  reports exact feature matrix, exact constraints, `max_absdiff=0.0` distances,
  and exact clusters.
- Tracked 50-signature a_silva index-count full-predict parity:
  `scratch/baseline/a_silva_single_query/tracked_gate_subset50_index_counts_20260520/compare_full_predict_arrow_parity_subset50_index_counts_20260520.json`
  explicitly exercises `name_counts_index/` and reports exact feature matrix,
  exact constraints, `max_absdiff=0.0` distances, and exact clusters.
- 50-signature a_silva complete-Arrow full-predict parity:
  `scratch/baseline/a_silva_single_query/complete_arrow_subset50_arrow_name_artifacts/compare_full_predict_complete_arrow_subset50_20260520.json`
  reports `max_absdiff=0.0`, `nonzero_absdiff_count=0`, and exact clusters.
- 1000-signature a_silva complete-Arrow full-predict parity:
  `scratch/baseline/a_silva_single_query/complete_arrow_subset1000_arrow_name_artifacts/compare_full_predict_complete_arrow_subset1000_featurecheck_20260520.json`
  reports exact 39-feature matrix parity, exact constraint parity,
  `max_absdiff=0.0` distance parity, and exact clusters over 499,500 pairs.
- Seeded 1000-signature a_silva:
  `scratch/baseline/a_silva_single_query/complete_arrow_subset1000_seeded_arrow_name_artifacts/compare_full_predict_complete_arrow_subset1000_seeded_featurecheck_20260520.json`
  reports 997 required seed assignments, exact constraints, exact feature
  matrix, exact distances, and exact clusters.
- No-SPECTER 1000-signature a_silva:
  `scratch/baseline/a_silva_single_query/complete_arrow_subset1000_no_specter_arrow_name_artifacts/compare_full_predict_complete_arrow_subset1000_no_specter_featurecheck_20260520.json`
  reports exact constraints, exact feature matrix including NaN placement, exact
  distances, and exact clusters.
- 1000-signature h_wang complete-Arrow full-predict parity:
  `scratch/baseline/h_wang_single_query/complete_arrow_subset1000_arrow_name_artifacts/compare_full_predict_complete_arrow_subset1000_featurecheck_20260520.json`
  reports exact 39-feature matrix parity, exact constraint parity, exact
  distance parity, and exact clusters over 499,500 pairs.
- Seeded and no-SPECTER h_wang checks:
  `scratch/baseline/h_wang_single_query/complete_arrow_subset1000_seeded_arrow_name_artifacts/compare_full_predict_complete_arrow_subset1000_seeded_featurecheck_20260520.json`
  and
  `scratch/baseline/h_wang_single_query/complete_arrow_subset1000_no_specter_arrow_name_artifacts/compare_full_predict_complete_arrow_subset1000_no_specter_featurecheck_20260520.json`
  are both exact across feature matrices, constraints, distances, and clusters.

When to consider it:
- Alongside `FeatureBlock` definition, before large artifact regeneration.

Verification bar:
- Dual-read loaders first.
- Tiny fixture round-trips before any large artifact regeneration.
- Gate with existing Rust baseline commands and raw candidate/scoring parity.

### Reference-features deprecation

Status:
- Effectively soft-deprecated in production, but not removed from the codebase.

Current state:
- The 39-dim feature contract still reserves reference-feature slots at indices `16..21`.
- Current production paths do not rely on reference features.
- Legacy training, reproducibility paths, and tests still support them.
- `Clusterer.predict_from_arrow_paths(...)` now fails fast when a model requests
  `reference_features`, because the Arrow direct path intentionally does not yet
  carry the citation-derived reference-feature artifacts.

Open decision:
- Decide whether to hard-deprecate `reference_features` everywhere, or keep the
  legacy `ANDData` path as the only supported route for models that still
  request them.

Guardrail:
- Preserve the feature index contract unless there is an explicit migration plan.

### Configuration surface cleanup

Status:
- Partially improved, not finished.

What remains:
- Env var parsing and validation are still spread across multiple modules and scripts.
- Some scripts still set runtime knobs through ambient env vars instead of explicit parameters.

Why it matters:
- Reproducibility is better when run-critical settings live in CLI or typed API surfaces instead of implicit process state.

## Backlog

### Rust frontier ideas (Ask-first)

1. **Fused constraint and featurize pipeline in Rust**
   - Replace the remaining Python-side per-pair orchestration with one Rust batch call that applies constraints internally and returns features or distances.
   - Full `predict` profiling now shows Python chunk orchestration, constraint
     array conversion, and LightGBM call overhead are material at larger pair
     counts. Revisit after the `FeatureBlock`/raw scoring wrapper removes
     full-block setup costs, not before.

2. **Further Vec-backed internal storage refactors**
   - Some hot-path Rust structures already moved in this direction; do more only if profiling shows the remaining hash-map overhead is still material.

### Small refactor candidates

- `s2and/model.py`: further separate clusterer, incremental assignment, constraints, and pairwise orchestration responsibilities.
- `s2and/featurizer.py`: keep `many_pairs_featurize` as the public orchestrator but continue extracting cache lifecycle and worklist construction pieces.
- `s2and/data.py`: instrument and simplify `ANDData.__init__`, but avoid
  turning Rust work into a full `ANDData` port. Treat `ANDData` as the
  reference/compatibility layer and carve out `FeatureBlock` explicitly.
- `s2and/feature_port.py`: further separate runtime gating, artifact selection, cache IO, constraints, and batch bridge logic.
- `s2and_rust/src/lib.rs`: reduce duplication between `from_dataset` and `from_json_paths` when there is clear shared stage logic.

### Separate blocked track

- Normalization migration remains blocked: `docs/normalization_migration_blocked.md`
