# Work Plan

Status date: 2026-05-27

This is the active Rust/Arrow platform backlog. Stable architecture and artifact
contracts live in:

- [rust/inference_architecture.md](rust/inference_architecture.md)
- [rust/public_surface_inventory.md](rust/public_surface_inventory.md)
- [rust/artifact_formats.md](rust/artifact_formats.md)
- [rust/arrow_dataset_spec.md](rust/arrow_dataset_spec.md)
- [rust/runtime.md](rust/runtime.md)
- [rust/baselines.md](rust/baselines.md)

## Current Decisions

| Topic | Decision |
|---|---|
| `ANDData` | Keep as Python reference, training/eval, parity, fixture, and compatibility surface. Do not port all of `ANDData` to Rust. |
| Production inference | Production Rust inference should enter through raw Arrow IPC artifacts. JSON, Python objects, and `RustFeaturizer.from_dataset(...)` are compatibility surfaces. |
| Arrow preprocessing | Production Arrow rows are runtime inputs, not preprocessed `ANDData` caches. Rust owns local normalization, ngram construction, unidecode, name handling, and language detection from raw Arrow inputs. |
| Name counts | Use manifest-backed `name_counts_index/` for hot-path lookups. Do not satisfy strict production bundles from ambient package/global fallbacks. |
| Batch indexes | Filtered production Arrow reads require raw-planner batch lookup indexes. Full scans are explicit test/compatibility opt-ins only. |
| SPECTER | Missing embedding rows are valid. Present rows are real vectors, including all-zero rows. Select `specter` or `specter2` through the manifest/path mapping. |
| Seeds | Incremental production requires a seed source, but not necessarily a physical `cluster_seeds.arrow`; request/dataset seed mappings may be materialized into request-local Arrow. |
| Optional sidecars | Missing `cluster_seed_disallows` means no seed-disallow constraints. Missing `altered_cluster_signatures` means no altered claimed profiles. If a sidecar key is declared, its file must exist and validate. |

## Canonical Arrow Input Surface

`s2and.arrow_inputs` is the strict production validation authority. Call sites
may resolve manifests or request-local overlays, but they should not reimplement
required-artifact, path-kind, missing-file, or batch-index policy.

The canonical surface owns:

- Path normalization and structured `MissingArrowArtifactError` diagnostics.
- Required and optional artifact policy for prediction, subblocking,
  incremental prediction, feature generation, script profiling, and eval.
- Runtime schema validation policy for string/int/bool/list fields, null
  handling, duplicates, and id semantics. Today the checks still live in the
  table readers, subblocking, and Rust implementation; centralize only when it
  removes duplicated policy.
- Batch lookup index requirements and explicit full-scan opt-ins.
- Signature subset/filtering semantics and request-local seed overlays.
- SPECTER path selection, dimensions, all-zero vectors, and missing-vector
  semantics.
- Manifest-backed `name_counts_index/`, name tuple policy, and alias policy.
- Text normalization/unidecode, local language detection from raw titles, name
  splitting, paper-author ordering, null position, and duplicate-position
  semantics.
- Seed sidecars and request-local seed materialization.
- Subblocking strictness, telemetry keys, and producer hints.

## Active Sequence

1. Keep the local canonical Arrow replay/profiling source at
   `s2and/data/s2and_and_big_blocks_linker_dataset_20260525`.
2. Treat release artifact validation for the current Arrow S3 prefix as
   complete; keep future publication smokes explicit and network opt-in.
3. Convert production-facing scripts to Arrow routes where that is a direct
   replacement; relabel raw/reference parity scripts instead of expanding them.
4. Preserve strict production routing through `s2and.arrow_inputs`; do not add
   another strict/compatibility discovery layer.
5. Delete or isolate remaining non-Arrow Rust loaders after their scripts/tests
   are clearly labeled compatibility-only.
6. Split oversized Rust and Arrow incremental runtime modules only after the
   production boundary is locked and tested.

## Open Work

### 1. Production Artifact Generation

Current state:

- The current S3-synced public Arrow release under `s2and/data` has the
  manifest-backed `name_counts_index/generations/<generation-id>/` layout.
- The canonical local replay/profiling bundle is
  `s2and/data/s2and_and_big_blocks_linker_dataset_20260525`.

Remaining:

- For the next full regeneration, rebuild durable Arrow bundles from the full
  schema in [rust/arrow_dataset_spec.md](rust/arrow_dataset_spec.md) and run
  `refresh-root-manifest` before publication.
- Keep production-scale `name_counts_index/` in S3, not Git/LFS.
- Keep `name_counts.arrow` for generation, inspection, and parity debugging,
  not as the default request-time read.

Done when:

- A reproducible `uv run ...` conversion builds a local Arrow release root.
- Every dataset manifest, required Arrow IPC file, batch lookup index, and
  `name_counts_index/manifest.json` validates for the benchmark datasets and
  canonical linker replay bundle.

### 2. Arrow S3 Release

Target prefix:
`s3://ai2-s2-research-public/s2and-release-arrow`.

Current state:

- The prefix was published and no-auth spot-checked on 2026-05-25.
- The refreshed root manifest and root helper files were uploaded on
  2026-05-26 after default and size-only dry runs showed no changed large
  Arrow objects. A final dry run returned no pending uploads.
- A bounded public-release smoke on 2026-05-26 copied the root, `qian`, and
  `name_counts_index` manifests with `--no-sign-request`, then ran
  `scripts/eval_prod_models.py --dataset full --use-arrow --datasets qian
  --specter-suffixes _specter2.pkl --seed 42 --n_jobs 2` against the
  S3-converged local release root. The direct Arrow
  `predict_from_arrow_paths(...)` route completed with qian B3
  `(0.978, 0.964, 0.971)`.
- Keep the canonical replay subbundle name
  `s2and_and_big_blocks_linker_dataset_20260525/` for the current public Arrow
  release.
- The release is Arrow-native. Do not mirror `.feature_cache/`, raw JSON,
  pickle embeddings, or precomputed `features_corrected/`.
- Local release-manifest generation now records root-manifest checksums,
  per-dataset audit counts, generator commit metadata, validation requirements,
  and the exact validation commands.
- `refresh-root-manifest` refreshes those checksums/audits/validation commands
  in a local S3-synced checkout while preserving the logical S3 `output_root`
  and nested replay-bundle metadata.
- `scripts/verification/validate_local_arrow_release.py` provides the
  non-network local release-root smoke: root helper files, root-manifest
  checksum fields, dataset manifest references, required Arrow artifact paths,
  raw-planner batch-index sidecars, replay-bundle manifests, and
  `name_counts_index/manifest.json` targets.
- The local tiny-fixture release-layout regression covers root manifest
  checksums, dataset manifest references, readable Arrow IPC files, a
  batch-index sidecar, production model manifest, and
  `name_counts_index/manifest.json`.

Remaining:

- Keep S3/network validation as an explicit release smoke command, not default
  pytest, until CI has a dedicated network-enabled release job.
- For future publications, rerun the bounded public-release smoke after upload.

Done when:

- The public prefix has a root manifest with checksums.
- Every dataset manifest references existing files.
- A bounded smoke command can read the published files and run
  `predict_from_arrow_paths(...)`.

### 3. Strict Production Routing

Already landed:

- `Clusterer.predict_from_arrow_paths(...)` validates complete Arrow artifacts
  and raises structured missing-artifact errors.
- Rust `Clusterer.predict(...)` fails closed in production Rust mode when Arrow
  artifacts are incomplete, including subblocked prediction.
- Rust `Clusterer.predict_incremental(...)` requires base Arrow artifacts and a
  seed source before entering promoted incremental scoring.
- Arrow graph subblocking requires batch indexes and no longer falls back after
  Arrow read/prepare/call failures on the Rust production path.
- Rust Arrow subblocking is native-graph only. The legacy
  `subblocking_fallback_mode` knob and the Rust `fallback_cluster_fn` callback
  path were removed, so Rust no longer calls Python during Arrow subblocking.
- Raw payload / Python `FeatureBlock` Rust scoring bridges were removed.
- Bounded script-level smoke tests cover
  `scripts/convert_to_arrow.py service-json`, the Arrow production tutorial,
  Arrow production eval, and the Arrow rust-suite production benchmark
  dispatch.
- A canonical large-block fixture proves Arrow graph subblocking, Arrow
  featurizer reuse, and promoted incremental completion run without `ANDData`
  fallback.
- `scripts/eval_prod_models.py --use-arrow` was run on the local mini Arrow
  bundle for `pubmed`, `qian`, and `zbmath` with no `ANDData` construction:
  B3 F1 was 0.943, 0.971, and 0.976 respectively.
- The same Arrow eval smoke now runs against the S3-synced public release root
  in `s2and/data`: `pubmed`, `qian`, and `zbmath` with SPECTER2 produced B3
  F1 0.943, 0.971, and 0.959 respectively.
- Existing-release feature parity was spot-checked on 2026-05-26 with
  `scripts/verification/compare_existing_arrow_anddata_feature_parity.py`
  against `s2and/data-backup` raw JSON/pickle and `s2and/data` Arrow bundles
  for `pubmed`, `qian`, and `zbmath`. The bounded SPECTER2 run used 64
  signatures and 128 sampled pairs per dataset, and all three had 39 feature
  columns, zero NaN mismatches, and max absolute drift `0.0` at tolerance
  `1e-5`.

Remaining:

- Keep future production call sites on `s2and.arrow_inputs`; do not duplicate
  strict validation in scripts or model helpers.

Done when:

- Arrow full prediction, subblocked prediction, and promoted incremental smoke
  tests run without `ANDData` fallback.
- Compatibility routes still preserve legacy behavior for training, fixtures,
  parity checks, and explicitly labeled legacy scripts.

### 4. Compatibility And Python-Heavy Paths

Remaining:

- Prefer `Clusterer.predict_from_arrow_paths(...)` or Arrow-routed
  `predict(...)` for production inference.
- Prefer `RawBlockQueryCandidatePlanner.plan(...)` plus typed Arrow request
  tables for single-query/seeded incremental requests.
- Use `scripts/verification/compare_graph_subblocking_arrow_quality.py` for
  Python graph vs Rust graph subblocking checks. It no longer compares against
  the old SPECTER fallback or any Rust-calls-Python callback path.
- Split the broad raw-Arrow incremental runtime entrypoint into narrow
  planner-owned execution and preplanned scoring surfaces. Use a typed
  `RawArrowPlanBundle` or equivalent only after typed Arrow request-table
  fixtures pin the boundary and prove the split does not revive raw payload /
  mini-`ANDData` bridges.
- Keep `feature_block_from_arrow_paths(...)` implementation-only for
  fixture/parity validation until typed Arrow request-table coverage makes it
  unnecessary. Do not re-export it as public production API.
- Keep `RustFeaturizer.from_dataset(...)` as the Python-reference,
  training/eval, parity, and fixture surface.
- Keep text `altered_cluster_signatures.txt` fallback only for training/legacy
  fixtures until those fixtures migrate to Arrow sidecars.

Done when:

- Production docs and scripts recommend only Arrow-backed Rust inference.
- Removed raw payload / `FeatureBlock` scoring bridges have no callers.
- Remaining Python-object helpers are explicitly fixture, conversion, parity, or
  training utilities.

### 5. Rust Surface Cleanup

Already landed:

- Removed unused or duplicate public/debug APIs including
  `RustHybridCentroidRetriever.summary_count(...)`,
  `raw_block_query_candidate_plan_arrow(...)`,
  the legacy list-of-tuples linker aggregate API,
  the aggregate-only pair stats PyO3 method,
  the string-pair constraint API, and direct retriever debug APIs.
- Removed `RustFeaturizer.from_json_paths(...)` and the Python JSON-ingest
  lifecycle from active constructors.
- Removed direct Rust handling of tuple-shaped SPECTER pickle payloads from
  `RustFeaturizer.from_dataset(...)`; Python `ANDData` owns pickle loading and
  normalization before Rust feature generation.
- Production pairwise training now defaults `S2AND_BACKEND=rust` before
  importing `s2and`, so `ANDData` loading/preprocessing can delegate feature
  generation to `RustFeaturizer.from_dataset(...)`.
- Removed `RustFeaturizer.save(...)` and `RustFeaturizer.load(...)`; the
  counter-data measurement helper now reports build-time RSS deltas instead of
  depending on Rust featurizer serialization.
- Removed the legacy `make_subblocks_with_telemetry_arrow(...)` PyO3 export and
  its Python fallback callback. The Python wrapper now probes only
  `make_subblocks_with_telemetry_arrow_native_graph(...)`.
- Removed the `s2and.rust_capabilities` compatibility shim.
- Extracted the promoted non-pairwise linker feature kernel into
  `s2and_rust/src/promoted_linker.rs` with its focused Rust unit test and PyO3
  registration helper.
- Extracted manifest-backed name-count index loading and lookup into
  `s2and_rust/src/name_counts.rs`, keeping `lib.rs` on the existing
  `RawNameCountMaps` / `NameCountsData` contract.
- Extracted Python-compatible text/name normalization helpers into
  `s2and_rust/src/text_compat.rs`.
- Current inventory:
  [rust/public_surface_inventory.md](rust/public_surface_inventory.md).

Remaining:

- Remove or isolate JSON ingest helpers once their remaining parity and
  benchmark uses are explicitly labeled compatibility-only.
- Keep the indexed pairwise feature API as the Python Rust batching boundary.
  The Python `featurize_pair_rust(...)` and string-pair matrix wrappers have
  been removed; direct Rust `featurize_pair(...)`, `featurize_pairs(...)`, and
  `featurize_pairs_matrix(...)` PyO3 debug methods have also been removed.
- Split `s2and_rust/src/lib.rs` mechanically, one low-coupling module at a
  time. Completed extractions: `promoted_linker`, `name_counts`,
  `text_compat`. Candidate next modules: `arrow_io`, `ingest_arrow`,
  `ingest_json`, `ingest_dataset`, `features`, `constraints`, `retrieval`,
  `linker`, and `subblocking`.
- Move focused tests with extracted modules where that preserves private
  visibility without unnecessary `pub(crate)` churn.

Done when:

- `lib.rs` is mostly PyO3 export wiring.
- Extracted modules own focused tests.
- Rust unit tests and focused Python/Rust integration tests pass after each
  extraction.

### 6. Rust Ingest Deduplication

Current inventory:
[rust/ingest_source_policy_inventory.md](rust/ingest_source_policy_inventory.md).

Remaining:

- Deduplicate Arrow/JSON/`ANDData` staging records only where source semantics
  match: unidecode, language handling, name splitting, name-count telemetry,
  paper-author handling, ORCID/source-id derivation, malformed positions,
  reference features, `preprocess=false`, and filtering order.
- Keep source-specific differences documented instead of hiding them behind a
  broad shared helper.

Done when:

- Equivalent ingest semantics share staging helpers.
- Source-specific behavior remains explicit.
- Rust library tests and focused Python/Rust integration tests pass.
- Bounded real-dataset Arrow-vs-`ANDData` feature parity is recorded for
  datasets that have both original JSON inputs and generated Arrow bundles,
  with feature matrix drift no larger than `1e-5` except for documented,
  intentional source-policy differences.

### 7. Incremental Helper Shim

Current decision:

- Treat `Clusterer._predict_incremental_helper(...)` as internal-only test
  plumbing for now. It is not an external compatibility API.

Remaining:

- Before any rename/removal, migrate tests that monkeypatch the helper to a
  public routing surface, dependency parameter, or explicit test hook.

Done when:

- Monkeypatched tests use the chosen call surface and focused incremental tests
  pass.

### 8. Performance Targets

Current evidence:

- `scripts/rust_suite.py promoted-incremental-arrow-profile` ran 5 isolated
  runs on the canonical local `pubmed` `r agarwal` block with 25 synthetic seed
  clusters and 25 query signatures because the canonical replay bundle has no
  `clusters` artifact.
- Result summary:
  p50 predict wall time 11.15s, min 11.12s, max 11.63s, max RSS 3.72 GB,
  625 candidate rows, 1 query batch, p50 final predicted peak delta
  18,712,216 bytes.
- The largest measured contributors in the Arrow telemetry were reusable
  window featurizer work and manifest-backed name-count reads during window
  planning. Treat these as the next optimization candidates only if a focused
  before/after profile shows at least a 10% wall-time or allocation impact.
- Evidence: [rust/profiling/2026-05-27-promoted-incremental-arrow.md](rust/profiling/2026-05-27-promoted-incremental-arrow.md).
  This was a dirty-worktree/debug-assertions run, so it is operational
  prioritization evidence rather than a release-grade performance claim.

Next profiling target:

- Arrow read/summary construction and reusable component summaries on the
  canonical local promoted-incremental workload:
  `s2and/data/s2and_and_big_blocks_linker_dataset_20260525`.
- Use `scripts/rust_suite.py promoted-incremental-arrow-profile`, not the
  deleted JSON/`ANDData` big-block command.

Required metrics:

- p50 wall time over at least five isolated runs.
- Peak RSS.
- Summary-construction allocation volume from a stack-level allocation profiler
  where available.

Act only when:

- Arrow read or summary construction is at least a 10% contributor to p50 wall
  time or allocation volume, or the change removes a real `ANDData` dependency.
- Stop optimizing once measured improvement falls below 10% for the selected
  workload.
