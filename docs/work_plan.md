# Work Plan

Status date: 2026-05-25

This file is the active Rust/platform backlog. Stable architecture and artifact
contracts live in:

- [rust/inference_architecture.md](rust/inference_architecture.md)
- [rust/public_surface_inventory.md](rust/public_surface_inventory.md)
- [rust/artifact_formats.md](rust/artifact_formats.md)
- [rust/arrow_dataset_spec.md](rust/arrow_dataset_spec.md)
- [rust/runtime.md](rust/runtime.md)
- [rust/baselines.md](rust/baselines.md)

## Active Decisions

| Topic | Current decision |
|---|---|
| `ANDData` role | Keep as Python reference, training/eval, compatibility, and fallback object. Do not port all of `ANDData` to Rust. |
| Fast inference boundary | Prefer direct Arrow IPC inputs consumed by Rust. Avoid Arrow-to-Python-object materialization on the hot path. |
| Name counts | Use `s2and/data/name_counts_index/` for Rust hot-path lookups. Do not embed per-signature name-count columns in `signatures.arrow`. |
| Name aliases | Use the packaged filtered alias text by default. Keep per-dataset alias artifacts experimental only. |
| Reference features | Direct Arrow prediction fails fast when a model requires reference features. Current production models do not use them. |
| Optional seed sidecars | Missing `cluster_seed_disallows` means no seed-disallow constraints, and missing `altered_cluster_signatures` means no altered claimed profiles. Strict mode requires these sidecars only when the caller or manifest declares the condition; an explicit mapping key must point to an existing file. Converters may emit canonical empty tables, but strict routing does not require empty tables for empty sets. |

## Verification Invariants

- Keep tiny Arrow fixture tests for schema, row signals, and name-count index
  behavior.
- Keep bounded full-predict parity gates that compare features, constraints,
  distances, and clusters against the `ANDData` oracle.
- Keep raw Arrow incremental gates that compare candidate ids, pair ids, row
  signals, probabilities, and final decisions on bounded fixtures.
- Do not add complexity for less than a 10% improvement unless it removes a
  real bottleneck or an `ANDData` dependency on a hot path.

## Execution Sequence

The current Rust/platform work should move in this order:

1. Generate local Arrow-native runtime artifacts and regenerated
   `name_counts_index/` sidecars.
2. Publish and validate the Arrow S3 release layout.
3. Convert production-facing script entrypoints to Arrow routes, or relabel
   them as compatibility/parity tools with smoke tests.
4. Enforce strict Arrow production routing through one approved strictness
   authority; keep public compatibility fallback unchanged until an API change
   is approved.
5. Demote non-Arrow Rust entrypoints to compatibility/training/test surfaces.
6. Split oversized Rust and Arrow incremental runtime modules after the
   production boundary is locked.

Do not make strict Arrow routing the default before shippable Arrow bundles and
their validation checks exist. Do not enable strict routing for generic
`predict(...)` callers until the production scripts that exercise it have been
converted or explicitly gated behind a compatibility flag.

## Open Work

### Production Artifact Generation

- Regenerate durable Arrow bundles from the complete schema in
  [rust/arrow_dataset_spec.md](rust/arrow_dataset_spec.md).
- Regenerate `s2and/data/name_counts_index/` into the manifest-backed
  `generations/<generation-id>/` layout before publishing runtime bundles; the
  current direct-file layout is a legacy source artifact.
- Include the regenerated `name_counts_index/` as a required shared runtime
  artifact in the S3 release for bundles that use name-count features.
- Do not add the production `name_counts_index/` to Git LFS. It is release
  artifact scale, and S3 is the canonical distribution channel. Keep only tiny
  bounded fixtures in Git/LFS when tests need them.
- Keep `name_counts.arrow` available for generation, inspection, and parity
  debugging, not as the default request-time read.
- Done when: a reproducible conversion command builds a local Arrow release
  root whose manifests, required table files, batch lookup indexes, and
  `name_counts_index/` generation validate for every benchmark dataset and
  linker replay dataset in the release plan.

### Arrow S3 Release

Target a new public release prefix:
`s3://ai2-s2-research-public/s2and-release-arrow`.

This prefix was published and no-auth spot-checked on 2026-05-25. The current
linker replay subbundle is
`s2and_and_big_blocks_linker_dataset_20260525/`.

This release should be an Arrow-native data/runtime artifact release, not a
mirror of the legacy JSON/pickle bucket. It should omit `.feature_cache/`.

The native production bundle directory is currently named
`production_model_v1.21/` in `s2and/data`; keep that exact directory name
unless the release versioning scheme is intentionally changed. Embedding file
names are selected per dataset by the release manifest and should match the
source embedding generation, such as `specter.arrow` for legacy SPECTER or
`specter2.arrow` for SPECTER2. Do not infer the filename from whether the
dataset is a benchmark or replay dataset.

The source of truth for release layout and manifest shape is
[rust/arrow_dataset_spec.md](rust/arrow_dataset_spec.md) and
[rust/artifact_formats.md](rust/artifact_formats.md). Keep this section as the
active publication checklist, not as a second artifact spec.

Release contents:

- Root `manifest.json`, `LICENSE.txt`, and `lid.176.bin`.
- Current native production model bundle, `production_model_v1.21/`.
- Legacy compatibility model pickles:
  `production_model_v1.0.pickle`, `production_model_v1.1.pickle`, and
  `production_model_v1.2.pickle`.
- Shared manifest-backed `name_counts_index/`.
- Benchmark dataset directories for `aminer`, `arnetminer`, `inspire`, `kisti`,
  `medline`, `pubmed`, `qian`, and `zbmath`, each following the Arrow dataset
  manifest contract.
- `s2and_and_big_blocks_linker_dataset_20260525/` with Arrow runtime tables under
  `datasets/<dataset>/` and existing typed offline `components/`, `labels/`,
  and `splits/`.

- Convert table-shaped runtime inputs.
  - Convert each benchmark `<dataset>_signatures.json` to
    `signatures.arrow`.
  - Convert each benchmark `<dataset>_papers.json` to `papers.arrow` and
    `paper_authors.arrow`.
  - Convert each benchmark embedding payload to the manifest-declared embedding
    filename, for example `specter.arrow` or `specter2.arrow`.
  - Convert linker replay `raw/<dataset>/signatures.json` to
    `s2and_and_big_blocks_linker_dataset_20260525/datasets/<dataset>/signatures.arrow`.
  - Convert linker replay `raw/<dataset>/papers.json` to `papers.arrow` and
    `paper_authors.arrow`.
  - Convert linker replay embeddings to the manifest-declared embedding
    filename, preserving `specter2.arrow` when the source is SPECTER2.
  - Convert `name_counts.pickle` to a manifest-backed generation under
    `name_counts_index/`; do not publish `name_counts.arrow` in this release.
    The checked-in direct-file `s2and/data/name_counts_index/` layout is a
    legacy source artifact until regenerated.
  - Generate and ship Arrow batch lookup indexes beside the Arrow files using
    `<table-stem>.<path-key>.bin` names, for example
    `signatures.signatures_batch_index.bin` and
    `<specter-stem>.specter_batch_index.bin`.
- Keep small metadata and offline evaluation artifacts in their existing
  formats.
  - Keep `LICENSE.txt` and `lid.176.bin` unchanged.
  - Copy `production_model_v1.21/` exactly as the current native production
    model bundle.
  - Copy `production_model_v1.0.pickle`, `production_model_v1.1.pickle`, and
    `production_model_v1.2.pickle` exactly as legacy compatibility artifacts.
  - Keep `<dataset>_clusters.json` as eval-only truth.
  - Keep train/test split keys, pair CSVs, replay split CSVs, replay
    `summary.json`, Arrow-only `bundle.json`, and manifests, with paths and
    checksums updated where they refer to converted artifacts.
  - Keep linker replay `components/*.parquet` and `labels/*.parquet` as-is
    because they are already typed columnar offline artifacts.
- Omit or quarantine artifacts that are not part of the Arrow-native release.
  - Omit `.feature_cache/`; it is a historical precomputed feature cache and
    should not be copied into the Arrow release.
  - Omit `full_union_seed_*.pickle` from the Arrow-native release, or place it
    under an explicit `legacy/` prefix if paper-era compatibility requires it.
  - Do not duplicate raw JSON files after conversion. The existing
    `s2and-release` bucket remains the legacy JSON source.
  - Do not ship replay `raw/`, pickle `embeddings/`, or precomputed
    `features_corrected/` directories in the Arrow replay subbundle. Promoted
    feature rows are regenerated during replay for the selected pairwise model.
- Verification and publication runbook.
  - Before any full conversion or upload, run a tiny fixture/sample conversion
    and validation. Current expected command shape:

    ```powershell
    uv run python scripts/convert_to_arrow.py benchmark `
      --source-root s2and/data/s2and_mini `
      --output-root scratch/s2and_release_arrow_sample `
      --datasets pubmed `
      --overwrite `
      --overwrite-name-counts-index

    uv run python scripts/convert_to_arrow.py validate `
      --dataset-dir scratch/s2and_release_arrow_sample/pubmed `
      --require-embeddings `
      --require-name-counts-index
    ```

    `--overwrite-name-counts-index` regenerates the shared index once for this
    bounded sample. For a staged release with a prebuilt shared index, pass
    `--name-counts-index-root <root>` instead, or run the name-counts-index
    subcommand before converting datasets:
    `uv run python scripts/convert_to_arrow.py name-counts-index --output-root <root> --overwrite`.

  - For the full release, record the resolved source roots, output staging root,
    exact `uv run ...` commands, expected runtime, log path, and publish target
    before starting. If a conversion or upload is expected to exceed ten
    minutes, run it detached with stdout/stderr captured to a log and record the
    PID/job id.
  - Validate every generated dataset manifest with
    `scripts/convert_to_arrow.py validate`, requiring embeddings and
    `name_counts_index` when the selected production model uses those features.
    `--require-embeddings` requires the embedding table to exist and validate
    structurally; it does not require every referenced paper to have an
    embedding. Missing per-paper embeddings are valid for some sources and must
    be captured in the validation metrics as `missing_specter_paper_count` and
    reviewed in the release audit. Use `--require-complete-embeddings` only for
    datasets whose source contract guarantees full embedding coverage.
  - Produce a compact release audit file with per-dataset signature, paper,
    paper-author, embedding, cluster-seed, disallow, altered-signature, missing
    embedding, and batch-index counts; include root manifest path, generator git
    commit, source snapshot ids, and validation command lines.
  - Before making the S3 prefix public, do a dry-run or staging upload, then
    spot-check that `manifest.json`, one benchmark dataset manifest, one
    manifest-declared embedding Arrow file, one batch-index sidecar, the
    production model manifest, and `name_counts_index/manifest.json` can be
    read from the published prefix. The 2026-05-25 publication passed these
    checks.
  - Add a checked-in release-layout regression test, tentatively
    `tests/test_arrow_release_layout.py`. By default it should validate a tiny
    local release fixture: root manifest, one dataset manifest, required Arrow
    IPC files, one batch-index sidecar, production model manifest, and
    `name_counts_index/manifest.json`.
  - Keep S3/network validation as an explicit release smoke command, not a
    default pytest path: open the staged public prefix, verify the same layout
    targets, then run a bounded `predict_from_arrow_paths(...)` smoke test.
    Add it to CI only when CI has stable network access and an explicit
    release-validation job owns the required environment.
- Done when: the public prefix contains a root manifest with checksums, every
  dataset manifest references existing files, a validation command can open the
  required Arrow IPC files and sidecars from the published prefix, and docs point
  production users at this prefix instead of the legacy JSON bucket.

### Retire Non-Arrow Production Inference Path

Goal: production inference must enter Rust through Arrow artifacts only. JSON,
`ANDData`, `FeatureBlock`, and `RustFeaturizer.from_dataset` remain allowed for
training, fixtures, parity tests, and compatibility scripts, but not for
production inference.

- Define the production boundary.
  - Production full-block prediction requires complete Arrow paths:
    `signatures`, `papers`, `paper_authors`, required embedding table, and
    `name_counts_index` when the model uses name-count features.
  - Production seeded/incremental prediction additionally requires a seed source:
    either a valid `cluster_seeds.arrow` sidecar or a normalized request/dataset
    seed mapping that production can materialize into request-local Arrow.
  - `cluster_seed_disallows` is required only when pairwise seed-disallow
    constraints are present or the manifest declares the path. Missing
    `cluster_seed_disallows` means no disallow constraints.
  - `altered_cluster_signatures` is required only when altered claimed profiles
    are present or the manifest declares the path. Missing
    `altered_cluster_signatures` means no altered claimed profiles.
  - If either optional sidecar key is present in a manifest or explicit path
    mapping, the referenced file must exist and validate.
  - Missing Arrow artifacts should fail with an explicit error in production
    mode rather than silently falling back to `ANDData`.
- Update script entrypoints.
  - Status 2026-05-25: `scripts/tutorial_for_predicting_with_the_prod_model.py`
    accepts an Arrow bundle root and routes Arrow input through
    `Clusterer.predict_from_arrow_paths(...)`; JSON/`ANDData` remains an
    explicit compatibility route.
  - Status 2026-05-25: `scripts/_rust_suite/prod_inference_cmd.py` benchmarks
    Arrow `predict_from_arrow_paths(...)` by default; Python and
    `from_dataset` baselines are opt-in legacy comparisons.
  - Status 2026-05-25: `scripts/eval_prod_models.py` uses Arrow automatically
    for supported non-training evals when complete Arrow artifacts exist,
    including released full benchmark bundles. `--no-arrow` and `--train`
    preserve the raw/reference routes.
  - Status 2026-05-25: `scripts/_rust_suite/featurizer_reuse_cmd.py` runs
    repeated Arrow production-model evaluation by default; JSON/`ANDData`
    reuse checks remain under `--input-format json`.
  - Status 2026-05-25: `scripts/_rust_suite/stress_rebuild_cmd.py` defaults
    to `RustFeaturizer.from_arrow_paths`; `from_json_paths` and
    `from_dataset` remain explicit legacy stress targets.
  - Status 2026-05-25: `scripts/_rust_suite/largest_block_cmd.py` has an
    explicit Rust/Arrow single-run route for `predict_from_arrow_paths(...)`.
    Compare mode and constraint parity remain JSON/`ANDData` reference
    workflows because they compare against the Python object path.
  - Status 2026-05-25: `scripts/_rust_suite/compare_cmd.py` remains
    Python-vs-Rust `many_pairs_featurize(...)` parity, and
    `scripts/_rust_suite/big_block_incremental_cmd.py` remains JSON/`ANDData`
    until its subset/truth-bundle contract is redesigned for Arrow artifacts.
  - Redesign follow-up: do not bolt `--input-format arrow` onto
    `scripts/_rust_suite/compare_cmd.py`. Its current contract is
    Python-vs-Rust parity through `many_pairs_featurize(...)`; Arrow parity
    should be a separate command or an extension of
    `scripts/verification/compare_full_predict_arrow_parity.py` that compares
    Arrow artifacts against the incumbent oracle without changing the legacy
    feature-parity gate.
  - Redesign follow-up: do not partially convert
    `scripts/_rust_suite/big_block_incremental_cmd.py` while it still selects
    JSON subsets, raw signatures/papers, synthetic seed maps, and parquet truth
    rows independently. Define an Arrow subset/truth-bundle contract first:
    how selected query/candidate signatures map to `datasets/<dataset>/`
    Arrow rows, where `cluster_seeds`, optional `cluster_seed_disallows`,
    optional `altered_cluster_signatures`, labels, splits, and component
    parquet live, and which command produces the bounded fixture. Convert the
    script only after that artifact contract exists.
  - Convert remaining production/profiling scripts where the Arrow route is a
    direct replacement, and relabel scripts whose purpose is raw/reference
    parity.
  - Retire or relabel non-Arrow production-inference claims in rust-suite docs.
- Make artifact conversion mandatory before production inference.
  - Keep `scripts/convert_to_arrow.py service-json` as the supported bridge
    from service-shaped JSON to Arrow, alongside the benchmark and
    linker-replay conversion subcommands used for release assembly.
  - Ensure the converter always emits the artifacts required by production
    models, including `name_counts_index`.
  - Add a small validation command or test that opens the produced Arrow bundle
    and checks the production-required keys.
- Tighten runtime routing.
  - Status 2026-05-25: `Clusterer.predict_from_arrow_paths(...)` is the first
    strict production authority. It validates required mapping keys and declared
    files before building the Rust featurizer, and raises structured
    `MissingArrowArtifactError` with `context`, `required_keys`,
    `missing_keys`, `missing_files`, and `producer_hint`.
  - Status 2026-05-25: Rust `Clusterer.predict(...)`, with or without
    subblocking, now raises `MissingArrowArtifactError` when required Arrow
    artifacts are incomplete, instead of falling back to `ANDData`.
    Python/compatibility callers should select the Python route explicitly.
  - Status 2026-05-25: Rust `Clusterer.predict_incremental(...)` now requires
    base Arrow paths plus a seed source before seed sync or helper fallback, and
    `_predict_incremental_promoted_linker(...)` only calls the Arrow promoted
    implementation. The non-Arrow promoted linker remains a compatibility unit
    tested directly, not a production route.
  - Status 2026-05-25: production Rust `Clusterer.predict(...)`, including
    subblocked large-block prediction, no longer silently falls back when Arrow
    artifacts are incomplete.
  - Subblocked strict-routing follow-up:
    - Keep graph/legacy fallbacks only behind explicit compatibility, parity,
      training, or test paths, not behind `backend="rust"` production
      inference.
    - Expand bounded large-block fixtures that verify Arrow graph subblocking,
      Arrow featurizer reuse, and promoted incremental completion without
      `ANDData` fallback.
  - Preserve fallback only through approved compatibility/test routes.
  - Ensure errors name the missing Arrow artifact keys and the caller/script
    that should generate them.
- Strict routing error contract.
  - Do not add a standalone strict helper or new strict exception until the
    first approved strict production call site is being wired. A tested private
    resolver with no production caller is scaffold, and it risks duplicating
    compatibility discovery policy before the strictness authority is settled.
  - When strict routing is approved, introduce the structured missing-artifact
    error in the same change as the call site that raises it. It should carry
    enough context for tests and callers: `context`, `required_keys`,
    `missing_keys`, `missing_files`, and `producer_hint`.
  - The message should name the prediction context, list missing mapping keys
    separately from missing files, and name the command or script expected to
    produce the artifacts.
- Implementation map for strict production routing.
  - Current `_resolve_dataset_arrow_paths(...)` already validates explicit
    mappings, auto-discovers sibling Arrow dataset directories, adds raw-planner
    batch-index paths, resolves `name_counts_index`, supports
    `require_name_counts_index`, supports `require_cluster_seeds`, and adds
    optional `cluster_seed_disallows` / `altered_cluster_signatures` sidecars
    when files exist.
  - The remaining strict-routing work is narrower: at the approved call site,
    distinguish absent mapping keys from mapping keys whose files are missing,
    raise the structured strict-routing error instead of returning `None`, and
    carry structured context for callers and tests.
  - Current fallback surfaces to tighten:
    - `_resolve_dataset_arrow_paths(...)` in `s2and/model.py` can return
      `None` for incomplete auto-discovered Arrow artifacts and for missing
      required artifacts such as `cluster_seeds` or `name_counts_index`.
    - `Clusterer.predict(...)` now fails closed in Rust mode when Arrow paths
      are unavailable, including the subblocked route.
    - `_predict_subblocked(...)` still accepts `arrow_paths=None` for explicit
      non-production compatibility callers and direct unit tests, but
      `backend="rust"` production no longer reaches that fallback when base
      Arrow artifacts are missing.
    - `_predict_incremental_promoted_linker(...)` no longer falls back to the
      non-Arrow promoted linker. That legacy function remains callable as a
      compatibility/test surface.
    - `Clusterer.predict_incremental(...)` now fails closed when Rust mode lacks
      base Arrow artifacts or any seed source. Remaining compatibility fallback
      questions are about generic full-block `predict(...)` and Python-only
      helper usage.
    - Incremental altered-profile pre-split and residual Phase B reclustering
      call `predict_from_arrow_paths(...)` when `arrow_paths` is present, but
      otherwise fall back through Python reclustering. Strict promoted
      incremental routing must require validated Arrow paths before entering
      those subpaths.
  - Avoid a parallel strict/compatibility Arrow discovery implementation. The
    first strict-routing change should either refactor the existing resolver so
    it can emit structured diagnostics while preserving compatibility behavior,
    or perform minimal call-site validation without reimplementing discovery.
  - In strict production mode, do not silently satisfy a missing bundle
    `name_counts_index` from the checked-in `s2and/data/name_counts_index/`
    fallback. That fallback is useful for development and compatibility, but it
    can hide broken release bundles. Allow local name-count overrides only when
    the caller passes an explicit path.
  - Define one strictness authority before adding public parameters. Status
    2026-05-25: direct Arrow prediction, Rust `predict(...)`, and Rust
    `predict_incremental(...)` use the structured strict error. Python backend
    selection remains the compatibility path for generic callers.
  - Do not add public strictness kwargs during the first rollout. If callers
    later need public strictness controls, handle that as a separate approved
    API migration after the strict Rust route is stable.
  - In Rust production mode with compatibility fallback disabled:
    - Decision 2026-05-25: a physical `cluster_seeds.arrow` is not mandatory,
      but a seed source is mandatory. Seed source formats can include a
      request/dataset seed mapping, and production code may convert that mapping
      into request-local Arrow for Rust.
    - `predict(...)` validates required Arrow artifacts before selecting the
      Arrow route, for both full-block and subblocked Rust prediction.
    - `predict_incremental(...)` requires base Arrow artifacts plus an
      incremental seed source before seed sync or helper fallback. Accepted seed
      sources are a valid `cluster_seeds.arrow` sidecar or a request/dataset seed
      mapping such as `dataset.cluster_seeds_require`; production may materialize
      request-local `cluster_seeds.arrow` from that mapping. It requires
      `cluster_seed_disallows` only when disallow constraints are present or
      when an explicit manifest/path key declares it.
    - `_predict_incremental_promoted_linker(...)` does not call the non-Arrow
      promoted linker.
    - Altered-profile pre-split and residual Phase B reclustering receive the
      same validated Arrow path payload; under strict authority, `arrow_paths=None`
      raises `MissingArrowArtifactError` instead of selecting Python
      reclustering.
    - `predict_from_arrow_paths(...)` remains strict and should not gain
      compatibility fallback.
  - Preserve current fallback behavior only through the approved compatibility
    path, for tests, parity checks, training fixtures, and legacy scripts.
- Keep verification gates focused.
  - Keep bounded Arrow full-predict parity tests against the `ANDData` oracle.
  - Keep raw Arrow incremental tests for candidate rows, pair rows, row
    signals, probabilities, and final decisions.
  - Add or adjust script-level smoke tests for
    `scripts/convert_to_arrow.py service-json`, the Arrow prod-model tutorial
    flow, the Arrow prod eval flow, and the Arrow rust-suite production
    benchmark flow.
  - Update existing fallback-expecting tests:
    - Status 2026-05-25:
      `tests/test_rust_distance_matrix_blockwise.py::test_predict_auto_requires_arrow_paths_with_name_counts_index`
      expects a missing `name_counts_index` error in full-block Rust production
      mode.
    - `tests/test_cluster_incremental.py` should expect missing seed-source
      errors in Rust production mode, plus compatibility-mode tests for
      `_predict_incremental_helper(...)`. Add separate tests proving absent
      `cluster_seed_disallows` means no disallows, while a missing declared
      disallow artifact fails explicitly.
    - Add strict-routing call-site tests for missing `signatures`, `papers`,
      `paper_authors`, `specter`, `name_counts_index`, `cluster_seeds`,
      conditionally `cluster_seed_disallows`, and conditionally
      `altered_cluster_signatures`, using the approved production call site or
      refactored resolver as the authority.
    - Status 2026-05-25:
      `tests/test_rust_distance_matrix_blockwise.py::test_predict_subblocked_rust_requires_arrow_paths`
      verifies missing base Arrow artifacts fail before graph/legacy fallback.
      Valid Arrow paths still exercise Arrow graph subblocking and Arrow
      featurizer construction; explicit Python/compatibility routes continue to
      cover state restoration and fallback invariants.
- Clean up after migration.
  - Status 2026-05-25: `docs/production_inference.md` states that production
    Rust inference requires Arrow artifacts, and dataset-based
    `warm_rust_featurizer(...)` is compatibility-only warmup.
  - Status 2026-05-25: `docs/rust/inference_architecture.md` describes
    non-Arrow Rust loaders as compatibility, training, parity, or test surfaces
    only.
  - Status 2026-05-25: compatibility Rust entrypoints remain callable for
    training, fixtures, parity, and legacy scripts, but production docs no
    longer present `RustFeaturizer.from_dataset(...)`,
    `RustFeaturizer.from_json_paths(...)`, or the removed
    `RustFeaturizer.from_feature_block(...)` as production inference APIs.
  - Status 2026-05-25: no `s2and/feature_port.py` code change is needed for
    dispatcher demotion. Strict production call sites use
    `build_rust_featurizer_from_arrow_paths(...)`; `build_rust_featurizer(...)`
    and `_resolve_requested_build_path` remain a compatibility/training
    dispatcher for `ANDData` routes. Keep `__all__` alphabetized and exported
    for compatibility; do not use export ordering as a production-routing
    signal.
  - Status 2026-05-25: `scripts/README.md` labels the remaining JSON/`ANDData`
    big-block incremental command as a legacy profiling fixture pending the
    Arrow subset/truth-bundle redesign.
  - Keep training/materialization scripts on `ANDData`; they are not part of
    the production inference removal.
- Done when: with `backend="rust"` or `S2AND_BACKEND=rust` and the approved
  strict production authority enabled, missing production artifacts raise
  `MissingArrowArtifactError`; the approved compatibility path preserves the old
  fallback behavior for legacy/parity callers; Arrow full-predict, subblocked
  predict, and promoted incremental smoke tests run without `ANDData` fallback;
  and
  `scripts/eval_prod_models.py --use-arrow` runs end-to-end on at least three
  benchmark datasets without `ANDData` fallback telemetry.

### Compatibility And Python-Heavy Paths

- Prefer `Clusterer.predict_from_arrow_paths(...)` or Arrow-routed
  `predict(...)` for production inference.
- Prefer `RawBlockQueryCandidatePlanner.plan(...)` plus typed Arrow request
  tables for single-query/seeded incremental requests. Do not restore raw
  payload-to-`FeatureBlock` scoring adapters.
- Status 2026-05-25: `RustFeaturizer.from_feature_block(...)`,
  `feature_port.build_rust_featurizer_from_feature_block(...)`, and raw payload
  scoring wrappers were removed after a repo-local no-caller scan. Keep the
  lower-level Python `FeatureBlock` construction and query-adapter helpers only
  as fixture/parity utilities until typed Arrow request-table coverage makes
  them unnecessary.
- Keep JSON loaders and remaining Python-object adapters as compatibility
  surfaces unless profiling shows they are still on a production hot path. In
  particular, `feature_block_from_arrow_paths(...)` is a bridge/test surface
  unless a caller still needs Python `FeatureBlock` compatibility.
- Keep `RustFeaturizer.from_dataset(...)` as the Python-reference,
  training/eval, and parity surface. Do not add production-only behavior there.
- Treat `RustFeaturizer.from_json_paths(...)` as a compatibility and benchmark
  surface until the Arrow release and strict Arrow routing are complete; then
  remove it from core runtime capability checks before deleting the loader.
- Done when: production scripts and docs recommend only Arrow-backed Rust
  inference, removed raw payload / `FeatureBlock` scoring bridges have no code
  callers, and remaining Python-object helpers are explicitly fixture,
  conversion, parity, or training utilities.

### Rust Surface Cleanup

Goal: after strict Arrow production routing lands, make the Rust codebase easier
to maintain without changing runtime behavior.

- Current surface notes:
  - `s2and_rust/src/lib.rs` is roughly 15.7k lines in the current working tree
    and owns PyO3 exports, Arrow IO, JSON ingest, Python object ingest, text
    compatibility, name-count indexes, pair features, constraints, retrieval,
    promoted linker row features, and subblocking. The branch diff against
    `origin/main` is large enough that line-count reductions should target
    public surface area first, not only module splitting.
  - No Rust training surface exists today. Keep training, calibration, model
    fitting, and release replay in Python unless a measured bottleneck justifies
    a separate port.
  - Do not delete the `RustNameCompatibleSubblockSelector` helper trio
    (`from_py`, `allowed_component_keys`,
    `select_candidate_indices_for_summaries`). They are live internal helpers
    used by `top_k_hybrid_centroid_pair_plan(...)` when retrieval subblock
    filtering is enabled.
  - Status 2026-05-25: `RustHybridCentroidRetriever.summary_count(...)` was
    removed after a repo-local no-caller scan found no maintained consumer and
    no external compatibility promise.
- Use the Rust public-surface inventory to choose one good API per behavior
  before moving code:
  - Status 2026-05-25: the current inventory is checked in at
    [rust/public_surface_inventory.md](rust/public_surface_inventory.md).
  - Status 2026-05-25: raw Arrow candidate planning now exposes the reusable
    `RawBlockQueryCandidatePlanner` class as the canonical public surface. The
    one-shot `raw_block_query_candidate_plan_arrow(...)` wrapper was removed
    after runtime callers moved to the planner and Python preserved the
    single-request telemetry merge.
  - Status 2026-05-25: canonical linker pair aggregation keeps the numpy-array
    `linker_pair_index_arrays_and_aggregate_stats(...)` path used by promoted
    incremental linking. The legacy list-of-tuples
    `linker_pair_features_and_aggregate_stats_indexed(...)` API and Python
    wrapper were removed after repo-local callers moved to the array API.
  - Status 2026-05-25: aggregate-only pair stats now use
    `linker_pair_index_arrays_and_aggregate_stats(..., emit_matrix=False)`.
    The separate `linker_pair_index_arrays_aggregate_stats(...)` PyO3 method
    was deleted, and capability probes key off the canonical array API.
  - Status 2026-05-25: canonical constraint APIs are indexed and
    block-upper-triangle. The string-pair `get_constraints_matrix(...)` Rust
    method and `get_constraints_matrix_rust(...)` Python wrapper were removed,
    and core capability checks no longer require the string-pair matrix API.
  - Canonical pairwise feature API: prefer matrix/indexed APIs. Keep
    `featurize_pair(...)` only as a parity/debug helper, and keep
    `featurize_pairs(...)` only while `s2and/featurizer.py` still requires the
    legacy row-by-row fallback.
  - Canonical retriever API: prefer
    `RustHybridCentroidRetriever.top_k_hybrid_centroid_pair_plan(...)` for
    runtime retrieval.
  - Status 2026-05-25: direct retriever debug APIs
    `top_k_hybrid_centroid(...)` and `chooser_feature_rows_subset(...)` were
    removed after capability probes and tests moved to pair-plan coverage.
- Deletion order:
  1. Lock strict Arrow production routing so compatibility fallbacks are no
     longer confused with production Rust.
  2. Remove or consolidate duplicate linker pair aggregate APIs.
  3. Remove the string-pair constraint API from core/public routing.
  4. Status 2026-05-25: raw payload / Python `FeatureBlock` scoring bridges
     were deleted. Continue with typed Arrow request-table assembly for any
     future ad hoc request callers rather than restoring raw payload scoring.
  5. Demote `from_json_paths(...)` from core runtime capability checks before
     removing JSON ingest helpers.
- Split `s2and_rust/src/lib.rs` after the production boundary is locked.
  - Start with a compact public-surface inventory before moving code: list each
    PyO3 export, Rust helper used from Python, and test-only helper, plus the
    owning caller class or module.
  - Before drawing final module boundaries, inventory the helpers referenced by
    the in-file `mod tests` in `lib.rs`. Existing tests currently reach private
    helpers by module scope, so extraction will otherwise force unnecessary
    `pub(crate)` churn. Prefer relocating focused tests with moved code when
    that preserves private visibility cleanly.
  - First split should be mechanical and behavior-preserving. Suggested module
    boundaries: `arrow_io`, `name_counts`, `text_compat`, `ingest_arrow`,
    `ingest_json`, `ingest_dataset`, `features`, `constraints`, `retrieval`,
    `linker`, and `subblocking`.
  - Extract one low-coupling module at a time, starting with the inventory's
    clearest leaf module. Do not attempt the whole module list in one change.
  - Keep `lib.rs` as the PyO3 export surface plus small wiring only. Avoid
    public Rust API redesign during the first split.
  - Verification: run Rust unit tests plus focused Python/Rust integration
    tests after each small module extraction. Prefer moving tests with the code
    they exercise rather than creating broad snapshot tests.
- Treat `s2and/incremental_linking/runtime.py` as a second cleanup target after
  strict Arrow production routing is locked. It owns Arrow-routed promoted
  incremental runtime and builder orchestration, so include it in the
  public-surface inventory, but do not split it in the same change as the first
  Rust module extraction.
- Use the Rust public-surface inventory to decide which Python wrappers should
  remain public and which should become compatibility helpers.
- Done when: `lib.rs` is reduced to PyO3 exports and small wiring, extracted
  modules own their focused tests, the Rust public-surface inventory is checked
  in, and the same Rust library plus focused Python/Rust integration tests pass
  after each mechanical extraction.

### Rust Ingest Deduplication

Evidence: `s2and_rust/src/lib.rs` has shared staging structs and preprocessing
helpers for Arrow construction, while `from_json_paths(...)` still keeps local
`SignatureInput`, `PaperInput`, and `PaperPreprocessed` records.
`from_dataset(...)` also keeps a local `PaperInput` record, so include both
non-Arrow ingest paths in the inventory. The former `FeatureBlock` Rust scoring
bridge is removed and should not be restored as an ingest surface.

- Start with a source-policy inventory for unidecode, language handling, name
  splitting, name-count telemetry/defaulting, paper-author handling,
  ORCID/source-id derivation, missing/null/malformed positions, reference
  features, `preprocess=false`, and filtering order.
  - Status 2026-05-25: the initial inventory is checked in at
    [rust/ingest_source_policy_inventory.md](rust/ingest_source_policy_inventory.md).
    It identifies Arrow/JSON no-reference preprocessing as the only plausible
    short-term reuse target and calls out source-specific decisions for
    `ANDData`, JSON reference features, language handling, paper-author
    positions, and name-count default telemetry.
- Deduplicate staging records only after the inventory names equivalent
  semantics and intentional source-specific differences.
- Done when: JSON, Arrow, and `ANDData` ingest paths share the reusable staging
  helpers where semantics match; source-specific differences are documented;
  Rust library tests and focused Python/Rust integration tests, including
  source-specific `paper_authors.position` expectations, pass.

### Incremental Helper Shim Decision

Evidence: `Clusterer._predict_incremental_helper(...)` still exists in
`s2and/model.py`, and tests still monkeypatch it directly. Current monkeypatch
sites are test plumbing: mutation capture, failure injection, and routing
recording in `tests/test_cluster_incremental.py`.

- Current decision: treat `_predict_incremental_helper(...)` as an internal-only
  test seam for now. Leave it in place and document that it is not an external
  compatibility API.
  - Status 2026-05-25: `s2and/model.py` documents the helper as internal-only
    and not an external compatibility API. Existing direct monkeypatches remain
    test plumbing until a future ask-first removal/rename.
- Before any future rename/removal, migrate tests that monkeypatch
  `Clusterer._predict_incremental_helper(...)`. Prefer fault injection through a
  public routing surface, dependency parameter, or explicit test hook over
  patching the private helper directly.
- Treat public-surface removal as ask-first.
- Done when: monkeypatched tests use the chosen call surface, focused
  incremental-linking tests pass, and docs or code comments identify whether
  `_predict_incremental_helper(...)` is internal-only or compatibility surface.

### Performance Targets

- Next profiling should target Arrow read/summary construction and reusable
  component summaries on a realistic Arrow promoted-incremental workload:
  raw single-query or small query-batch prediction against a published
  `s2and_and_big_blocks_linker_dataset_20260525` dataset, after first
  sanity-checking the profiler on the tiny Arrow fixture.
  - Status 2026-05-25: tiny promoted-incremental Arrow preflight passed via
    `tests/test_eval_prod_models.py::test_pubmed_specter2_arrow_fixture_incremental_smoke_matches_expected_b3`;
    see
    [rust/profiling/2026-05-25-promoted-incremental-preflight.md](rust/profiling/2026-05-25-promoted-incremental-preflight.md).
    Full profiling is blocked on choosing the local data source/runner because
    this checkout has `s2and_and_big_blocks_linker_dataset_20260513_arrow`, not
    a local `s2and_and_big_blocks_linker_dataset_20260525`, and the existing
    big-block measurement command is still JSON/`ANDData`-based.
- Primary metrics: p50 wall time over at least five isolated runs, peak RSS, and
  summary-construction allocation volume from a stack-level allocation profiler
  (`heaptrack`/`perf` on Linux or ETW allocation tracing on Windows).
- Act only when Arrow read or summary construction is at least a 10% contributor
  to p50 wall time or allocation volume, or when a change removes a real
  `ANDData` dependency. Stop iterating once the measured improvement is below
  10% for the selected workload.
- Earlier profiling suggested pairwise/model scoring and the old Python
  row-signal bridge were not the main raw single-query bottlenecks. Keep this
  as a hypothesis to re-check before optimizing them again.
- SPECTER/vector clone cleanup is deferred based on bounded profiling from
  2026-05-23; see
  [rust/profiling/2026-05-23.md](rust/profiling/2026-05-23.md). Do not change
  code for this now: it is measurable on the JSON compatibility path, but not a
  clear current bottleneck on the production-oriented Arrow path.
