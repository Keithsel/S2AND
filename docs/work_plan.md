# Work Plan

Status date: 2026-05-24

This file is the active Rust/platform backlog. Stable architecture and artifact
contracts live in:

- [rust/inference_architecture.md](rust/inference_architecture.md)
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
3. Enforce strict Arrow production routing through one approved strictness
   authority; keep public compatibility fallback unchanged until an API change
   is approved.
4. Demote non-Arrow Rust entrypoints to compatibility/training/test surfaces.
5. Split the oversized Rust module after the production boundary is locked.

Do not make strict Arrow routing the default before shippable Arrow bundles and
their validation checks exist.

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

Treat this prefix as planned until the release validation command confirms the
uploaded contents.

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
- `linker_replay_20260513/` with Arrow runtime tables under
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
    `linker_replay_20260513/datasets/<dataset>/signatures.arrow`.
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
    `summary.json`, `bundle.json`, and manifests, with paths and checksums
    updated where they refer to converted artifacts.
  - Keep linker replay `components/*.parquet` and `labels/*.parquet` as-is
    because they are already typed columnar offline artifacts.
- Omit or quarantine artifacts that are not part of the Arrow-native release.
  - Omit `.feature_cache/`; it is a historical precomputed feature cache and
    should not be copied into the Arrow release.
  - Omit `full_union_seed_*.pickle` from the Arrow-native release, or place it
    under an explicit `legacy/` prefix if paper-era compatibility requires it.
  - Do not duplicate raw JSON files after conversion. The existing
    `s2and-release` bucket remains the legacy JSON source.
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

  - For the full release, record the resolved source roots, output staging root,
    exact `uv run ...` commands, expected runtime, log path, and publish target
    before starting. If a conversion or upload is expected to exceed ten
    minutes, run it detached with stdout/stderr captured to a log and record the
    PID/job id.
  - Validate every generated dataset manifest with
    `scripts/convert_to_arrow.py validate`, requiring embeddings and
    `name_counts_index` when the selected production model uses those features.
  - Produce a compact release audit file with per-dataset signature, paper,
    paper-author, embedding, cluster-seed, disallow, altered-signature, missing
    embedding, and batch-index counts; include root manifest path, generator git
    commit, source snapshot ids, and validation command lines.
  - Before making the S3 prefix public, do a dry-run or staging upload, then
    spot-check that `manifest.json`, one benchmark dataset manifest, one
    manifest-declared embedding Arrow file, one batch-index sidecar, the
    production model manifest, and `name_counts_index/manifest.json` can be
    read from the published prefix.
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
  - Production seeded/incremental prediction additionally requires
    `cluster_seeds`.
  - `cluster_seed_disallows` is required only when pairwise seed-disallow
    constraints are present. Missing `cluster_seed_disallows` means no disallow
    constraints unless the producer contract chooses to emit a canonical empty
    table.
  - `altered_cluster_signatures` is required only when altered claimed profiles
    are present. Missing `altered_cluster_signatures` means no altered claimed
    profiles unless the producer contract chooses to emit a canonical empty
    table.
  - Missing Arrow artifacts should fail with an explicit error in production
    mode rather than silently falling back to `ANDData`.
- Update script entrypoints.
  - Convert `scripts/tutorial_for_predicting_with_the_prod_model.py` to accept
    Arrow input paths or an Arrow bundle root and route through
    `Clusterer.predict_from_arrow_paths(...)`.
  - Expand `scripts/eval_prod_models.py --use-arrow` beyond the current mini
    restriction, or split the mini-only smoke test from real production eval.
  - Convert `scripts/_rust_suite/prod_inference_cmd.py` to benchmark Arrow as
    the production path; keep `from_dataset` only as an explicit legacy/parity
    mode if still needed.
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
  - In production Rust mode, remove silent fallback from Arrow-routed
    `Clusterer.predict(...)` when Arrow artifacts are incomplete.
  - Preserve fallback only through approved compatibility/test routes.
  - Ensure errors name the missing Arrow artifact keys and the caller/script
    that should generate them.
- Strict routing error contract.
  - Add `MissingArrowArtifactError` near `_resolve_dataset_arrow_paths(...)` in
    `s2and/model.py`. It should be raised by strict production routing when
    required Arrow artifacts are absent or mapped to missing files.
  - The error should carry enough structured context for tests and callers:
    `context`, `required_keys`, `missing_keys`, `missing_files`, and
    `producer_hint`.
  - The message should name the prediction context, list missing mapping keys
    separately from missing files, and name the command or script expected to
    produce the artifacts.
- Implementation map for strict production routing.
  - Current fallback surfaces to tighten:
    - `_resolve_dataset_arrow_paths(...)` in `s2and/model.py` can return
      `None` for incomplete auto-discovered Arrow artifacts and for missing
      required artifacts such as `cluster_seeds` or `name_counts_index`.
    - `Clusterer.predict(...)` currently attempts Arrow routing in Rust mode,
      then falls through to subblocked or normal `ANDData` prediction when
      Arrow paths are unavailable.
    - `_predict_subblocked(...)` accepts `arrow_paths=None` and can use
      graph/legacy `ANDData` fallbacks.
    - `_predict_incremental_promoted_linker(...)` can fall back to the
      non-Arrow promoted linker, which uses `RustFeaturizer.from_dataset`.
    - `Clusterer.predict_incremental(...)` can fall back to
      `_predict_incremental_helper(...)` when Rust mode lacks seed inputs.
  - Add a strict helper near `_resolve_dataset_arrow_paths(...)`, tentatively
    `_require_dataset_arrow_paths(...)`, that wraps discovery, computes missing
    required keys, distinguishes missing mapping keys from missing files, and
    raises an explicit error naming the missing keys and the Arrow conversion
    command/script that should produce them.
  - In strict production mode, do not silently satisfy a missing bundle
    `name_counts_index` from the checked-in `s2and/data/name_counts_index/`
    fallback. That fallback is useful for development and compatibility, but it
    can hide broken release bundles. Allow local name-count overrides only when
    the caller passes an explicit path.
  - Define one strictness authority before adding public parameters. Preferred
    first step: production scripts and production-only routing call the strict
    helper directly, while general public `predict(...)` keeps current
    compatibility fallback behavior until a public API change is approved.
  - If the strictness decision needs to flow across several layers, prefer a
    `RuntimeContext` policy field over adding public kwargs to multiple
    prediction methods. Add a new public kwarg only after confirming the
    context-based path is insufficient.
  - If a public switch is approved, preserve current behavior at introduction
    time, for example `require_arrow_paths: bool = False` or
    `allow_non_arrow_fallback: bool = True`, then change defaults only in a
    separately approved API migration.
  - In Rust production mode with compatibility fallback disabled:
    - `predict(...)` calls the strict helper before selecting the Arrow route.
    - `predict_incremental(...)` requires base Arrow artifacts plus
      `cluster_seeds` before seed sync or helper fallback. It requires
      `cluster_seed_disallows` only when disallow constraints are present or
      when the converter contract requires canonical empty tables.
    - `_predict_incremental_promoted_linker(...)` does not call the non-Arrow
      promoted linker.
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
    - `tests/test_rust_distance_matrix_blockwise.py` should expect a missing
      `name_counts_index` error in Rust production mode, plus a sibling
      compatibility-mode test that preserves the old fallback.
    - `tests/test_cluster_incremental.py` should expect missing `cluster_seeds`
      errors in Rust production mode, plus compatibility-mode tests for
      `_predict_incremental_helper(...)`. Add separate tests proving absent
      `cluster_seed_disallows` means no disallows, while a missing declared
      disallow artifact fails explicitly.
    - Add strict-helper tests for missing `signatures`, `papers`,
      `paper_authors`, `specter`, `name_counts_index`, `cluster_seeds`,
      conditionally `cluster_seed_disallows`, and conditionally
      `altered_cluster_signatures`.
- Clean up after migration.
  - Update `docs/production_inference.md` to state that production Rust
    inference requires Arrow artifacts.
  - Update `docs/rust/inference_architecture.md` so non-Arrow Rust loaders are
    described as compatibility, training, or test surfaces only.
  - Demote compatibility Rust entrypoints from the normal public Python surface
    after callers are migrated. Keep `RustFeaturizer.from_dataset(...)`,
    `RustFeaturizer.from_json_paths(...)`, and
    `RustFeaturizer.from_feature_block(...)` callable for training,
    fixtures, parity, and legacy scripts, but do not present them as production
    inference APIs.
  - Retarget the `s2and/feature_port.py` cleanup at dispatcher behavior, not
    `__all__`: `build_rust_featurizer(...)` and `_resolve_requested_build_path`
    should not select non-Arrow production paths once strict Arrow routing is
    enabled. Keep compatibility builders internal or explicitly named as
    compatibility helpers.
  - Remove or quarantine stale non-Arrow production-inference commands from
    `scripts/README.md`.
  - Keep training/materialization scripts on `ANDData`; they are not part of
    the production inference removal.
- Done when: with `backend="rust"` or `S2AND_BACKEND=rust` and the approved
  strict production authority enabled, missing production artifacts raise
  `MissingArrowArtifactError`; the approved compatibility path preserves the old
  fallback behavior for legacy/parity callers; Arrow full-predict and promoted
  incremental smoke tests run without `ANDData` fallback; and
  `scripts/eval_prod_models.py --use-arrow` runs end-to-end on at least three
  benchmark datasets without `ANDData` fallback telemetry.

### Compatibility And Python-Heavy Paths

- Prefer upgrading callers to `Clusterer.predict_from_arrow_paths(...)` or
  Arrow-routed `predict(...)` before optimizing `RustFeaturizer.from_dataset`.
- Prefer the raw Arrow wrapper for single-query/seeded incremental requests
  before optimizing raw payload to Python `FeatureBlock` adapters.
- Keep JSON loaders and Python-object adapters as compatibility surfaces unless
  profiling shows they are still on a production hot path.
- Done when: production scripts and docs no longer recommend
  `RustFeaturizer.from_dataset(...)`, `RustFeaturizer.from_json_paths(...)`, or
  `RustFeaturizer.from_feature_block(...)` for inference, and remaining direct
  uses are labeled training/reference, compatibility, parity, or test-only.

### Rust Surface Cleanup

Goal: after strict Arrow production routing lands, make the Rust codebase easier
to maintain without changing runtime behavior.

- Split `s2and_rust/src/lib.rs` after the production boundary is locked.
  - Current state: `lib.rs` is roughly 15k lines and owns PyO3 exports, Arrow
    IO, JSON ingest, Python object ingest, text compatibility, name-count
    indexes, pair features, constraints, retrieval, promoted linker row
    features, and subblocking.
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
- Use the Rust public-surface inventory to decide which Python wrappers should
  remain public and which should become compatibility helpers.
- Done when: `lib.rs` is reduced to PyO3 exports and small wiring, extracted
  modules own their focused tests, the Rust public-surface inventory is checked
  in, and the same Rust library plus focused Python/Rust integration tests pass
  after each mechanical extraction.

### Rust Ingest Deduplication

Evidence: `s2and_rust/src/lib.rs` has shared staging structs and preprocessing
helpers for Arrow and FeatureBlock construction, while `from_json_paths(...)`
still keeps local `SignatureInput`, `PaperInput`, and `PaperPreprocessed`
records.

- Start with a source-policy inventory for unidecode, language handling, name
  splitting, name-count telemetry/defaulting, paper-author handling,
  ORCID/source-id derivation, missing/null/malformed positions, reference
  features, `preprocess=false`, and filtering order.
- Deduplicate staging records only after the inventory names equivalent
  semantics and intentional source-specific differences.
- Done when: JSON, Arrow, FeatureBlock, and `ANDData` ingest paths share the
  reusable staging helpers where semantics match; source-specific differences
  are documented; Rust library tests and focused Python/Rust integration tests,
  including source-specific `paper_authors.position` expectations, pass.

### Incremental Helper Shim Decision

Evidence: `Clusterer._predict_incremental_helper(...)` still exists in
`s2and/model.py`, and tests still monkeypatch it directly.

- Decide whether the shim is removable, narrowable, or still an external
  compatibility surface.
- Migrate tests that monkeypatch `Clusterer._predict_incremental_helper(...)`
  before any rename/removal. Prefer fault injection through a public routing
  surface, dependency parameter, or explicit test hook over patching the private
  helper directly.
- Treat public-surface removal as ask-first.
- Done when: monkeypatched tests use the chosen call surface, focused
  incremental-linking tests pass, and docs or code comments identify whether
  `_predict_incremental_helper(...)` is internal-only or compatibility surface.

### Performance Targets

- Next profiling should target Arrow read/summary construction and reusable
  component summaries.
- Earlier profiling suggested pairwise/model scoring and the old Python
  row-signal bridge were not the main raw single-query bottlenecks. Keep this
  as a hypothesis to re-check before optimizing them again.
- SPECTER/vector clone cleanup is deferred based on bounded profiling from
  2026-05-23; see
  [rust/profiling/2026-05-23.md](rust/profiling/2026-05-23.md). Do not change
  code for this now: it is measurable on the JSON compatibility path, but not a
  clear current bottleneck on the production-oriented Arrow path.
