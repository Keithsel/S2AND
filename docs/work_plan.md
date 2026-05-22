# Work Plan

Status date: 2026-05-22

This file is only the active Rust/platform backlog. Current architecture and
artifact decisions live in:

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

## Open Work

### Stabilize Direct Arrow Gates

- Keep tiny Arrow fixture tests for schema, row signals, and name-count index
  behavior.
- Keep bounded full-predict parity gates that compare features, constraints,
  distances, and clusters against the `ANDData` oracle.
- Keep raw Arrow incremental gates that compare candidate ids, pair ids, row
  signals, probabilities, and final decisions on bounded fixtures.

### Production Artifact Generation

- Regenerate durable Arrow bundles from the complete schema in
  [rust/arrow_dataset_spec.md](rust/arrow_dataset_spec.md).
- Regenerate `s2and/data/name_counts_index/` into the manifest-backed
  `generations/<generation-id>/` layout before publishing runtime bundles; the
  current direct-file layout is a legacy source artifact.
- Include the regenerated `s2and/data/name_counts_index/` as the shared
  name-count artifact for runtime bundles that use name-count features.
- Keep `name_counts.arrow` available for generation, inspection, and parity
  debugging, not as the default request-time read.

### Arrow S3 Release

Target a new public release prefix:
`s3://ai2-s2-research-public/s2and-release-arrow`.

This release should be an Arrow-native data/runtime artifact release, not a
mirror of the legacy JSON/pickle bucket. It should omit `.feature_cache/`.

Preferred layout:

```text
s2and-release-arrow/
  manifest.json
  LICENSE.txt
  lid.176.bin
  production_model_v1.21/
    manifest.json
    clusterer.json
    pairwise/
    incremental_linker/
  production_model_v1.0.pickle
  production_model_v1.1.pickle
  production_model_v1.2.pickle
  name_counts_index/
    manifest.json
    generations/<generation-id>/
      first.bin
      last.bin
      first_last.bin
      last_first_initial.bin
  aminer/
  arnetminer/
  inspire/
  kisti/
  medline/
  pubmed/
  qian/
  zbmath/
    manifest.json
    signatures.arrow
    papers.arrow
    paper_authors.arrow
    specter.arrow
    signatures.signatures_batch_index.bin
    papers.papers_batch_index.bin
    paper_authors.paper_authors_batch_index.bin
    specter.specter_batch_index.bin
    <dataset>_clusters.json
    splits/
  # ^ Every benchmark dataset directory (aminer ... zbmath) has the same
  # internal layout shown under zbmath/ above.
  linker_replay_20260513/
    manifest.json
    bundle.json
    datasets/<dataset>/
      manifest.json
      signatures.arrow
      papers.arrow
      paper_authors.arrow
      specter2.arrow
      signatures.signatures_batch_index.bin
      papers.papers_batch_index.bin
      paper_authors.paper_authors_batch_index.bin
      specter2.specter_batch_index.bin
    components/
    labels/
    splits/
```

- Convert table-shaped runtime inputs.
  - Convert each benchmark `<dataset>_signatures.json` to
    `signatures.arrow`.
  - Convert each benchmark `<dataset>_papers.json` to `papers.arrow` and
    `paper_authors.arrow`.
  - Convert each benchmark `<dataset>_specter.pickle` to `specter.arrow`.
  - Convert linker replay `raw/<dataset>/signatures.json` to
    `linker_replay_20260513/datasets/<dataset>/signatures.arrow`.
  - Convert linker replay `raw/<dataset>/papers.json` to `papers.arrow` and
    `paper_authors.arrow`.
  - Convert linker replay `embeddings/<dataset>/specter2.pkl` to
    `specter2.arrow`.
  - Convert `name_counts.pickle` to a manifest-backed generation under
    `name_counts_index/`; do not publish `name_counts.arrow` in this release.
    The checked-in direct-file `s2and/data/name_counts_index/` layout is a
    legacy source artifact until regenerated.
  - Generate and ship Arrow batch lookup indexes beside the Arrow files using
    `<table-stem>.<path-key>.bin` names, for example
    `signatures.signatures_batch_index.bin` and
    `specter2.specter_batch_index.bin`.
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
    `cluster_seeds`, `cluster_seed_disallows`, and
    `altered_cluster_signatures` when altered claimed profiles are present.
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
  - Preserve fallback only under an explicit compatibility/test mode.
  - Ensure errors name the missing Arrow artifact keys and the caller/script
    that should generate them.
- Keep verification gates focused.
  - Keep bounded Arrow full-predict parity tests against the `ANDData` oracle.
  - Keep raw Arrow incremental tests for candidate rows, pair rows, row
    signals, probabilities, and final decisions.
  - Add or adjust script-level smoke tests for
    `scripts/convert_to_arrow.py service-json`, the Arrow prod-model tutorial
    flow, the Arrow prod eval flow, and the Arrow rust-suite production
    benchmark flow.
- Clean up after migration.
  - Update `docs/production_inference.md` to state that production Rust
    inference requires Arrow artifacts.
  - Update `docs/rust/inference_architecture.md` so non-Arrow Rust loaders are
    described as compatibility, training, or test surfaces only.
  - Remove or quarantine stale non-Arrow production-inference commands from
    `scripts/README.md`.
  - Keep training/materialization scripts on `ANDData`; they are not part of
    the production inference removal.

### Remaining Python-Heavy Paths

- Prefer upgrading callers to `Clusterer.predict_from_arrow_paths(...)` or
  Arrow-routed `predict(...)` before optimizing `RustFeaturizer.from_dataset`.
- Prefer the raw Arrow wrapper for single-query/seeded incremental requests
  before optimizing raw payload to Python `FeatureBlock` adapters.
- Keep JSON loaders and Python-object adapters as compatibility surfaces unless
  profiling shows they are still on a production hot path.

### Performance Targets

- Next profiling should target Arrow read/summary construction and reusable
  component summaries.
- Pairwise/model scoring and the old Python row-signal bridge are no longer the
  main raw single-query bottlenecks.
- No new complexity for less than a 10% improvement unless it removes a real
  bottleneck or an `ANDData` dependency on a hot path.
