# Rust Inference Architecture

Status date: 2026-05-25

This is the current map for Rust-backed inference. It replaces the older
raw-candidate-plan design log and the promoted-incremental design note.

The target is not a Rust clone of all `ANDData`. `ANDData` remains the Python
reference object for training, legacy loading, pair sampling, compatibility
tests, and explicit Python/legacy inference. The production Rust target is a
narrow set of typed inference inputs read directly by Rust.

## Before / After

| Area | Before | After |
|---|---|---|
| Inference boundary | Most production paths started from full `ANDData` and crossed into Rust after Python had built namedtuples, blocks, and feature objects. | Direct Arrow paths feed signatures, papers, paper authors, SPECTER, and seed rows into Rust without full-block `ANDData`. |
| Full-block prediction | `Clusterer.predict(...)` built or reused `ANDData`, then used Rust mainly for pairwise feature and clustering kernels. | `Clusterer.predict_from_arrow_paths(...)` and Arrow-routed `predict(...)` build the filtered Rust featurizer from Arrow IPC and reuse Rust blockwise feature, constraint, and clustering logic. Rust mode requires complete Arrow artifacts and fails fast when they are missing. |
| Raw incremental requests | The raw path first materialized Python signal objects or mini compatibility objects before scoring. | `RawBlockQueryCandidatePlanner` performs retrieval and row-signal construction in Rust from Arrow IPC, then the raw Arrow scoring route runs without full-block `ANDData`. |
| Candidate scope | Giant blocks could lead to broad query-vs-seed-signature work before the linker saw compact candidates. | Rust retrieval builds a bounded query-to-component candidate plan before pair scoring. |
| Pairwise feature build | Python object materialization and `ANDData` construction dominated some profiles before Rust pairwise work began. | `RustFeaturizer.from_arrow_paths(...)` (Rust method exposed via PyO3; Python wrapper is `build_rust_featurizer_from_arrow_paths` in `s2and/feature_port.py`) constructs only the selected scoring rows from Arrow and global sidecars. Production filtered reads require batch indexes. |
| Row signals | Several promoted link/abstain row signals were assembled in Python after Rust retrieval. | Rust emits the promoted native row signals needed by the raw Arrow planner/scoring route, including name-count rarity and paper-author overlap signals. |
| Name counts | Docs and tests previously preferred embedding four per-signature count columns in `signatures.arrow`; Rust could skip global artifacts if all selected rows had embedded counts. | Embedded Arrow name-count columns are not a supported direction. Runtime bundles should provide `s2and/data/name_counts_index/`; `name_counts.arrow` is only for generation, inspection, and parity debugging. |
| Name alias data | Some paths could pass per-dataset Arrow `name_pairs` / `name_tuples` overrides. | Runtime aliases now come from the explicit `name_tuples` argument; production path bundles must not carry alias override paths. |
| SPECTER | Pickle remained common in Python paths; Rust paths handled some payloads through Python objects. | Direct Arrow uses fixed-size-list `float32` embedding tables. Safetensors is still only a future benchmark if SPECTER read time becomes material. |
| Cluster seeds | Seed semantics were mostly Python maps on the incremental path. | Seeded/incremental Arrow requires a seed source: either `cluster_seeds.arrow` or a normalized seed mapping that production materializes into request-local Arrow. `cluster_seed_disallows.arrow` is optional unless disallow constraints are declared. Unseeded full predict can omit both. |
| Reference features | Legacy feature slots and training paths still supported citation-derived reference features. | Direct Arrow predict fails fast if a model requests reference features; current production models do not use them. |
| Data ingestion | JSON/pickle plus `ANDData` preprocessing was the default ingestion shape. | Arrow IPC is the preferred table-shaped ingestion format when the hot path stays Rust/columnar. JSON remains compatibility/test input. |
| Verification | Performance and parity evidence lived across several design logs. | Current gates should point to this architecture doc, `arrow_dataset_spec.md`, `artifact_formats.md`, `runtime.md`, and `baselines.md`. |

## Name-Count Decision

Use the sorted exact-verified `s2and/data/name_counts_index/` sidecar as the
Rust hot-path artifact. It is a better fit than SQLite for the current workload
because Rust does exact point lookups against four static dictionaries; the
binary index is memory-map friendly, has exact string verification after hash
lookup, and avoids shipping a query engine or managing SQLite runtime state.

Reconsider SQLite only if the requirement changes to ad hoc querying, partial
updates, cross-process transactional writes, or richer offline inspection.

## Compatibility And Python-Heavy Paths

Production Rust inference uses `Clusterer.predict_from_arrow_paths(...)`,
Arrow-routed `Clusterer.predict(...)`, or promoted
`Clusterer.predict_incremental(...)` with complete base Arrow artifacts. The
paths below remain useful, but they are compatibility, training, parity, or test
surfaces rather than production inference APIs.

Removed bridge surfaces: `RustFeaturizer.from_feature_block(...)` and raw
payload scoring wrappers are no longer Rust inference APIs. They built or
traversed Python `FeatureBlock` objects before scoring; production raw requests
must materialize typed Arrow request rows before entering Rust.

| Path | Current Python dependency | Production status |
|---|---|---|
| `Clusterer.predict(...)` without Arrow paths | Explicit Python/legacy routes can still build normal `ANDData` and use Python block orchestration. | Rust production now raises `MissingArrowArtifactError`. Provide complete Arrow artifacts or select `backend="python"` for compatibility/reference execution. |
| `Clusterer.predict_incremental(...)` without base Arrow paths or seed source | Explicit Python/legacy routes can still use Python incremental helpers and `ANDData` seed state. | Rust production now raises `MissingArrowArtifactError`. Provide `signatures`, `papers`, `paper_authors`, required embedding/name-count artifacts, and a seed source via `cluster_seeds` or `dataset.cluster_seeds_require`. |
| `RustFeaturizer.from_dataset(...)` | Traverses Python `ANDData` objects over PyO3. | Keep as incumbent/reference, training/eval, parity, and compatibility surface; do not present or optimize it as the production hot path. |
| JSON Rust loaders | Avoid `ANDData`, but still read compatibility JSON and call Python text normalization helpers. | Fixture, legacy script, and benchmark surface only; Arrow IPC is the production table-shaped target. |
| Training and release replay | Python owns data cleaning, feature table materialization, LightGBM training, calibration, and metrics. | Not a no-`ANDData` inference target; keep Python unless runtime profiling shows a training bottleneck worth porting. |

Profiling on the `f_matsen` inference payload shows that the Arrow incremental
path spent its material residual time in altered-profile seed pre-splitting:
raw Arrow retrieval and scoring were small, while the altered pre-split called
back through pairwise prediction for hundreds of seed signatures. Production
request producers should still emit the canonical Arrow inputs, including
`cluster_seed_disallows.arrow` when seed disallow constraints exist and
`altered_cluster_signatures.arrow` when altered claimed profiles are present.
The runtime pre-splits altered claimed profiles in process before promoted
incremental linking; request producers do not provide any separate altered split
artifact. This pre-split intentionally runs before retrieval/linking so the
candidate components match the naturalized seed map used by the Python
incremental path.

For true incremental Arrow requests, `cluster_seeds.arrow` is the seed
membership source when no Python seed map has been materialized, and optional
`cluster_seed_disallows.arrow` is merged into partial supervision for altered
pre-split and residual scoring. After altered-profile pre-split, the runtime
writes a request-local temporary seed table for Rust retrieval. For
bulk full-block prediction with subblocking, the single-letter synthetic
incremental pass suppresses the original required seed memberships from
`cluster_seeds.arrow` and the altered-profile list from
`altered_cluster_signatures.arrow`, but preserves optional
`cluster_seed_disallows.arrow` as pairwise negative supervision.

## Current Verification Focus

- Tiny Arrow fixture tests for schema and row-signal behavior.
- Exact parity gates for direct Arrow full predict: feature matrix, constraints,
  distances, and clusters.
- Raw Arrow incremental checks for candidate rows, pair rows, row signals,
  probabilities, and final link/abstain decisions.
- Stage telemetry that separates Arrow read, name-count index load, retrieval,
  featurizer construction, pair scoring, and raw row-signal construction. Final
  logistic-gate decisions are covered by result telemetry but are not currently
  timed as a separate stage.
