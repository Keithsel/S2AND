# Rust Inference Architecture

Status date: 2026-05-20

This is the current map for Rust-backed inference. It replaces the older
raw-candidate-plan design log and the promoted-incremental design note.

The target is not a Rust clone of all `ANDData`. `ANDData` remains the Python
reference object for training, legacy loading, pair sampling, compatibility
tests, and fallback inference. The fast runtime target is a narrow set of typed
inference inputs read directly by Rust.

## Before / After

| Area | Before | After |
|---|---|---|
| Inference boundary | Most production paths started from full `ANDData` and crossed into Rust after Python had built namedtuples, blocks, and feature objects. | Direct Arrow paths feed signatures, papers, paper authors, SPECTER, and seed rows into Rust without full-block `ANDData`. |
| Full-block prediction | `Clusterer.predict(...)` built or reused `ANDData`, then used Rust mainly for pairwise feature and clustering kernels. | `Clusterer.predict_from_arrow_paths(...)` and Arrow-routed `predict(...)` build the filtered Rust featurizer from Arrow IPC and reuse Rust blockwise feature, constraint, and clustering logic. |
| Raw incremental requests | The raw path first materialized Python signal objects or mini compatibility objects before scoring. | `raw_block_query_candidate_plan_arrow(...)` performs retrieval and row-signal construction in Rust from Arrow IPC, then the public raw Arrow wrapper scores without full-block `ANDData`. |
| Candidate scope | Giant blocks could lead to broad query-vs-seed-signature work before the linker saw compact candidates. | Rust retrieval builds a bounded query-to-component candidate plan before pair scoring. |
| Pairwise feature build | Python object materialization and `ANDData` construction dominated some profiles before Rust pairwise work began. | `RustFeaturizer.from_arrow_paths(...)` constructs only the selected scoring rows from Arrow and global sidecars. |
| Row signals | Several promoted link/abstain row signals were assembled in Python after Rust retrieval. | Rust emits the promoted native row signals needed by the raw Arrow wrapper, including name-count rarity and paper-author overlap signals. |
| Name counts | Docs and tests previously preferred embedding four per-signature count columns in `signatures.arrow`; Rust could skip global artifacts if all selected rows had embedded counts. | Embedded Arrow name-count columns are not a supported direction. Runtime bundles should provide `s2and/data/name_counts_index/`; `name_counts.arrow` is a generation, inspection, or parity fallback. |
| Name alias data | Some paths could pass per-dataset Arrow `name_pairs` / `name_tuples` overrides. | Production uses the packaged filtered alias text as the default shared runtime resource. Overrides are for experiments only. |
| SPECTER | Pickle remained common in Python paths; Rust paths handled some payloads through Python objects. | Direct Arrow uses fixed-size-list `float32` embedding tables. Safetensors is still only a future benchmark if SPECTER read time becomes material. |
| Cluster seeds | Seed semantics were mostly Python maps on the incremental path. | Seeded/incremental Arrow uses `cluster_seeds.arrow`; unseeded full predict can omit it. |
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

## Rust Paths Still Heavy In Python

| Path | Current Python dependency | No-`ANDData` upgrade status |
|---|---|---|
| `Clusterer.predict(...)` without Arrow paths | Falls back to normal `ANDData` construction and Python block orchestration. | Upgraded when callers provide complete Arrow artifacts or `dataset.arrow_paths`; otherwise keep as compatibility fallback. |
| `Clusterer.predict_incremental(...)` without seed-bearing Arrow paths | Uses promoted Rust retrieval/scoring from `ANDData`-derived state when Arrow seed inputs are unavailable. | Upgrade by providing `signatures`, `papers`, `paper_authors`, optional `specter`, and `cluster_seeds` Arrow paths plus `s2and/data/name_counts_index/`. |
| `RustFeaturizer.from_dataset(...)` | Traverses Python `ANDData` objects over PyO3. | Keep as incumbent/reference path; do not optimize as the production hot path. |
| `RustFeaturizer.from_feature_block(...)` | Avoids full `ANDData`, but still traverses a Python `FeatureBlock` object and uses Python text compatibility helpers. | Useful bridge for raw payload compatibility; Arrow is the preferred no-`ANDData` runtime path. |
| Raw payload wrappers | Build a Python `FeatureBlock` before entering Rust. | Replace with Arrow/request-table assembly when the caller can provide typed rows. |
| JSON Rust loaders | Avoid `ANDData`, but still read compatibility JSON and call Python text normalization helpers. | Keep for fixtures and compatibility; Arrow IPC is the table-shaped target. |
| Training and release replay | Python owns data cleaning, feature table materialization, LightGBM training, calibration, and metrics. | Not a no-`ANDData` inference target; keep Python unless runtime profiling shows a training bottleneck worth porting. |

Profiling on the `f_matsen` inference payload shows that the Arrow incremental
path spent its material residual time in altered-profile seed pre-splitting:
raw Arrow retrieval and scoring were small, while the altered pre-split called
back through pairwise prediction for hundreds of seed signatures. Production
request producers should still emit the canonical Arrow inputs, including
`altered_cluster_signatures.arrow` when altered claimed profiles are present.
The runtime pre-splits altered claimed profiles in process before promoted
incremental linking; request producers do not provide any separate altered split
artifact. The Arrow path first asks Rust retrieval which original seed
components are candidate-relevant for the incoming queries, then only pre-splits
altered claimed profiles in those components.

## Current Verification Focus

- Tiny Arrow fixture tests for schema and row-signal behavior.
- Exact parity gates for direct Arrow full predict: feature matrix, constraints,
  distances, and clusters.
- Raw Arrow incremental checks for candidate rows, pair rows, row signals,
  probabilities, and final link/abstain decisions.
- Stage telemetry that separates Arrow read, name-count index load, retrieval,
  featurizer construction, pair scoring, and final gate time.
