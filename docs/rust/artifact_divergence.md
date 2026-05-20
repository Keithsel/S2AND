# Rust Artifact Divergence and Format Migration Plan

Status date: 2026-05-20

Tracks artifact-level divergences between Python and Rust paths and the planned
format migration. This supersedes the older MessagePack-first plan. See also:
`docs/work_plan.md` and `docs/rust/raw_block_query_candidate_plan.md`.

---

## Current Direction

The target is **not** a Rust reimplementation of all `ANDData`.

`ANDData` remains the broad Python compatibility/reference object for legacy
loading, training/eval splits, pair sampling, Sinonym mutation, cluster
bookkeeping, and parity tests. The Rust path should consume a narrower
inference contract, tentatively `FeatureBlock`, containing only the data needed
for retrieval, constraints, pair features, and link/abstain scoring.

Initial code boundary: `s2and/incremental_linking/feature_block.py`.

The raw scoring wrappers are also in place:
`predict_incremental_link_or_abstain_from_raw_feature_block(...)` and
`predict_incremental_link_or_abstain_from_raw_payloads(...)`. They build a mini
`FeatureBlock` from the raw candidate plan or raw payloads, build a
`RustFeaturizer` directly through `RustFeaturizer.from_feature_block(...)`, and
use the existing Rust pairwise-feature and constraint-label kernels. On the
bounded a_silva and h_wang fixtures the earlier mini-`ANDData` bridge had exact
normalized decision, linked-cluster, probability, 53-feature-matrix,
candidate-row, and pair-count parity with the current promoted path. The current
wrapper removes that mini-`ANDData` layer; the tiny parity gate compares
`FeatureBlock -> RustFeaturizer` against the mini-`ANDData` constructor.

The direct Arrow path is now first-class rather than adapter-specific:

- `RustFeaturizer.from_arrow_paths(...)` reads signatures, papers,
  paper-authors, cluster seeds, optional SPECTER, optional author
  email/block/source ids, optional abstracts, and optional embedded name-count
  columns directly from Arrow IPC.
- `predict_incremental_link_or_abstain_from_raw_arrow_paths(...)` runs raw Arrow
  retrieval plus downstream link/abstain scoring without full-block `ANDData`.
- `Clusterer.predict_from_arrow_paths(...)` runs full-block predict from Arrow
  by reusing the existing Rust blockwise feature/constraint APIs and clustering
  logic with a filtered Rust featurizer.

Important schema note: the older `scratch/baseline/*/arrow_full_specter`
fixtures were generated before the complete inference schema. They are good
speed probes, but exact parity fixtures must include all feature-bearing fields,
especially abstract presence and embedded name-count columns, and should include
`cluster_seeds.arrow` only when the caller wants seed semantics.

Current complete-Arrow scratch fixtures now prove that schema for bounded
full-predict runs. The checks compare incumbent `ANDData` against direct
Arrow/Rust over the full 39-feature matrix, upper-triangle constraint labels,
distance matrices, and final clusters.

Latest hot-path finding: `name_counts.arrow` is a valid Rust-readable global
lookup, but it should not be cold-read inside each request. On h_wang, the
1.4GB lookup added 25.73s to raw retrieval and 35.51s to filtered featurizer
construction when the signatures table lacked embedded counts. The runtime
bundle should carry embedded per-signature count columns in `signatures.arrow`
when possible. If embedding is impractical, use the sorted binary
`name_counts_index/` sidecar: the same h_wang raw incremental profile drops to
17.04s predict time with name-count setup at 0.028s, and full predict over
1000 signatures is 2.31s. The global Arrow lookup remains useful for artifact
generation, parity fallback, or inspection. The Rust Arrow readers skip global
lookup artifacts when all selected signatures already have embedded counts, and
prefer `name_counts_index/` over `name_counts.arrow` when both are present.

The raw Arrow probes changed the format decision:

- Direct Arrow IPC reads consumed by Rust are fast.
- Arrow IPC read into Python and then rebuilt as dict/list objects is slow.
- Therefore, columnar formats are useful only when the hot path stays columnar
  across the Rust boundary.

MessagePack can still be useful as a compatibility or service transport for
dict-shaped legacy payloads, but it should no longer be the primary universal
target for table-shaped inference data.

---

## Format Selection Rationale

| Format | Use case | Rationale |
|---|---|---|
| **Arrow IPC / RecordBatch** | `FeatureBlock` tables: signatures, papers, paper authors, cluster seeds when needed, clusters when needed, name pairs, embedded name-count columns or a canonical `name_counts` lookup table | Typed, columnar, read natively by Python and Rust, avoids PyO3 traversal of nested Python dict/list objects, supports memory-mapped local files, and now powers both raw incremental and full predict direct-Rust paths. |
| **Arrow fixed-size-list or Safetensors** | SPECTER embeddings | Both avoid pickle and preserve compact `float32` storage. Arrow keeps embeddings in the same schema family as the request tables; Safetensors remains a good tensor-specific option. Choose with a benchmark before regenerating artifacts. |
| **LightGBM native text** | Trained models | Cross-language target for future Rust-native model inference. Not urgent until setup and chunk orchestration costs are reduced. |
| **Plain text or Arrow table** | `name_tuples` | Existing text is already shared; a single default variant matters more than the container format. Arrow is reasonable if bundled with a larger `FeatureBlock` artifact. |
| **MessagePack** | Compatibility transport for legacy dict-shaped payloads | Better than JSON for nested dicts, but it preserves the object shape we are trying to avoid on the hot path. Do not make it the central table-shaped inference artifact. |

### Formats Deprioritized Or Rejected

| Format | Reason |
|---|---|
| **Pickle** | Python-only and unsafe as a cross-language target. Rust should not rely on Python pickle FFI for production ingest. |
| **JSON** | Acceptable as a compatibility input and test fixture format, but not the target hot-path runtime format for large inference blocks. |
| **Arrow -> Python dict/list -> Rust** | Measured slower than the current JSON/pickle load on the h_wang fixture. This defeats the columnar advantage. |
| **MessagePack as universal target** | Improves legacy nested-object serialization but does not solve Python object materialization or typed columnar Rust ingestion. |
| **Parquet as request/runtime hot path** | Useful for analytics/offline inspection, but Arrow IPC is the simpler runtime target for local files and request bundles. Revisit only with a direct Rust benchmark and a concrete pushdown/scan use case. |
| **NPZ** | Fast in Python, but less ergonomic as a shared Rust/Python schema than Arrow or Safetensors. |

---

## Format Decision Table

| Artifact / data family | Current Python | Current Rust | Target | Notes |
|---|---|---|---|---|
| Signatures | JSON -> `ANDData` namedtuples | JSON paths or `from_dataset` PyO3 traversal | Arrow `FeatureBlock.signatures` table | Keep JSON adapter for compatibility; do not rebuild Python objects on the Arrow path. |
| Papers | JSON -> `ANDData` namedtuples | JSON paths or `from_dataset` PyO3 traversal | Arrow `papers` plus `paper_authors` tables | Split repeated authors into a child table for simpler Rust reads. Include an abstract-presence string for exact scoring because `abstract_count` is a promoted pairwise feature. |
| Clusters | JSON | Limited native use | Arrow membership table when needed | Raw incremental retrieval mostly needs seed components, not full clusters. |
| Cluster seeds | JSON require/disallow map | JSON or synced from `ANDData` | Arrow `cluster_seeds` table | Should support both require and disallow constraints. |
| ORCID constraints | JSON | Limited native use | Fold into constraint/seed tables as needed | Keep separate only if a caller needs the legacy artifact boundary. |
| `name_counts.pickle` / `name_counts_rust.json` | Pickle | Rust JSON, embedded Arrow columns, Arrow lookup, or sorted binary index | Embedded per-signature columns for hot request/block bundles; sorted `name_counts_index/` for exact fallback; Arrow `name_counts` for generation/inspection/parity | The direct Arrow path accepts a long-form Arrow table with `kind`, `name`, `count`, but cold global lookup load is too expensive for per-request use. Rust skips global lookup artifacts when embedded counts are complete and prefers the exact-verified index over the Arrow table when both are present. |
| SPECTER pickle files | Pickle | Pickle via Python FFI or tuple payloads in incumbent paths | Arrow fixed-size-list or Safetensors | Direct Arrow fixed-size-list parity is exact on bounded full-predict checks; Safetensors is still an optional future benchmark. |
| `first_k_letter_counts_from_orcid.json` | JSON | Not loaded by Rust | Regenerated JSON or folded into canonical constraints | Lower priority than inference table schemas. |
| FastText `lid.176.bin` | FastText binary | FastText binary | Keep as-is | Already cross-language enough for current needs. |
| `name_tuples` text | Text | Text or Arrow pair table | One default packaged filtered text file | Direct Arrow accepts `name_pairs` / `name_tuples` path keys for experiments, but production Arrow bundles should not duplicate aliases per dataset. The filtered text file is small enough that mmap/indexing is not currently justified. |
| LightGBM model | Pickled `Clusterer` state | Not loaded natively | LightGBM native text later | Useful only when moving model inference itself cross-language. |
| Train/val/test pairs | CSV | Not used by Rust | Keep CSV | Outside the inference hot path. |

---

## Current Divergence Map

| Area | Current divergence | Updated resolution target |
|---|---|---|
| `name_counts` artifact | Python loads `name_counts.pickle`; Rust native JSON ingest expects `name_counts_rust.json` shape; direct Arrow can consume embedded columns, `name_counts_index/`, or `name_counts.arrow`. | Use embedded per-signature columns on the hot path. Use the sorted exact-verified binary index when counts cannot be embedded. Keep Arrow as the generation/inspection/parity fallback. Delete dual-path pickle/JSON compatibility only after broader parity gates and artifact generation are settled. |
| `name_tuples` source | Python supports filtered and full files; Rust defaults to filtered; direct Arrow can consume an override path. | Collapse to one runtime default. Keep non-default variants for offline experiments only; do not create per-dataset `name_pairs.arrow` artifacts. |
| ORCID first-k counts normalization | Artifact predates current normalization and runtime has compatibility lookup. | Regenerate or fold into the canonical constraint input after the `FeatureBlock` schema stabilizes. |
| SPECTER embedding load path | Python loads pickle; Rust handles dict and tuple payloads in the incumbent path; direct Arrow uses fixed-size-list. | Keep Arrow fixed-size-list for the direct path. Benchmark Safetensors only if SPECTER read time remains material. |
| Signatures/papers object shape | Python `ANDData` uses namedtuples and many derived fields; raw Rust Arrow path uses typed columns. | Keep `ANDData` as the reference adapter. The target runtime schema is the narrow inference contract, not every `ANDData` field. |

---

## Execution Order

1. Define the `FeatureBlock` schema and explicitly mark which `ANDData`
   responsibilities are out of scope.
2. Add tiny fixture round-trips for `ANDData -> FeatureBlock -> Rust` and Arrow
   IPC -> `FeatureBlock -> Rust`.
3. Promote the raw Arrow candidate-plan schema into the same complete
   `FeatureBlock` tables. This now includes abstract presence, paper language
   fields, SPECTER fixed-size-list, and name-count support. Name aliases remain
   the single packaged filtered runtime default unless an experiment explicitly
   passes an override path.
4. Keep the public raw-only downstream scoring wrappers and the full
   `predict_from_arrow_paths(...)` endpoint as the first production endpoints.
   They avoid both full-block `ANDData` and the earlier mini-`ANDData`
   compatibility layer.
5. Keep SPECTER Arrow fixed-size-list as the direct-path default; benchmark
   Safetensors only if profiling shows SPECTER read time remains material.
6. Use embedded per-signature name-count columns in request/block Arrow bundles
   when possible. Use sorted `name_counts_index/` when counts cannot be embedded.
   Keep Arrow `name_counts` as the canonical generation/inspection/parity table,
   not as the default hot-path read.
7. Collapse `name_tuples` to one runtime default.
8. Move additional row-signal or transport setup into Rust only after
   production-like profiling shows it is still material.
9. Remove legacy pickle/JSON divergence only after dual-read parity gates are
   green on bounded h_wang and non-h_wang checks.

---

## Verification Bar

- Tiny fixtures first, then bounded h_wang and non-h_wang checks. Current
  complete full-predict evidence: a_silva 50 signatures and 1000 signatures,
  seeded a_silva 1000 signatures, no-SPECTER a_silva 1000 signatures, h_wang
  1000 signatures, seeded h_wang 1000 signatures, and no-SPECTER h_wang 1000
  signatures all have exact feature-matrix, constraint, distance, and cluster
  parity with the incumbent path. The tracked 50-signature embedded-count and
  index-count gates also have exact feature, constraint, distance, and cluster
  parity.
- Exact row counts, pair counts, candidate ids, feature matrices,
  probabilities, and normalized final link/abstain decisions.
- Stage telemetry for load/read time, `FeatureBlock` construction, Rust feature
  prep, retrieval, constraints, pair feature matrix build, LightGBM prediction,
  and final gate.
- No performance claim without before/after profile deltas.
