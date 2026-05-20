# Rust-Only Raw Block Candidate Plan

Status: design spec + implemented direct Arrow/raw scoring evidence

## Use Case

Some production requests contain one or more new query signatures plus one
random author-name block. The single-query case is the most common, but the
API must natively support N ≥ 1 queries against the same block so that
seed-side prep (feature extraction, summary aggregation, retriever build)
amortizes across the batch. The caller only needs the current promoted
incremental linker's top candidate seed clusters per query, followed by the
existing linker scoring/link-or-abstain logic.

The current `Clusterer.predict_incremental(...)` path is too expensive for this
shape because it assumes an `ANDData` object has been built for the block and a
Rust featurizer can be built or reused for that dataset. Even for a single
query, full dataset loading, paper/signature preprocessing, Rust featurizer
construction, seed-summary construction, and retrieval setup can dominate the
actual top-25 candidate search.

This spec defines a maximum-speed, minimum-memory Rust API for:

```text
raw query + raw block + seed map -> current-linker candidate plan
```

The Rust API output is the retrieved candidate plan plus the row signals needed
by downstream link/abstain scoring. A Python bridge proves that the id-based raw
candidate plan can feed existing pairwise scoring, 53-feature assembly,
promoted linker prediction, and logistic link/abstain without rerunning
retrieval. Public raw scoring wrappers now remove the incumbent full-block
`ANDData`/linker-input setup from that handoff by scoring only the query plus
retrieved candidate members.

## Baseline Measurement (Prerequisite)

Before any Rust work begins, profile the current
`Clusterer.predict_incremental(...)` path on a realistic single-query +
one-block request and report per-stage wall time and peak RSS for:

- raw payload deserialization
- `ANDData` construction for the request block
- `RustFeaturizer` build/reuse
- seed cluster summary construction
- `RustHybridCentroidRetriever` construction
- `top_k_hybrid_centroid_pair_plan(...)` execution
- downstream pairwise scoring + promoted linker + logistic gate (out of scope
  for this shortcut, but measured for context so the share of time the
  shortcut can actually remove is unambiguous)

This profile is the justification for the shortcut. It must show that the
stages this spec skips (`ANDData`, `RustFeaturizer`, summary materialization,
retriever setup) dominate end-to-end latency for this request shape. If they
do not, revise this spec — or fold the optimization into the existing entry
point — before writing Rust. The baseline numbers also set the latency and
peak-RSS targets the new API's telemetry is compared against in the parity
and acceptance tests below.

### Initial Findings (h_wang single-query, n=1 fixture)

Fixture: query `100036391-4` ("Huiliang Wang"), seed 42, h_wang block of
419,641 signatures / 37,152 seed components / 418,749 seed signatures.
Backend: Rust, n_jobs=8. Harness:
[scratch/baseline/profile_predict_incremental.py](../../scratch/baseline/profile_predict_incremental.py).
Raw report: [profile.json](../../scratch/baseline/h_wang_single_query/profile.json).

End-to-end (total 1442.7 s, peak RSS 29.6 GB):

| Stage | Seconds | % of total |
|---|---:|---:|
| Raw payload deserialization (sigs + papers + specter + seeds) | 10.3 | 0.7% |
| **`ANDData` construction** | **1141.9** | **79.1%** |
| Load production model | 1.7 | 0.1% |
| `predict_incremental` total | 286.2 | 19.8% |

Inside `predict_incremental` (286.2 s):

| Internal stage | Seconds | n | % of predict |
|---|---:|---:|---:|
| `build_incremental_linker_inputs` (total) | 116.0 | 1 | 40.5% |
| ↳ `extract_query_features` | 80.5 | 418,750 | 28.1% |
| ↳ `build_cluster_summary` | 98.8 | 37,152 | 34.5% |
| ↳ `build_rust_hybrid_centroid_retriever` | 16.2 | 1 | 5.7% |
| `build_linker_retrieval_batch_rust` (the actual top-k call) | **1.3** | 1 | **0.5%** |
| `get_rust_featurizer` | ~0 | 2 | 0.0% |
| Unaccounted (pairwise scoring + promoted linker + logistic gate) | ~169 | – | ~59% |

Implications for this spec:

- **The premise is strongly confirmed.** The skip-target stages cost
  ~1257 s combined (`ANDData` 1142 s + linker-inputs build 116 s) while the
  *actual* top-25 retrieval is 1.3 s. The "heavy setup dominates the search"
  pattern the spec describes is real and large.
- **`ANDData` is the elephant**, not one cost among several. ~79% of wall
  time and ~18 GB of RSS sit in `ANDData` construction alone. The single
  biggest win from skipping it dwarfs every other line item.
- **`RustFeaturizer` build is essentially free** (0.0001 s, two cache-hit
  calls). The Goal "Avoid building a full `RustFeaturizer`" is not
  load-bearing for this request shape and could be downgraded to a
  memory/cleanup concern rather than a latency win.
- **~169 s of unaccounted time inside `predict_incremental`** is the
  pairwise scoring + promoted linker + logistic gate, i.e. the work Phase 2
  of this spec defers. Even a perfect Phase 1 leaves that ~169 s in place,
  so end-to-end goes from ~1443 s → roughly 169 s + thin payload parse
  (~10 s) + new-API setup bounded by top-k component size. That is an
  ~8× wall-clock speedup *before* doing Phase 2.

Caveat: single fixture (one query, one block). Before locking the baseline
in, re-run on at least one additional random h_wang query plus one smaller
block (e.g. `a_silva`, `s_park`) to confirm the qualitative shape holds.
The conclusion that `ANDData` dominates and pure retrieval is sub-2 s is
very unlikely to flip, but the exact ratios will move.

### Forward-Looking Baseline (Sinonym Off, n_jobs=20)

Sinonym overwrite is expected to be disabled by default going forward, so the
most relevant baseline is the same h_wang single-query fixture with Sinonym off,
`n_jobs=20`, release Rust, and an explicit large RAM budget
(`total_ram_bytes=1000000000000`) to avoid memory-budget throttling. Harness:
[scratch/baseline/profile_predict_incremental.py](../../scratch/baseline/profile_predict_incremental.py).
Raw report:
[profile.json](../../scratch/baseline/h_wang_single_query/njobs20_no_sinonym_unbounded_20260519/profile.json).

End-to-end (total 335.5 s, peak RSS 29.6 GB):

| Stage | Seconds | % of total |
|---|---:|---:|
| Raw payload deserialization (sigs + papers + specter + seeds) | 8.8 | 2.6% |
| **`ANDData` construction** | **202.7** | **60.4%** |
| Load production model | 1.6 | 0.5% |
| `predict_incremental` total | 120.3 | 35.9% |

Inside `predict_incremental` (120.3 s):

| Internal stage | Seconds | n | % of predict |
|---|---:|---:|---:|
| `build_incremental_linker_inputs` (total) | 95.5 | 1 | 79.3% |
| -> `extract_query_features` | 74.8 | 418,750 | 62.1% |
| -> `build_cluster_summary` | 92.0 | 37,152 | 76.4% |
| -> `build_rust_hybrid_centroid_retriever` | 2.6 | 1 | 2.1% |
| `build_linker_retrieval_batch_rust` (the actual top-k call) | **0.2** | 1 | **0.2%** |
| `get_rust_featurizer` | ~0 | 2 | 0.0% |
| Unaccounted (pairwise scoring + promoted linker + logistic gate) | ~24.7 | - | ~20.5% |

Updated implications:

- **The shortcut premise still holds with Sinonym off.** The skip-target stages
  are no longer 20+ minutes, but they still dominate: `ANDData` construction
  plus linker input construction cost about 298 s while actual retrieval is
  0.2 s.
- **The most urgent Rust work is feature/summary construction, not nearest-neighbor
  search.** Retrieval is already effectively free for the single-query request
  shape; query feature extraction and seed cluster summary construction are the
  hot stages inside `predict_incremental`.
- **End-to-end scoring is smaller than originally measured under this
  configuration.** The remaining non-setup work is roughly 25 s, so a retrieval
  candidate-plan shortcut can plausibly make the whole request fast enough to
  be useful before an all-Rust scoring port.
- **File loading is not the biggest bottleneck, but it is worth fixing if it is
  cheap.** JSON/pickle deserialization is about 9 s here. That is small beside
  `ANDData`, but large enough to matter once the raw shortcut removes the
  preprocessing stages.

### Arrow IPC Load Probe

A scratch Arrow IPC probe was run on the same full h_wang fixture, including
419,641 signatures, 412,708 papers, 5,788,327 paper-author rows, 418,749
cluster-seed rows, and 349,430 SPECTER rows. Python conversion report:
[report.json](../../scratch/baseline/h_wang_single_query/arrow_full_specter/report.json).
Rust IPC read report:
[rust_read_report.json](../../scratch/baseline/h_wang_single_query/arrow_full_specter/rust_read_report.json).

| Path | Seconds |
|---|---:|
| Current JSON + pickle load | 8.9 |
| Python Arrow IPC table read | 0.2 |
| Rust Arrow IPC table read | 0.5 |
| Arrow tables converted back to Python objects | 164.7 |

Implication: **Arrow is a good hot-path format only when Rust consumes Arrow
columns directly.** Reading Arrow and then rebuilding Python dict/list objects
defeats the purpose and is slower than the current JSON/pickle load. Arrow
should therefore be validated as an input adapter to the Rust typed structs, not
as an early Python `ANDData` materialization format.

### Direct Rust Arrow Candidate-Plan Probe

A first retrieval-only prototype now exists in
[`s2and_rust/src/lib.rs`](../../s2and_rust/src/lib.rs):
`raw_block_query_candidate_plan_arrow(...)`. It reads the Arrow IPC signature,
paper, paper-author, cluster-seed, and SPECTER tables directly in Rust; builds
the same retrieval query and seed-summary structs used by
`RustHybridCentroidRetriever`; and returns the flat pair-plan schema plus
signature-id strings and telemetry.

Focused regression tests:
[`tests/test_raw_block_candidate_plan_arrow.py`](../../tests/test_raw_block_candidate_plan_arrow.py).
They now cover:

- tiny Arrow IPC parity against the existing Rust retriever;
- ORCID override behavior;
- multi-query batching with `query_view="auto"` and SPECTER exemplars;
- query-as-seed exclusion plus missing optional metadata.

Full h_wang probe command:

```powershell
uv run python scratch\baseline\run_raw_arrow_candidate_plan.py `
  --fixture-dir scratch\baseline\h_wang_single_query\arrow_full_specter `
  --query-signature-id 100036391-4 `
  --output-json scratch\baseline\h_wang_single_query\arrow_full_specter\raw_candidate_plan_njobs20_orcid_on_all25_20260519.json `
  --top-k 25 `
  --query-view auto `
  --n-jobs 20 `
  --orcid-enabled `
  --max-exemplars 4
```

Full h_wang results:
[raw_candidate_plan_njobs20_orcid_on_all25_20260519.json](../../scratch/baseline/h_wang_single_query/arrow_full_specter/raw_candidate_plan_njobs20_orcid_on_all25_20260519.json).

| Stage | Seconds |
|---|---:|
| Read signatures Arrow | 0.48 |
| Read papers Arrow | 0.30 |
| Read paper-authors Arrow | 1.40 |
| Read cluster seeds Arrow | 0.13 |
| Read SPECTER Arrow | 1.04 |
| Build text/unidecode context | 0.75 |
| Build query/seed features | 0.96 |
| Build seed summaries + retriever state | 2.84 |
| Retrieval pair plan | 0.14 |
| **Rust-reported total** | **8.47** |
| **Harness wall time** | **11.92** |

Counts: 419,641 signatures, 412,708 papers, 5,788,327 paper-author rows
(grouped under 412,708 papers), 418,749 seed signatures, 37,152 seed
components, and 349,430 SPECTER rows. The raw path returned 25 candidate rows
and 318 candidate pairs.

Parity check against the current Python-built retrieval path:
[raw_candidate_plan_compare_current_orcid_on_all25_20260519.json](../../scratch/baseline/h_wang_single_query/arrow_full_specter/raw_candidate_plan_compare_current_orcid_on_all25_20260519.json).

| Check | Result |
|---|---:|
| Current `ANDData` construction | 199.1 s |
| Current `build_incremental_linker_inputs` | 93.4 s |
| Current Rust retrieval batch | 0.3 s |
| Current candidate rows / pairs | 25 / 318 |
| Raw candidate rows / pairs | 25 / 318 |
| Top-25 component order parity | exact |
| Top-25 retrieval score max abs diff | 0.0 |

Deeper multi-query h_wang parity check:
[raw_candidate_plan_njobs20_orcid_on_3query_deep_20260520.json](../../scratch/baseline/h_wang_single_query/arrow_full_specter/raw_candidate_plan_njobs20_orcid_on_3query_deep_20260520.json)
and
[raw_candidate_plan_compare_current_orcid_on_3query_deep_20260520.json](../../scratch/baseline/h_wang_single_query/arrow_full_specter/raw_candidate_plan_compare_current_orcid_on_3query_deep_20260520.json).

Queries: `100036391-4`, `104839473-2`, and `115155813-0`; `query_view="auto"`;
ORCID enabled; `n_jobs=20`; top-k 25.

| Check | Result |
|---|---:|
| Rust Arrow candidate rows / pairs | 75 / 1213 |
| Current candidate rows / pairs | 75 / 1213 |
| Row count parity | exact |
| Pair count parity | exact |
| Component order parity | exact |
| Retrieval rank parity | exact |
| Row query-offset parity | exact |
| Row component-size parity | exact |
| Pair row-index parity | exact |
| Left signature-id parity | exact |
| Right signature-id parity | exact |
| Retrieval score max abs diff | 0.0 |
| Row-signal max abs diff | 0.0 for ORCID, middle, affiliation, coauthor, venue, year, title, SPECTER centroid, and SPECTER exemplar signals |

Current-path timings for this deeper compare:

| Stage | Seconds |
|---|---:|
| Current `ANDData` construction | 198.3 |
| Current `build_incremental_linker_inputs` | 93.4 |
| Current Rust retrieval batch | 0.4 |

The matching pair signature ids matter: this does not only prove that the same
75 component rows were ranked the same way. It also proves that the flat
query-to-member pair plan handed to downstream scoring is identical for this
three-query full-block request.

No-SPECTER h_wang parity check:
[raw_candidate_plan_njobs20_orcid_on_all25_no_specter_20260520.json](../../scratch/baseline/h_wang_single_query/arrow_full_specter/raw_candidate_plan_njobs20_orcid_on_all25_no_specter_20260520.json)
and
[raw_candidate_plan_compare_current_orcid_on_all25_no_specter_20260520.json](../../scratch/baseline/h_wang_single_query/arrow_full_specter/raw_candidate_plan_compare_current_orcid_on_all25_no_specter_20260520.json).

Query: `100036391-4`; `query_view="auto"`; ORCID enabled; `n_jobs=20`; top-k
25; SPECTER omitted from both the Rust Arrow path and the current Python-built
path.

| Check | Result |
|---|---:|
| Rust Arrow candidate rows / pairs | 25 / 228 |
| Current candidate rows / pairs | 25 / 228 |
| Row count parity | exact |
| Pair count parity | exact |
| Component order parity | exact |
| Retrieval rank parity | exact |
| Row query-offset parity | exact |
| Row component-size parity | exact |
| Pair signature-plan parity | exact |
| Retrieval score max abs diff | 0.0 |
| Row-signal max abs diff | 0.0 for ORCID, middle, affiliation, coauthor, venue, year, title, SPECTER centroid, and SPECTER exemplar signals |
| Rust Arrow `specter_count` | 0 |

Current-path timings for this no-SPECTER compare:

| Stage | Seconds |
|---|---:|
| Current `ANDData` construction | 205.1 |
| Current `build_incremental_linker_inputs` | 84.8 |
| Current Rust retrieval batch | 0.2 |

Non-h_wang block parity check:
[raw_candidate_plan_njobs20_orcid_on_all25_20260520.json](../../scratch/baseline/a_silva_single_query/arrow_full_specter/raw_candidate_plan_njobs20_orcid_on_all25_20260520.json)
and
[raw_candidate_plan_compare_current_orcid_on_all25_20260520.json](../../scratch/baseline/a_silva_single_query/arrow_full_specter/raw_candidate_plan_compare_current_orcid_on_all25_20260520.json).

Fixture: `a_silva`, query `10027229-7` ("Arlindo da Silva"), 88,933
signatures, 88,552 seed signatures, 9,687 seed components, and 67,003 SPECTER
rows. `query_view="auto"` resolved to `full`; ORCID enabled; `n_jobs=20`;
top-k 25.

| Check | Result |
|---|---:|
| Rust Arrow candidate rows / pairs | 25 / 570 |
| Current candidate rows / pairs | 25 / 570 |
| Row count parity | exact |
| Pair count parity | exact |
| Component order parity | exact |
| Retrieval rank parity | exact |
| Row query-offset parity | exact |
| Row component-size parity | exact |
| Pair signature-plan parity | exact |
| Retrieval score max abs diff | 0.0 |

Rust Arrow timings for this a_silva check:

| Stage | Seconds |
|---|---:|
| Rust-reported total | 1.66 |
| Harness wall time | 2.03 |
| Build seed summaries + retriever state | 0.51 |
| Retrieval pair plan | 0.03 |

Current-path timings for this a_silva compare:

| Stage | Seconds |
|---|---:|
| Current `ANDData` construction | 49.8 |
| Current `build_incremental_linker_inputs` | 16.6 |
| Current Rust retrieval batch | 0.08 |

### Downstream Scoring Bridge Probe

A Python bridge now converts the raw id-based candidate plan into the numeric
`LinkerRetrievalBatch` expected by the existing scoring runtime. This bridge
maps request-local query offsets through `query_signature_ids` and maps
`left_signature_ids` / `right_signature_ids` through the active featurizer's
signature-id order. It does not rerun retrieval.

The first comparison harness runs the current production private path and the
raw-plan bridge path against the same `ANDData`, seed setup, featurizer,
constraint backend, pairwise model, promoted linker, and logistic gate. This
proves downstream parity for the candidate-plan handoff. The public wrapper now
builds a `FeatureBlock` directly from raw payloads, or consumes raw Arrow inputs
directly through `RustFeaturizer.from_arrow_paths(...)`. The raw Arrow plan now
carries the remaining row signals itself, so that path no longer builds a
Python signal `FeatureBlock`. Both wrappers then run the same Rust
pairwise-feature and constraint-label kernels. That proves the raw scoring path
can skip full-block `ANDData`, full linker-input construction, full constraint
backend setup, and the earlier mini-`ANDData` compatibility layer while
preserving exact scoring semantics.

Non-h_wang a_silva link/abstain bridge check:
[raw_candidate_plan_njobs20_orcid_on_all25_link_abstain_20260520.json](../../scratch/baseline/a_silva_single_query/arrow_full_specter/raw_candidate_plan_njobs20_orcid_on_all25_link_abstain_20260520.json)
and
[raw_candidate_plan_compare_link_abstain_orcid_on_all25_20260520.json](../../scratch/baseline/a_silva_single_query/arrow_full_specter/raw_candidate_plan_compare_link_abstain_orcid_on_all25_20260520.json).

| Check | Result |
|---|---:|
| Current candidate rows / pairs | 25 / 570 |
| Raw bridge candidate rows / pairs | 25 / 570 |
| Linked signature clusters | exact |
| Final link/abstain decisions | exact |
| Probability max abs diff | 0.0 |
| 53-feature matrix max abs diff | 0.0 |
| Current retrieval + scoring | 0.46 s |
| Raw-plan bridge scoring | 0.33 s |
| Shared `ANDData` setup still paid | 50.3 s |
| Shared linker inputs for current path + extra signals | 17.4 s |

Full h_wang link/abstain bridge check:
[raw_candidate_plan_njobs20_orcid_on_all25_link_abstain_20260520.json](../../scratch/baseline/h_wang_single_query/arrow_full_specter/raw_candidate_plan_njobs20_orcid_on_all25_link_abstain_20260520.json)
and
[raw_candidate_plan_compare_link_abstain_orcid_on_all25_20260520.json](../../scratch/baseline/h_wang_single_query/arrow_full_specter/raw_candidate_plan_compare_link_abstain_orcid_on_all25_20260520.json).

| Check | Result |
|---|---:|
| Current candidate rows / pairs | 25 / 318 |
| Raw bridge candidate rows / pairs | 25 / 318 |
| Linked signature clusters | exact |
| Final link/abstain decisions | exact |
| Probability max abs diff | 0.0 |
| 53-feature matrix max abs diff | 0.0 |
| Current retrieval + scoring | 2.27 s |
| Raw-plan bridge scoring | 1.34 s |
| Shared `ANDData` setup still paid | 220.3 s |
| Shared linker inputs for current path + extra signals | 103.5 s |

Mini-`FeatureBlock` raw scoring wrapper checks:

a_silva:
[raw_candidate_plan_compare_link_abstain_mini_feature_block_20260520.json](../../scratch/baseline/a_silva_single_query/arrow_full_specter/raw_candidate_plan_compare_link_abstain_mini_feature_block_20260520.json).

| Check | Result |
|---|---:|
| Current candidate rows / pairs | 25 / 570 |
| Raw mini candidate rows / pairs | 25 / 570 |
| Linked signature clusters | exact |
| Normalized final link/abstain decisions | exact |
| Probability max abs diff | 0.0 |
| 53-feature matrix max abs diff | 0.0 |
| Current full `ANDData` setup | 50.88 s |
| Current seed/featurizer/constraint setup | 3.86 s |
| Current linker-input + extra-signal setup | 17.50 s |
| Current retrieval + scoring | 0.40 s |
| Raw `FeatureBlock` build | 0.098 s |
| Raw mini scoring wrapper | 0.614 s |
| Mini `ANDData` inside wrapper | 0.262 s |
| Mini Rust featurizer inside wrapper | 0.061 s |
| Current vs raw pairwise feature seconds | 0.042 / 0.00067 |
| Current vs raw constraint seconds | 0.042 / 0.00062 |

h_wang:
[raw_candidate_plan_compare_link_abstain_mini_feature_block_20260520.json](../../scratch/baseline/h_wang_single_query/arrow_full_specter/raw_candidate_plan_compare_link_abstain_mini_feature_block_20260520.json).

| Check | Result |
|---|---:|
| Current candidate rows / pairs | 25 / 318 |
| Raw mini candidate rows / pairs | 25 / 318 |
| Linked signature clusters | exact |
| Normalized final link/abstain decisions | exact |
| Probability max abs diff | 0.0 |
| 53-feature matrix max abs diff | 0.0 |
| Current full `ANDData` setup | 209.70 s |
| Current seed/featurizer/constraint setup | 21.90 s |
| Current linker-input + extra-signal setup | 97.98 s |
| Current retrieval + scoring | 1.84 s |
| Raw `FeatureBlock` build | 0.256 s |
| Raw mini scoring wrapper | 0.429 s |
| Mini `ANDData` inside wrapper | 0.173 s |
| Mini Rust featurizer inside wrapper | 0.039 s |
| Current vs raw pairwise feature seconds | 0.209 / 0.0013 |
| Current vs raw constraint seconds | 0.208 / 0.00050 |

`decisions_exact_match` remains false in those JSON reports only because the
numeric query signature index is local to the active featurizer. The normalized
decision payload maps that index back to `query_signature_id`, and that semantic
decision payload matches exactly.

### Complete Arrow Full-Predict Parity

The direct Arrow path has also been checked outside the single-query raw
incremental shape. The complete-Arrow harness builds a bounded incumbent
`ANDData`, writes typed Arrow tables from the same bounded payload, attaches
name-count artifacts, uses the packaged filtered name-alias default, builds
`RustFeaturizer.from_arrow_paths(...)`, and compares:

- the full 39-feature matrix for every upper-triangle pair;
- NaN placement in the feature matrix;
- upper-triangle constraint labels;
- distance matrices;
- final cluster assignments.

All runs used `n_jobs=20`, `total_ram_bytes=1000000000000`, and 1000-signature
blocks (499,500 pairs), except the earlier 50-signature smoke check.

| Fixture | Variant | Report | Result |
|---|---|---|---|
| a_silva | complete schema, SPECTER | [report](../../scratch/baseline/a_silva_single_query/complete_arrow_subset1000_arrow_name_artifacts/compare_full_predict_complete_arrow_subset1000_featurecheck_20260520.json) | exact features, constraints, distances, clusters |
| a_silva | seeded, 997 required assignments | [report](../../scratch/baseline/a_silva_single_query/complete_arrow_subset1000_seeded_arrow_name_artifacts/compare_full_predict_complete_arrow_subset1000_seeded_featurecheck_20260520.json) | exact features, constraints, distances, clusters |
| a_silva | no SPECTER | [report](../../scratch/baseline/a_silva_single_query/complete_arrow_subset1000_no_specter_arrow_name_artifacts/compare_full_predict_complete_arrow_subset1000_no_specter_featurecheck_20260520.json) | exact features, constraints, distances, clusters |
| h_wang | complete schema, SPECTER | [report](../../scratch/baseline/h_wang_single_query/complete_arrow_subset1000_arrow_name_artifacts/compare_full_predict_complete_arrow_subset1000_featurecheck_20260520.json) | exact features, constraints, distances, clusters |
| h_wang | seeded, 999 required assignments | [report](../../scratch/baseline/h_wang_single_query/complete_arrow_subset1000_seeded_arrow_name_artifacts/compare_full_predict_complete_arrow_subset1000_seeded_featurecheck_20260520.json) | exact features, constraints, distances, clusters |
| h_wang | no SPECTER | [report](../../scratch/baseline/h_wang_single_query/complete_arrow_subset1000_no_specter_arrow_name_artifacts/compare_full_predict_complete_arrow_subset1000_no_specter_featurecheck_20260520.json) | exact features, constraints, distances, clusters |

The remaining explicit boundary is `reference_features`: production models do
not use them, and `Clusterer.predict_from_arrow_paths(...)` now fails fast if a
model requests them. Supporting them in the Arrow path would require carrying
the citation-derived reference artifacts as part of the narrow inference schema.

Updated implication: **the central idea is good, and the retrieval-to-scoring
handoff is now proven on the checked fixtures.** The direct Arrow Rust path
reproduces the current top-25 retrieval candidate plan exactly on h_wang
single-query, h_wang three-query, h_wang without SPECTER, and a_silva
non-h_wang block checks while cutting retrieval-input construction by one to two
orders of magnitude. The raw candidate plan also drives the existing downstream
pairwise scoring, promoted linker, and link/abstain gate with exact parity on
both h_wang and a_silva single-query checks. The remaining unproved part is no
longer retrieval or scoring parity, nor full-block setup removal. The public raw
endpoint now removes the mini-`ANDData` compatibility bridge as well:
`RustFeaturizer.from_feature_block(...)` builds the existing Rust featurizer
state directly from the narrow `FeatureBlock` contract. The open optimization is
whether more row-signal or transport setup should move into Rust after profiling
production-like request traffic.

The profiling evidence says direct `FeatureBlock` scoring is a useful cleanup
but not the dominant win. In the checked fixtures, the earlier mini `ANDData`
plus mini Rust featurizer setup was 0.323 s for a_silva and 0.212 s for h_wang,
while the full-block setup removed by the wrapper was 72.24 s and 329.58 s
respectively. Removing the mini layer is still the right simplification now that
we have the boundary, but the next work should be driven by a fresh wrapper
profile rather than by intuition.

## Strategic Direction

The shortcut's unique contribution is narrow: **skip full `ANDData`
construction for raw small-N requests**. The rest of the work should be built as
shared S2AND inference infrastructure wherever practical, but that must not mean
reimplementing all of `ANDData` in Rust.

`ANDData` is intentionally broad. It owns legacy artifact loading, train/eval
splits, pair sampling, Sinonym mutation, name-count compatibility semantics,
SPECTER shaping, paper/signature preprocessing, cluster bookkeeping, and many
debugging/reproducibility behaviors. The Rust work should carve out the smaller
inference contract hidden inside it, tentatively called `FeatureBlock`.

Build the reusable part around a Rust-native inference core:

```text
raw request or ANDData adapter
  -> FeatureBlock
  -> typed Rust inference structs
  -> query feature extraction
  -> seed component summaries
  -> temporary hybrid centroid retriever
  -> candidate component plan
  -> optional mini-block pair scoring inputs
```

Then expose two adapters into the same contract:

- a raw-payload adapter for the small-N production request shape, which avoids
  constructing `ANDData`;
- an incumbent `ANDData -> FeatureBlock` adapter so existing callers can use the
  same Rust feature/summary/retrieval core and so `ANDData` remains the parity
  oracle while the Rust path is hardened.

This avoids building a parallel one-off implementation. The raw path gets the
large latency win from skipping `ANDData`, while the existing path can still
benefit from the Rust port of `extract_query_features(...)` and
`build_cluster_summary(...)`.

Explicitly out of scope for the Rust shortcut:

- training/validation/test split construction;
- pair sampling policy;
- legacy cluster artifact migration;
- Sinonym overwrite execution, beyond accepting already-mutated input when a
  caller opts into that non-default mode;
- reference-feature generation unless a future model requires it;
- general-purpose `ANDData` object lifecycle or mutation semantics.

Treat these as separate decisions:

- **Raw shortcut:** use it when the caller already has one or more query
  signatures plus one raw block payload and does not otherwise need a full
  `ANDData`.
- **Faster `ANDData`:** investigate and optimize as a general S2AND project.
  The baseline shows this is the biggest incumbent-path lever, but the raw
  shortcut should not wait for a lazy/incremental `ANDData` redesign.
- **Post-retrieval scoring:** keep using the existing `LinkerCandidateBatch`
  scoring path for now; the bridge has exact parity. Full-block
  `ANDData`/linker-input setup is now removable through the mini-`FeatureBlock`
  wrapper, and the wrapper now builds the Rust featurizer directly from
  `FeatureBlock`. The next scoring task is production-traffic profiling to
  decide whether more row-signal or transport setup should move into Rust.
- **Arrow IPC:** make direct Rust Arrow consumption an early adapter target,
  because the full-block Rust read probe is already sub-second. Keep the
  JSON-shaped adapter for compatibility and parity tests, but do not treat
  Arrow-to-Python materialization as a performance path.

## ASAP Work Items

1. **Freeze the `FeatureBlock` boundary before adding more Rust surface.**
   List required retrieval/scoring fields, optional precomputed fields, schema
   versioning, and the `ANDData` responsibilities that are intentionally out of
   scope. This is the guardrail against rebuilding all of `ANDData` in Rust.
   Initial Python boundary and tests live in
   [`s2and/incremental_linking/feature_block.py`](../../s2and/incremental_linking/feature_block.py)
   and [`tests/test_feature_block.py`](../../tests/test_feature_block.py).

2. **Profile the public raw-only downstream scoring wrappers.**
   The public wrappers now build a mini `FeatureBlock` from the raw candidate
   plan or raw payloads, build `RustFeaturizer` state directly from
   `FeatureBlock`, map raw ids into the Rust featurizer order, and reuse the
   existing Rust pairwise-feature and constraint-label kernels. The previous
   mini-`ANDData` proof had exact feature matrix, probability,
   normalized-decision, linked-cluster, candidate-row, and pair-count parity on
   a_silva and h_wang. The production work is now live/request-shaped profiling
   and deciding whether more row-signal or transport setup should move into
   Rust.

3. **Promote the retrieval and downstream parity evidence into stable gates.**
   The local tests and scratch full-block checks now cover missing metadata,
   query-as-seed exclusion, multi-query batching, ORCID fanout, SPECTER
   operation, no-SPECTER operation, a non-h_wang block, and exact downstream
   link/abstain parity on h_wang plus a_silva. Keep the acceptance target as
   exact candidate ids, pair ids, row signals, retrieval scores, probabilities,
   feature matrices, and final decisions unless a deliberate scoring-policy
   change is approved. Convert the scratch commands that should be rerun
   regularly into a bounded pytest or documented CI/manual gate.

4. **Harden the direct Arrow IPC API boundary.**
   Promote the scratch Arrow schema into the `FeatureBlock` schema, add explicit
   schema validation errors, and decide whether the public function is
   Arrow-only or a small dispatcher over Arrow plus JSON compatibility inputs.

5. **Profile before moving more scoring internals.**
   Full-block `ANDData`, full featurizer order, constraint backend setup, and
   current linker-input construction are no longer required by the raw scoring
   wrapper. The mini-`ANDData` compatibility bridge has been replaced by
   direct `RustFeaturizer.from_feature_block(...)` construction. Move more
   row-signal or scoring orchestration into Rust only if production request
   profiling still shows it is material.

6. **Scope `ANDData` fast/lazy construction independently.**
   Measure `ANDData` initialization sub-stages first: JSON load, namedtuple
   materialization, paper preprocessing, signature preprocessing, Sinonym,
   specter filtering, name counts, and block/signature indexing. Optimize or
   defer the expensive pieces only after the measurements identify them.

7. **Only then consider a broader Rust scoring port.**
   The retrieval candidate plan is fast, and the bridge checks show existing
   scoring/gating is near 1-2 s once incumbent setup has already happened. Port
   more scoring logic to Rust only after the raw-only wrapper removes `ANDData`
   and linker-input setup and profiling still shows Python orchestration is
   material.

## Goals

- Match the promoted production linker's retrieval semantics: use `query_view="full"`
  when the query has a multi-letter first name, otherwise `query_view="initial_only"`.
- Use ORCID in production retrieval whenever it is present. Disable ORCID only for
  labeled-dataset calibration/evaluation runs where ORCID would leak labels or
  make the frozen benchmark incomparable.
- Treat ORCID matches as a production override: if the query ORCID matches any
  seed component, retrieve all matching components, skip middle/year hard
  filters for that query, exclude non-matching components, and force a link to
  one of the ORCID-matching components rather than abstaining.
- Avoid constructing `ANDData` for the full request block.
- Avoid requiring a full `RustFeaturizer` when the caller does not already have
  an `ANDData`; this is primarily a memory/API-boundary cleanup, not the
  measured latency lever for the current fixture.
- Avoid Python `QueryFeatures` and `ClusterSummary` object materialization by
  moving equivalent feature extraction and seed summary construction into the
  shared Rust inference core.
- Keep request-time memory proportional to the incoming block and the returned
  top candidates.
- Require no global retrieval service, persistent per-block index, or corpus-wide
  infra.
- Return enough information for a follow-up scoring wrapper to construct a
  mini-dataset and run the existing downstream scoring path without rerunning
  retrieval. The current proof bridge does this against the incumbent featurizer
  order; the production wrapper should use a mini-dataset or raw-slice order.

## Non-Goals

- This is not a global ANN service or cross-block index.
- This does not guarantee identical results to the full end-to-end linker if the
  input omits fields that the current retriever uses.
- The Rust candidate-plan API itself does not return a final cluster assignment;
  final link/abstain remains a downstream bridge/wrapper responsibility.
- This does not replace the promoted linker model, pairwise model, or logistic
  gate.
- This does not load millions of blocks or maintain a live index across random
  block arrivals.

## Existing Retrieval Semantics To Preserve

The Python promoted path currently:

1. Builds one masked query feature row from the query signature.
2. Groups seed signatures by cluster/component.
3. Builds one cluster summary per seed component.
4. Constructs `RustHybridCentroidRetriever`.
5. Calls `top_k_hybrid_centroid_pair_plan(...)`.
6. Receives candidate component rows plus a query-to-member pair plan.

The Rust-only shortcut should preserve this shape. "Top 25 nearest neighbors"
means top 25 candidate seed clusters/components under the current hybrid centroid
retriever, not top 25 individual signatures.

## Proposed Rust API

Expose a PyO3 function from `s2and_rust`:

```python
plan = s2and_rust.raw_block_query_candidate_plan(
    query_signatures=query_signatures,   # dict[sig_id, payload]; len >= 1
    query_papers=query_papers,            # dict[paper_id, payload]; missing papers use current missing-metadata semantics
    block_signatures=block_signatures,
    block_papers=block_papers,
    cluster_seeds_require=cluster_seeds_require,
    specter_embeddings=specter_embeddings,
    top_k=25,
    query_view="auto",
    orcid_enabled=True,
    num_threads=8,
)
```

The API is intrinsically a batch: `query_signatures` is always a mapping (or
columnar table in the Arrow shape), even when it contains a single entry.
Seed-side work (feature extraction, summary aggregation, retriever build) runs
once per request and is shared across every query in the batch — the
single-query call pays the same prep cost as a 10-query call, so callers that
have multiple queries against the same block should send them together.

The Python wrapper should be intentionally thin. It should validate required
top-level keys and pass raw mappings/lists through to Rust without building
`ANDData`.

### Input Contract

`query_signatures`
: Mapping from query signature id to raw signature payload, using the same
  field names as S2AND signatures JSON. Must contain at least one entry. The
  batch order is the iteration order of the mapping; output arrays are indexed
  by this order via `row_query_offsets`.

`query_papers`
: Mapping from paper id to raw paper payload, ideally covering every
  `query_signatures[i]["paper_id"]`. May share entries with `block_papers`
  when a query's paper is already in the block. If a query paper is missing,
  Rust must preserve the current promoted retriever's explicit missing-metadata
  behavior and report telemetry rather than raising.

`block_signatures`
: Raw signatures for candidate seed signatures in the current block. It may
  include any of the query signatures, but Rust must exclude each query from
  candidate seed summaries unless it appears in `cluster_seeds_require`.

`block_papers`
: Raw papers referenced by `block_signatures` and by queries whose papers are
  not in `query_papers`. Missing seed papers are represented with the current
  retriever's missing year/title/venue/author-list semantics and telemetry,
  rather than a hard failure.

`cluster_seeds_require`
: Mapping from seed signature id to seed cluster/component id. Only signatures
  in this mapping are eligible candidate members.

`specter_embeddings`
: Optional mapping from paper id to vector. If omitted or missing for a paper,
  the same missing-specter behavior as the current retriever must be used.

`top_k`
: Number of candidate components to return. Default is 25 and should match the
  incremental linker artifact metadata.

`query_view`
: Support `"auto"`, `"full"`, and `"initial_only"`. `"auto"` is the production
  default and is resolved per-query: preserve the normalized full first name
  and middle-name evidence when the query has a multi-letter first name;
  otherwise mask to the first initial. `"full"` and `"initial_only"` apply
  uniformly to every query in the batch.

`orcid_enabled`
: Boolean retrieval option. Production must pass `true` so query ORCID and seed
  component ORCIDs participate in retrieval. Labeled calibration/evaluation runs
  should pass `false`. When enabled and at least one seed component contains the
  query ORCID, ORCID overrides normal retrieval pruning: the returned candidate
  set is all matching components, not the top-k scoring subset.

`num_threads`
: Thread count for block summarization and retrieval.

## Ideal Input Data Formats

The shared runtime format should be **Apache Arrow IPC**. Use Arrow IPC/Feather
for durable local files and Arrow `RecordBatch`/table objects for request-local
payloads. This should become the common schema family for the narrow
`FeatureBlock` inference contract, Rust runtime readers, and the raw-block
candidate-plan shortcut. `ANDData` can derive or validate a `FeatureBlock`, but
the Arrow schema should not attempt to serialize every `ANDData` field or
mutation behavior.

The fastest path should avoid parsing large nested Python dictionaries inside
Rust for every request. The logical contract above can be exposed first for
compatibility, but the target production input is a compact request-local,
columnar Arrow payload.

Do not validate Arrow performance by reading Arrow into Python objects and then
feeding those objects through the JSON-shaped compatibility adapter. The h_wang
probe showed that table reads are sub-second, while Arrow-to-Python
materialization is much slower than the current JSON/pickle load. Arrow's value
comes from keeping the data columnar across the Rust boundary.

### Shared Runtime Format Recommendation

Use one Arrow IPC file or stream per logical table:

| Data | Recommended format | Shape |
|---|---|---|
| Signatures | Arrow IPC table | One row per signature, primitive/string/list columns |
| Papers | Arrow IPC tables | `papers` table plus `paper_authors` child table |
| Specter embeddings | Arrow IPC table | `paper_id` plus fixed-size `float32` vector |
| Cluster seeds | Arrow IPC table | `signature_id`, `cluster_id`, optional constraint type |
| Clusters | Arrow IPC table | One `cluster_id`, `signature_id` member row per membership |
| Name counts | Arrow IPC table | `kind`, `name`, `count` |
| Name pairs | Arrow IPC table | `name_1`, `name_2` |

Parquet can remain useful for analytics and offline data inspection, but it
should not be the hot request/runtime format. Protobuf, MessagePack, and CBOR are
acceptable wire formats only if a service boundary requires them; Rust should
convert them into the same typed columnar structs before retrieval. JSON and
pickle should be compatibility inputs, not target runtime formats.

Why Arrow IPC:

- Python and Rust can read the same typed buffers.
- Uncompressed Arrow IPC can be memory-mapped for local runtime use.
- Columns avoid per-row Python dict/list traversal across PyO3.
- Nullable primitive columns and list columns map cleanly to current S2AND data.
- Specter embeddings can be stored as contiguous `float32` vectors.
- The same schemas can feed the raw path and an `ANDData -> FeatureBlock`
  parity adapter without making Rust own full `ANDData` semantics.

### Preferred Request Shape

Use one query *table* (one row per query) plus one block payload split into
typed arrays:

```text
RawBlockQueryRequest
  query_signature_ids: string array        # length = N_queries (>= 1)
  query_paper_ids: string array            # aligned with query_signature_ids
  query_signature_columns: SignatureColumns  # one row per query
  query_paper_columns: PaperColumns          # one row per distinct query paper
  seed_signature_ids: string array
  seed_cluster_ids: string/int array
  seed_paper_ids: string array
  signature_columns: SignatureColumns       # one row per seed signature
  paper_columns: PaperColumns               # block papers
  specter: optional EmbeddingColumns
```

The arrays should be aligned by row position. For example,
`seed_signature_ids[i]`, `seed_cluster_ids[i]`, and `seed_paper_ids[i]` describe
the same seed signature; `query_signature_ids[j]` and `query_paper_ids[j]`
describe the j-th query. Single-query callers send arrays of length 1.

For Arrow IPC, this request shape is a bundle of `RecordBatch` objects: one
for the query signatures table, one for query papers, one for seed signatures,
one for block papers, one for paper authors, and one optional embedding batch.

### Signature Columns

Prefer arrays of primitive/string fields instead of nested JSON objects:

```text
signature_id: string
paper_id: string
author_first: string
author_middle: string
author_last: string
author_suffix: string
author_affiliations: list[string]
author_email: string | null
author_orcid: string | null
author_position: int32
author_block: string | null
source_author_ids: list[string]
```

Name-count fields are not required for Phase 1 retrieval. The current Rust
retriever does not consume them directly; they are used later by the promoted
linker row-signal builder. Include them when the caller wants the Phase 2
downstream scoring wrapper to avoid recomputing or defaulting rarity features:

```text
name_count_first: float32
name_count_last: float32
name_count_first_last: float32
name_count_last_first_initial: float32
```

Optional pre-normalized fields may also be included to skip repeated normalization:

```text
first_normalized: string
middle_normalized: string
last_normalized: string
coauthor_blocks: list[string]
affiliation_terms: list[string]
```

If pre-normalized fields are supplied, the request must include a
`normalization_version` string. Rust should reject unknown versions rather than
silently mixing incompatible retrieval semantics.

### Paper Columns

Use one row per paper referenced by the query or seed signatures when the paper
metadata is available:

```text
paper_id: string
title: string
abstract: string | null       # needed for scoring; retrieval-only callers may omit it
venue: string
journal_name: string
year: int32 | null
authors: list[AuthorRecord]
```

`AuthorRecord` should be compact:

```text
position: int32
author_name: string
```

References are not required for the current retrieval-only candidate plan unless
a future retriever feature explicitly uses them. `abstract` is not consumed by
retrieval, but it is required for exact downstream scoring because the promoted
pairwise feature set includes `abstract_count`. If the source only has an
abstract-presence marker, a non-empty string is sufficient.

### Embedding Columns

Use an Arrow fixed-size list column:

```text
specter:
  paper_id: utf8
  embedding: fixed_size_list<float32>[embedding_dim]
```

For non-Arrow compatibility transport, use:

```text
paper_ids: string array
matrix: float32[row_count, embedding_dim]
```

Do not use `dict[paper_id] -> list[float]` as a target runtime shape. It is
acceptable only as a compatibility shim.

### Python Compatibility Input

The first implementation may accept existing raw JSON-shaped dictionaries:

```text
query_signatures: dict[signature_id, signature_payload]   # len >= 1
query_papers: dict[paper_id, paper_payload]
block_signatures: dict[signature_id, signature_payload]
block_papers: dict[paper_id, paper_payload]
cluster_seeds_require: dict[signature_id, cluster_id]
specter_embeddings: dict[paper_id, vector]
```

This compatibility shape is easier to integrate, but it is not the maximum-speed
format. It still pays nested-object traversal and string-key lookup costs across
the Python/Rust boundary.

### Format Preference Order

1. Arrow IPC / typed columnar arrays consumed directly by Rust, with optional
   pre-normalized fields and contiguous Arrow `fixed_size_list<float32>`
   embeddings.
2. Typed columnar arrays held in process and consumed directly by Rust.
3. Existing JSON-shaped Python dictionaries, accepted as a compatibility shim.

Do not add "Arrow read -> Python dict/list -> Rust" as an optimization path; the
measured Arrow-to-Python materialization cost is too high.

The Rust core should be implemented against the typed columnar structs. The
JSON-shaped API should be a thin adapter that converts into those structs.

### Migration Order

1. Define the `FeatureBlock` schema first, including the exact fields required
   for retrieval, constraints, pair features, and link/abstain scoring.
2. Add an `ANDData -> FeatureBlock` adapter for parity tests and incumbent-path
   measurement. This adapter may read Python objects, but it is not the
   production speed path.
3. Add Rust Arrow readers for the same `FeatureBlock` tables and validate row
   counts, required columns, null handling, SPECTER vector shape, and ID
   alignment.
4. Add `raw_block_query_candidate_plan(...)` adapters for both direct Arrow IPC
   and JSON-shaped compatibility input, backed by the same typed structs.
5. Migrate specter first in production request construction because it is the
   largest contiguous numeric payload and should not be rebuilt as Python lists.
6. Migrate signatures, papers, paper authors, seeds, and name-count inputs after
   retrieval and downstream scoring parity are stable.
7. Treat Python `ANDData` Arrow loaders as a separate incumbent-path project.
   They are useful only if they avoid expensive Python object materialization or
   replace enough `ANDData` preprocessing to justify the added path.

### Output Contract

Return a JSON/Python-mapping-compatible object. Multi-query results share one
flat candidate-row array; each row is tagged with its originating query via
`row_query_offsets` (an index into `query_signature_ids`).

Important: `row_query_offsets` are request-local query offsets, not numeric
signature indices. Do not feed them directly into `LinkerCandidateBatch`.
`LinkerCandidateBatch.row_query_signature_indices`, `left_signature_indices`,
and `right_signature_indices` are numeric indices in the active scoring
signature order. The proof bridge maps through the incumbent featurizer's full
signature order. The production Phase 2 wrapper should build a mini-dataset
signature order from the query plus retrieved component members, map signature
ids to those mini-dataset indices, and only then construct `LinkerCandidateBatch`.

```text
{
  "schema_version": "raw_block_query_candidate_plan_v1",
  "query_signature_ids": [str],          # length = N_queries
  "query_views_resolved": [str],         # resolved per-query view (e.g. "full"/"initial_only")
  "top_k": int,
  "candidate_row_count": int,
  "pair_count": int,
  "row_component_keys": [str],
  "row_component_sizes": [int],
  "row_query_offsets": uint32 array/list,  # row -> index in query_signature_ids
  "left_signature_ids": [str],
  "right_signature_ids": [str],
  "pair_row_indices": uint32 array/list,
  "retrieval_scores": float32 array/list,
  "retrieval_ranks": uint16 array/list,
  "row_signals": {
    "...": array/list
  },
  "component_members": {
    "<component_key>": [signature_id, ...]
  },
  "telemetry": {
    "query_count": int,
    "seed_signature_count": int,
    "seed_component_count": int,
    "normalization_seconds": float,
    "summary_seconds": float,
    "retriever_build_seconds": float,
    "retrieval_seconds": float,
    "total_seconds": float
  }
}
```

Single-query callers see `query_signature_ids` of length 1; every row's
`row_query_offsets` entry is 0. No special-case shape.

Arrays may be returned as NumPy arrays through PyO3 where existing downstream
code benefits from zero-copy or low-copy conversion. The schema should remain
logically stable even if the physical transport uses arrays.

## Algorithm

Seed-side prep (steps 1–4) runs once per request regardless of query count.
Query-side prep (step 5) and retrieval (step 6) scale with the batch size.

1. Iterate `cluster_seeds_require` and load each seed signature from
   `block_signatures`.
2. For each seed signature, extract the same retrieval features currently used
   by Python `extract_query_features(...)`:
   - first, middle, first initial, middle initials
   - coauthor blocks
   - affiliation terms
   - venue terms
   - title terms
   - year
   - ORCID, enabled in production and disabled only for labeled calibration/eval
   - specter vector presence/value
   - name-count fields only when scoring-wrapper readiness is requested
   - paper author count, paper author names, author position, local author
     window names
3. Aggregate seed features into one Rust-native cluster summary per component.
4. Build the temporary `RustHybridCentroidRetriever` directly from Rust-native
   summaries.
5. For each query in `query_signatures`:
   - Parse the raw query signature and paper into a Rust-native query feature
     struct.
   - Resolve the query view: `auto` -> `full` for a multi-letter first name,
     `initial_only` otherwise; `full`/`initial_only` use the caller-supplied
     value. Record the resolved view per query so it can be returned in
     `query_views_resolved`.
   - Apply the masking policy. `full` keeps full-first and middle-name
     evidence; `initial_only` masks first name to the first initial and clears
     full-first/middle evidence. ORCID is controlled separately by
     `orcid_enabled` and applies to all queries in the batch.
6. Run `top_k_hybrid_centroid_pair_plan(...)` once over the batched queries
   against the shared retriever; the existing Rust pair-plan method already
   accepts a list of queries. The raw API should expose request-local
   `row_query_offsets`, not full-dataset query signature indices.
   If `orcid_enabled` is true and a query ORCID matches one or more seed
   components, return all matching components regardless of `top_k` and mark
   those rows with `orcid_match`.
7. Return candidate rows, row signals, candidate component ids, component
   member ids, and pair ids using signature ids rather than full-dataset
   integer indices. Each row carries the `row_query_offsets` entry that maps it
   back to the input query. Numeric dataset/featurizer indices are introduced
   only by the Phase 2 mini-dataset wrapper.
8. Downstream link/abstain handling must apply the promoted constraint decision
   policy after row scoring:
   - `orcid_match` rows force-link, return all ORCID-matching components before
     truncation, and are exempt from disallow vetoes. If multiple ORCID-matching
     components exist, choose among those rows deterministically by the existing
     row ordering/scoring tie-breaks; do not allow a non-ORCID row or the
     abstain gate to override the ORCID match.
   - `get_constraint` require labels force-link the matching candidate row.
   - `get_constraint` disallow labels veto a candidate row only when the row has
     one member pair and that pair disallows, all member pairs disallow, or at
     least three member pairs exist and the disallow fraction is at least `0.8`.
     ORCID and require rows are exempt.
   - If the top row is vetoed, recompute the gate over eligible rows for that
     query; if all rows are vetoed, abstain.

## Memory Requirements

The implementation should not allocate structures proportional to any corpus
outside the supplied block.

Target request memory:

```text
O(raw block payload)
+ O(seed signatures in block)
+ O(seed components in block)
+ O(query_count * top_k candidate rows), except ORCID override queries may
  return all ORCID-matching components
+ O(members in retrieved candidate components)
```

The seed-side terms are shared across the batch; only the candidate-row term
scales with `query_count`.

Avoid:

- full `ANDData`
- full `RustFeaturizer`
- pairwise feature matrices
- broad query-vs-all-seed pair materialization
- retaining normalized per-signature records after cluster summaries are built,
  unless needed for `component_members`

## Error Handling

Use explicit typed failures surfaced through PyO3 exceptions:

- `MissingSeedSignature`
- `InvalidSignaturePayload`
- `InvalidPaperPayload`
- `UnsupportedQueryView`
- `InvalidSpecterVector`

For optional metadata, including missing query or seed papers, prefer
current-linker missing-value semantics over adapter-specific compatibility
behavior. The
current Python path treats absent papers as explicit missing metadata for year,
title, venue, paper authors, and local author windows. Log telemetry counts for
missing papers, missing specter vectors, empty coauthors, and empty
affiliations. Raise only for missing seed signatures, malformed required
payload fields, unsupported query views, or invalid specter vectors.

## Parity Requirements

Add tests against existing Python retrieval construction:

1. Dummy fixture parity:
   - Build `ANDData`.
   - Run existing `build_incremental_linker_inputs(...)` plus Rust retrieval.
   - Run new raw-block Rust API on equivalent raw payloads.
   - Assert identical component keys, ranks, and pair member ids.

2. Metadata-missing fixture:
   - Omit selected optional fields.
   - Assert missing-value behavior matches the current promoted retriever.

3. Single-query giant-block sample:
   - Use a small bounded sample from a large block.
   - Assert candidate row count is `<= top_k` unless the query has an ORCID
     match, in which case assert all ORCID-matching components are returned.
   - Assert no broad pairwise feature generation occurs.
   - Capture telemetry and peak RSS.

4. Multi-query batch parity:
   - Build a request with N > 1 query signatures against the same block.
   - Assert that running each query individually and then merging produces the
     same candidate rows, ranks, and component members as the single batched
     call (modulo deterministic ordering of `row_query_offsets`).
   - Assert seed-side telemetry (summary_seconds, retriever_build_seconds) is
     reported once for the batch, not per query.

5. Query exclusion:
   - Include one or more query signatures in `block_signatures`.
   - Assert each query is not used as a seed candidate unless explicitly
     seeded via `cluster_seeds_require`.

## Retrieval Recall Evaluation

The existing `s2and_and_big_blocks_linker_dataset_20260513` labels are censored
to the incumbent retriever's top-25 candidates. That dataset can measure whether
a new retriever preserves or improves the rank of already-observed positives,
but it cannot by itself prove that newly retrieved candidates are true negatives.

Use a calibration-only manual labeling loop as the first tractable evaluation:

1. Freeze the calibration query set.
2. Run the incumbent promoted retriever and each proposed retriever with
   `top_k=25`.
3. For each query, reuse existing labels for candidates already present in the
   frozen top-25 label table.
4. Create manual review packets for every newly retrieved
   `(query_id, candidate_component_key)` pair that appears in a proposed
   retriever's top 25 and is absent from the frozen labels.
5. Manually label those new candidates.
6. Merge the new labels with the frozen labels.
7. Report recall@1/5/10/25, positive-first rate, candidate count, and abstain
   impact for each retriever on the same expanded calibration label set.

This is not a global relabeling effort. The annotation unit is only a query and
one candidate component that a proposed retriever actually surfaced in the top
25.

### Manual Labeling Contract

Review unit:
: `(dataset, query_signature_id, query_group_id, candidate_component_key)`

Allowed labels:
: `same_author`, `different_author`, `uncertain_or_unresolvable`

Required packet context:

- Query signature metadata: author name, paper title, venue, year, coauthors,
  affiliation, ORCID if present, and specter availability.
- Candidate component evidence: member signatures, member papers, years,
  venues, coauthors, affiliations, ORCIDs if present, and cluster size.
- Retrieval evidence: retriever policy name, rank, score, row signals, and
  whether the candidate was present in the incumbent frozen top 25.
- Existing known-positive candidate for the query, when available, to help
  adjudicate split-cluster cases.

Output schema:

```text
dataset
query_signature_id
query_group_id
candidate_component_key
retriever_policy
retrieval_rank
label
confidence
notes
needs_adjudication
source_urls
```

External web evidence should be disabled by default. Enable it only for a
separate adjudication pass so labels remain reproducible from packet contents.

### Calibration Labeling Guardrails

- Start with a tiny packet build and validation run before generating the full
  calibration review set.
- Deduplicate review units across retriever policies before labeling.
- Preserve the frozen source labels; write new labels to a separate artifact.
- Validate exactly one manual label per assigned review unit, no duplicate ids,
  and only allowed label values.
- Report label counts, uncertain count, duplicate count, missing count, and
  invalid-enum count after each batch.
- Treat calibration results as proposal evidence, not final generalization
  evidence, if retriever weights are tuned using the same calibration labels.

### What This Solves

This labeling loop directly answers the question the fixed-top-25 dataset cannot:
when a proposed retriever replaces an incumbent top-25 candidate with a new
candidate, was the new candidate actually a true author link?

It keeps labeling cost bounded because only newly surfaced top-25 candidates are
reviewed. It also keeps the proposed Rust shortcut aligned with the evaluation:
the raw-block candidate-plan API must expose retriever policy name, rank, score,
component id, and enough packet evidence to support manual adjudication.

## Integration Plan

Phase 0: `FeatureBlock` boundary, adapters, and measurement gates

- Define the narrow `FeatureBlock` contract for query signatures, seed
  signatures, papers, paper authors, specter vectors, seed membership, name
  counts needed for scoring, and optional clusters/constraints.
  Initial Python contract: `s2and/incremental_linking/feature_block.py`.
- Document the `ANDData` responsibilities that are explicitly out of scope for
  this Rust path so the implementation does not become a full `ANDData` port.
- Add an `ANDData -> FeatureBlock` adapter that can feed the same Rust core
  without changing public incumbent surfaces.
- Add typed Rust request structs behind `FeatureBlock`; the Rust core should not
  depend on Python namedtuples or raw JSON object layout.
- Port the production retrieval feature extractor and seed summary aggregation
  into Rust against those structs.
- Add direct Arrow IPC readers for the same typed structs, starting with the
  schema slices needed for retrieval parity. Validate read-time row counts and
  SPECTER vector shape before wiring retrieval semantics.
- Add stage telemetry in the incumbent path before and after the core is wired
  in, so general-path wins are measurable separately from raw-shortcut wins.
- Keep the Arrow IPC schema work behind the same typed structs rather than
  coupling the core to one transport format.

Phase 1: retrieval-only Rust API

- Add direct Arrow IPC and JSON-shaped raw-payload adapters for the shared Rust
  core. Use JSON compatibility tests to simplify parity setup, but use Arrow
  for max-speed evidence.
- Add `raw_block_query_candidate_plan(...)`.
- Add Python tests comparing output to current linker retrieval.
- Add telemetry to report normalization, summary, retriever build, and retrieval
  time separately.

Phase 2: downstream scoring wrapper

- Current proof bridge: convert the raw candidate plan into the existing
  numeric `LinkerRetrievalBatch` without rerunning retrieval, using the incumbent
  featurizer's signature-id order. This has exact downstream parity on h_wang
  and a_silva single-query checks.
- Current mini-order bridge: derive `FeatureBlockSignatureOrder` from the raw
  candidate plan and map ids into deterministic mini-block numeric indices.
- Current direct `FeatureBlock` scoring bridge: build `RustFeaturizer` state
  directly from `FeatureBlock` through
  `RustFeaturizer.from_feature_block(...)` so existing pairwise/constraint code
  can be reused while full-block and mini-`ANDData` setup are removed.
- Production raw-only wrapper status: the public wrappers
  `predict_incremental_link_or_abstain_from_raw_feature_block(...)` and
  `predict_incremental_link_or_abstain_from_raw_payloads(...)` build a mini
  `FeatureBlock` containing the query plus returned component members, build the
  Rust featurizer directly from that contract, and map `left_signature_ids`,
  `right_signature_ids`, and request-local row query offsets into the numeric
  `left_signature_indices`, `right_signature_indices`, and
  `row_query_signature_indices` expected by `LinkerCandidateBatch`.
- Direct raw Arrow wrapper status:
  `predict_incremental_link_or_abstain_from_raw_arrow_paths(...)` now performs
  raw Arrow retrieval, builds the filtered scoring featurizer through
  `RustFeaturizer.from_arrow_paths(...)`, and then reuses the same Rust
  pairwise/constraint/link-abstain scoring path. Rust now emits the remaining
  row signals that previously required a Python signal `FeatureBlock`:
  name-count rarity, candidate max paper-author count, paper-author-list
  overlap, local author-window overlap, and author-count delta. There is no
  full-block `ANDData`, no mini-`ANDData` fallback, and no Python signal
  `FeatureBlock` in the raw Arrow wrapper.
- Full predict generalization:
  `Clusterer.predict_from_arrow_paths(...)` uses the same Arrow featurizer
  constructor for ordinary full-block `predict`, then runs the existing Rust
  block upper-triangle feature and constraint APIs directly from that featurizer.
  This applies the raw-path lesson to general S2AND Rust predict, not only
  `predict_incremental`.
- Reuse existing pairwise scoring, 53-feature assembly, promoted linker, and
  logistic gate until profiling shows this orchestration is still a material
  bottleneck after `ANDData` and current linker-input construction are gone.

Latest direct Arrow performance evidence, release build, n_jobs=20,
total_ram_bytes=1e12:

- h_wang raw incremental, embedded per-signature name-count columns and the
  default packaged filtered `name_tuples` aliases:
  `scratch/baseline/h_wang_single_query/arrow_full_specter_embedded_counts/direct_raw_arrow_wrapper_native_signals_embedded_counts_20260520.json`
  reports 17.41s direct wrapper predict time after model load, including
  16.04s raw retrieval/summary build, 1.34s filtered Arrow scoring-featurizer
  build, and `raw_arrow_signal_seconds=0.00019`. The output linked the query to
  the same component as the exact raw FeatureBlock parity report.
- h_wang raw incremental with the old signatures table lacking embedded counts
  but with the 1.4GB global `name_counts.arrow` present:
  `direct_raw_arrow_wrapper_native_signals_external_name_artifacts_20260520.json`
  reports 85.86s predict time. The global lookup read costs 25.73s in raw
  retrieval and 35.51s in filtered featurizer construction. This is now treated
  as a fallback/build-time path, not the hot-path target.
- h_wang raw incremental with the same old signatures table lacking embedded
  counts but with a sorted exact-verified `name_counts_index/` sidecar:
  `scratch/baseline/h_wang_single_query/arrow_full_specter/direct_raw_arrow_wrapper_native_signals_name_counts_index_20260520.json`
  reports 17.04s predict time. The global name-count setup step drops from
  25.73s to 0.028s in raw retrieval, and filtered Arrow scoring-featurizer
  construction drops from 35.51s to 1.27s.
- h_wang full predict, 1000 signatures / 499,500 pairs, embedded counts and the
  default packaged filtered `name_tuples` aliases:
  `scratch/baseline/h_wang_single_query/arrow_full_specter_embedded_counts/direct_predict_embedded_counts_subset1000_20260520.json`
  reports 2.23s for `predict_from_arrow_paths(...)` after model load.
- h_wang full predict, 1000 signatures / 499,500 pairs, old signatures table
  plus `name_counts_index/`:
  `scratch/baseline/h_wang_single_query/arrow_full_specter/direct_predict_name_counts_index_subset1000_20260520.json`
  reports 2.31s for `predict_from_arrow_paths(...)` after model load.
- Older a_silva full predict, 1000 signatures / 499,500 pairs, direct Arrow,
  no global name-count load: 1.31s for `predict_from_arrow_paths(...)` after
  model load. The prior incumbent full-scope profile for the same pair count
  spent 50.86s in `ANDData` construction plus 6.19s in predict.

Hot-path rule: prefer embedded per-signature name counts in `signatures.arrow`
when producing request/block bundles. When embedding is impractical, provide the
sorted binary `name_counts_index/` sidecar. Keep `name_counts.arrow` for
artifact generation, parity fallback, and inspection; do not cold-read it per
request.

Complete Arrow full-predict parity update:

- The older `arrow_full_specter` scratch fixtures remain speed probes, but the
  complete Arrow fixture path now carries abstract presence, paper language
  fields, SPECTER fixed-size-list embeddings, and name-count artifacts. Name
  aliases are no longer treated as dataset-local Arrow artifacts; the hot path
  uses the default packaged filtered `name_tuples` file unless a parity
  experiment explicitly passes an override path.
- Tracked 50-signature a_silva embedded-count gate:
  `scratch/baseline/a_silva_single_query/tracked_gate_subset50_embedded_counts_20260520/compare_full_predict_arrow_parity_subset50_embedded_counts_20260520.json`
  reports exact feature-matrix parity, exact constraint parity,
  `max_absdiff=0.0` distance parity, and exact clusters.
- Tracked 50-signature a_silva index-count gate:
  `scratch/baseline/a_silva_single_query/tracked_gate_subset50_index_counts_20260520/compare_full_predict_arrow_parity_subset50_index_counts_20260520.json`
  exercises `name_counts_index/` explicitly and reports exact feature-matrix
  parity, exact constraint parity, `max_absdiff=0.0` distance parity, and exact
  clusters.
- 50-signature a_silva complete-Arrow full-predict parity:
  `scratch/baseline/a_silva_single_query/complete_arrow_subset50_arrow_name_artifacts/compare_full_predict_complete_arrow_subset50_20260520.json`
  reports `max_absdiff=0.0`, `nonzero_absdiff_count=0`, and exact clusters.
- 1000-signature a_silva complete-Arrow full-predict parity:
  `scratch/baseline/a_silva_single_query/complete_arrow_subset1000_arrow_name_artifacts/compare_full_predict_complete_arrow_subset1000_featurecheck_20260520.json`
  reports exact 39-feature matrix parity, exact constraint parity,
  `max_absdiff=0.0`, `nonzero_absdiff_count=0`, and exact clusters over
  499,500 pairs.
- The 50-signature run first generated `name_counts.arrow` with 35,419,433
  rows. The 1000-signature run reused the name-count artifacts and measured
  31.18s for Arrow featurizer construction plus 1.65s for Arrow distance
  construction. A later mini `zbmath` microbenchmark measured the packaged
  filtered text alias fallback at ~22ms over no aliases, while `name_pairs.arrow`
  was slower, so a name-count-style mmap/index sidecar is not justified for name
  aliases.

Phase 3: incumbent-path `ANDData` optimization

- Profile and optimize `ANDData` construction sub-stages: JSON loading,
  namedtuple materialization, paper preprocessing, signature preprocessing,
  specter filtering, name counts, and indexing. Treat Sinonym as an explicit
  non-default mode rather than the forward-looking baseline.
- Consider lazy/incremental `ANDData` construction only where the caller still
  needs an `ANDData` object and the measured sub-stage costs justify the
  additional complexity.
- Keep this separate from the raw shortcut. Improving `ANDData` is worthwhile
  for broad S2AND functionality, but it is not a prerequisite for raw requests
  that can enter through `FeatureBlock` directly.

Phase 4: further Rust-native raw scoring cleanup

- Raw row-signal construction and row-level name-count rarity are now emitted by
  Rust for the direct Arrow candidate plan. Further Rust work should target
  reusable/raw component-summary artifacts or faster full-block Arrow summary
  setup if production-like profiles still show raw retrieval/summary as
  material.
- Keep the same parity gate: exact normalized decisions, linked clusters,
  probabilities, 53-feature matrix, candidate-row counts, and pair counts.

## Acceptance Criteria

- The Baseline Measurement profile exists, is checked in alongside the
  implementation, and shows the skipped stages dominate end-to-end latency on
  the target request shape.
- The `FeatureBlock` contract exists and explicitly lists which `ANDData`
  responsibilities are out of scope.
- The raw shortcut and incumbent `ANDData -> FeatureBlock` adapter both use the
  same Rust feature/summary/candidate-plan core, or the doc explicitly records
  why one adapter is blocked.
- Direct Rust Arrow IPC input reads the full h_wang fixture schema slices in
  sub-second order and does not materialize Python dict/list objects on the hot
  path.
- The raw-block API returns the same top candidate components as the current
  promoted retrieval path on parity fixtures.
- The raw candidate-plan bridge returns the same final normalized link/abstain
  decisions, probabilities, 53-feature matrix, candidate-row counts, and pair
  counts as the current promoted path on h_wang and a non-h_wang fixture.
- A production single-query raw request does not construct full-block `ANDData`.
- A production single-query raw request does not build a full-block scoring
  `RustFeaturizer`; it builds only the filtered query-plus-candidate scoring
  featurizer after raw retrieval. Retrieval summary construction still scans
  the full seed block unless/until precomputed component summaries are added.
- Telemetry separates summary construction from retriever build and retrieval.
- Incumbent-path telemetry reports `ANDData` construction sub-stages separately
  enough to decide whether lazy/incremental construction is worthwhile.
- Memory is bounded by the request block and top candidate output.
- Phase 2 can pass the output to downstream scoring without invoking retrieval
  again by constructing numeric `LinkerCandidateBatch` index arrays explicitly;
  the final production version should do this against a deterministic
  mini-dataset/raw-slice signature order rather than the full incumbent
  featurizer order.

## Open Questions

- Should output arrays use NumPy arrays, Python lists, or a mixed transport based
  on size?
- Which remaining scoring inputs should be carried directly by the raw candidate
  plan, and which should be rebuilt from a mini dataset or raw-slice view?
