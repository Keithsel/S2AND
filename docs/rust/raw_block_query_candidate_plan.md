# Rust-Only Raw Block Candidate Plan

Status: design spec

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

The output is only the retrieved candidate plan. Final pairwise scoring, 53-feature
assembly, promoted linker prediction, and logistic link/abstain can be wired as a
separate follow-up.

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

## Strategic Direction

The shortcut's unique contribution is narrow: **skip full `ANDData`
construction for raw small-N requests**. The rest of the work should be built as
shared S2AND inference infrastructure wherever practical.

Build the reusable part as a Rust-native inference core:

```text
raw or ANDData-backed signature/paper/seeds
  -> typed Rust request structs
  -> query feature extraction
  -> seed component summaries
  -> temporary hybrid centroid retriever
  -> candidate component plan
```

Then expose two adapters:

- a raw-payload adapter for the small-N production request shape, which avoids
  constructing `ANDData`;
- an incumbent `ANDData` adapter so existing `predict_incremental(...)` callers
  can use the same Rust feature/summary/retrieval core.

This avoids building a parallel one-off implementation. The raw path gets the
large latency win from skipping `ANDData`, while the existing path can still
benefit from the Rust port of `extract_query_features(...)` and
`build_cluster_summary(...)`.

Treat these as separate decisions:

- **Raw shortcut:** use it when the caller already has one or more query
  signatures plus one raw block payload and does not otherwise need a full
  `ANDData`.
- **Faster `ANDData`:** investigate and optimize as a general S2AND project.
  The baseline shows this is the biggest incumbent-path lever, but the raw
  shortcut should not wait for a lazy/incremental `ANDData` redesign.
- **Post-retrieval scoring:** first bridge raw candidate plans into the existing
  `LinkerCandidateBatch` scoring path. Only port the full scoring/gate pipeline
  to Rust after profiling proves Python orchestration still dominates.
- **Arrow IPC:** keep it as the preferred durable/runtime schema direction, but
  do not make it a Phase 1 gate. Start with a compatibility adapter from the
  existing JSON-shaped payload into the same typed Rust structs.

## ASAP Work Items

1. **Build a shared Rust feature/summary/candidate-plan core.**
   Implement typed Rust request structs and the production-equivalent
   normalizer/feature extractor once. Both the raw shortcut and incumbent
   `ANDData` path should call this core.

2. **Prototype the retrieval-only raw small-N candidate-plan API.**
   Add `raw_block_query_candidate_plan(...)` as the fastest path for raw
   production requests. Keep v1 narrow: parse raw signatures/papers/seeds,
   build summaries, retrieve candidate components, return ids and telemetry.
   Do not block this on Arrow or all-Rust scoring.

3. **Add stage-level telemetry and parity gates.**
   Report payload parse, `ANDData`, query feature extraction, seed summary
   build, retriever build, retrieval, pairwise scoring, linker/gate time, and
   RSS. Add parity tests for dummy data, missing metadata, ORCID override,
   query exclusion, and multi-query batching.

4. **Bridge raw candidate plans into existing downstream scoring.**
   Build the Phase 2 mini-dataset wrapper that maps returned signature ids into
   deterministic mini-dataset numeric indices and constructs
   `LinkerCandidateBatch` without rerunning retrieval.

5. **Scope `ANDData` fast/lazy construction independently.**
   Measure `ANDData` initialization sub-stages first: JSON load, namedtuple
   materialization, paper preprocessing, signature preprocessing, Sinonym,
   specter filtering, name counts, and block/signature indexing. Optimize or
   defer the expensive pieces only after the measurements identify them.

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
  retrieval.

## Non-Goals

- This is not a global ANN service or cross-block index.
- This does not guarantee identical results to the full end-to-end linker if the
  input omits fields that the current retriever uses.
- This does not return a final cluster assignment.
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
payloads. This should become the common schema family for Python `ANDData`, Rust
runtime readers, and the raw-block candidate-plan shortcut.

The fastest path should avoid parsing large nested Python dictionaries inside
Rust for every request. The logical contract above can be exposed first for
compatibility, but the target production input is a compact request-local,
columnar Arrow payload.

### Shared Runtime Format Recommendation

Use one Arrow IPC file or stream per logical table:

| Data | Recommended format | Shape |
|---|---|---|
| Signatures | Arrow IPC table | One row per signature, primitive/string/list columns |
| Papers | Arrow IPC tables | `papers` table plus `paper_authors` child table |
| Specter embeddings | Arrow IPC table | `paper_id` plus fixed-size `float32` vector |
| Cluster seeds | Arrow IPC table | `signature_id`, `cluster_id`, optional constraint type |
| Clusters | Arrow IPC table | One `cluster_id`, `signature_id` member row per membership |
| Name counts | Arrow IPC table | `count_type`, `key`, `count` |
| Name tuples | Arrow IPC table | `first`, `second` |

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
- The same schemas can feed `ANDData`, Rust featurizer construction, and the new
  raw-block retrieval shortcut.

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

References and abstract text are not required for the current retrieval-only
candidate plan unless a future retriever feature explicitly uses them.

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

1. Typed columnar arrays with optional pre-normalized fields and contiguous
   Arrow `fixed_size_list<float32>` embeddings.
2. Typed columnar arrays with raw fields normalized in Rust.
3. Existing JSON-shaped Python dictionaries, accepted as a compatibility shim.

The Rust core should be implemented against the typed columnar structs. The
JSON-shaped API should be a thin adapter that converts into those structs.

### Migration Order

1. Define and test Arrow schemas for signatures, papers, paper authors, specter,
   cluster seeds, clusters, name counts, and name tuples.
2. Add Python `ANDData` loaders that accept Arrow IPC paths while preserving the
   current in-memory `ANDData` contract.
3. Add Rust readers for the same Arrow schemas.
4. Use the same Arrow request tables in `raw_block_query_candidate_plan(...)`.
5. Migrate specter first because pickle is the largest cross-language startup
   and compatibility problem.
6. Migrate signatures, papers, and seeds after specter parity is stable.

### Output Contract

Return a JSON/Python-mapping-compatible object. Multi-query results share one
flat candidate-row array; each row is tagged with its originating query via
`row_query_offsets` (an index into `query_signature_ids`).

Important: `row_query_offsets` are request-local query offsets, not full-dataset
signature indices. Do not feed them directly into `LinkerCandidateBatch`.
`LinkerCandidateBatch.row_query_signature_indices`, `left_signature_indices`,
and `right_signature_indices` are numeric indices in the dataset/featurizer
signature order. The Phase 2 wrapper must build a mini-dataset signature order
from the query plus retrieved component members, map signature ids to those
mini-dataset indices, and only then construct `LinkerCandidateBatch`.

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
current-linker missing-value semantics over fallback-heavy behavior. The
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

Phase 0: shared Rust inference core and measurement gates

- Define typed Rust request structs for query signatures, seed signatures,
  papers, paper authors, specter vectors, and seed membership.
- Port the production retrieval feature extractor and seed summary aggregation
  into Rust against those structs.
- Add an `ANDData` adapter that can feed the same Rust core without changing
  the public `predict_incremental(...)` surface.
- Add stage telemetry in the incumbent path before and after the core is wired
  in, so general-path wins are measurable separately from raw-shortcut wins.
- Keep the Arrow IPC schema work behind the same typed structs rather than
  coupling the core to one transport format.

Phase 1: retrieval-only Rust API

- Add a JSON-shaped raw-payload adapter for the shared Rust core.
- Add `raw_block_query_candidate_plan(...)`.
- Add Python tests comparing output to current linker retrieval.
- Add telemetry to report normalization, summary, retriever build, and retrieval
  time separately.

Phase 2: downstream scoring wrapper

- Add a Python wrapper that converts the candidate plan into a
  `LinkerCandidateBatch` without rerunning retrieval.
- Materialize a mini dataset containing the query plus returned component
  members, build a deterministic mini-dataset signature order, and map
  `left_signature_ids`, `right_signature_ids`, and `row_query_offsets` into the
  numeric `left_signature_indices`, `right_signature_indices`, and
  `row_query_signature_indices` expected by `LinkerCandidateBatch`.
- Reuse existing pairwise scoring, 53-feature assembly, promoted linker, and
  logistic gate.

Phase 3: incumbent-path `ANDData` optimization

- Profile and optimize `ANDData` construction sub-stages: JSON loading,
  namedtuple materialization, paper preprocessing, signature preprocessing,
  Sinonym, specter filtering, name counts, and indexing.
- Consider lazy/incremental `ANDData` construction only where the caller still
  needs an `ANDData` object and the measured sub-stage costs justify the
  additional complexity.

Phase 4: optional all-Rust scoring

- Only after Phase 2 profiling shows Python mini-dataset/scoring orchestration
  still dominates, port the final scoring path into Rust-native raw-slice APIs.

## Acceptance Criteria

- The Baseline Measurement profile exists, is checked in alongside the
  implementation, and shows the skipped stages dominate end-to-end latency on
  the target request shape.
- The raw shortcut and incumbent `ANDData` adapter both use the same Rust
  feature/summary/candidate-plan core, or the doc explicitly records why one
  adapter is blocked.
- The raw-block API returns the same top candidate components as the current
  promoted retrieval path on parity fixtures.
- A single-query request does not construct `ANDData`.
- A single-query raw request does not build `RustFeaturizer`.
- Telemetry separates summary construction from retriever build and retrieval.
- Incumbent-path telemetry reports `ANDData` construction sub-stages separately
  enough to decide whether lazy/incremental construction is worthwhile.
- Memory is bounded by the request block and top candidate output.
- Phase 2 can pass the output to downstream scoring without invoking retrieval
  again by constructing the mini-dataset and numeric `LinkerCandidateBatch`
  index arrays explicitly.

## Open Questions

- Should output arrays use NumPy arrays, Python lists, or a mixed transport based
  on size?
- Should the candidate plan include enough row signals for immediate 53-feature
  assembly, or should some row signals be rebuilt during downstream scoring from
  the mini dataset?
