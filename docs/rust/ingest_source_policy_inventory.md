# Rust Ingest Source Policy Inventory

Status date: 2026-05-25

This inventory documents the maintained Rust ingest sources after removing the
legacy JSON constructor. The goal is to keep Arrow and `ANDData` ownership
separate unless a policy has an explicit parity contract.

## Current Ingest Owners

| Source | Entry point | Role |
|---|---|---|
| Arrow IPC | `RustFeaturizer.from_arrow_paths(...)` | Production Arrow constructor for selected rows. |
| `ANDData` | `RustFeaturizer.from_dataset(...)` | Python-reference, training/eval, parity, and compatibility constructor. |

Shared Arrow staging exists today in `StageSignatureInput`,
`StagePaperInput`, `preprocess_stage_papers(...)`, and
`preprocess_stage_signatures(...)` in `s2and_rust/src/lib.rs`. `ANDData`
intentionally reads mostly precomputed Python fields.

## Policy Matrix

| Policy | Arrow | `ANDData` |
|---|---|---|
| Unidecode | Eagerly preloads a char map for selected signatures, needed papers, and paper authors through `ensure_unidecode_for_raw_arrow_inputs(...)`. | Lazily preloads only when existing preprocessed fields or counters are missing and must be recomputed. |
| Language | Trusts `predicted_language` when present and defaults missing `is_reliable` to `false`; detects from raw title only when language is missing. | Preserves existing `predicted_language`; detects only for in-signature papers missing language. |
| Name normalization | Recomputes first, middle, and last normalization through shared stage helpers. | Reads normalized name fields from Python objects and recomputes only missing pieces. |
| Name counts | Uses `name_counts_index` when present; rejects a `name_counts.arrow` path without the index. | Reads attached `author_info_name_counts`; overlay is handled later by `update_signature_name_counts(...)`. |
| Name tuples | Arrow `name_pairs` / `name_tuples` file wins over the Python argument; otherwise falls back to the argument/default text file. | Reads `dataset.name_tuples`. |
| Paper authors | Requires a separate `paper_authors` table, requires non-null positions, sorts by position, and rejects duplicate `(paper_id, position)`. | Reads Python paper-author objects/tuples with required positions; preserves the Python-provided order. |
| Signature positions | Nullable Arrow position defaults to `0` with telemetry. | Requires `author_info_position` to extract as an integer. |
| Reference features | Rejects `compute_reference_features=True`. | Supports reference features by consuming existing `reference_details` counters. |
| `preprocess=false` | Still normalizes titles/authors and builds title word ngrams; skips title char, venue, journal, affiliation, and coauthor ngrams. | Paper recomputation follows `preprocess`; signature fallback can still compute missing coauthor/affiliation ngrams from existing Python fields. |
| Filtering and order | Optional requested `signature_ids` are deduplicated in requested order; absent ids sort all loaded signatures. Papers, paper authors, SPECTER, and seed constraints are filtered to selected rows. | Loads all dataset signatures and sorts ids; uses dataset-level seed maps. |
| SPECTER | Reads fixed-size-list Arrow rows for needed papers; missing rows count as missing embeddings. | Uses `dataset.specter_embeddings` when present; missing rows count as missing embeddings. |

## Safe Reuse Candidates

- Arrow staging helpers are the authority for production file-backed ingest.
- `ANDData` should keep consuming Python-owned precomputed state unless a
  focused parity test proves a field can be recomputed in Rust without drift.

## Do Not Merge Without A Decision

- Do not route `from_dataset(...)` through the shared Arrow stage helpers
  without deciding how much precomputed Python state to preserve. Its purpose is
  to load `ANDData`, not to re-run full raw ingest.
- Do not merge paper-author policies unless the repo explicitly chooses whether
  `ANDData` should become as strict as Arrow's non-null/unique position rules.

## Verification Targets

Before any code deduplication, keep these focused tests in scope:

- `tests/test_rust_from_dataset_contract.py`
- `tests/test_raw_block_candidate_plan_arrow.py`
- Rust unit tests in `s2and_rust/src/lib.rs`
