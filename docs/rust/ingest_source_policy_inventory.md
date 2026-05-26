# Rust Ingest Source Policy Inventory

Status date: 2026-05-25

This inventory is the prerequisite for any Rust ingest deduplication. The goal
is to document which policies are actually equivalent across Arrow, JSON, and
`ANDData` before reusing staging structs or preprocessing helpers.

## Current Ingest Owners

| Source | Entry point | Role |
|---|---|---|
| Arrow IPC | `RustFeaturizer.from_arrow_paths(...)` | Production Arrow constructor for selected rows. |
| JSON | `RustFeaturizer.from_json_paths(...)` | Compatibility, fixture, and benchmark constructor. |
| `ANDData` | `RustFeaturizer.from_dataset(...)` | Python-reference, training/eval, parity, and compatibility constructor. |

Shared Arrow staging exists today in `StageSignatureInput`,
`StagePaperInput`, `preprocess_stage_papers(...)`, and
`preprocess_stage_signatures(...)` in `s2and_rust/src/lib.rs`. JSON still has
local `SignatureInput`, `PaperInput`, and `PaperPreprocessed` records.
`ANDData` intentionally reads mostly precomputed Python fields.

## Policy Matrix

| Policy | Arrow | JSON | `ANDData` |
|---|---|---|---|
| Unidecode | Eagerly preloads a char map for selected signatures, needed papers, and paper authors through `ensure_unidecode_for_raw_arrow_inputs(...)`. | Eagerly preloads while parsing signature names, affiliations, paper titles, venues, journals, and embedded paper authors. | Lazily preloads only when existing preprocessed fields or counters are missing and must be recomputed. |
| Language | Trusts `predicted_language` when present and defaults missing `is_reliable` to `false`; detects from raw title only when language is missing. | Always detects language from raw title for loaded papers. | Preserves existing `predicted_language`; detects only for in-signature papers missing language. |
| Name normalization | Recomputes first, middle, and last normalization through shared stage helpers. | Recomputes equivalent first, middle, and last normalization inline. | Reads normalized name fields from Python objects and recomputes only missing pieces. |
| Name counts | Uses `name_counts_index` when present; rejects a `name_counts.arrow` path without the index; otherwise can use the legacy JSON name-count path/default. | Loads JSON name-count artifacts with optional normalization-version validation and records default telemetry. | Reads attached `author_info_name_counts`; overlay is handled later by `update_signature_name_counts(...)`. |
| Name tuples | Arrow `name_pairs` / `name_tuples` file wins over the Python argument; otherwise falls back to the argument/default text file. | Loads from the text path/default. | Reads `dataset.name_tuples`. |
| Paper authors | Requires a separate `paper_authors` table, requires non-null positions, sorts by position, and rejects duplicate `(paper_id, position)`. | Reads embedded `paper["authors"]`, skips non-object author entries, defaults missing/null positions to `0`, and keeps input order. | Reads Python paper-author objects/tuples with required positions; preserves the Python-provided order. |
| Signature positions | Nullable Arrow position defaults to `0` with telemetry. | Missing/null JSON position defaults to `0` with telemetry; malformed non-integers fail. | Requires `author_info_position` to extract as an integer. |
| Reference features | Rejects `compute_reference_features=True`. | Supports reference features by deriving counters from loaded referenced papers. | Supports reference features by consuming existing `reference_details` counters. |
| `preprocess=false` | Still normalizes titles/authors and builds title word ngrams; skips title char, venue, journal, affiliation, and coauthor ngrams. | Same as Arrow for the local preprocessing loop. | Paper recomputation follows `preprocess`; signature fallback can still compute missing coauthor/affiliation ngrams from existing Python fields. |
| Filtering and order | Optional requested `signature_ids` are deduplicated in requested order; absent ids sort all loaded signatures. Papers, paper authors, SPECTER, and seed constraints are filtered to selected rows. | Loads all signatures, filters papers to referenced paper IDs, errors on missing referenced papers, and sorts signature ids. Cluster seed JSON is not selected-row filtered. | Loads all dataset signatures and sorts ids; uses dataset-level seed maps. |
| SPECTER | Reads fixed-size-list Arrow rows for needed papers; missing rows count as missing embeddings. | Accepts pickle path, dict, or `None`; missing rows count as missing embeddings. | Uses `dataset.specter_embeddings` when present; missing rows count as missing embeddings. |

## Safe Reuse Candidates

- Arrow and JSON no-reference paper/signature preprocessing are semantically
  close for title normalization, author normalization, name splitting,
  coauthor extraction, affiliation filtering, and `preprocess=false`.
- JSON local `SignatureInput` and `PaperInput` could be converted to the shared
  stage records only if name-count default telemetry is preserved and reference
  features stay source-specific.
- The shared `StagePaperPreprocessed` record would need reference metadata
  extensions before it could replace JSON's local `PaperPreprocessed` in the
  reference-feature path.

## Do Not Merge Without A Decision

- Do not route `from_dataset(...)` through the shared Arrow/JSON stage helpers
  without deciding how much precomputed Python state to preserve. Its purpose is
  to load `ANDData`, not to re-run full raw ingest.
- Do not merge paper-author policies until the repo explicitly chooses between
  Arrow's strict non-null/unique positions and JSON's compatibility defaulting.
- Do not merge language policy until JSON either remains a redetecting
  compatibility loader or starts honoring serialized `predicted_language`.
- Do not merge name-count policy until JSON default telemetry has an equivalent
  output in shared helpers.

## Verification Targets

Before any code deduplication, keep these focused tests in scope:

- `tests/test_rust_from_json_paths.py`
- `tests/test_rust_from_dataset_contract.py`
- `tests/test_raw_block_candidate_plan_arrow.py`
- Rust unit tests in `s2and_rust/src/lib.rs`
