# Rust Artifact Formats

Status date: 2026-05-20

This is the current artifact-format decision table for Rust-backed inference.
It replaces the older artifact-divergence migration log.

## Current Targets

| Artifact / data family | Target format | Runtime role |
|---|---|---|
| Signatures | `signatures.arrow` Arrow IPC table | Required direct-Rust input. Contains identity, paper id, author name fields, affiliations, ORCID, position, optional block/email/source ids. It does not contain embedded name-count columns. |
| Papers | `papers.arrow` Arrow IPC table | Required direct-Rust input. Contains title, abstract-presence signal, venue, journal, year, language, and reliability fields. |
| Paper authors | `paper_authors.arrow` Arrow IPC table | Required for coauthor and paper-author row signals. |
| Cluster seeds | `cluster_seeds.arrow` Arrow IPC table | Required for seeded/incremental Arrow prediction. Omit for unseeded full prediction. |
| SPECTER | Arrow fixed-size-list `float32` table | Preferred direct-path embedding input. Include the embedding version required by the model. |
| Name counts | `s2and/data/name_counts_index/` sorted binary sidecar | Preferred Rust hot-path lookup artifact for models that use name-count features. |
| Name-count Arrow table | `name_counts.arrow` long-form Arrow table | Generation, inspection, and parity fallback only. Do not cold-read it per request when the index is available. |
| Name aliases | Packaged filtered text file | Shared runtime default. Avoid per-dataset alias artifacts unless running an explicit experiment. |
| Pairwise and linker models | Native LightGBM text plus JSON metadata | Current production model-bundle format. |
| Eval clusters | Existing clusters JSON | Offline evaluation truth only; not part of production inference scoring. |

## Name Counts

The preferred production layout is:

```text
s2and/data/name_counts_index/
  manifest.json
  first.bin
  last.bin
  first_last.bin
  last_first_initial.bin
```

`manifest.json` must have `schema_version: "name_counts_index_v1"`.

Do not embed per-signature name-count values in `signatures.arrow`. That path
has been removed from the runtime direction. Do not build a production request
path that loads `name_counts.arrow` into Python dicts/lists.

## Deprioritized Or Rejected

| Format / approach | Current decision |
|---|---|
| Embedded `name_count_*` columns in `signatures.arrow` | Removed as a preferred/supporting Arrow hot path. Use `name_counts_index/`. |
| SQLite for name counts | Not better for the current exact static point-lookup workload. Revisit only for ad hoc queries, updates, or transaction requirements. |
| Pickle | Keep only for legacy compatibility. It is Python-only and not a production cross-language target. |
| JSON | Fine for fixtures and compatibility loaders; not the runtime target for large table-shaped inference data. |
| Arrow read into Python dict/list before Rust | Defeats the columnar boundary and was measured slower than keeping the hot path in Rust. |
| MessagePack as universal target | Better than JSON for nested legacy payloads, but it preserves the object shape the Rust path is trying to avoid. |
| Parquet as request/runtime hot path | Useful offline, but Arrow IPC is simpler for local runtime bundles and direct Rust readers. |
| Per-dataset `name_pairs.arrow` | Avoid for production. The default packaged filtered aliases are small enough as text. |

## Format Ownership

- `docs/rust/arrow_dataset_spec.md` owns the table schemas and manifest
  checklist.
- This document owns artifact-format choices and rejected alternatives.
- `docs/rust/inference_architecture.md` owns the runtime boundary and
  remaining Python-heavy paths.
