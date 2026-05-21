# Arrow Dataset Specification

Status date: 2026-05-21

This document defines the Arrow artifact contract for engineers assembling
datasets for the direct Rust S2AND inference path. These artifacts are used by
`Clusterer.predict(...)`, `Clusterer.predict_from_arrow_paths(...)`, and the
promoted phase of `Clusterer.predict_incremental(...)`.

The goal is parity with the current `ANDData(preprocess=True)` representation
without requiring production inference to materialize full `ANDData`.

---

## Summary

Each Arrow dataset is a directory of Arrow IPC file-format tables plus a
manifest. The hot path reads these files directly from Rust or through
memory-mapped Arrow readers.

Required for full-block prediction:

- `signatures.arrow`
- `papers.arrow`
- `paper_authors.arrow`
- `specter.arrow` or `specter2.arrow` when the model uses
  `embedding_similarity`

Required in addition for seeded prediction or incremental prediction promoted
through Arrow:

- `cluster_seeds.arrow`
- `cluster_seed_disallows.arrow`

Required when incremental input contains altered claimed profiles:

- `altered_cluster_signatures.arrow`

Offline evaluation datasets may also include:

- `<dataset>_clusters.json`

Do not create per-dataset `name_pairs.arrow` files for production datasets.
Name aliases are a shared runtime resource.

---

## Layout

Preferred on-disk layout:

```text
<arrow_root>/
  manifest.json
  <dataset>/
    manifest.json
    signatures.arrow
    papers.arrow
    paper_authors.arrow
    specter.arrow
    specter2.arrow
    signatures.signatures_batch_index.bin
    papers.papers_batch_index.bin
    paper_authors.paper_authors_batch_index.bin
    specter.specter_batch_index.bin
    specter2.specter_batch_index.bin
    cluster_seeds.arrow
    cluster_seed_disallows.arrow
    altered_cluster_signatures.arrow
    <dataset>_clusters.json
```

Notes:

- `cluster_seeds.arrow` is required only for seeded/incremental datasets. It can
  be omitted for unseeded full prediction and offline eval.
- `cluster_seed_disallows.arrow` preserves pairwise seed disallow constraints.
  Write an empty table when the request has no seed disallows.
- `altered_cluster_signatures.arrow` is required for incremental datasets whose
  seed clusters include altered claimed profiles. It is the producer-owned
  request artifact for this condition. `altered_cluster_signatures.txt` is
  accepted only as a legacy compatibility fallback.
- `<dataset>_clusters.json` is ground truth for offline evaluation only. It is
  not part of production inference scoring.
- `specter.arrow` is the SPECTER v1 embedding table. `specter2.arrow` is the
  SPECTER v2 embedding table. Include whichever model family will be used; eval
  bundles usually include both.
- The Arrow files must be Arrow IPC file format, not Arrow stream format. The
  current writer uses `pyarrow.ipc.new_file(...)`; readers use
  `pyarrow.ipc.open_file(...)` and memory maps.

The Python API may also pass explicit paths through `dataset.arrow_paths`,
`dataset.feature_block_arrow_paths`, or `dataset.rust_arrow_paths`. In that case
the path mapping should use these keys:

| Key | Meaning |
|---|---|
| `signatures` | Path to `signatures.arrow` |
| `papers` | Path to `papers.arrow` |
| `paper_authors` | Path to `paper_authors.arrow` |
| `specter` | Path to the embedding table selected for the current model, even if the file is physically named `specter2.arrow` |
| `cluster_seeds` | Path to `cluster_seeds.arrow` for incremental/seeded prediction |
| `cluster_seed_disallows` | Path to `cluster_seed_disallows.arrow` for pairwise seed disallow constraints |
| `altered_cluster_signatures` | Path to `altered_cluster_signatures.arrow` when altered claimed profiles are present; text fallback is accepted for legacy callers |
| `clusters` | Path to eval-only ground-truth clusters JSON |
| `name_counts_index` | Optional shared/global name-count index directory, normally `s2and/data/name_counts_index` |
| `name_counts` | Optional long-form Arrow name-count table for generation/inspection/parity, not preferred on the hot path |
| `signatures_batch_index` | Optional S2AND-generated lookup index for `signatures.arrow`, used by indexed raw candidate planning |
| `papers_batch_index` | Optional S2AND-generated lookup index for `papers.arrow`, used by indexed raw candidate planning |
| `paper_authors_batch_index` | Optional S2AND-generated lookup index for `paper_authors.arrow`, used by indexed raw candidate planning |
| `specter_batch_index` | Optional S2AND-generated lookup index for the selected embedding path passed as `specter`, used by indexed raw candidate planning |

---

## Large-Block Physical Layout

The schema above is the semantic artifact contract. Large-block incremental
serving also needs a physical layout that makes indexed raw candidate planning
cheap. This layout is not required for correctness, but it is required for the
scalable performance path on large blocks such as common family-name blocks.

For large block artifacts, producers should write the lookup tables below as
Arrow IPC file-format files with bounded record batches. Do not write these
tables as one giant record batch when the row count exceeds the limit.

| Table | Lookup key | Maximum rows per IPC record batch |
|---|---|---:|
| `signatures.arrow` | `signature_id` | 16,384 |
| `papers.arrow` | `paper_id` | 16,384 |
| `paper_authors.arrow` | `paper_id` | 16,384 |
| `specter.arrow` / `specter2.arrow` | `paper_id` | 2,048 |

The smaller request-scoped tables do not need a random-access physical layout:

| Table | Layout guidance |
|---|---|
| `cluster_seeds.arrow` | Read fully by the raw planner; no bounded-batch requirement. |
| `cluster_seed_disallows.arrow` | Read fully when present; no bounded-batch requirement. |
| `altered_cluster_signatures.arrow` | Read as request metadata; bounded batches do not address altered-profile pre-splitting cost. |

Implementation notes for producers:

- Use Arrow IPC file format, not stream format.
- In PyArrow, call `pyarrow.ipc.new_file(...)` and write tables with
  `writer.write_table(table, max_chunksize=<limit>)`.
- Preserve `signatures.arrow` row order. Record-batch boundaries must not
  change row contents or row order.
- Keep `paper_authors.arrow` grouped by `paper_id`, then ordered by `position`
  where practical. This improves locality when all authors for a paper are read.
- One record batch is acceptable only when
  `row_count <= maximum rows per IPC record batch`.
- For embedding files, the 2,048-row limit is intentionally lower because each
  row contains a dense vector. If the embedding dimension changes enough that a
  batch becomes much larger than roughly 8-16 MiB, lower this limit rather than
  raising it.

S2AND binary batch indexes are derived artifacts over the final Arrow files.
The preferred handoff is for producers to supply bounded Arrow IPC files and
for an S2AND prep step to generate these indexes. Producers may include indexes
only when they are generated with S2AND tooling, such as
`s2and.incremental_linking.feature_block.write_raw_arrow_batch_lookup_indexes`.
Do not hand-write the binary format in an independent pipeline. Do not generate
these indexes before a later rewrite or deployment copy that changes the source
Arrow file metadata; regenerate the indexes from the final files in their
serving location.

Recommended sidecar filenames are stem-qualified:

```text
signatures.signatures_batch_index.bin
papers.papers_batch_index.bin
paper_authors.paper_authors_batch_index.bin
specter.specter_batch_index.bin
specter2.specter_batch_index.bin
```

When both `specter.arrow` and `specter2.arrow` are present, write one embedding
index per file. At runtime, the selected embedding file is passed under the
`specter` path key, and S2AND uses the adjacent
`<embedding-stem>.specter_batch_index.bin` sidecar when present.

The batch-index format is S2AND-owned and versioned by
`arrow_batch_lookup_index_v1`. Each record maps a 64-bit FNV-1a hash of the
lookup key to an IPC record-batch index; the Rust reader verifies exact ids
after loading the selected batches, so hash collisions do not change results.

---

## Source Semantics

Rows must match the values that S2AND would expose after normal preprocessing:

- `preprocess=True`
- `use_sinonym_overwrite=False`
- `use_orcid_id=True`
- `name_tuples="filtered"`
- `name_counts_index/` available when the selected model uses name-count features

Use the existing `FeatureBlock` writer as the reference implementation:
`s2and.incremental_linking.feature_block.write_feature_block_arrow_from_anddata`.
An independent assembly pipeline is fine, but it must produce the same values as
that writer.

Important parity details:

- Preserve source signature order. The current converter writes
  `signature_ids=list(dataset_obj.signatures)` for this reason.
- Store ids as strings, even if an upstream source stores numeric ids.
- Use the same post-preprocessing author and paper fields as `ANDData`.
- Keep `abstract` as an abstract-presence signal, not raw abstract text. The
  current `FeatureBlock` encoding writes `"Has Abstract"` when the preprocessed
  paper has an abstract and `""` otherwise.
- Include all paper-author rows needed for coauthor features.
- Do not include embedded name-count columns in `signatures.arrow`; use the
  shared `name_counts_index/` sidecar.

---

## Table Schemas

### `signatures.arrow`

One row per signature. Required columns:

| Column | Arrow type | Nulls | Meaning |
|---|---:|---:|---|
| `signature_id` | `string` | no | Stable signature id |
| `paper_id` | `string` | no | Referenced paper id |
| `author_first` | `string` | yes | Preprocessed author first name |
| `author_middle` | `string` | yes | Preprocessed author middle name |
| `author_last` | `string` | yes | Preprocessed author last name |
| `author_suffix` | `string` | yes | Preprocessed author suffix |
| `author_affiliations` | `list<string>` | yes | Author affiliations; prefer empty list over null |
| `author_orcid` | `string` | yes | ORCID value used by S2AND |
| `author_position` | `int64` | yes | Author position on the paper |
| `author_block` | `string` | yes | S2 block key, needed for block reconstruction/eval |
| `author_email` | `string` | yes | Author email |
| `source_author_ids` | `list<string>` | yes | Upstream author ids |

Name-count values are intentionally not part of the signature table.

### `papers.arrow`

One row per paper referenced by `signatures.arrow`. Required columns:

| Column | Arrow type | Nulls | Meaning |
|---|---:|---:|---|
| `paper_id` | `string` | no | Stable paper id |
| `title` | `string` | yes | Preprocessed title |
| `abstract` | `string` | yes | Abstract-presence signal: `"Has Abstract"` or `""` |
| `venue` | `string` | yes | Preprocessed venue |
| `journal_name` | `string` | yes | Preprocessed journal name |
| `year` | `int64` | yes | Publication year |
| `predicted_language` | `string` | yes | Predicted language if available |
| `is_reliable` | `bool` | yes | S2AND reliability flag if available |

### `paper_authors.arrow`

One row per paper-author child row. Required columns:

| Column | Arrow type | Nulls | Meaning |
|---|---:|---:|---|
| `paper_id` | `string` | no | Referenced paper id |
| `position` | `int64` | no | Author position |
| `author_name` | `string` | no | Author display/name string used by coauthor features |

Rows should be ordered by `paper_id` then `position` where practical. Ordering is
not the identity contract, but stable ordering makes diffs and validation easier.

### `specter.arrow` and `specter2.arrow`

One row per embedded paper. Required columns:

| Column | Arrow type | Nulls | Meaning |
|---|---:|---:|---|
| `paper_id` | `string` | no | Referenced paper id |
| `embedding` | `fixed_size_list<float32>[dimension]` | no | SPECTER vector |

All vectors in one file must have the same dimension. If the model uses
`embedding_similarity`, every paper referenced by `signatures.arrow` should have
an embedding row for the selected embedding version. Missing embeddings can
change scores and should fail validation unless the target model explicitly
permits them.

### `cluster_seeds.arrow`

Required for incremental/seeded prediction through the Arrow promoted path.
Optional for unseeded full prediction.

| Column | Arrow type | Nulls | Meaning |
|---|---:|---:|---|
| `signature_id` | `string` | no | Seed signature id |
| `cluster_id` | `string` | no | Required seed component/cluster id |

Only required seed assignments are persisted here. Pairwise seed disallow
constraints are persisted separately in `cluster_seed_disallows.arrow`.

### `cluster_seed_disallows.arrow`

Required for incremental/seeded prediction through the Arrow promoted path.
Write an empty table when no seed disallows are present.

| Column | Arrow type | Nulls | Meaning |
|---|---:|---:|---|
| `signature_id_1` | `string` | no | First signature id in the disallow pair |
| `signature_id_2` | `string` | no | Second signature id in the disallow pair |

Each id must exist in `signatures.arrow`. Runtime treats the pair as
undirected, matching existing `cluster_seeds_disallow` semantics.

### `altered_cluster_signatures.arrow`

Required for incremental prediction when the request includes altered claimed
profiles. Omit it, or write an empty table, when no altered profiles are
present.

Required columns:

| Column | Arrow type | Nulls | Meaning |
|---|---:|---:|---|
| `signature_id` | `string` | no | Seed signature id belonging to an altered claimed profile |

Each id must exist in `signatures.arrow` and in `cluster_seeds.arrow`. At
runtime, S2AND maps these signature ids through the seed assignments to identify
the claimed seed components that need altered-profile pre-splitting.

`altered_cluster_signatures.txt` with one signature id per line is still
accepted by the Python runtime for older fixtures and ANDData-compatible tooling,
but new production Arrow producers should emit the Arrow table.

### `<dataset>_clusters.json`

Eval-only truth data. Keep the same shape as existing S2AND clusters JSON:

```json
{
  "cluster_id": {
    "signature_ids": ["signature_a", "signature_b"]
  }
}
```

Production prediction does not need this file.

---

## Name Counts

Preferred production layout:

1. Provide a shared/global `s2and/data/name_counts_index/` sidecar when the
   selected model uses name-count features.
2. Keep `name_counts.arrow` only for generation, inspection, and parity
   debugging.

The shared binary index layout is:

```text
s2and/data/name_counts_index/
  manifest.json
  first.bin
  last.bin
  first_last.bin
  last_first_initial.bin
```

`manifest.json` must have `schema_version: "name_counts_index_v1"`.

Do not build a pipeline that loads `name_counts.arrow` into Python dicts/lists
for production request handling. That defeats the purpose of this contract.

---

## Name Aliases

Production datasets must not contain per-dataset `name_pairs.arrow` files. The
runtime default is the packaged filtered alias file:

```text
s2and_name_tuples_filtered.txt
```

If a non-default alias set is ever needed, make it an explicit shared/global
runtime artifact, not something duplicated into every dataset directory.

---

## Manifests

Each dataset directory must contain `manifest.json`. The manifest is not the hot
path source of truth, but it is required for auditability and validation.

Required fields for semantic Arrow artifacts:

```json
{
  "schema": "feature_block_arrow_v2",
  "dataset": "dataset_name",
  "signature_count": 0,
  "paper_count": 0,
  "paths": {
    "signatures": "signatures.arrow",
    "papers": "papers.arrow",
    "paper_authors": "paper_authors.arrow",
    "cluster_seeds": "cluster_seeds.arrow",
    "cluster_seed_disallows": "cluster_seed_disallows.arrow"
  },
  "name_tuples": "default packaged filtered aliases"
}
```

Large-block optimized artifacts should also include:

```json
{
  "paths": {
    "signatures_batch_index": "signatures.signatures_batch_index.bin",
    "papers_batch_index": "papers.papers_batch_index.bin",
    "paper_authors_batch_index": "paper_authors.paper_authors_batch_index.bin",
    "specter_batch_index": "specter.specter_batch_index.bin"
  },
  "physical_layout": {
    "schema": "s2and_arrow_physical_v1",
    "optimized_for": "incremental_raw_candidate_planning",
    "tables": {
      "signatures": {
        "key": "signature_id",
        "max_record_batch_rows": 16384,
        "row_count": 0,
        "record_batch_count": 0,
        "actual_max_batch_rows": 0,
        "batch_index_path_key": "signatures_batch_index",
        "batch_index_present": true
      }
    }
  }
}
```

Recommended additional fields:

- `cluster_count` for eval datasets.
- `source_dir` or source snapshot identifier.
- `generated_at`.
- `generator_version` or git commit.
- `specter` metadata with `row_count`, `dimension`, and source artifact id for
  each embedding file.
- `name_counts` metadata with the shared index path and schema version.
- `physical_layout.tables.<table>` entries for every large lookup table:
  `row_count`, `record_batch_count`, `actual_max_batch_rows`,
  `max_record_batch_rows`, lookup `key`, and batch-index presence.
- `raw_planner_batch_indexes` metrics when S2AND-generated sidecars are present.
- `validation` summary with row counts, duplicate counts, missing reference
  counts, physical-layout checks, and parity-check command/output location.

Root-level `manifest.json` should list dataset directories and their manifest
paths when an artifact bundle contains multiple datasets.

---

## Validation Checklist

Validate every generated dataset before handing it to model evaluation or
production inference.

Required checks:

- Every Arrow file opens with `pyarrow.ipc.open_file(...)`.
- Required files exist for the intended use case.
- Required columns exist with the exact Arrow types above.
- `signature_id` values are unique.
- `paper_id` values in `papers.arrow` are unique.
- Every `signatures.paper_id` exists in `papers.arrow`.
- Every `paper_authors.paper_id` exists in `papers.arrow`.
- When embeddings are required, every referenced paper has an embedding in the
  selected SPECTER file.
- `cluster_seeds.signature_id` is a subset of `signatures.signature_id`.
- `cluster_seed_disallows.signature_id_1` and
  `cluster_seed_disallows.signature_id_2` are subsets of
  `signatures.signature_id`.
- `name_counts_index/manifest.json` exists when the selected model uses
  `name_counts`.
- `author_block` is present when the dataset will be used for block
  reconstruction or offline eval.
- Signature row order matches the source `ANDData` order or the documented
  source order for that dataset.
- Eval-only clusters JSON references only signatures present in
  `signatures.arrow`.

Required physical-layout checks for large-block optimized artifacts:

- `signatures.arrow`, `papers.arrow`, `paper_authors.arrow`, and the selected
  embedding file are bounded as specified in
  [Large-Block Physical Layout](#large-block-physical-layout).
- `physical_layout.schema` is `s2and_arrow_physical_v1`.
- `physical_layout.tables.<table>.actual_max_batch_rows` is less than or equal
  to `physical_layout.tables.<table>.max_record_batch_rows`.
- One-batch lookup tables have
  `row_count <= max_record_batch_rows`; otherwise they should be rejected as
  unoptimized for indexed raw planning.
- If batch-index sidecars are present, they were generated from the final Arrow
  files and the manifest path keys point to those sidecars.

Recommended smoke checks:

```powershell
uv run python scripts/convert_s2and_mini_to_arrow.py `
  --source-root s2and/data/s2and_mini `
  --output-root s2and/data/s2and_mini_arrow `
  --n-jobs 20 `
  --overwrite
```

```powershell
$env:S2AND_BACKEND='rust'
uv run python scripts/eval_prod_models.py --dataset mini --n_jobs 20 --seed 42
```

The eval command should report `use_arrow=True` when the Arrow bundle is
complete.

---

## Non-Goals

This Arrow dataset contract is not a full `ANDData` replacement. Do not include
training pair samples, train/val/test split construction artifacts, reference
features, sinonym overwrite outputs, or pair-sampling policy state unless a
separate training/eval contract explicitly asks for them.

The direct Rust inference path should consume only the narrow feature-block
inputs it needs for scoring and clustering.
