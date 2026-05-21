# Work Plan

Status date: 2026-05-20

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
- Include `s2and/data/name_counts_index/` as the shared name-count artifact for
  runtime bundles that use name-count features.
- Keep `name_counts.arrow` available for generation, inspection, and parity
  debugging, not as the default request-time read.

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
