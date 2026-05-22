# Subblocking

Subblocking is used by full-block `Clusterer.predict(...)` to keep distance-matrix construction bounded for large
blocks. `make_subblocks(...)` is the block-construction boundary.

## Single-letter pass

Bulk subblocked prediction partitions blocks by first-name initial. Subblocks whose first-name key is a single
letter (typically initial-only signatures) are deferred: the multi-letter subblocks are predicted first, the
resulting clusters become temporary cluster seeds, and the single-letter subblocks then run through a synthetic
incremental pass that can merge those signatures back into the established clusters instead of forming their own
shallow components.

## Incremental routing

Incremental prediction has two supported routes:

- **Promoted Rust linker.** Used when the resolved backend is Rust, the Rust extension exposes the required
  promoted-incremental capabilities, and seed inputs are available (either an `ANDData` seed map or a
  `cluster_seeds` entry in `dataset.arrow_paths`). When complete Arrow artifacts are also available, retrieval and
  scoring run directly against the Arrow tables; otherwise the runtime builds the Rust featurizer from the
  Python state.
- **Python fallback helper.** Used when the backend resolves to Python, the Rust extension lacks the required
  capabilities, or no seed inputs are provided. This path covers partition coverage but does not implement
  batched incremental routing.

`batching_threshold` controls two things at once. For full-block prediction it caps subblock size so each subblock
fits the configured `desired_memory_use`. For promoted Rust incremental prediction it caps the number of
unassigned query signatures per linker batch. The standalone Python incremental fallback rejects
`batching_threshold` with a `ValueError` — pass `batching_threshold=None` on that path or use the Rust backend
with seed inputs.

See [production_inference.md](production_inference.md#large-blocks-and-incremental-inference) for the full
caller-facing contract.
