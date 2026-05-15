# Subblocking

Subblocking is used by full-block `Clusterer.predict(...)` to keep distance-matrix construction bounded for large
blocks. `make_subblocks(...)` is the block-construction boundary; prediction dispatch no longer has a separate
single-letter incremental phase-split path.

Incremental prediction now has two supported routes:

- promoted Rust linker when the runtime backend resolves to Rust and cluster seeds are present
- one Python fallback helper for no-seed coverage or Python backend execution

`batching_threshold` still matters for promoted Rust incremental query batching. In the Python incremental fallback it is
rejected because that path does not implement batched incremental routing.
