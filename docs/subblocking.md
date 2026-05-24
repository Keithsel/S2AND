# Subblocking

Subblocking is used by full-block `Clusterer.predict(...)` to keep pairwise distance construction bounded for large
blocks. The core boundary is `make_subblocks_with_telemetry(...)`; `make_subblocks(...)` returns only the final
subblock dictionary.

For `Clusterer.predict(..., batching_threshold=N)`, `N` becomes the maximum subblock size. Every emitted subblock is
expected to have at most `N` signatures.

## Current default

The current default is graph-first subblocking fallback:

- Prefix and middle-name splitting still define the main subblock structure.
- Oversized groups that cannot be split by names call the graph fallback instead of the old SPECTER fallback.
- If graph preparation or a graph fallback call fails, the old Python `cluster_with_specter(...)` path runs for that
  fallback group.
- Set `clusterer.subblocking_fallback_mode = "legacy"` to bypass graph fallback and use the old behavior directly.

The default is configured on `Clusterer`:

```python
clusterer.subblocking_fallback_mode = "graph"
clusterer.subblocking_graph_config = GraphSubblockingConfig()
```

## Partition flow

Subblocking proceeds in this order:

1. Split signatures by first-name prefix.
2. For first-name groups that remain too large, split by middle-name prefix.
3. For groups that are still too large, call the fallback cluster function. By default this is the graph fallback.
4. Merge compatible small subblocks back up to the maximum size.
5. Run the optional same-ORCID repair pass.

The single-letter first-name path is handled later in bulk prediction. Multi-letter subblocks are predicted first,
their resulting clusters become temporary cluster seeds, and single-letter subblocks run through a synthetic
incremental pass so initial-only signatures can attach back to established clusters.

## ORCID policy

`make_subblocks(...)` and `make_subblocks_with_telemetry(...)` have a `use_orcid_subblocking` flag. When enabled, the
final repair pass groups same-ORCID signatures into one subblock only when an existing target subblock can absorb the
whole ORCID group without exceeding `maximum_size`. Otherwise the split is preserved and telemetry records the
capacity skip.

The ORCID key is canonicalized to match Rust Arrow ingestion: keep digits and `X`/`x`, require exactly 16 ORCID
characters, uppercase the check digit, and format as `0000-0000-0000-0000`. Blank or invalid values are ignored.
This subblocking policy is independent from same-ORCID hard-link distance constraints.

## Graph fallback

The graph fallback builds a capacity-constrained graph over only the oversized fallback group. It uses normalized
SPECTER embeddings plus coauthor and affiliation evidence to score candidate edges, then greedily unions edges while
respecting `target_subblock_size`.

Default `GraphSubblockingConfig` behavior:

- `neighbor_mode="projection"` with `projection_count=12` and `projection_window=12`.
- `min_edge_score=0.30`.
- `component_pack_strategy="edge-greedy"` and `pack_components=True`.
- Exact kNN remains available with `neighbor_mode="exact"`, but it is capped by `max_exact_knn_group_size`.
- Sparse-evidence edges, adaptive projection, aggregate packing, and local moves are still experimental knobs and are
  off by default.

When Arrow paths are available, the graph fallback is Arrow-backed. Before fallback calls, `prepare(...)` receives the
actual fallback-signature groups and loads the union of required `signatures`, `paper_authors`, and `specter` rows
into memory. Each fallback call then slices that in-memory evidence for its group. Without Arrow paths, the graph
fallback uses the active `ANDData` object.

Graph fallback is intentionally wrapped with the old Python SPECTER fallback. Graph failures are not swallowed:
warnings are logged, telemetry records the failure, and then `cluster_with_specter(...)` runs for the affected group.

## Python and Rust routing

Subblocking has two supported routes:

- **Pure Python.** Direct calls to `make_subblocks(...)` and `make_subblocks_with_telemetry(...)` use the original
  Python implementation.
- **Arrow-native Rust.** `Clusterer.predict(...)` can route oversized blocks to Rust when
  the call's resolved backend is Rust and indexed Arrow signature rows are available.

For `Clusterer.predict(...)` with Arrow paths, Rust Arrow subblocking is used only when all of these are true:

- `Clusterer.predict(..., backend="rust")` is used, or its `runtime_context` resolves to Rust.
- `arrow_paths["signatures"]` is present.
- `arrow_paths["signatures_batch_index"]` is present.

That path calls `make_subblocks_arrow_rust(...)`, so Rust loads the name and ORCID rows needed for subblocking from
Arrow. If those conditions are not met, prediction falls back to Python subblocking orchestration. In both Rust and
Python orchestration, the oversized fallback groups still call the configured fallback callable, so graph fallback
remains the default unless `subblocking_fallback_mode="legacy"`.

## Telemetry

`make_subblocks_with_telemetry(...)` returns the final subblocks plus telemetry for the partition process, including:

- input and single-letter/multi-letter signature counts
- first-name dead-end counts
- fallback candidate and invocation counts
- pre-merge and final subblock counts
- ORCID repair capacity skips
- final SPECTER-labeled subblock counts

`Clusterer.predict(...)` also records graph hook telemetry on `_last_graph_subblocking_telemetry` and
`_last_arrow_graph_subblocking_telemetry`:

- `enabled`, `mode`, `source`, and `candidate_signature_count`
- `arrow_load_seconds` and `arrow_load_metrics`
- `fallback_invocation_count` and per-group `fallback_stats`
- `legacy_fallback_invocation_count`
- `graph_prepare_failed`, `graph_prepare_error`, and `graph_fallback_errors`

`fallback_stats` includes per-group graph details such as candidate edge count, raw and packed component counts,
maximum component size, edge-build seconds, and total fallback seconds.

## Incremental routing

Incremental prediction has two supported routes:

- **Promoted Rust linker.** Used when the resolved backend is Rust, the Rust extension exposes the required
  promoted-incremental capabilities, and seed inputs are available, either as an `ANDData` seed map or a
  `cluster_seeds` entry in `dataset.arrow_paths`. When complete Arrow artifacts are also available, retrieval and
  scoring run directly against the Arrow tables. Otherwise the runtime builds the Rust featurizer from Python state.
- **Python fallback helper.** Used when the backend resolves to Python, the Rust extension lacks the required
  capabilities, or no seed inputs are provided. This path covers partition coverage but does not implement batched
  incremental routing.

`batching_threshold` controls two things. For full-block prediction it caps subblock size. For promoted Rust
incremental prediction it caps the number of unassigned query signatures per linker batch. The standalone Python
incremental fallback rejects `batching_threshold` with a `ValueError`; pass `batching_threshold=None` on that path or
use the Rust backend with seed inputs.

See [production_inference.md](production_inference.md#large-blocks-and-incremental-inference) for the full
caller-facing contract.
