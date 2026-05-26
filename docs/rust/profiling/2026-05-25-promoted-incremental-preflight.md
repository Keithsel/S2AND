# Promoted Incremental Profiling Preflight (2026-05-25)

Date: 2026-05-25

## Goal

Prepare the next profiling pass for Arrow promoted-incremental inference. The
work-plan target is Arrow read/summary construction and reusable component
summaries on a realistic `s2and_and_big_blocks_linker_dataset_20260525`
workload.

## Tiny Preflight

Command:

```powershell
uv run pytest -q tests/test_eval_prod_models.py::test_pubmed_specter2_arrow_fixture_incremental_smoke_matches_expected_b3
```

Result:

- `1 passed`
- Existing `hyperopt` / `pkg_resources` deprecation warning only.

The fixture exercises `Clusterer.predict_incremental(...)` through the promoted
Arrow/Rust path on `tests/fixtures/arrow/pubmed_specter2`. The test asserts
`arrow_promoted_incremental == 1`, `seed_setup_cluster_seeds_source == "arrow"`,
`seed_arrow_reused_source == 1`, total query count `127`, nonzero candidate
rows, and the expected B3 score.

## Full-Run Blocker

The local checkout currently has:

- `s2and/data/s2and_and_big_blocks_linker_dataset_20260513_arrow`
- no `s2and/data/s2and_and_big_blocks_linker_dataset_20260525`

The existing `scripts/_rust_suite/big_block_incremental_cmd.py` remains a
JSON/`ANDData` measurement path and is not a valid benchmark for the Arrow-only
promoted-incremental performance target.

## Decision Needed

Before starting a multi-hour profiling run, choose the data and runner:

- Sync or otherwise materialize the published
  `s2and_and_big_blocks_linker_dataset_20260525` bundle locally, then run a
  dedicated Arrow promoted-incremental profiler against it.
- Or explicitly treat the local
  `s2and_and_big_blocks_linker_dataset_20260513_arrow` directory as the
  refreshed `20260525` equivalent for profiling, and record that alias in the
  profiling report.

Do not use the legacy JSON/`ANDData` big-block command for this target unless
the goal changes to compatibility-path profiling.
