## Refactor Backlog

## Focused scope: deferred structural refactors (2026-05-22)

These four items are real refactors, not quick cleanup. Treat them as separate
PR-sized changes unless a later implementation pass proves two are naturally
coupled. The goal is to shrink API ambiguity and duplication without changing
scoring behavior, Arrow formats, or public data contracts.

### Plan-wide implementation gates

- Treat "behavior-preserving" as a testable claim. Each implementation PR must
  include output/telemetry parity checks and either run a relevant bounded perf
  workload with no >5% regression on the same command/environment, or explicitly
  state that perf is out of scope because the changed code is not on a measured
  hot path.
- Every PR that touches Python files, tests, or wrappers must run
  `uv run ruff check <touched-python-paths>` and
  `uv run ty check <touched-python-paths-or-.>`. Rust-only PRs do not need
  Python lint/typecheck unless they also update Python tests or wrapper code.
- Items 2, 3, and 4 all edit [`lib.rs`](s2and_rust/src/lib.rs). Treat them as
  sequential work, not parallelizable branches. Item 2 should rebase over Items
  3 and 4 if either has landed, because helper/state cleanup may change the
  constructor surface it is deduplicating.

### 1. `TemporaryArrowPaths` contract rewrite

**Current surface.** `TemporaryArrowPaths` lives in
[`feature_block_arrow.py`](s2and/incremental_linking/feature_block_arrow.py) and is
re-exported through [`feature_block.py`](s2and/incremental_linking/feature_block.py).
`arrow_paths_with_temporary_cluster_seeds(...)` returns either an owning bundle
with a tempdir or a non-owning bundle when existing seed paths can be reused.
Production callers manually close the bundle in
[`model.py`](s2and/model.py) and
[`production.py`](s2and/incremental_linking/production.py). Tests monkeypatch
the factory directly. The class already implements `__enter__` / `__exit__`,
but those methods yield the bundle object, not the Arrow path payload dict that
production callers actually pass through the planner.

**Problem to solve.** The type has two modes but one interface, so callers must
remember to call `.close()` even when the current instance owns no resource.
That is safe today because `.close()` is a no-op for non-owning bundles, but the
ownership contract is implicit.

**Scoped change.**
- Add an explicit context-manager factory, for example
  `temporary_arrow_paths_with_cluster_seeds(...)`, that yields `dict[str, str]`.
- Keep `TemporaryArrowPaths` and `arrow_paths_with_temporary_cluster_seeds(...)`
  as a compatibility layer for at least one staged patch. They are public
  re-exports and tests monkeypatch the factory directly. If the compatibility
  layer is retained, keep one regression test on the old factory path; otherwise
  delete it in the same patch only as an explicit approved public-API removal.
- Convert production callers first so temp ownership is lexical:
  `with temporary_arrow_paths_with_cluster_seeds(...) as arrow_path_payload: ...`.
- In `production.py`, the `with` scope must cover every use of
  `arrow_path_payload`, including raw planning/scoring and
  `_finish_incremental_with_optional_split_inverse(...)`.
- Update tests to assert cleanup through the context manager rather than by
  manually calling `.close()`.

**Non-goals.** Do not change Arrow path key names, seed/disallow file contents,
or reuse policy. Do not alter temporary directory prefix behavior except where
needed to preserve existing tests. Do not move temporary Arrow creation inside
batch, block, or window loops.

**Verification.**
- Focused pytest: `tests/test_cluster_incremental.py` seed-bundle cleanup tests
  and `tests/test_feature_block.py` temporary Arrow path tests.
- Include `tests/test_regression_fixes.py`, which covers the altered-presplit
  temporary seed path and restoration behavior.
- Add or update a regression target for the `production.py` scope: the temp
  bundle must remain live through raw planning/scoring and
  `_finish_incremental_with_optional_split_inverse(...)`.
- Include `tests/test_promoted_*.py` only if the implementation touches promoted
  training, materializer, or CLI surfaces; the direct production caller coverage
  is otherwise in `tests/test_cluster_incremental.py` plus altered-presplit
  coverage in `tests/test_regression_fixes.py`.
- `uv run ruff check s2and/incremental_linking/feature_block_arrow.py s2and/model.py s2and/incremental_linking/production.py tests/test_cluster_incremental.py tests/test_feature_block.py`.
- `uv run ty check s2and/incremental_linking/feature_block_arrow.py s2and/model.py s2and/incremental_linking/production.py tests/test_cluster_incremental.py tests/test_feature_block.py tests/test_regression_fixes.py`.

**Risk.** Medium. The behavior is simple, but missing a production close path can
leak temp dirs; changing the return type too aggressively can break tests and
downstream imports. Closing the temp bundle before the residual finish path is a
correctness bug. Latency risk is low if the new factory remains request-scoped
and does not create temp Arrow files inside nested loops. Prefer a staged
compatibility patch.

### 2. Rust staging-record deduplication

**Current surface.** [`lib.rs`](s2and_rust/src/lib.rs) has separate staging
record families for:
- JSON/path construction: `SignatureInput`, `PaperInput`, `PaperPreprocessed`.
- Arrow construction: `ArrowSignatureInput`, `ArrowPaperInput`,
  `ArrowPaperPreprocessed`.
- FeatureBlock construction: `FeatureBlockSignatureInput`,
  `FeatureBlockPaperInput`, `FeatureBlockPaperPreprocessed`.

The fields and preprocessing shape are nearly identical, but extraction/default
rules are duplicated in each constructor.

**Problem to solve.** Duplication lets parsing/defaulting drift between ingestion
paths. The earlier `position` extraction bug came from exactly this shape: one
path accepted malformed values that others rejected.

**Scoped change.**
- Introduce shared internal structs for the common post-extraction shape, for
  example `SignatureStageInput`, `PaperStageInput`, and
  `PaperStagePreprocessed`.
- Keep source-specific extraction, validation, defaulting, ORCID policy,
  language-source policy, reference-feature handling, and name-count telemetry
  outside the shared helper or behind explicit policy parameters:
  `signature_stage_from_arrow(...)`, `signature_stage_from_feature_block(...)`,
  `signature_stage_from_json(...)`.
- Centralize shared preprocessing from `StageInput -> Preprocessed` so
  unidecode, language detection, name splitting, name counts, and paper-author
  handling use one implementation where the source policies match.
- Migrate one constructor first, preferably `from_arrow_paths`, then
  `from_feature_block`, then JSON/path construction. Each migration should be
  behavior-preserving.
- Do not introduce shared stage types until at least two constructors use them.
  During transition, avoid a half-shared helper that serves one constructor and
  can drift from the others.
- Keep `from_dataset` out of scope for this pass because it has distinct
  partial-precompute behavior. It stays on the old construction path, and no
  helper called by `from_dataset` should be forced to depend on the new shared
  stage types.

**Non-goals.** Do not rewrite Arrow readers, change Python-visible method
signatures, change telemetry keys, or change missing/null default policy except
where tests already require stricter behavior. Do not move filtering later in
the pipeline: Arrow construction must still filter selected signatures and
needed papers before preprocessing. Do not pull Rayon-backed preprocessing back
into serial extraction, and do not extend JSON ingest lifetimes for loaded
signature/paper rows.

**Verification.**
- `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test --manifest-path s2and_rust/Cargo.toml -q --lib`.
- Focused Python/Rust integration tests covering all constructors:
  `tests/test_rust_from_json_paths.py`,
  `tests/test_feature_block.py`,
  `tests/test_rust_distance_matrix_blockwise.py`, and raw Arrow ingest tests.
- Add mandatory regressions for malformed `paper_authors.position` across
  Arrow, FeatureBlock, and JSON paths.
- Add parity checks for telemetry keys/default counts, language fields,
  ORCID/source-id behavior, unidecode/non-ASCII names, name-count
  presence/default telemetry, and `preprocess=false`.
- After each constructor migration, capture a per-step parity snapshot covering
  representative feature vectors, constraints, constructor telemetry keys, and
  relevant default/error cases before moving to the next constructor.
- Run `uv run ruff check` and `uv run ty check` on any Python wrappers/tests
  touched by the migration.
- Run the relevant bounded constructor/build perf workload and report no >5%
  regression, or explicitly document that no perf claim was measured for that
  patch.

**Risk.** High. This touches all Rust featurizer construction paths. Correctness
risk is concentrated in source-specific policy drift: Arrow rejects some
malformed values that JSON currently treats as missing, JSON has different
language and ORCID derivation behavior, and JSON carries name-count default
telemetry that Arrow/FeatureBlock do not. Keep each constructor migration
separate and compare telemetry/feature parity after each step.

### 3. Rust matrix-entrypoint scaffolding refactor

**Current surface.** [`lib.rs`](s2and_rust/src/lib.rs) has several Python-facing
matrix entrypoints that repeat the same pattern: validate indices, resolve
selected columns, allocate a flat buffer, compute rows, build `Array2` via
`from_shape_vec`, and map shape errors. This occurs across pair feature
matrices, indexed pair matrices, linker pair aggregate methods, and block
upper-triangle matrix methods.

**Problem to solve.** The repeated scaffolding makes error handling and bounds
checks drift. It also makes smaller performance work harder because each
entrypoint must be patched separately.

**Scoped change.**
- Start with tiny shared helpers, not a generic framework:
  - `array2_from_vec(context, rows, cols, values) -> PyResult<Array2<T>>`.
  - `validate_feature_indices(indices, full_cols, arg_name, context)`.
  - `matrix_positions_for_indices(matrix_indices, selected_indices, context)`,
    defined as order-preserving and non-deduplicating.
- In helper names/errors, `arg_name` means the Python argument label such as
  `selected_indices`, `matrix_indices`, or `aggregate_indices`; `context` means
  the Rust entrypoint or operation name for diagnosis.
- Replace only the `Array2::from_shape_vec(...).map_err(...)` boilerplate first.
- In a second patch, factor selected-index validation where the resulting helper
  removes duplicated branches without hiding control flow.
- Do not combine pair-feature matrix and aggregate-stat kernels until the helper
  boundaries are proven by tests.

**Non-goals.** Do not change numerical kernels, row ordering, NaN handling,
threading, or Python return tuple shapes. Preserve selected-index order and
duplicates exactly; do not sort, dedupe, or convert requested indices to sets.
Keep `matrix_indices` and `aggregate_indices` semantics distinct, including the
existing error shape when aggregate indices are absent from matrix indices. Do
not abstract NaN policy across aggregate paths yet. Do not introduce
trait-heavy generic abstractions unless the helper call sites stay easy to
read.

**Verification.**
- Rust library tests plus focused Python tests for distance matrices, linker
  runtime batches, and raw candidate plans:
  `tests/test_rust_distance_matrix_blockwise.py`,
  `tests/test_linker_runtime_batch.py`,
  `tests/test_raw_block_candidate_plan_arrow.py`.
- Add focused tests for out-of-range `selected_indices`, out-of-range
  `aggregate_indices`, duplicate selected indices preserving duplicate output
  columns, including `selected_indices=[2, 2, 3]` producing a 3-column matrix,
  row order in upper-triangle block APIs, and unchanged 5-tuple/6-tuple wrapper
  contracts.
- Run `uv run ruff check` and `uv run ty check` on Python wrappers/tests touched
  by the helper extraction.
- Run a bounded matrix-entrypoint perf workload and report no >5% regression, or
  explicitly state that the patch was array/error scaffolding only and did not
  measure a perf claim.

**Risk.** Medium-high. The safe first step is helper extraction for array/error
construction only. Helper extraction should stay outside hot loops and preserve
the current `py.allow_threads` / optional Rayon boundaries. Kernel consolidation
is a separate, riskier change.

### 4. Manual `parallel_drop` restructuring

**Current surface.** At the end of `raw_block_query_candidate_plan_arrow(...)` in
[`lib.rs`](s2and_rust/src/lib.rs), many large staging collections are manually
destructured and dropped with `parallel_drop_hashmap(...)` /
`parallel_drop_vec(...)`.

**Problem to solve.** The manual drop list is easy to forget when new large
fields are added. It also mixes cleanup policy into the planner body, making the
function harder to review.

**Scoped change.**
- Introduce a private staging-state struct for the raw Arrow planner, for
  example `RawArrowCandidatePlanState`, that owns the large maps/vectors created
  during planning.
- Implement cleanup mechanically around the current manual drop list. The first
  patch should not add omitted collections such as `required_signature_ids` or
  the unidecode character map unless profiling proves they matter.
- Do not implement `Drop` for the first pass. Use an explicit success-path
  cleanup function/method that consumes the state after payload conversion.
- Move fields into the state only when they are fully built and no longer need
  independent ownership, then borrow from the state during scoring/assembly.
- On the success path, explicitly drop the state after `payload.unbind()` under
  `py.allow_threads(...)`, preserving `install_with_optional_rayon_pool(...)`,
  then record `drop_secs` and `wall_secs` exactly as today.
- On error paths before the explicit cleanup block, either accept ordinary Rust
  drop as a correctness-over-latency fallback and document that choice, or route
  errors through the same `allow_threads` cleanup path. Do not leave this
  behavior implicit.
- Keep `RustHybridCentroidRetriever` ownership semantics explicit. If the state
  owns it, destructure it exactly as today: parallel-drop `summaries`, plain-drop
  `component_index_by_key`, `coauthor_cluster_df`,
  `non_mega_coauthor_cluster_df`, and `affiliation_cluster_df`.
- Keep output assembly and telemetry unchanged.

**Non-goals.** Do not change retrieval scoring, candidate row ordering, telemetry
fields, Arrow read behavior, or the existing `parallel_drop_*` helper semantics.
Do not move the entire planner into `py.allow_threads` in the same patch. Do not
put Python-bound objects in the state.

**Verification.**
- `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test --manifest-path s2and_rust/Cargo.toml -q --lib`.
- Focused Python tests:
  `tests/test_raw_block_candidate_plan_arrow.py`,
  `tests/test_cluster_incremental.py` raw Arrow promoted-window cases.
- Add a focused raw-plan smoke assertion that telemetry timings still contain
  numeric `drop_secs` and `wall_secs`. Do not try to assert memory freeing
  itself.
- Run `uv run ruff check` and `uv run ty check` on Python tests/wrappers touched
  by the restructuring.
- Run a bounded raw-plan perf workload and report no >5% regression on the
  success path, or explicitly state that perf was not measured for that patch.

**Risk.** Medium. The main risk is Rust borrow/lifetime churn while moving maps
into a state struct. A lexical `Drop` at function return is not acceptable for
the first pass because it would run under the wrong GIL/threading conditions and
would break timing semantics. Keep the first patch mechanical: same fields, same
drop behavior, explicit success-path cleanup after payload conversion, no
scoring changes.

### Recommended order

1. `TemporaryArrowPaths` context-manager contract. Smallest behavioral surface,
   mostly Python tests.
2. Rust matrix scaffolding helper extraction. Good cleanup with narrow helper
   boundaries if limited to array/error helpers first.
3. Manual `parallel_drop` state struct with explicit success-path cleanup.
   Isolated to one large function, but borrow-checker risk is real.
4. Rust staging-record deduplication. Highest blast radius; do this last and in
   constructor-by-constructor patches.

### Companion backlog hygiene

Small one-off items from the earlier review, such as the deprecated
`predict_incremental_helper` shim, SPECTER clone cleanup, and
`_raw_plan_query_views` side-effect cleanup, are not part of these four scoped
structural refactors. Track them in a separate quick-wins list or handle them as
explicit one-off PRs; do not hide them inside the large refactor PRs.
