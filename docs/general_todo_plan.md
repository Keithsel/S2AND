# General TODO Plan

Status date: 2026-05-27

This is a code-checked consolidation of the general TODO pile described across
the repository documentation. It does not replace
[work_plan.md](work_plan.md), which remains the active Rust/Arrow execution
plan. It exists to show what the docs collectively say is still open, what is
blocked, and what the current code confirms.

## Source Scan

Scanned:

- [README.md](../README.md)
- [AGENTS.md](../AGENTS.md)
- `docs/` recursively
- [scripts/README.md](../scripts/README.md)
- [scripts/production/README.md](../scripts/production/README.md)
- [s2and_rust/README.md](../s2and_rust/README.md)
- Doc-like legacy files:
  [paper_experiments_env.txt](../paper_experiments_env.txt) and
  [tutorial.ipynb](../scripts/tutorial.ipynb)

Excluded from backlog interpretation: vendored third-party docs under
`s2and_rust/vendor/`, archived notebooks/scripts under `scripts/archive/`,
license files, and data text files under `s2and/data/`.

Code-check commands used:

```powershell
rg --files --hidden --glob '!**/.venv/**' --glob '!**/.git/**' --glob '!**/data/**' --glob '!**/data-backup/**' --glob '!dist/**' --glob '!scratch/**' --glob '!s2and_rust/vendor/**' --glob '*.md' .
rg -n --hidden --max-filesize 4M --glob '!**/.venv/**' --glob '!**/.git/**' --glob '!**/data/**' --glob '!**/data-backup/**' --glob '!dist/**' --glob '!scratch/**' --glob '!s2and_rust/vendor/**' --glob '!scripts/archive/**' --glob '*.md' "Remaining:|Done when:|Open Work|blocked|TODO|FIXME|future|must|should|need" README.md AGENTS.md docs scripts s2and_rust
rg -n --hidden --max-filesize 4M --glob '!**/.venv/**' --glob '!**/.git/**' --glob '!**/data/**' --glob '!**/data-backup/**' --glob '!dist/**' --glob '!scratch/**' --glob '!s2and_rust/vendor/**' --glob '!scripts/archive/**' --glob '*.txt' --glob '*.ipynb' "TODO|FIXME|Remaining|Open Work|blocked|future|should|need|NotImplemented" paper_experiments_env.txt scripts
rg -n --hidden --max-filesize 4M --glob '!**/.venv/**' --glob '!**/.git/**' --glob '!**/data/**' --glob '!**/data-backup/**' --glob '!dist/**' --glob '!scratch/**' --glob '!s2and_rust/vendor/**' --glob '!scripts/archive/**' --glob '*.py' --glob '*.rs' --glob '*.toml' --glob '*.md' "predict_from_arrow_paths|_predict_incremental_helper|feature_block_from_arrow_paths|RawBlockQueryCandidatePlanner|RawArrowPlanBundle|_canonicalize_last_for_counts|_lasts_equivalent_for_constraint|signature_name_parts_for_subblocking|split_first_middle_hyphen_aware|from_json_paths|arrow_batch_lookup" s2and s2and_rust tests scripts docs
rg -n --hidden --max-filesize 4M --glob '!**/.venv/**' --glob '!**/.git/**' --glob '!**/data/**' --glob '!**/data-backup/**' --glob '!dist/**' --glob '!scratch/**' --glob '!s2and_rust/vendor/**' --glob '!scripts/archive/**' --glob '*.py' --glob '*.rs' --glob '*.toml' --glob '*.md' "TODO|FIXME|XXX|NotImplemented|raise NotImplemented" s2and s2and_rust scripts tests docs
```

## Current Read

The active TODO pile is mostly not literal `TODO` comments in docs. It is
structured as:

1. Rust/Arrow production execution backlog in [work_plan.md](work_plan.md).
2. Blocked normalization migration in
   [normalization_migration_blocked.md](normalization_migration_blocked.md).
3. Operational cleanup items implied by production docs and script docs.
4. Code-only TODO comments that mostly point back to the blocked
   normalization migration, plus one explicit compact-linker feature gap.
5. A small number of stale or potentially confusing docs issues.

## Priority Plan

### P1: Regenerate And Validate Future Arrow Releases

Goal: make the next public Arrow release reproducible and locally verifiable
before any network/S3 smoke.

Doc evidence:

- [work_plan.md](work_plan.md) keeps the canonical local replay/profiling
  bundle at `s2and/data/s2and_and_big_blocks_linker_dataset_20260525`.
- [arrow_dataset_spec.md](rust/arrow_dataset_spec.md) defines the durable
  Arrow layout, manifests, batch indexes, and validation checklist.
- [scripts/README.md](../scripts/README.md) points production artifact work at
  `scripts/convert_to_arrow.py` and
  `scripts/verification/validate_local_arrow_release.py`.

Code check:

- [convert_to_arrow.py](../scripts/convert_to_arrow.py) imports
  `require_name_counts_index_artifact`.
- [validate_local_arrow_release.py](../scripts/verification/validate_local_arrow_release.py)
  checks root manifests, required files, batch indexes, replay bundle
  references, and `name_counts_index/manifest.json`.
- [test_arrow_release_layout.py](../tests/test_arrow_release_layout.py) covers
  a tiny release-layout regression.

Concrete next actions:

- On the next release build, regenerate durable Arrow bundles from the full
  schema, then run `refresh-root-manifest`.
- Keep production-scale `name_counts_index/` in S3, not Git/LFS.
- Keep S3/no-auth validation as an explicit release smoke until there is a
  network-enabled release CI job.

Verification gate:

```powershell
uv run python scripts/verification/validate_local_arrow_release.py --release-root s2and/data
uv run python scripts/convert_to_arrow.py validate --dataset-dir <dataset-dir>
```

### Done: Retire Or Rename The Incremental Helper After A Final Search

Goal: avoid preserving `_predict_incremental_helper(...)` as accidental API.

Doc evidence:

- [work_plan.md](work_plan.md) says `_predict_incremental_helper(...)` is
  internal-only test plumbing and any rename/removal needs a final scoped
  monkeypatch search.

Code check:

- [model.py](../s2and/model.py) now routes the Python fallback through the
  explicit internal `_predict_incremental_python_fallback(...)` method.
- The final scoped search found no `_predict_incremental_helper(...)`
  references or monkeypatches under `tests` or `s2and`.

Completed check:

```powershell
rg -n --hidden --glob '!**/.venv/**' --glob '!**/.git/**' --glob '!**/data/**' --glob '!**/data-backup/**' --glob '!dist/**' --glob '!scratch/**' "monkeypatch.*_predict_incremental_helper|_predict_incremental_helper" tests s2and
```

- Behavior coverage stays on `Clusterer.predict_incremental(...)`.

Verification gate:

```powershell
uv run pytest -q tests/test_cluster_incremental.py tests/test_incremental_linking_production.py tests/test_incremental_linking_runtime.py
```

### P1: Continue Rust Surface Cleanup One Module At A Time

Goal: reduce `s2and_rust/src/lib.rs` without changing public behavior.

Doc evidence:

- [work_plan.md](work_plan.md) lists completed extractions:
  `promoted_linker`, `name_counts`, `text_compat`, `arrow_batch_lookup`,
  `constraints`, `orcid`, `pair_indexing`, `rayon_pool`,
  `language_detection`, `raw_arrow::arrow_io`, `raw_arrow::readers`,
  `raw_arrow::paths`, `raw_arrow_features`, `raw_candidate_planner`,
  `retrieval`, `ingest_dataset`, `rust_featurizer` (including linker-distance
  helpers), `features` helpers, and `subblocking`.
- Candidate next modules: none identified for this cleanup pass; `lib.rs`
  now keeps shared core data types, tests, build info, and PyO3 registration.
- [public_surface_inventory.md](rust/public_surface_inventory.md) says to keep
  `RawBlockQueryCandidatePlanner` and not delete
  `RustNameCompatibleSubblockSelector` internals.

Code check:

- [s2and_rust/src](../s2and_rust/src) contains the extracted modules; `lib.rs`
  now owns shared core data types, tests, build info, and PyO3 registration.

Concrete next actions:

- Extract one low-coupling Rust module at a time.
- Move focused Rust unit tests with the extracted module when it avoids
  unnecessary visibility broadening.
- Do not fold Arrow, JSON, and `ANDData` ingest semantics together during a
  mechanical split.

Verification gate:

```powershell
uv run maturin develop -m s2and_rust/Cargo.toml --release
uv run --active --no-project cargo test --manifest-path s2and_rust/Cargo.toml
uv run pytest -q tests/test_compare_python_vs_rust.py tests/test_rust_from_dataset_contract.py tests/test_raw_block_candidate_plan_arrow.py
```

### P1: Deduplicate Ingest Only Where Semantics Match

Goal: share staging helpers without erasing source-specific policy.

Doc evidence:

- [ingest_source_policy_inventory.md](rust/ingest_source_policy_inventory.md)
  says Arrow staging owns production file-backed ingest while `ANDData` keeps
  Python-owned precomputed state.
- It explicitly says not to route `from_dataset(...)` through shared Arrow
  staging without deciding how much Python state to preserve.

Code check:

- [data.py](../s2and/data.py) still owns `ANDData` preprocessing, count
  semantics, constraints, and paper preprocessing.
- Rust staging and ingestion helpers now live in
  [ingest_dataset.rs](../s2and_rust/src/ingest_dataset.rs), with
  `RustFeaturizer` wiring in [rust_featurizer.rs](../s2and_rust/src/rust_featurizer.rs).
- Rust and Python normalization parity tests are covered by
  [test_text.py](../tests/test_text.py) and
  [test_rust_from_dataset_contract.py](../tests/test_rust_from_dataset_contract.py).

Concrete next actions:

- Start with helpers whose semantics are already identical and covered by
  parity tests.
- Keep source-specific language, paper-author, name-count, and reference
  feature behavior visible until a repo decision says otherwise.
- Record bounded real-dataset Arrow-vs-`ANDData` feature parity when a shared
  helper changes behavior-sensitive fields.

Verification gate:

```powershell
uv run pytest -q tests/test_rust_from_dataset_contract.py tests/test_raw_block_candidate_plan_arrow.py tests/test_text.py
```

### P2: Run A Release-Grade Performance Pass Before Optimizing

Goal: only optimize measured bottlenecks that clear the documented threshold.

Doc evidence:

- [work_plan.md](work_plan.md) says the next profiling target is Arrow
  read/summary construction and reusable component summaries on the canonical
  promoted-incremental workload.
- [2026-05-27-promoted-incremental-arrow.md](rust/profiling/2026-05-27-promoted-incremental-arrow.md)
  shows p50 wall time around 11.15s and max RSS around 3.72 GB, but explicitly
  says it was dirty-worktree/debug-assertions operational evidence, not a
  release-grade performance claim.

Code check:

- [promoted_incremental_arrow_profile_cmd.py](../scripts/_rust_suite/promoted_incremental_arrow_profile_cmd.py)
  is the current profiling runner.
- [rust_suite.py](../scripts/rust_suite.py) defaults production/rust suite
  model paths to `production_model_v1.21`.

Concrete next actions:

- Build the Rust extension in release mode.
- Run at least five isolated promoted-incremental Arrow profile runs on the
  canonical local bundle.
- Act only if Arrow read, summary construction, or name-count work is at least
  a 10% contributor to wall time or allocation volume, or if the change removes
  a real `ANDData` dependency.

Verification gate:

```powershell
uv run maturin develop -m s2and_rust/Cargo.toml --release
uv run python scripts/rust_suite.py promoted-incremental-arrow-profile `
  --runs 5 `
  --arrow-root s2and/data/s2and_and_big_blocks_linker_dataset_20260525 `
  --dataset pubmed `
  --target-block "r agarwal" `
  --query-limit 25 `
  --max-seed-clusters 25 `
  --synthetic-seeds-when-clusters-missing `
  --require-rust-release
```

### Blocked: Normalization Canonicalization Migration

Goal: keep legacy compatibility stable until canonical artifacts and retraining
can move together.

Doc evidence:

- [normalization_migration_blocked.md](normalization_migration_blocked.md) says
  the migration is blocked until required data/artifacts are ready.
- Open decisions remain around the compatibility-mode decommission window and
  threshold tightening.
- The current ASCII/non-ASCII dash behavior is a measured legacy-compatibility
  repair, not the canonical target.

Code check:

- [data.py](../s2and/data.py) still contains `_canonicalize_last_for_counts`
  and `_lasts_equivalent_for_constraint`.
- [text.py](../s2and/text.py) still contains
  `split_first_middle_hyphen_aware(...)` and
  `first_names_name_compatible(...)`.
- [subblocking.py](../s2and/subblocking.py) still contains
  `signature_name_parts_for_subblocking(...)`.
- [text_compat.rs](../s2and_rust/src/text_compat.rs) contains the Rust
  compatibility normalization helpers.
- Transitional tests exist in
  [test_surname_hyphen_aware.py](../tests/test_surname_hyphen_aware.py),
  [test_subblocking_telemetry.py](../tests/test_subblocking_telemetry.py),
  [test_text.py](../tests/test_text.py), and
  [test_rust_from_dataset_contract.py](../tests/test_rust_from_dataset_contract.py).

Concrete next actions when unblocked:

- Freeze canonical examples for first/middle/last, apostrophes, dash-like
  forms, initials, compound surnames, and particles.
- Regenerate name counts, name tuples, and ORCID prefix counts with provenance
  and `normalization_version` metadata.
- Only then remove compatibility shims, tuple probing fallbacks, ORCID prefix
  fallbacks, and block compaction workarounds.
- Treat title/venue/journal/source-ID normalization as field-specific,
  versioned feature work, not as a quick change to `normalize_text(...)`.

Verification gate:

```powershell
uv run pytest -q tests/test_surname_hyphen_aware.py tests/test_subblocking_telemetry.py tests/test_text.py tests/test_rust_from_dataset_contract.py tests/test_cluster_incremental.py
```

### Watchlist: Training Reference Features

Goal: keep the Rust training fast path honest when reference features are
requested.

Doc evidence:

- [rust/runtime.md](rust/runtime.md) says training-mode deferred paper
  preprocessing is gated on `compute_reference_features=False`, because
  reference-details preprocessing remains Python-only.

Code check:

- [data.py](../s2and/data.py) owns Python paper preprocessing.
- [runtime.py](../s2and/runtime.py) owns Rust capability detection.
- [rust_lifecycle.py](../s2and/rust_lifecycle.py) owns lifecycle policy.

Concrete next actions:

- If reference-feature training becomes required, first add focused tests that
  prove the current gate behavior.
- Then decide whether to port reference-detail preprocessing to Rust or keep
  that workload on Python preprocessing.

Verification gate:

```powershell
uv run pytest -q tests/test_rust_from_dataset_contract.py tests/test_preprocess_papers_parallel_defaults.py tests/test_rust_lifecycle.py tests/test_rust_capabilities.py
```

### Watchlist: Compact Incremental Partial Supervision

Goal: keep an unsupported compact-linker mode explicit instead of letting it
look accidentally broken.

Doc evidence:

- [production_inference.md](production_inference.md) defines promoted
  incremental seed routing and telemetry as caller-visible contract.
- [rust/arrow_dataset_spec.md](rust/arrow_dataset_spec.md) defines
  `cluster_seed_disallows.arrow` as the optional seed-disallow table.

Code check:

- [runtime.py](../s2and/incremental_linking/runtime.py) raises
  `NotImplementedError` when compact-linker retrieved-candidate scoring receives
  `partial_supervision`.
- [test_incremental_linking_runtime.py](../tests/test_incremental_linking_runtime.py)
  asserts that failure mode.
- This is separate from `FastCluster.transform(...)`, which is intentionally
  unsupported inductive-mode API and covered by
  [test_model_pairwise_exceptions.py](../tests/test_model_pairwise_exceptions.py).

Concrete next actions:

- Do nothing unless a production compact-linker request path actually needs
  partial supervision.
- If needed, first add a typed request fixture proving the desired merge
  semantics, then wire the compact runtime behavior with explicit tests for
  require/disallow conflicts.

Verification gate:

```powershell
uv run pytest -q tests/test_incremental_linking_runtime.py::test_private_retrieved_candidate_slice_rejects_partial_supervision
uv run pytest -q tests/test_model_pairwise_exceptions.py
```

## Documentation Cleanup Items

These are small but user-facing.

1. If licensing policy is corrected, update [README.md](../README.md),
   [pyproject.toml](../pyproject.toml), root [LICENSE](../LICENSE), and dataset
   docs together. The current MIT / CC-BY-4.0 / ODC-BY mismatch is already
   preserved in README as a known issue.
2. Code TODO comments in [data.py](../s2and/data.py) and the production count
   scripts point at the same blocked normalization migration above; do not
   schedule them as separate cleanup work before canonical artifacts exist.

## Standing Guardrails

These are not TODOs, but they should shape future work:

- Keep production artifact validation routed through `s2and.arrow_inputs`.
- Keep production Rust inference on `Clusterer.predict_from_arrow_paths(...)`
  or complete Arrow paths to `Clusterer.predict(...)`.
- Keep full scans and compatibility fallbacks explicit test-only or parity-only
  options.

## Explicit Non-Goals For Now

- Do not revive SPECTER/vector clone work without a fresh allocation profile.
- Do not remove normalization shims before regenerated canonical artifacts are
  validated.
- Do not add another strict/compatibility discovery layer beside
  `s2and.arrow_inputs`.
- Do not run S3/network release smokes as default pytest.
- Do not optimize performance items below the documented 10% threshold unless
  they remove a real production dependency.
