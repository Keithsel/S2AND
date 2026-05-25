# Normalization Unification Migration Plan

Execution status (last reconfirmed 2026-05-24; originally entered blocked state 2026-03-02)
- Blocked: normalization work is on hold until the required data/artifacts are ready.
- Keep this plan separate from the active execution plan in `docs/work_plan.md`.

Status
- Draft updated from issue notes through August 31, 2025.
- Rust compatibility-alignment notes added on February 20, 2026; helper porting is no longer tracked here as a separate blocking work item.
- Reviewed on February 24, 2026 alongside big-block execution planning; no normalization-policy changes in that workstream.

Scope
- Unify name normalization for first/middle/last across data preparation, modeling, subblocking, and auxiliary datasets (name counts, name tuples, ORCID prefix counts).
- Ensure training-time and inference-time normalization are identical, including Sinonym-dependent behavior.

Decided (from issue history)
- Apostrophes: canonical fields should remove apostrophes globally (no dual stream).
- Chinese given names: Sinonym-aware handling is part of canonicalization; hyphenated given names should stay together.
- First-name compatibility checks should use multi-token prefix logic (`same_prefix_tokens`), not single-token-only rules.
- Canonical surname storage:
  - Persist canonical last names as normalized, space-separated tokens (e.g., `ou yang`, `de la cruz`).
  - Treat hyphen/space variants equivalently during canonicalization.
  - Use compact surname projection (remove spaces/hyphens) only where required (block key and legacy-compatible count keys during transition).
- Surname particles/prefixes:
  - Preserve particles in canonical surname form (`de la`, `van der`, `bin`, etc. are not dropped or rewritten).
- Artifact regeneration is required after normalization policy changes:
  - Name counts.
  - Name tuples.
  - ORCID prefix counts.
- Retraining requirement: production retraining data must go through the same normalization + Sinonym path as production inference.

Open Decisions (remaining before migration freeze)
1) Compatibility-mode decommission window
   - finalize the exact number of releases/runs required before removing compatibility mode.
2) Threshold tightening
   - decide whether adopted Rust-alignment thresholds should be tightened for release-grade datasets.

Rust Alignment Decisions (effective February 20, 2026; refreshed 2026-05-23)
1) Canonical cutover contract
   - Rust ingestion paths must stay compatible with current Python + compatibility-shim behavior until canonical artifacts are regenerated and versioned.
   - Switch Rust and Python to canonical-only behavior only after the same canonical artifact gates pass.
2) Version-compatibility contract
   - `normalization_version=legacy_compat`:
     - allows current code paths + legacy artifacts + compatibility shims.
   - `normalization_version=canonical_v2`:
     - requires regenerated canonical artifacts for name counts, name tuples, and ORCID prefix counts.
     - must fail fast on code/artifact version mismatch unless an explicit temporary compatibility override is enabled.
   - old code + `canonical_v2` artifacts is unsupported.
3) Removal contract for compatibility shims
   - Do not remove `_canonicalize_last_for_counts`, `_lasts_equivalent_for_constraint`,
     name-tuple compatibility probing, or ORCID first-token fallback until canonical artifacts are validated in rollout.
4) Retraining contract
   - Before enabling canonical mode by default, production retraining and production inference must use the same canonical normalization + Sinonym path.
5) Rust coupling
   - Any Rust ingestion change that affects normalized names, name-count keys, ORCID fallbacks, or block keys must be treated as a policy-sensitive change, not a pure performance refactor.

Current State (post-Sinonym hyphen pass)
- Given-name canonicalization currently preserves hyphenated Chinese given names:
  - `s2and.text.split_first_middle_hyphen_aware`.
- Backward-compat shims exist for artifacts built with legacy normalization:
  - Name counts (first): when raw first had a hyphen, join spaces in canonical first for count keys (e.g., `qi xin` -> `qixin`).
  - Name counts (last): join spaces in canonical last for compound/hyphenated surnames (e.g., `ou yang` -> `ouyang`).
    - Helper: `_canonicalize_last_for_counts(...)`.
  - Constraints: last-name disallow uses space-insensitive comparison (`ou yang` == `ouyang`).
    - Helper: `_lasts_equivalent_for_constraint(...)`.
  - Subblocking: ORCID prefix map lookup has a first-token fallback for multi-token first names.
  - Name tuples in constraints and incremental new-name guarding: shared helper
    `first_names_name_compatible(...)` probes exact, joined, and first-token forms for compatibility with legacy tuples.
  - Sinonym overwrite block recomputation preserves spaced compound surnames for blocking (`q ou yang`) when overwriting
    blocks.

Fix during the blocked canonical migration (real-data findings)
- These are intentionally deferred from `legacy_compat` unless called out elsewhere as a compatibility repair. Fix them
  when artifacts, caches, and production models can move together under a versioned normalization contract.
- Title/text feature normalization is too destructive for some paper fields:
  - Real titles with formulas, identifiers, and enumerated parts collapse because `normalize_text(...)` drops digits and
    punctuation (`Co3O4`, `H2O2`, `CCDC 619488`, `Part 1`/`Part 2`).
  - Python locations to audit/change under a versioned feature contract:
    - Generic normalizer: `s2and/text.py::normalize_text`.
    - Paper preprocessing: `s2and/data.py::preprocess_paper_1`.
    - Incremental query/summary title and venue terms:
      `s2and/incremental_linking/query_adapter.py::_normalize_term_set`.
    - Any training/reference feature code that consumes normalized titles or title n-grams.
  - Rust locations to audit/change in the same release:
    - Generic compatibility normalizer: `s2and_rust/src/lib.rs::normalize_text_compat_from_map`.
    - Paper preprocessing and raw Arrow/JSON feature extraction paths that normalize titles, venues, journals,
      paper authors, or reference details before hashing/feature construction.
  - Do not change global `normalize_text(...)` in legacy mode. Introduce field-specific canonical title/venue
    normalization only with cache/version bumps and production-model validation.
- Name canonicalization needs a single versioned first/middle/last policy:
  - Python locations:
    - `s2and/text.py::split_first_middle_hyphen_aware` or its canonical replacement.
    - `s2and/data.py::ANDData.preprocess_signatures` and `ANDData._compute_signature_name_counts`.
    - `s2and/data.py::_canonicalize_last_for_counts` and `_lasts_equivalent_for_constraint`.
    - `s2and/text.py::first_names_name_compatible`.
    - `s2and/subblocking.py::signature_name_parts_for_subblocking`.
    - Pairwise/incremental consumers of `author_info_first_normalized`,
      `author_info_first_normalized_without_apostrophe`, and middle/last normalized fields.
  - Rust locations:
    - `s2and_rust/src/lib.rs::split_first_middle_hyphen_aware_compat`.
    - `s2and_rust/src/lib.rs::build_name_counts_data_from_artifact`.
    - `s2and_rust/src/lib.rs::canonical_last_for_counts`.
    - Rust constraint/name-tuple helpers and pairwise/incremental feature extraction paths that consume normalized
      first/middle/last values.
  - Compatibility repairs inside `legacy_compat` may keep current behavior correct, but canonical-only semantics must
    wait for regenerated name counts, name tuples, and ORCID prefix counts.
- `preprocess=False` is semantically misleading:
  - Today Python `s2and/data.py::preprocess_paper_1(..., preprocess=False)` still normalizes titles and authors,
    builds title word n-grams, and computes language for signature papers, while leaving venue/journal and some
    character n-gram fields raw/unset.
  - Rust stage/from-JSON paths intentionally mirror that behavior for parity.
  - During migration, replace the boolean with explicit modes such as `raw`, `minimal_legacy`, and `full`, or keep the
    legacy mode name explicit. Tests should assert exactly which fields are normalized in each mode.
- Subblock-token fallback parsing is case/punctuation preserving:
  - Python: `s2and/incremental_linking_training/query_support.py::_subblock_tokens`.
  - Rust: `s2and_rust/src/lib.rs::subblock_tokens_from_key`.
  - Generated current indexes appear to feed normalized keys, so this is not an observed generated-data failure.
    Canonical migration should either normalize parsed fallback tokens in both languages or fail fast on raw keys.
- Missing/non-informative text values collapse to empty strings:
  - `normalize_text(None)`, empty strings, digit-only strings, and punctuation-only strings can all become `""`.
  - During canonical migration, distinguish true missingness from normalized-empty nonmissing values where that matters
    for paper titles, venues, journals, and affiliation evidence. Any schema/cache change must be versioned.
- Source identifiers are not text:
  - `source_author_ids`, MAG IDs, DBLP suffixes, ACM IDs, and ORCIDs must never use `normalize_text(...)`.
  - Python locations carrying source IDs: `s2and/incremental_linking/feature_block_contract.py`,
    `s2and/incremental_linking/feature_block_bridges.py`, and
    `s2and/incremental_linking/feature_block_arrow.py`.
  - Rust raw Arrow/JSON contracts should preserve source IDs verbatim unless an identifier-specific canonicalizer is
    explicitly selected.

Target End State
- One canonical normalization path for first/middle/last consumed by all codepaths.
- No semantic distinction between `author_info_first_normalized` and `author_info_first_normalized_without_apostrophe`.
- Canonical last names are stored in spaced normalized form, with compact projections derived only for specific downstream keys.
- No runtime compatibility shims for legacy artifacts.
- All generated artifacts are built from the same canonical normalization logic.
- Field-specific text canonicalizers are explicit; title/venue/journal/source-ID behavior is not implicitly inherited
  from person-name normalization.

Migration Plan (phased, verifiable)
1) Lock policy and examples
   - Resolve all Open Decisions above.
   - Freeze a canonical example table covering:
     - `Jo Ann`, `Jo-Ann`, `JoAnn`.
     - `Yu Zhong`, `Yu-Zhong`, `YuZhong`, `Y. Z.`.
     - Apostrophe-like forms (`O'Brien`, ``O`Brien``, curly apostrophes).
     - Multi-initial cases (`H. G.`-style).
   - Output: explicit normalization invariants used by tests and artifact builders.

2) Implement unified canonicalization
   - Provide one canonicalization routine for first/middle/last (extend or replace `split_first_middle_hyphen_aware`).
   - Remove dual-read usage of `author_info_*_normalized*` fields in featurizer/subblocking/constraints and standardize on canonical fields.
   - Keep migration-scoped feature/version switch only if needed for safe rollout.

3) Regenerate artifacts with canonical logic
   - Regenerate name counts (`first`, `last`, `first_last`, `last_first_initial`).
   - Regenerate name tuples aligned with canonical forms.
   - Regenerate `s2and/data/first_k_letter_counts_from_orcid.json` using canonical first names (no token fallback).
   - Record reproducibility metadata: source snapshot, script/version hash, generation date.

4) Cut over and remove compatibility code
   - Remove `_canonicalize_last_for_counts`.
   - Remove `_lasts_equivalent_for_constraint`.
   - Remove name-tuple compatibility probing (joined/first-token fallback) from
     `first_names_name_compatible(...)`.
   - Remove subblocking first-token ORCID count probe.
   - Remove inference-only block compaction workaround once blocks are canonical everywhere.

5) Validate, benchmark, and roll out
   - Pairwise and clustering evaluation on representative datasets; compare to pinned baseline.
   - Subblocking checks: size distribution, merge behavior, and ORCID co-location sanity checks.
   - Performance checks: runtime and memory for preprocessing/subblocking/featurization.
   - Cache/version bump as needed (featurizer cache and artifact versioning).

6) Rust canonical alignment track
   - Audit Rust ingestion paths against the frozen canonical example table before cutover.
   - Update Rust helpers or constructor policies only as needed for `canonical_v2` artifacts.
   - Verify parity against Python outputs while compatibility shims are still enabled, then add no-shim canonical tests before enabling canonical mode.

Required Evidence / Exit Criteria
- Behavior:
  - Targeted pytest coverage for canonical examples and no-shim paths.
  - Existing transitional tests replaced or updated for the end state.
- Quality:
  - No-regression thresholds for pairwise and clustering metrics are met.
- Quality thresholds (adopted for Rust alignment):
  - Pairwise: `AUC delta <= 0.001`, `F1 delta <= 0.005` versus pinned baseline.
  - Clustering: `B3 delta <= 0.005` versus pinned baseline.
- Runtime:
  - No unexpected slowdown beyond agreed threshold.
- Runtime thresholds (adopted for Rust alignment):
  - Subblocking/preprocess runtime regression `<=10%` versus pinned baseline on the active benchmark protocol.
  - Peak RSS regression `<=10%` unless explicitly accepted for a release candidate.
- Data integrity:
  - Artifact generation logs include counts, key cardinalities, and basic spot checks.
- Versioning integrity:
  - Every regenerated artifact includes `normalization_version` metadata and generation provenance.
  - Code/artifact mismatch behavior is validated (fail-fast by default).

Compatibility/Rollback Notes
- Use explicit artifact normalization versioning during transition.
- Prefer fail-fast on code/artifact version mismatch unless a temporary compatibility flag is intentionally enabled.
- Decommission compatibility mode after one validated release window.
- Rust rollout note:
  - Treat any remaining Rust canonical-cutover work as a separate release action from legacy compatibility-shim removal.

References in code (as of this migration doc)
- Given-name canonicalization: `s2and.text.split_first_middle_hyphen_aware`.
- Surname count shim: `_canonicalize_last_for_counts` in `s2and/data.py`.
- Last-name constraint shim: `_lasts_equivalent_for_constraint` in `s2and/data.py`.
- Constraint and incremental new-name tuple fallback logic (exact/joined/first-token forms):
  `first_names_name_compatible(...)` in `s2and/text.py`, consumed by `ANDData.get_constraint`
  and incremental clustering guards.
- ORCID prefix fallback in subblocking: lookup path in `s2and/subblocking.py` during merge-pair scoring.
- Sinonym overwrite gating/application: `compute_sinonym_overwrite_allowlist`, `apply_sinonym_overwrites` in `s2and/data.py`.

Tests (current)
- `tests/test_surname_hyphen_aware.py`
  - Transitional regression coverage for surname count canonicalization, last-name constraint equivalence,
    name-tuple compatibility forms, and block compaction behavior under Sinonym overwrites.
- `tests/test_cluster_incremental.py`
  - Transitional regression coverage that incremental new-name guarding accepts the same legacy
    name-tuple compatibility forms as constraints.

Tests (required for end state)
- Canonical first-name equivalence cases from the frozen example table.
- Canonical surname policy tests for spaced storage + compact projection sites.
- Tests proving removal of compatibility fallbacks does not break expected behavior with regenerated artifacts.
