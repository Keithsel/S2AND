# Reproducibility

This document covers the paper-era environment and compatibility notes for older released artifacts.

## Paper-era branch

The original paper experiments were run on the `s2and_paper` branch with a Python `3.7.9` environment captured in `paper_experiments_env.txt`.

If you need to reproduce the paper-era setup:

```bash
git checkout s2and_paper
uv venv --python 3.7.9
```

Then install the pinned environment from `paper_experiments_env.txt` inside that isolated environment and rerun the paper experiment command set from the `s2and_paper` branch. The current branch keeps a reference copy at `scripts/archive/paper_experiments.sh`, but that file is historical and not the supported entrypoint for current `main` development.

## Old released model artifacts

Older released pickles such as:

- `production_model.pickle`
- `full_union_seed_*.pickle`

are paper-era artifacts and only run on the `s2and_paper` branch, not on `main`.

Those pickles store a dictionary with a `clusterer` key rather than a bare clusterer object.

## Current branch

For current work on `main`, prefer the checked-in native production bundle:

- `production_model_v1.21/`

Legacy pickles are still present for compatibility and parity checks:

- `production_model_v1.2.pickle`
- `production_model_v1.1.pickle`

The v1.21 bundle includes the promoted incremental linker under
`incremental_linker/`. Its replay target is tracked separately at
`production_model_v1.21/reproducibility/incremental_linker_training_target.json`;
replay scripts should not depend on machine-local analysis artifacts.

For repeated promoted-linker replay, materialized feature bundles can be reused
only through the explicit `precomputed-promoted` mode. The bundle must be
portable and validated against the replay target JSON; local scratch paths are
not accepted as artifact metadata.

See [production_inference.md](production_inference.md) for the current inference contract.
