# Production Release Artifacts

Scripts in this directory create or validate checked-in production artifacts
under `s2and/data/`. The model release artifact is a native
`production_model_vX.Y/` directory. Do not create production pickles.

## 1. Train Pairwise

```powershell
uv run python scripts/production/model/train_pairwise.py `
  --production-version 1.3 `
  --output-dir s2and/data/production_model_v1.3 `
  --run-full
```

This writes the pairwise-only bundle stage:

```text
production_model_v1.3/
  clusterer.json
  manifest.json
  pairwise/
    main.lgb
    nameless.lgb
    metadata.json
    main_prediction_fixture.json
    nameless_prediction_fixture.json
  reproducibility/
    pairwise_training_config.json
    pairwise_training_summary.json
```

This stage is loadable for training/finalization, but it is not a complete
runtime production model until the linker is added.

## 2. Train Linker And Finalize

```powershell
uv run python scripts/production/model/train_linker_and_finalize.py `
  --production-bundle-version 1.3 `
  --pairwise-model-path s2and/data/production_model_v1.3 `
  --save-production-bundle-to s2and/data/production_model_v1.3 `
  --run-full
```

This writes:

```text
production_model_v1.3/
  incremental_linker/
    booster.lgb
    metadata.json
  reproducibility/
    incremental_linker_training_target.json
  manifest.json
```

After this step, users load the model with:

```python
from s2and.production_model import load_production_model

clusterer = load_production_model("s2and/data/production_model_v1.3")
```

## Count Artifacts

The `counts/` scripts document or export production count artifacts:

- `counts/generate_name_counts.py`: documents the internal query used to build
  `name_counts.pickle`.
- `counts/generate_orcid_name_prefix_counts.py`: documents the internal query
  used to build `first_k_letter_counts_from_orcid.json`.
- `counts/export_name_counts_for_rust.py`: converts the current Python
  `name_counts.pickle` into the Rust JSON ingest shape until a shared native
  count format replaces both.
