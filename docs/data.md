# Data and Models

This document covers dataset download, checked-in model artifacts, and `path_config.json`.

## Full dataset download

Download the full S2AND release into `s2and/data/`:

```bash
aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and-release s2and/data/
```

Expected size is about `55.5 GiB`.

The release includes dataset files. The current production model bundle is also
checked into this repo under `s2and/data/production_model_v1.21/`.

## Production model bundle

The current production model is a native bundle directory:

- `s2and/data/production_model_v1.21/manifest.json`
- `s2and/data/production_model_v1.21/pairwise/main.lgb`
- `s2and/data/production_model_v1.21/pairwise/nameless.lgb`
- `s2and/data/production_model_v1.21/incremental_linker/booster.lgb`
- `s2and/data/production_model_v1.21/incremental_linker/metadata.json`

This bundle is included in package data, so prediction does not require a
separate model download. The older `production_model_v1.2.pickle` is retained
for legacy pairwise compatibility, but the documented production loader points
at the v1.21 bundle.

New production releases should be built as native bundle directories with
`scripts/production/model/train_pairwise.py` followed by
`scripts/production/model/train_linker_and_finalize.py`; do not create new
production pickles.

The replay target for rebuilding/auditing the promoted incremental linker lives
at:

```text
s2and/data/production_model_v1.21/reproducibility/incremental_linker_training_target.json
```

It is not a runtime model; it records feature order and training params for the
replay script.

The promoted linker train/calibrate/eval replay data is also published under
the same S3 release prefix. Download it when you need to rebuild or audit the
promoted linker artifact:

```bash
aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and-release/s2and_and_big_blocks_linker_dataset_20260513 s2and/data/s2and_and_big_blocks_linker_dataset_20260513
```

This source bundle is the default `--source-bundle-root` for
`scripts/production/model/linker_train_calibrate_eval.py`.

## Configuring `s2and/data/path_config.json`

Some scripts look up the main data root through `s2and/data/path_config.json`
or the `S2AND_PATH_CONFIG` environment variable. This config points at the
downloaded benchmark dataset root; it is separate from the package data checked
in under `s2and/data/`.

Example:

```json
{
  "main_data_dir": "absolute path to your downloaded S2AND data",
  "internal_data_dir": ""
}
```

Guidance:

- Set `main_data_dir` to the directory containing your downloaded S2AND datasets.
- `internal_data_dir` is only relevant for internal AI2 workflows and can be left empty.
- If your data lives in this repo's `s2and/data/` directory, the default placeholder config already resolves there.

## Dataset file expectations

Most workflows use the standard S2AND JSON files for:

- signatures
- papers
- clusters
- optional cluster seeds
- SPECTER embeddings

The tutorial script supports both:

- mini-dataset naming such as `<dataset>_papers.json`
- plain fixture naming such as `papers.json`

See [production_inference.md](production_inference.md) for the minimal inference input contract, and [training.md](training.md) for training-mode dataset requirements.
