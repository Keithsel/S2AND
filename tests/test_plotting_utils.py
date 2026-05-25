from __future__ import annotations

import pytest

import s2and.consts as consts
from s2and import plotting_utils


def test_plotting_utils_deferred_config_load_raises_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    missing_config = tmp_path / "missing_path_config.json"
    monkeypatch.setenv(consts.CONFIG_LOCATION_ENV, str(missing_config))
    monkeypatch.setattr(consts, "_CONFIG", None)

    with pytest.raises(FileNotFoundError, match="Could not find S2AND path config"):
        plotting_utils._experiment_dir()
