from __future__ import annotations

import pytest

import s2and.plotting_utils as plotting_utils


def test_plotting_utils_deferred_config_load_raises_when_missing(monkeypatch):
    import s2and.consts as consts_module

    monkeypatch.setattr(consts_module, "CONFIG_LOCATION", "/definitely/missing/path_config.json")
    monkeypatch.setattr(consts_module, "_CONFIG", None)
    with pytest.raises(FileNotFoundError):
        plotting_utils._experiment_dir()
