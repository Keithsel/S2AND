from __future__ import annotations

import importlib

import pytest

import s2and.consts as consts_module


def test_consts_deferred_config_load_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.reload(consts_module)
    monkeypatch.delenv(module.CONFIG_LOCATION_ENV, raising=False)
    monkeypatch.setattr(module, "CONFIG_LOCATION", "/definitely/missing/path_config.json")
    module.__dict__["_CONFIG"] = None

    with pytest.raises(FileNotFoundError, match="path config"):
        _ = module.CONFIG["main_data_dir"]
