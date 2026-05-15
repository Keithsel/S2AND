from __future__ import annotations

import logging
import sys
from unittest.mock import patch

import s2and  # noqa: F401
from scripts import make_inventors_hf_specter_embeddings, rust_suite
from scripts.production.model import train_pairwise


def test_s2and_import_does_not_force_debug_logging() -> None:
    logger = logging.getLogger("s2and")

    assert logger.level == logging.NOTSET
    assert not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers)


def test_inventors_embedding_script_log_level_defaults_to_debug() -> None:
    with patch.object(sys, "argv", ["make_inventors_hf_specter_embeddings.py"]):
        args = make_inventors_hf_specter_embeddings.parse_args()

    assert args.log_level == "DEBUG"


def test_pairwise_training_cli_configures_debug_logging_by_default() -> None:
    with (
        patch.object(train_pairwise, "train_pairwise_bundle") as train_pairwise_bundle,
        patch.object(train_pairwise.logging, "basicConfig") as basic_config,
    ):
        train_pairwise.main(["--production-version", "test"])

    basic_config.assert_called_once_with(level=logging.DEBUG)
    train_pairwise_bundle.assert_called_once()


def test_pairwise_training_search_space_records_eps_under_eps_label() -> None:
    eps_node = train_pairwise._search_space()["eps"]
    hyperopt_param = eps_node.pos_args[0]

    assert hyperopt_param.name == "hyperopt_param"
    assert hyperopt_param.pos_args[0].obj == "eps"


def test_rust_suite_file_logging_preserves_existing_logger_level(tmp_path) -> None:
    logger = logging.getLogger("s2and")
    previous_level = logger.level
    log_path = tmp_path / "rust-suite.log"

    logger.setLevel(logging.DEBUG)
    handler = rust_suite._configure_file_logging(str(log_path))
    try:
        assert logger.level == logging.DEBUG
        assert handler is not None
        assert handler.level == logging.NOTSET
    finally:
        if handler is not None:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(previous_level)


def test_rust_suite_main_removes_file_handler_after_run(monkeypatch, tmp_path) -> None:
    logger = logging.getLogger("s2and")
    log_path = tmp_path / "rust-suite.log"

    monkeypatch.setattr(rust_suite, "_dispatch", lambda *_args, **_kwargs: 0)

    assert rust_suite.main(["--log-file", str(log_path), "compare"]) == 0
    assert not any(
        isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_path)
        for handler in logger.handlers
    )
