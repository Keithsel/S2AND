from __future__ import annotations

import logging

from scripts import rust_suite


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
