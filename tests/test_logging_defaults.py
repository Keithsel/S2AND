from __future__ import annotations

import logging
import sys
from unittest.mock import patch

import s2and  # noqa: F401
from scripts import make_inventors_hf_specter_embeddings
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
