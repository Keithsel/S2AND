from __future__ import annotations

from scripts import eval_prod_models


def test_eval_prod_models_parser_defaults_to_inventors_s2and() -> None:
    args = eval_prod_models._build_parser().parse_args([])

    assert args.dataset == "inventors_s2and"
