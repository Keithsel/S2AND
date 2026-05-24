from __future__ import annotations

import s2and.incremental_linking.production as production_module


def test_raw_arrow_plan_window_enabled_when_query_batch_is_smaller_than_query_count() -> None:
    query_batch_size = 2
    plan_window_size = production_module._raw_arrow_plan_window_size(  # noqa: SLF001
        query_count=9,
        query_batch_size=query_batch_size,
        plan_window_multiplier=production_module._RAW_ARROW_PLAN_WINDOW_MULTIPLIER,  # noqa: SLF001
    )

    assert production_module._RAW_ARROW_PLAN_WINDOW_MULTIPLIER > 1  # noqa: SLF001
    assert plan_window_size > query_batch_size
    assert int(plan_window_size > query_batch_size) == 1
