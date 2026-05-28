from __future__ import annotations

import pytest

from s2and import text as s2and_text


@pytest.fixture(autouse=True)
def _disable_fasttext_loading_for_tests():
    previous_enabled = s2and_text.fasttext_loading_enabled()
    previous_model = s2and_text._FASTTEXT_MODEL  # noqa: SLF001
    previous_initialized = s2and_text._FASTTEXT_MODEL_INITIALIZED  # noqa: SLF001
    previous_load_failed = s2and_text._FASTTEXT_LOAD_FAILED  # noqa: SLF001
    s2and_text.set_fasttext_loading_enabled(False)
    try:
        yield
    finally:
        s2and_text.set_fasttext_loading_enabled(previous_enabled)
        s2and_text._FASTTEXT_MODEL = previous_model  # noqa: SLF001
        s2and_text._FASTTEXT_MODEL_INITIALIZED = previous_initialized  # noqa: SLF001
        s2and_text._FASTTEXT_LOAD_FAILED = previous_load_failed  # noqa: SLF001
