from __future__ import annotations

import pytest

from s2and import text as s2and_text


@pytest.fixture(autouse=True)
def _disable_fasttext_loading_for_tests():
    previous_enabled = s2and_text.fasttext_loading_enabled()
    s2and_text.set_fasttext_loading_enabled(False)
    try:
        yield
    finally:
        s2and_text.set_fasttext_loading_enabled(previous_enabled)
