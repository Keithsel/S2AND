from __future__ import annotations

import random
import re
from collections.abc import Iterable
from pathlib import Path

import pytest
from text_unidecode import unidecode

from s2and.text import normalize_text

s2and_rust = pytest.importorskip("s2and_rust")
rust_normalize_text = getattr(s2and_rust, "normalize_text_compat", None)


EXAMPLE_CASES: list[str | None] = [
    None,
    "",
    "TeXt",
    "te'漢字xt",
    "te'xt",
    "A1 B-2",
    "Élodie Brontë",
    "François L'Ouverture",
    "Γιάννης Παπαδόπουλος",
    "Мария Иванова",
    "東京大学 漢字",
    "中文 标题 123",
    "O'Neil",
    "O’Neil",
    "Jean-Luc Picard",
    "Amin-ul-Haq",
    "Arif‐ullah",
    "Hua－li",
    " spaced\tand\nwrapped   text ",
    "COVID-19: A 2-year follow-up",
    "Müller–Lüdenscheidt",
    "naïve coöperate soufflé",
    "D'Arcy O`Connor",
    "Straße",
    "李小龍",
    "Δx/Δt ≥ 0",
    "№ 42 — résumé",
]


def _manual_python_normalize(text: str | None, special_case_apostrophes: bool = False) -> str:
    if text is None or len(text) == 0:
        return ""
    norm_text = unidecode(text).lower()
    if special_case_apostrophes:
        norm_text = norm_text.replace("'", "")
    norm_text = re.sub(r"[^a-zA-Z\s]+", " ", norm_text)
    return re.sub(r"\s+", " ", norm_text).strip()


def _assert_normalization_parity(cases: Iterable[str | None]) -> int:
    assert callable(rust_normalize_text), "s2and_rust.normalize_text_compat is unavailable"
    mismatches: list[str] = []
    checked = 0
    for text in cases:
        for special_case_apostrophes in (False, True):
            expected = normalize_text(text, special_case_apostrophes=special_case_apostrophes)
            manual_expected = _manual_python_normalize(text, special_case_apostrophes)
            actual = rust_normalize_text(text, special_case_apostrophes)
            checked += 1
            if expected != manual_expected or actual != expected:
                mismatches.append(
                    f"text={text!r} special={special_case_apostrophes} "
                    f"normalize_text={expected!r} manual_unidecode={manual_expected!r} "
                    f"rust={actual!r}"
                )
    assert not mismatches, f"{len(mismatches)}/{checked} normalization mismatches:\n" + "\n".join(mismatches[:20])
    return checked


def _bounded_unicode_cases() -> list[str]:
    rng = random.Random(8675309)
    deterministic_codepoints = list(range(0, 0x10000, 257))
    random_codepoints = [rng.randrange(0, 0x110000) for _ in range(768)]
    codepoints = deterministic_codepoints + random_codepoints
    chars = [chr(codepoint) for codepoint in codepoints if not 0xD800 <= codepoint <= 0xDFFF]

    alphabet = [
        "A",
        "z",
        " ",
        "\t",
        "'",
        "’",
        "-",
        "‐",
        "é",
        "ø",
        "ß",
        "Γ",
        "Ж",
        "漢",
        "字",
        "中",
        "文",
        "１",
        "₂",
        "€",
        "🧬",
    ]
    strings = ["".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 14))) for _ in range(256)]
    return [*chars, *strings]


def test_rust_normalize_text_matches_python_examples_and_scripts() -> None:
    assert _assert_normalization_parity(EXAMPLE_CASES) == len(EXAMPLE_CASES) * 2


def test_rust_normalize_text_matches_python_bounded_unicode_chars_and_strings() -> None:
    cases = _bounded_unicode_cases()

    assert _assert_normalization_parity(cases) == len(cases) * 2


def _arrow_fixture_strings(limit_per_file: int = 80) -> list[str]:
    pyarrow = pytest.importorskip("pyarrow")
    base = Path(__file__).parent / "fixtures" / "arrow" / "pubmed_specter2" / "pubmed"
    specs = [
        ("signatures.arrow", ("author_first", "author_middle", "author_last", "author_suffix", "author_affiliations")),
        ("papers.arrow", ("title", "venue", "journal_name")),
        ("paper_authors.arrow", ("author_name",)),
    ]
    values: list[str] = []
    for filename, columns in specs:
        path = base / filename
        if not path.exists():
            continue
        with pyarrow.ipc.open_file(path) as reader:
            if reader.num_record_batches == 0:
                continue
            batch = reader.get_batch(0).slice(0, limit_per_file)
            for column in columns:
                for value in batch.column(column).to_pylist():
                    if value is None:
                        continue
                    if isinstance(value, list):
                        values.extend(item for item in value if item)
                    elif value:
                        values.append(value)
    return values


def test_rust_normalize_text_matches_python_on_bounded_arrow_fixture_sample() -> None:
    values = _arrow_fixture_strings()
    if not values:
        raise pytest.skip.Exception("local Arrow text fixtures are unavailable")

    assert _assert_normalization_parity(values) == len(values) * 2
