"""Compare Python and Rust language detector components on raw dataset titles."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import pycld2 as cld2

from s2and.text import _get_fasttext_model, detect_language

DetectorAudit = tuple[str, str, str, bool]


def _combine_language_predictions(fasttext_label: str, cld2_label: str) -> tuple[str, bool]:
    if fasttext_label == "un_ft" and cld2_label == "un_2":
        return "un", False
    if fasttext_label == "un_ft":
        return cld2_label, True
    if cld2_label == "un_2":
        return fasttext_label, True
    if cld2_label != fasttext_label:
        return "un", False
    return cld2_label, True


def _python_language_audit(text: str) -> DetectorAudit:
    if len(text.split()) <= 1:
        return "un_ft", "un_2", "un", False

    isuppers = [char.isupper() for char in text if char.isalpha()]
    if not isuppers:
        return "un_ft", "un_2", "un", False

    ft_model = _get_fasttext_model()
    if ft_model is None:
        fasttext_label = "un_ft"
    else:
        fasttext_input = (
            text.lower().replace("\n", " ") if sum(isuppers) / len(isuppers) > 0.9 else text.replace("\n", " ")
        )
        fasttext_label = ft_model.predict(fasttext_input)[0][0].split("__")[-1]

    try:
        cld2_label = cld2.detect(text)[2][0][1]
        if cld2_label == "un":
            cld2_label = "un_2"
    except (UnicodeError, cld2.error):
        cld2_label = "un_2"

    predicted_language, is_reliable = _combine_language_predictions(fasttext_label, cld2_label)
    reference_reliable, _reference_english, reference_language = detect_language(text)
    if (is_reliable, predicted_language) != (reference_reliable, reference_language):
        raise AssertionError(
            "Python audit implementation disagrees with s2and.text.detect_language: "
            f"audit={(is_reliable, predicted_language)!r} "
            f"detect_language={(reference_reliable, reference_language)!r}"
        )
    return fasttext_label, cld2_label, predicted_language, is_reliable


def _dataset_paths(data_root: Path, dataset: str) -> tuple[Path, Path]:
    dataset_root = data_root / dataset
    papers_path = dataset_root / f"{dataset}_papers.json"
    signatures_path = dataset_root / f"{dataset}_signatures.json"
    if not papers_path.exists() or not signatures_path.exists():
        raise FileNotFoundError(
            f"Expected {papers_path} and {signatures_path}; pass --data-root pointing at the JSON dataset root."
        )
    return papers_path, signatures_path


def _load_in_signature_titles(data_root: Path, dataset: str) -> list[tuple[str, str]]:
    papers_path, signatures_path = _dataset_paths(data_root, dataset)
    signatures = json.loads(signatures_path.read_text(encoding="utf-8"))
    in_signature_paper_ids: list[str] = []
    seen: set[str] = set()
    for signature in signatures.values():
        paper_id = str(signature["paper_id"])
        if paper_id not in seen:
            seen.add(paper_id)
            in_signature_paper_ids.append(paper_id)

    papers = json.loads(papers_path.read_text(encoding="utf-8"))
    titles: list[tuple[str, str]] = []
    for paper_id in in_signature_paper_ids:
        paper = papers[paper_id]
        titles.append((paper_id, paper.get("title") or ""))
    return titles


def _load_rust_module(extension_path: Path | None) -> ModuleType:
    if extension_path is None:
        import s2and_rust

        return s2and_rust

    if (extension_path.parent / "__init__.py").exists():
        sys.modules.pop("s2and_rust", None)
        sys.modules.pop("s2and_rust._s2and_rust", None)
        sys.path.insert(0, str(extension_path.parent.parent.resolve()))
        import s2and_rust

        return s2and_rust

    loader = importlib.machinery.ExtensionFileLoader("_s2and_rust", str(extension_path))
    spec = importlib.util.spec_from_loader("_s2and_rust", loader)
    if spec is None:
        raise RuntimeError(f"Could not create import spec for Rust extension path: {extension_path}")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def _summarize(
    rows: list[tuple[str, str]],
    python_audits: list[DetectorAudit],
    rust_audits: list[DetectorAudit],
    *,
    max_examples: int,
) -> dict[str, Any]:
    mismatch_counts: Counter[str] = Counter()
    pair_counts: dict[str, Counter[str]] = {
        "fasttext_label": Counter(),
        "cld2_label": Counter(),
        "predicted_language": Counter(),
        "is_reliable": Counter(),
    }
    examples: list[dict[str, Any]] = []
    fields = ("fasttext_label", "cld2_label", "predicted_language", "is_reliable")

    for (paper_id, title), python_audit, rust_audit in zip(rows, python_audits, rust_audits, strict=True):
        mismatched_fields = [
            field
            for field, python_value, rust_value in zip(fields, python_audit, rust_audit, strict=True)
            if python_value != rust_value
        ]
        for field in mismatched_fields:
            mismatch_counts[field] += 1
            field_index = fields.index(field)
            pair_counts[field][f"{python_audit[field_index]!r} -> {rust_audit[field_index]!r}"] += 1
        if mismatched_fields and len(examples) < max_examples:
            examples.append(
                {
                    "paper_id": paper_id,
                    "title": title,
                    "mismatched_fields": mismatched_fields,
                    "python": dict(zip(fields, python_audit, strict=True)),
                    "rust": dict(zip(fields, rust_audit, strict=True)),
                }
            )

    return {
        "total_titles": len(rows),
        "mismatch_counts": dict(mismatch_counts),
        "pair_counts": {field: dict(counter.most_common(20)) for field, counter in pair_counts.items()},
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="s2and/data-backup", type=Path)
    parser.add_argument("--dataset", default="qian")
    parser.add_argument("--limit", type=int, default=1000, help="Number of titles to audit by default; use 0 for all.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--rust-extension-path",
        type=Path,
        default=None,
        help="Optional compiled _s2and_rust extension path, useful when the installed .pyd is locked.",
    )
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args()

    if os.environ.get("S2AND_SKIP_FASTTEXT", "").lower() in {"1", "true", "yes"}:
        raise RuntimeError("S2AND_SKIP_FASTTEXT is set; language detector parity audit requires fastText.")

    rust_module = _load_rust_module(args.rust_extension_path)
    rust_audit_fn = getattr(rust_module, "_debug_language_detector_audit", None)
    if rust_audit_fn is None:
        raise RuntimeError("s2and_rust._debug_language_detector_audit is unavailable; rebuild the Rust extension.")

    rows = _load_in_signature_titles(args.data_root, args.dataset)
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.limit > 0:
        rows = rows[: args.limit]
    titles = [title for _paper_id, title in rows]

    python_audits = [_python_language_audit(title) for title in titles]
    rust_audits = [tuple(row) for row in rust_audit_fn(titles)]
    summary = {
        "config": {
            "data_root": str(args.data_root),
            "dataset": args.dataset,
            "limit": args.limit,
        },
        **_summarize(rows, python_audits, rust_audits, max_examples=args.max_examples),
    }

    output = json.dumps(summary, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
