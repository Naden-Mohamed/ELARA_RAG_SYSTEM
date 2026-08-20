from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "src"
    / "evaluation"
    / "retrieval"
    / "dataset"
    / "evaluation_cases.jsonl"
)


def load_evaluation_cases(
    path: Path = DATASET_PATH,
) -> list[dict[str, Any]]:
    """
    Load evaluation_cases.jsonl.

    Supports both:
      1. JSONL: one JSON object per line
      2. JSON array: [ {...}, {...} ]
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}"
        )

    text = path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        raise ValueError(
            f"Evaluation dataset is empty: {path}"
        )

    # Support JSON array.
    if text.startswith("["):
        data = json.loads(text)

        if not isinstance(data, list):
            raise ValueError(
                "Evaluation dataset must contain a JSON list."
            )

        return data

    # JSONL.
    cases = []

    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        line = line.strip()

        if not line:
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON on line {line_number}: {exc}"
            ) from exc

        if not isinstance(item, dict):
            raise ValueError(
                f"Line {line_number} is not a JSON object."
            )

        cases.append(item)

    return cases


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    return " ".join(
        str(value)
        .lower()
        .split()
    )


def get_pages(payload: dict[str, Any]) -> list[int]:
    pages = payload.get(
        "page_numbers",
        payload.get(
            "page_number",
            [],
        ),
    )

    if pages is None:
        return []

    if isinstance(pages, int):
        return [pages]

    if isinstance(pages, str):
        try:
            return [int(pages)]
        except ValueError:
            return []

    if isinstance(pages, list):
        output = []

        for page in pages:
            try:
                output.append(int(page))
            except (
                TypeError,
                ValueError,
            ):
                continue

        return output

    return []


def get_sections(payload: dict[str, Any]) -> list[str]:
    sections = payload.get(
        "section_headings",
        payload.get(
            "section",
            [],
        ),
    )

    if sections is None:
        return []

    if isinstance(sections, str):
        return [sections]

    if isinstance(sections, list):
        return [
            str(section)
            for section in sections
            if section
        ]

    return [str(sections)]


def get_document_name(
    payload: dict[str, Any],
) -> str:
    for key in (
        "doc_name",
        "original_filename",
        "source",
        "document_id",
    ):
        value = payload.get(key)

        if value:
            return str(value)

    return ""


def keyword_coverage(
    text: str,
    expected_keywords: list[str],
) -> float:
    if not expected_keywords:
        return 0.0

    normalized = normalize_text(text)

    matched = 0

    for keyword in expected_keywords:
        if normalize_text(keyword) in normalized:
            matched += 1

    return matched / len(expected_keywords)