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


PAGE_TOLERANCE = 2


def document_token_overlap(
    payload: dict[str, Any],
    case: dict[str, Any],
) -> bool:
    """Token-overlap signal between target_doc and the retrieved chunk's
    stored document name. Reported for diagnostics but NOT used to gate
    relevance -- with today's corpus (renamed on upload, original title
    never persisted) it is nearly always False even for a correct hit,
    so gating on it reproduces the all-zero-metrics bug. Once uploads
    preserve the real filename, this can be promoted into a real gate."""
    target_doc = case.get("target_doc")
    if not target_doc:
        return True

    doc_name = get_document_name(payload)
    if not doc_name:
        return False

    target_tokens = {t for t in normalize_text(target_doc).split() if len(t) > 3}
    doc_tokens = {t for t in normalize_text(doc_name).split() if len(t) > 3}
    if not target_tokens or not doc_tokens:
        return False

    return bool(target_tokens & doc_tokens)


def page_is_close(
    payload: dict[str, Any],
    case: dict[str, Any],
    tolerance: int = PAGE_TOLERANCE,
) -> bool:
    target_page = case.get("target_page")
    if target_page is None:
        return False

    pages = get_pages(payload)
    if not pages:
        return False

    return any(abs(p - int(target_page)) <= tolerance for p in pages)


KEYWORD_RELEVANCE_THRESHOLD = 0.5


def label_relevance(
    point: dict[str, Any],
    case: dict[str, Any],
) -> dict[str, Any]:
    """Single source of truth for 'is this retrieved chunk relevant to this
    case', used by retrieval eval, the reranker comparison, and the chunk-
    config comparison so all three agree on the same ground truth.

    Relevant if EITHER:
      - the target page is within PAGE_TOLERANCE pages of a page the chunk
        actually covers, OR
      - keyword coverage on the chunk text is >= KEYWORD_RELEVANCE_THRESHOLD

    Document identity is reported (`document_token_overlap`) but does not
    gate relevance -- see that function's docstring for why.
    """
    if case.get("expected_status") != "answered":
        return {"relevant": False, "reason": "case is not answerable (refusal case)"}

    payload = point.get("payload", {})
    if not isinstance(payload, dict):
        return {"relevant": False, "reason": "no payload on retrieved point"}

    text = payload.get("text", "")
    coverage = keyword_coverage(text, case.get("expected_keywords", []))
    page_ok = page_is_close(payload, case)
    doc_overlap = document_token_overlap(payload, case)

    if page_ok:
        return {"relevant": True, "reason": f"target_page within tolerance ({PAGE_TOLERANCE})", "keyword_coverage": coverage, "document_token_overlap": doc_overlap}
    if coverage >= KEYWORD_RELEVANCE_THRESHOLD:
        return {"relevant": True, "reason": f"keyword_coverage={coverage:.2f} >= {KEYWORD_RELEVANCE_THRESHOLD}", "keyword_coverage": coverage, "document_token_overlap": doc_overlap}
    return {"relevant": False, "reason": f"keyword_coverage={coverage:.2f}, page not close, doc_overlap={doc_overlap}", "keyword_coverage": coverage, "document_token_overlap": doc_overlap}