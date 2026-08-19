# src/evaluation/retrieval/run_retrieval_eval.py

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests


API_URL = os.getenv(
    "ELARA_SEARCH_URL",
    "http://127.0.0.1:8000/rag/search",
)

DATASET_PATH = Path(
    os.getenv(
        "ELARA_EVAL_DATASET",
        "src/evaluation/retrieval/dataset/evaluation_cases.jsonl",
    )
)

RESULTS_DIR = Path(
    "src/evaluation/retrieval/results"
)

METRICS_CSV = (
    RESULTS_DIR
    / "eval_metrics.csv"
)

SUMMARY_CSV = (
    RESULTS_DIR
    / "eval_summary.csv"
)

TOP_K_CONFIGS = [3, 5, 10]


# ============================================================
# DATASET
# ============================================================

def load_dataset(
    path: Path,
) -> list[dict[str, Any]]:

    text = path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return []

    if text.startswith("["):
        return json.loads(text)

    return [
        json.loads(line)
        for line in text.splitlines()
        if line.strip()
    ]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(
    text: Any,
) -> str:

    if text is None:
        return ""

    return " ".join(
        str(text)
        .lower()
        .split()
    )


def get_pages(
    payload: dict,
) -> list[int]:

    pages = payload.get(
        "page_numbers",
        payload.get(
            "page_number",
            [],
        ),
    )

    if pages is None:
        return []

    if isinstance(
        pages,
        int,
    ):
        return [pages]

    if isinstance(
        pages,
        str,
    ):
        try:
            return [int(pages)]
        except ValueError:
            return []

    if isinstance(
        pages,
        list,
    ):
        normalized = []

        for page in pages:
            try:
                normalized.append(
                    int(page)
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

        return normalized

    return []


def get_sections(
    payload: dict,
) -> list[str]:

    sections = payload.get(
        "section_headings",
        payload.get(
            "section",
            [],
        ),
    )

    if sections is None:
        return []

    if isinstance(
        sections,
        str,
    ):
        return [sections]

    if isinstance(
        sections,
        list,
    ):
        return [
            str(section)
            for section in sections
            if section
        ]

    return [str(sections)]


# ============================================================
# RETRIEVAL RELEVANCE
# ============================================================

def keyword_coverage(
    retrieved_text: str,
    expected_keywords: list[str],
) -> float:

    if not expected_keywords:
        return 0.0

    text = normalize_text(
        retrieved_text
    )

    matched = sum(
        normalize_text(keyword)
        in text
        for keyword in expected_keywords
    )

    return matched / len(
        expected_keywords
    )


def document_match(
    payload: dict,
    case: dict,
) -> bool:

    target_doc = case.get(
        "target_doc"
    )

    if not target_doc:
        return False

    candidates = [
        payload.get("doc_name"),
        payload.get(
            "original_filename"
        ),
        payload.get("source"),
        payload.get("document_id"),
    ]

    target_norm = normalize_text(
        target_doc
    )

    return any(
        target_norm
        in normalize_text(candidate)
        for candidate in candidates
        if candidate
    )


def page_match(
    payload: dict,
    case: dict,
) -> bool:

    target_page = case.get(
        "target_page"
    )

    if target_page is None:
        return False

    return (
        int(target_page)
        in get_pages(payload)
    )


def is_relevant(
    point: dict,
    case: dict,
) -> bool:

    # Refusal cases do NOT have a retrieval
    # relevance target.
    if (
        case.get("expected_status")
        != "answered"
    ):
        return False

    payload = point.get(
        "payload",
        {},
    )

    text = payload.get(
        "text",
        "",
    )

    expected_keywords = case.get(
        "expected_keywords",
        [],
    )

    doc_ok = document_match(
        payload,
        case,
    )

    page_ok = page_match(
        payload,
        case,
    )

    coverage = keyword_coverage(
        text,
        expected_keywords,
    )

    # Strong relevance:
    #
    # 1. target document + target page
    # OR
    # 2. target document + >= 50% keywords
    #
    # We intentionally do NOT use:
    #
    #     score >= 0.60
    #
    # as ground truth.
    #
    # Similarity is a retrieval signal, not relevance
    # ground truth.

    if doc_ok and page_ok:
        return True

    if doc_ok and coverage >= 0.50:
        return True

    return False


# ============================================================
# API
# ============================================================

def search(
    query: str,
    limit: int,
) -> list[dict]:

    response = requests.post(
        API_URL,
        json={
            "text": query,
            "limit": limit,
        },
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    # Current ELARA response shape
    search_data = (
        data
        .get("data", {})
        .get(
            "search_results",
            {},
        )
    )

    if isinstance(
        search_data,
        dict,
    ):
        points = search_data.get(
            "points",
            [],
        )

    elif isinstance(
        search_data,
        list,
    ):
        points = search_data

    else:
        points = []

    return points


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    points: list[dict],
    case: dict,
    k: int,
):

    top_k = points[:k]

    relevant_ranks = []

    for rank, point in enumerate(
        top_k,
        start=1,
    ):

        if is_relevant(
            point,
            case,
        ):
            relevant_ranks.append(
                rank
            )

    relevant_count = len(
        relevant_ranks
    )

    precision = (
        relevant_count / k
        if k
        else 0.0
    )

    # For this dataset, each safe query has one
    # intended evidence target.
    #
    # Therefore recall is:
    #
    # 1 if at least one relevant chunk found
    # 0 otherwise.
    #
    recall = (
        1.0
        if relevant_ranks
        else 0.0
    )

    mrr = (
        1.0 / relevant_ranks[0]
        if relevant_ranks
        else 0.0
    )

    hit = int(
        bool(relevant_ranks)
    )

    return {
        "precision_at_k": precision,
        "recall": recall,
        "hit_at_k": hit,
        "reciprocal_rank": mrr,
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

def run_evaluation_pipeline():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cases = load_dataset(
        DATASET_PATH
    )

    if not cases:
        raise RuntimeError(
            "Evaluation dataset is empty."
        )

    run_id = (
        "RUN_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    rows = []

    for k in TOP_K_CONFIGS:

        print(
            f"\n{'=' * 70}"
        )

        print(
            f"Running retrieval evaluation "
            f"with Top-K={k}"
        )

        print(
            f"{'=' * 70}"
        )

        for case in cases:

            print(
                f"[{case['id']}] "
                f"{case['query']}"
            )

            try:

                points = search(
                    case["query"],
                    k,
                )

            except Exception as exc:

                print(
                    f"ERROR: {exc}"
                )

                points = []

            metrics = calculate_metrics(
                points,
                case,
                k,
            )

            retrieved_records = []

            for rank, point in enumerate(
                points[:k],
                start=1,
            ):

                payload = point.get(
                    "payload",
                    {},
                )

                pages = get_pages(
                    payload
                )

                sections = get_sections(
                    payload
                )

                text = payload.get(
                    "text",
                    "",
                )

                retrieved_records.append({

                    "rank": rank,

                    "chunk_id": point.get(
                        "id",
                        "N/A",
                    ),

                    "score": round(
                        float(
                            point.get(
                                "score",
                                0.0,
                            )
                        ),
                        4,
                    ),

                    "document_id": payload.get(
                        "document_id",
                        "N/A",
                    ),

                    "document": (
                        payload.get(
                            "doc_name"
                        )
                        or payload.get(
                            "original_filename"
                        )
                        or payload.get(
                            "source"
                        )
                        or "N/A"
                    ),

                    "pages": pages,

                    "sections": sections,

                    "is_relevant": (
                        is_relevant(
                            point,
                            case,
                        )
                    ),

                    "keyword_coverage": round(
                        keyword_coverage(
                            text,
                            case.get(
                                "expected_keywords",
                                [],
                            ),
                        ),
                        4,
                    ),

                    "text_preview": (
                        text[:300]
                        .replace(
                            "\n",
                            " ",
                        )
                    ),
                })

            top1 = (
                retrieved_records[0]
                if retrieved_records
                else {}
            )

            rows.append({

                "run_id": run_id,

                "timestamp": (
                    datetime.now()
                    .isoformat()
                ),

                "top_k": k,

                "query_id": case["id"],

                "category": case[
                    "category"
                ],

                "persona": case[
                    "persona"
                ],

                "query": case[
                    "query"
                ],

                "expected_status": case[
                    "expected_status"
                ],

                "is_failure_case": case.get(
                    "is_failure_case",
                    False,
                ),

                "failure_mode": (
                    case.get(
                        "failure_mode"
                    )
                    or ""
                ),

                "expected_gate_trigger": (
                    case.get(
                        "expected_gate_trigger"
                    )
                    or ""
                ),

                "precision_at_k": round(
                    metrics[
                        "precision_at_k"
                    ],
                    4,
                ),

                "recall": round(
                    metrics[
                        "recall"
                    ],
                    4,
                ),

                "hit_at_k": metrics[
                    "hit_at_k"
                ],

                "reciprocal_rank": round(
                    metrics[
                        "reciprocal_rank"
                    ],
                    4,
                ),

                "top1_score": (
                    top1.get(
                        "score"
                    )
                ),

                "top1_chunk_id": (
                    top1.get(
                        "chunk_id",
                        "N/A",
                    )
                ),

                "top1_document": (
                    top1.get(
                        "document",
                        "N/A",
                    )
                ),

                "top1_pages": json.dumps(
                    top1.get(
                        "pages",
                        [],
                    )
                ),

                "top1_sections": json.dumps(
                    top1.get(
                        "sections",
                        [],
                    ),
                    ensure_ascii=False,
                ),

                "retrieved_chunks_json": (
                    json.dumps(
                        retrieved_records,
                        ensure_ascii=False,
                    )
                ),
            })

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        METRICS_CSV,
        index=False,
    )

    aggregate_and_save_summary(
        df
    )

    print(
        "\nRetrieval evaluation complete."
    )

    print(
        f"Metrics: {METRICS_CSV}"
    )

    print(
        f"Summary: {SUMMARY_CSV}"
    )


# ============================================================
# SUMMARY
# ============================================================

def aggregate_and_save_summary(
    df: pd.DataFrame,
):

    # Only answerable questions have retrieval
    # relevance ground truth.
    safe_df = df[
        df["expected_status"]
        == "answered"
    ].copy()

    if safe_df.empty:
        print(
            "No answerable cases found."
        )
        return

    summary = (
        safe_df
        .groupby("top_k")
        .agg(
            mean_precision_at_k=(
                "precision_at_k",
                "mean",
            ),

            mean_recall=(
                "recall",
                "mean",
            ),

            hit_rate=(
                "hit_at_k",
                "mean",
            ),

            mean_mrr=(
                "reciprocal_rank",
                "mean",
            ),

            mean_top1_score=(
                "top1_score",
                "mean",
            ),

            total_queries=(
                "query_id",
                "count",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    print(
        "\nRetrieval summary:"
    )

    print(
        summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    run_evaluation_pipeline()