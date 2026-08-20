from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from evaluation.common import (
    DATASET_PATH,
    get_document_name,
    get_pages,
    get_sections,
    keyword_coverage,
    label_relevance,
    load_evaluation_cases,
    normalize_text,
)


API_URL = os.getenv(
    "ELARA_SEARCH_URL",
    "http://127.0.0.1:8000/rag/search",
)

RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results"
)

TOP_K_VALUES = [
    1,
    3,
    5,
    10,
]


# ============================================================
# RETRIEVAL GROUND TRUTH
# ============================================================

def is_answerable_case(
    case: dict[str, Any],
) -> bool:
    return (
        case.get("expected_status")
        == "answered"
    )


def is_relevant(
    point: dict[str, Any],
    case: dict[str, Any],
) -> bool:
    """Thin wrapper -- ground truth now lives in evaluation.common.label_relevance
    so the retrieval eval, reranker comparison, and chunk-config comparison
    all agree on what counts as a relevant chunk. See that function's
    docstring for why document identity no longer gates relevance."""
    return label_relevance(point, case)["relevant"]


# ============================================================
# API
# ============================================================

def search(
    query: str,
    limit: int,
) -> tuple[list[dict[str, Any]], float]:

    start = time.perf_counter()

    response = requests.post(
        API_URL,
        json={
            "text": query,
            "limit": limit,
        },
        timeout=60,
    )

    latency = (
        time.perf_counter()
        - start
    )

    response.raise_for_status()

    body = response.json()

    data = body.get(
        "data",
        {},
    )

    search_results = data.get(
        "search_results",
        {},
    )

    if isinstance(
        search_results,
        dict,
    ):
        points = search_results.get(
            "points",
            [],
        )

    elif isinstance(
        search_results,
        list,
    ):
        points = search_results

    else:
        points = []

    return points, latency


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    points: list[dict[str, Any]],
    case: dict[str, Any],
    k: int,
) -> dict[str, float]:

    top_k = points[:k]

    relevant_ranks = [
        rank
        for rank, point in enumerate(
            top_k,
            start=1,
        )
        if is_relevant(
            point,
            case,
        )
    ]

    retrieved_count = len(top_k)

    relevant_count = len(
        relevant_ranks
    )

    precision = (
        relevant_count
        / retrieved_count
        if retrieved_count
        else 0.0
    )

    # This dataset currently defines one target
    # evidence location per answerable case.
    recall = (
        1.0
        if relevant_count > 0
        else 0.0
    )

    hit = int(
        relevant_count > 0
    )

    reciprocal_rank = (
        1.0 / relevant_ranks[0]
        if relevant_ranks
        else 0.0
    )

    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "hit_at_k": hit,
        "mrr_at_k": reciprocal_rank,
    }


# ============================================================
# CASE EVALUATION
# ============================================================

def evaluate_case(
    case: dict[str, Any],
    k: int,
) -> dict[str, Any]:

    started = time.perf_counter()

    try:
        points, api_latency = search(
            case["query"],
            k,
        )

        error = ""

    except Exception as exc:
        points = []
        api_latency = 0.0
        error = str(exc)

    metrics = calculate_metrics(
        points,
        case,
        k,
    )

    retrieved = []

    for rank, point in enumerate(
        points[:k],
        start=1,
    ):

        payload = point.get(
            "payload",
            {},
        )

        text = payload.get(
            "text",
            "",
        )

        retrieved.append({
            "rank": rank,
            "chunk_id": str(
                point.get(
                    "id",
                    "",
                )
            ),
            "score": float(
                point.get(
                    "score",
                    0.0,
                )
            ),
            "document": get_document_name(
                payload
            ),
            "pages": get_pages(
                payload
            ),
            "sections": get_sections(
                payload
            ),
            "relevant": is_relevant(
                point,
                case,
            ),
            "keyword_coverage": keyword_coverage(
                text,
                case.get(
                    "expected_keywords",
                    [],
                ),
            ),
            "text_preview": normalize_text(
                text
            )[:300],
        })

    first_relevant_rank = next(
        (
            item["rank"]
            for item in retrieved
            if item["relevant"]
        ),
        None,
    )

    return {
        "id": case["id"],
        "category": case.get(
            "category"
        ),
        "query": case["query"],
        "expected_status": case.get(
            "expected_status"
        ),
        "target_doc": case.get(
            "target_doc"
        ),
        "target_page": case.get(
            "target_page"
        ),
        "top_k": k,
        "returned_count": len(points[:k]),
        "precision_at_k": metrics[
            "precision_at_k"
        ],
        "recall_at_k": metrics[
            "recall_at_k"
        ],
        "hit_at_k": metrics[
            "hit_at_k"
        ],
        "mrr_at_k": metrics[
            "mrr_at_k"
        ],
        "first_relevant_rank": (
            first_relevant_rank
        ),
        "api_latency_seconds": api_latency,
        "evaluation_latency_seconds": (
            time.perf_counter()
            - started
        ),
        "error": error,
        "retrieved_chunks_json": json.dumps(
            retrieved,
            ensure_ascii=False,
        ),
    }


# ============================================================
# EVALUATION
# ============================================================

def run_evaluation() -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cases = load_evaluation_cases(
        DATASET_PATH
    )

    answerable_cases = [
        case
        for case in cases
        if is_answerable_case(case)
    ]

    if not answerable_cases:
        raise RuntimeError(
            "No answerable evaluation cases found."
        )

    print(
        f"Dataset: {DATASET_PATH}"
    )

    print(
        f"Answerable cases: "
        f"{len(answerable_cases)}"
    )

    rows = []

    for k in TOP_K_VALUES:

        print(
            f"\n{'=' * 70}"
        )

        print(
            f"Evaluating Retrieval @ K={k}"
        )

        print(
            f"{'=' * 70}"
        )

        for index, case in enumerate(
            answerable_cases,
            start=1,
        ):

            print(
                f"[{index}/{len(answerable_cases)}] "
                f"{case['id']}"
            )

            row = evaluate_case(
                case,
                k,
            )

            rows.append(row)

    dataframe = pd.DataFrame(
        rows
    )

    metrics_path = (
        RESULTS_DIR
        / "retrieval_cases.csv"
    )

    dataframe.to_csv(
        metrics_path,
        index=False,
    )

    summary = (
        dataframe
        .groupby("top_k")
        .agg(
            precision_at_k=(
                "precision_at_k",
                "mean",
            ),
            recall_at_k=(
                "recall_at_k",
                "mean",
            ),
            hit_rate=(
                "hit_at_k",
                "mean",
            ),
            mrr=(
                "mrr_at_k",
                "mean",
            ),
            mean_latency_seconds=(
                "api_latency_seconds",
                "mean",
            ),
            queries=(
                "id",
                "count",
            ),
        )
        .reset_index()
    )

    summary_path = (
        RESULTS_DIR
        / "retrieval_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        "\nRetrieval Evaluation Complete"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        f"\nCases: {metrics_path}"
    )

    print(
        f"Summary: {summary_path}"
    )


if __name__ == "__main__":
    run_evaluation()