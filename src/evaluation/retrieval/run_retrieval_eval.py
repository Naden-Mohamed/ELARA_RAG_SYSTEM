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
    load_evaluation_cases,
    normalize_text,
)

API_URL = os.getenv(
    "ELARA_SEARCH_URL",
    "http://127.0.0.1:8000/rag/search",
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
TOP_K_VALUES = [1, 3, 5, 10]


# ============================================================
# RETRIEVAL GROUND TRUTH
# ============================================================


def is_relevant(point: dict[str, Any], case: dict[str, Any]) -> bool:
    """Evaluates chunk relevance based on expected case status."""
    expected_status = case.get("expected_status", "answered")

    # Refusal cases (unsafe, out_of_scope, ambiguous) should NOT retrieve relevant content
    if expected_status == "refuse":
        return False

    payload = point.get("payload", {})
    text = payload.get("text", "")
    expected_keywords = case.get("expected_keywords", [])

    # Check keyword coverage
    coverage = keyword_coverage(text, expected_keywords) if expected_keywords else 0.0
    if coverage >= 0.5:
        return True

    # Page offset check (+/- 2 pages tolerance)
    target_page = case.get("target_page")
    retrieved_pages = get_pages(payload)
    if target_page is None or not retrieved_pages:
        return False
    return any(abs(p - target_page) <= 2 for p in retrieved_pages if isinstance(p, int))


# ============================================================
# API
# ============================================================


def search(query: str, limit: int) -> tuple[list[dict[str, Any]], float]:
    start = time.perf_counter()
    response = requests.post(
        API_URL,
        json={"text": query, "limit": limit},
        timeout=60,
    )
    latency = time.perf_counter() - start
    response.raise_for_status()

    body = response.json()
    data = body.get("data", {})
    search_results = data.get("search_results", {})

    if isinstance(search_results, dict):
        points = search_results.get("points", [])
    elif isinstance(search_results, list):
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
    expected_status = case.get("expected_status", "answered")

    # Filter metrics strictly by answerable queries
    if expected_status == "answered":
        relevant_ranks = [
            rank
            for rank, point in enumerate(top_k, start=1)
            if is_relevant(point, case)
        ]

        # Binary target: hit is 1 if at least one relevant chunk found in top k
        hit = 1 if len(relevant_ranks) > 0 else 0
        recall = 1.0 if hit else 0.0
        mrr = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0
        precision = (1.0 if hit else 0.0) / k

        return {
            "precision_at_k": precision,
            "recall_at_k": recall,
            "hit_at_k": hit,
            "mrr_at_k": mrr,
        }
    else:
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "hit_at_k": 0.0,
            "mrr_at_k": 0.0,
        }


# ============================================================
# CASE EVALUATION
# ============================================================


def evaluate_case(case: dict[str, Any], k: int) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        points, api_latency = search(case["query"], k)
        error = ""
    except Exception as exc:
        points = []
        api_latency = 0.0
        error = str(exc)

    metrics = calculate_metrics(points, case, k)
    retrieved = []

    for rank, point in enumerate(points[:k], start=1):
        payload = point.get("payload", {})
        text = payload.get("text", "")

        retrieved.append(
            {
                "rank": rank,
                "chunk_id": str(point.get("id", "")),
                "score": float(point.get("score", 0.0)),
                "document": get_document_name(payload),
                "pages": get_pages(payload),
                "sections": get_sections(payload),
                "relevant": is_relevant(point, case),
                "keyword_coverage": keyword_coverage(
                    text, case.get("expected_keywords", [])
                ),
                "text_preview": normalize_text(text)[:300],
            }
        )

    first_relevant_rank = next(
        (item["rank"] for item in retrieved if item["relevant"]),
        None,
    )

    return {
        "id": case["id"],
        "category": case.get("category"),
        "query": case["query"],
        "expected_status": case.get("expected_status"),
        "target_doc": case.get("target_doc"),
        "target_page": case.get("target_page"),
        "top_k": k,
        "returned_count": len(points[:k]),
        "precision_at_k": metrics["precision_at_k"],
        "recall_at_k": metrics["recall_at_k"],
        "hit_at_k": metrics["hit_at_k"],
        "mrr_at_k": metrics["mrr_at_k"],
        "first_relevant_rank": first_relevant_rank,
        "api_latency_seconds": api_latency,
        "evaluation_latency_seconds": time.perf_counter() - started,
        "error": error,
        "retrieved_chunks_json": json.dumps(retrieved, ensure_ascii=False),
    }


# ============================================================
# RUN EVALUATION
# ============================================================


def run_evaluation() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cases = load_evaluation_cases(DATASET_PATH)

    if not cases:
        raise RuntimeError("No evaluation cases found.")

    print(f"Dataset: {DATASET_PATH}")
    print(f"Total evaluation cases loaded: {len(cases)}")

    rows = []
    for k in TOP_K_VALUES:
        print(f"\n{'=' * 70}\nEvaluating Retrieval @ K={k}\n{'=' * 70}")
        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] Case: {case['id']} ({case.get('category')})")
            row = evaluate_case(case, k)
            rows.append(row)

    dataframe = pd.DataFrame(rows)
    metrics_path = RESULTS_DIR / "all_retrieval_cases.csv"
    dataframe.to_csv(metrics_path, index=False)

    # Detailed summary broken down by top_k and category
    summary = (
        dataframe.groupby(["top_k", "category"])
        .agg(
            precision_at_k=("precision_at_k", "mean"),
            recall_at_k=("recall_at_k", "mean"),
            hit_rate=("hit_at_k", "mean"),
            mrr=("mrr_at_k", "mean"),
            mean_latency_seconds=("api_latency_seconds", "mean"),
            queries=("id", "count"),
        )
        .reset_index()
    )

    summary_path = RESULTS_DIR / "all_retrieval_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nRetrieval Evaluation Complete")
    print(summary.to_string(index=False))
    print(f"\nCases output: {metrics_path}")
    print(f"Summary output: {summary_path}")


if __name__ == "__main__":
    run_evaluation()
