from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from ..common import (
    DATASET_PATH,
    load_evaluation_cases,
)


API_URL = os.getenv(
    "ELARA_GENERATION_URL",
    "http://127.0.0.1:8000/rag/test-prompt",
)

RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results"
)


def call_generation_api(
    case: dict[str, Any],
) -> tuple[dict[str, Any], float]:

    payload = {
        "query": case["query"],
        "persona": case.get(
            "persona",
            "general",
        ),
        "language": "en",
        "context_chunks": [],
    }

    started = time.perf_counter()

    response = requests.post(
        API_URL,
        json=payload,
        timeout=120,
    )

    latency = (
        time.perf_counter()
        - started
    )

    response.raise_for_status()

    return response.json(), latency


def extract_status(
    body: dict[str, Any],
) -> str:

    return str(
        body.get(
            "status",
            body.get(
                "data",
                {},
            ).get(
                "status",
                "unknown",
            ),
        )
    )


def extract_data(
    body: dict[str, Any],
) -> dict[str, Any]:

    data = body.get(
        "data",
        {},
    )

    return (
        data
        if isinstance(data, dict)
        else {}
    )


def evaluate_case(
    case: dict[str, Any],
) -> dict[str, Any]:

    try:

        body, latency = call_generation_api(
            case
        )

        status = extract_status(
            body
        )

        data = extract_data(
            body
        )

        expected_status = case.get(
            "expected_status"
        )

        # Your endpoint currently returns:
        #
        # status="success"
        #
        # for successful generation.
        #
        # Therefore translate it into
        # the evaluation vocabulary.

        if (
            expected_status == "answered"
            and status == "success"
        ):
            normalized_status = "answered"

        elif (
            expected_status == "refuse"
            and status == "refused"
        ):
            normalized_status = "refuse"

        else:
            normalized_status = status

        return {
            "id": case["id"],
            "category": case.get(
                "category"
            ),
            "query": case["query"],
            "expected_status": expected_status,
            "actual_api_status": status,
            "normalized_status": normalized_status,
            "status_correct": (
                normalized_status
                == expected_status
            ),
            "answer": data.get(
                "answer",
                "",
            ),
            "is_refusal": data.get(
                "is_refusal",
                False,
            ),
            "validation_reason": data.get(
                "validation_reason",
                "",
            ),
            "gate_reason": data.get(
                "gate_reason",
                "",
            ),
            "top_similarity_score": data.get(
                "top_similarity_score"
            ),
            "latency_seconds": latency,
            "error": "",
        }

    except Exception as exc:

        return {
            "id": case["id"],
            "category": case.get(
                "category"
            ),
            "query": case["query"],
            "expected_status": case.get(
                "expected_status"
            ),
            "actual_api_status": "error",
            "normalized_status": "error",
            "status_correct": False,
            "answer": "",
            "is_refusal": None,
            "validation_reason": "",
            "gate_reason": "",
            "top_similarity_score": None,
            "latency_seconds": None,
            "error": str(exc),
        }


def run_generation_eval() -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cases = load_evaluation_cases(
        DATASET_PATH
    )

    rows = []

    print(
        f"Loaded {len(cases)} evaluation cases."
    )

    for index, case in enumerate(
        cases[10:20],
        start=1,
    ):

        print(
            f"[{index}/{len(cases)}] "
            f"{case['id']}"
        )

        row = evaluate_case(
            case
        )

        rows.append(row)

    dataframe = pd.DataFrame(
        rows
    )

    output_path = (
        RESULTS_DIR
        / "generation_results.csv"
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    accuracy = dataframe[
        "status_correct"
    ].mean()

    print(
        "\nGeneration Evaluation"
    )

    print(
        f"Status accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        "\nBy category:"
    )

    print(
        dataframe
        .groupby("category")[
            "status_correct"
        ]
        .mean()
    )

    answered = dataframe[
        dataframe["expected_status"]
        == "answered"
    ]

    if not answered.empty:

        print(
            "\nAnswered cases:"
        )

        print(
            f"Correct status: "
            f"{answered['status_correct'].mean():.4f}"
        )

    refusal = dataframe[
        dataframe["expected_status"]
        == "refuse"
    ]

    if not refusal.empty:

        print(
            "\nRefusal cases:"
        )

        print(
            f"Correct refusal: "
            f"{refusal['status_correct'].mean():.4f}"
        )

    print(
        f"\nResults: {output_path}"
    )


if __name__ == "__main__":
    run_generation_eval()