
import asyncio
import json
from pathlib import Path
from statistics import mean
from typing import Any

from main import app


COLLECTION_NAME = "ELARA"

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR / "evaluation_cases.jsonl" 

THRESHOLD_MIN = 0.30
THRESHOLD_MAX = 0.90
THRESHOLD_STEP = 0.01


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """
    Supports both:
        - JSON array
        - JSONL
    """

    text = path.read_text(encoding="utf-8").strip()

    if path.suffix == ".json":
        return json.loads(text)

    if not text:
        return []

    # JSON array
    if text.startswith("["):
        data = json.loads(text)

        if not isinstance(data, list):
            raise ValueError(
                "Evaluation dataset must contain a JSON list."
            )

        return data

    # JSONL
    cases = []

    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        line = line.strip()

        if not line:
            continue

        try:
            cases.append(
                json.loads(line)
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON on line {line_number}: {exc}"
            ) from exc

    return cases


async def get_top_score(
    app_state,
    query: str,
) -> float:

    from models.enums.LLMEnums import (
        DocumentTypeEnum,
    )

    embeddings = await (
        app_state
        .embedding_service
        .embed_text(
            query,
            DocumentTypeEnum.QUERY.value,
        )
    )

    vector = embeddings[0]

    if hasattr(vector, "tolist"):
        vector = vector.tolist()

    results = await (
        app_state
        .vectordb
        .search_by_vector(
            COLLECTION_NAME,
            vector,
            5,
        )
    )

    if not results or not results.points:
        return 0.0

    return max(
        float(point.score)
        for point in results.points
    )


async def collect_scores(
    app_state,
    eval_cases: list[dict[str, Any]],
):
    answer_scores = []
    refusal_scores = []

    records = []

    for case in eval_cases:

        score = await get_top_score(
            app_state,
            case["query"],
        )

        expected_status = case[
            "expected_status"
        ]

        record = {
            "id": case["id"],
            "category": case["category"],
            "expected_status": expected_status,
            "score": round(score, 6),
        }

        records.append(record)

        if expected_status == "answered":
            answer_scores.append(score)

        elif expected_status == "refuse":
            refusal_scores.append(score)

    return (
        answer_scores,
        refusal_scores,
        records,
    )


def evaluate_threshold(
    threshold: float,
    answer_scores: list[float],
    refusal_scores: list[float],
):

    true_positive = sum(
        score >= threshold
        for score in answer_scores
    )

    false_negative = sum(
        score < threshold
        for score in answer_scores
    )

    true_negative = sum(
        score < threshold
        for score in refusal_scores
    )

    false_positive = sum(
        score >= threshold
        for score in refusal_scores
    )

    total = (
        len(answer_scores)
        + len(refusal_scores)
    )

    accuracy = (
        (true_positive + true_negative)
        / total
        if total
        else 0.0
    )

    answer_recall = (
        true_positive
        / len(answer_scores)
        if answer_scores
        else 0.0
    )

    refusal_recall = (
        true_negative
        / len(refusal_scores)
        if refusal_scores
        else 0.0
    )

    # Balanced accuracy is preferable to raw accuracy
    # because the classes may become imbalanced later.
    balanced_accuracy = (
        (answer_recall + refusal_recall)
        / 2
    )

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "true_positive": true_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "false_positive": false_positive,
    }


async def calibrate(
    app_state,
    eval_cases: list[dict[str, Any]],
) -> dict[str, Any]:

    (
        answer_scores,
        refusal_scores,
        records,
    ) = await collect_scores(
        app_state,
        eval_cases,
    )

    candidates = [
        round(
            THRESHOLD_MIN
            + i * THRESHOLD_STEP,
            2,
        )
        for i in range(
            int(
                (
                    THRESHOLD_MAX
                    - THRESHOLD_MIN
                )
                / THRESHOLD_STEP
            )
            + 1
        )
    ]

    threshold_results = [
        evaluate_threshold(
            threshold,
            answer_scores,
            refusal_scores,
        )
        for threshold in candidates
    ]

    best = max(
        threshold_results,
        key=lambda x: (
            x["balanced_accuracy"],
            x["refusal_recall"],
            x["answer_recall"],
        ),
    )

    return {
        "recommended_threshold": best[
            "threshold"
        ],

        "accuracy_at_threshold": round(
            best["accuracy"],
            4,
        ),

        "balanced_accuracy": round(
            best["balanced_accuracy"],
            4,
        ),

        "answer_recall": round(
            best["answer_recall"],
            4,
        ),

        "refusal_recall": round(
            best["refusal_recall"],
            4,
        ),

        "confusion_matrix": {
            "TP": best["true_positive"],
            "FN": best["false_negative"],
            "TN": best["true_negative"],
            "FP": best["false_positive"],
        },

        "should_answer_mean_score": (
            round(
                mean(answer_scores),
                4,
            )
            if answer_scores
            else None
        ),

        "should_refuse_mean_score": (
            round(
                mean(refusal_scores),
                4,
            )
            if refusal_scores
            else None
        ),

        "min_answer_score": (
            round(
                min(answer_scores),
                4,
            )
            if answer_scores
            else None
        ),

        "max_refusal_score": (
            round(
                max(refusal_scores),
                4,
            )
            if refusal_scores
            else None
        ),

        "score_records": records,
    }


async def main():

    eval_cases = load_dataset(
        DATASET_PATH
    )

    result = await calibrate(
        app.state,
        eval_cases,
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())