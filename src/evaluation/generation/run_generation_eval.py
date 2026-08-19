from src.core.prompts import GENERATION_JUDGE_PROMPT
from typing import Dict, Any, List
import pandas as pd
def judge_generation(query: str, evidence: str, recommendation: str) -> Dict[str, Any]:
    user_prompt = f"QUESTION:\n{query}\n\nEVIDENCE:\n{evidence}\n\nRECOMMENDATION:\n{recommendation}"
    try:
        result = call_json_model(GENERATION_JUDGE_PROMPT, user_prompt)
        return {
            "faithful": bool(result.get("faithful", False)),
            "relevant": bool(result.get("relevant", False)),
            "reason": str(result.get("reason", "")).strip(),
        }
    except Exception as exc:
        return {"faithful": False, "relevant": False, "reason": f"Judge call failed: {exc}"}

def run_generation_eval(dataset: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in dataset:
        result = run_pipeline(item["query"], audience=None, top_k=TOP_K, run_llm_evidence_gate=True)
        status = result["status"]
        expected = item["expected_status"]

        row = {
            "id": item.get("id"),
            "query": item["query"],
            "expected_status": expected,
            "actual_status": status,
            "status_correct": status == expected,
            "validation_failed": "grounding/citation validation" in result["reason"],
            "faithful": None,
            "relevant_judge": None,
            "relevant_overlap": None,
        }

        if status == "answered":
            resp = result["response"]
            judge = judge_generation(item["query"], resp.evidence, resp.recommendation)
            row["faithful"] = judge["faithful"]
            row["relevant_judge"] = judge["relevant"]
            row["judge_reason"] = judge["reason"]
            if item.get("ground_truth"):
                row["relevant_overlap"] = compute_precision_at_k(
                    [{"text": resp.recommendation}], item["ground_truth"], k=1
                )

        rows.append(row)

    return pd.DataFrame(rows)


generation_eval = run_generation_eval(eval_dataset + labeled_failure_cases)
print("Refusal/answer status accuracy:", generation_eval["status_correct"].mean())
answered = generation_eval[generation_eval["actual_status"] == "answered"]
print("Citation-validation clean-pass rate:", 1 - generation_eval["validation_failed"].mean())
print("Faithfulness rate:", answered["faithful"].mean())
print("Judge-relevance rate:", answered["relevant_judge"].mean())