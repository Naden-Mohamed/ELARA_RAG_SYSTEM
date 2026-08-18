import json
import csv
import os
import requests
import pandas as pd
from datetime import datetime

API_URL = "http://127.0.0.1:8000/rag/search"
DATASET_PATH = "evaluation/retrieval/dataset/eval_cases.json"
RESULTS_DIR = "evaluation/retrieval/results"
METRICS_CSV = os.path.join(RESULTS_DIR, "eval_metrics.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "eval_summary.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)

TOP_K_CONFIGS = [3, 5, 10]
CHUNK_CONFIGS = ["docling_hierarchical_512", "sentence_chunker_300"]
EMBEDDING_MODELS = ["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]

def evaluate_relevance(retrieved_chunk: dict, case: dict) -> bool:
    if case["is_failure_case"]:
        return False
    
    payload = retrieved_chunk.get("payload", {})
    text = payload.get("text", "").lower()
    
    expected_keywords = case.get("expected_keywords", [])
    if expected_keywords:
        matches = sum(1 for kw in expected_keywords if kw.lower() in text)
        return matches > 0

    return False

def run_evaluation_pipeline():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)

    file_exists = os.path.isfile(METRICS_CSV)
    
    with open(METRICS_CSV, mode="a", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "run_id", "timestamp", "chunk_config", "embedding_model", "top_k",
            "query_id", "query", "is_failure_case", "failure_mode",
            "precision_at_k", "relevant_count", "reciprocal_rank",
            "retrieved_ranks", "retrieved_chunks_json"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        run_id = f"RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        for chunk_cfg in CHUNK_CONFIGS:
            for emb_model in EMBEDDING_MODELS:
                for k in TOP_K_CONFIGS:
                    print(f"Running: Config={chunk_cfg} | Model={emb_model} | Top-K={k}")
                    
                    for case in eval_cases:
                        payload = {"text": case["query"], "limit": k}
                        try:
                            resp = requests.post(API_URL, json=payload, timeout=120)
                            if resp.status_code != 200:
                                continue
                            data = resp.json().get("data", {})
                            results = data.get("result", [])
                        except Exception as e:
                            print(f"Error requesting query {case['query_id']}: {e}")
                            continue

                        relevant_count = 0
                        first_relevant_rank = 0
                        chunk_records = []

                        for rank, point in enumerate(results, start=1):
                            p_load = point.get("payload", {})
                            is_rel = evaluate_relevance(point, case)
                            
                            if is_rel:
                                relevant_count += 1
                                if first_relevant_rank == 0:
                                    first_relevant_rank = rank

                            chunk_records.append({
                                "rank": rank,
                                "chunk_id": point.get("id"),
                                "score": round(point.get("score", 0.0), 4),
                                "document": p_load.get("doc_name") or p_load.get("original_filename", "N/A"),
                                "page": p_load.get("page_numbers") or p_load.get("page_number", "N/A"),
                                "section": p_load.get("section_headings") or p_load.get("section_title", "N/A"),
                                "text_preview": p_load.get("text", "")[:140].replace("\n", " ") + "...",
                                "is_relevant": is_rel
                            })

                        prec_k = relevant_count / k if k > 0 else 0
                        mrr = 1.0 / first_relevant_rank if first_relevant_rank > 0 else 0.0

                        writer.writerow({
                            "run_id": run_id,
                            "timestamp": datetime.now().isoformat(),
                            "chunk_config": chunk_cfg,
                            "embedding_model": emb_model,
                            "top_k": k,
                            "query_id": case["query_id"],
                            "query": case["query"],
                            "is_failure_case": case["is_failure_case"],
                            "failure_mode": case.get("failure_mode", ""),
                            "precision_at_k": round(prec_k, 4),
                            "relevant_count": relevant_count,
                            "reciprocal_rank": round(mrr, 4),
                            "retrieved_ranks": [c["rank"] for c in chunk_records if c["is_relevant"]],
                            "retrieved_chunks_json": json.dumps(chunk_records, ensure_ascii=False)
                        })

    aggregate_and_save_summary()

def aggregate_and_save_summary():
    if not os.path.exists(METRICS_CSV):
        return
        
    df = pd.read_csv(METRICS_CSV)
    
    valid_df = df[df["is_failure_case"] == False]
    
    summary = valid_df.groupby(
        ["chunk_config", "embedding_model", "top_k"]
    ).agg(
        avg_precision=("precision_at_k", "mean"),
        mean_reciprocal_rank=("reciprocal_rank", "mean"),
        total_queries=("query_id", "count"),
        total_relevant_retrieved=("relevant_count", "sum")
    ).reset_index()

    summary["avg_precision"] = summary["avg_precision"].round(4)
    summary["mean_reciprocal_rank"] = summary["mean_reciprocal_rank"].round(4)

    summary.to_csv(SUMMARY_CSV, index=False)
    print("Retrieval benchmark complete. Aggregated summary written to:", SUMMARY_CSV)

if __name__ == "__main__":
    run_evaluation_pipeline()