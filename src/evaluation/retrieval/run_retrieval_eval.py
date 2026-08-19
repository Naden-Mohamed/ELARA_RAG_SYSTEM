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
EMBEDDING_MODELS = ["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "BAAI/bge-m3"]

 # Test with diff chunk size and overlap
def evaluate_relevance(retrieved_chunk: dict, case: dict) -> bool:
    if case.get("is_failure_case", False):
        return False
    
    score = retrieved_chunk.get("score", 0.0)
    payload = retrieved_chunk.get("payload", {})
    text = payload.get("text", "").lower()
    
    expected_keywords = case.get("expected_keywords", [])
    target_page = case.get("target_page")
    page_nums = payload.get("page_numbers", [])
    
    page_match = target_page and any(p == target_page for p in page_nums)
    score_match = score >= 0.60
    
    if not expected_keywords:
        return score_match
        
    keyword_matches = sum(1 for kw in expected_keywords if kw.lower() in text)
    
    return keyword_matches > 0 or page_match or score_match # Score thresholding

def run_evaluation_pipeline():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)

    with open(METRICS_CSV, mode="w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "run_id", "timestamp", "chunk_config", "embedding_model", "top_k",
            "query_id", "query", "is_failure_case", "failure_mode",
            "precision_at_k", "recall", "reciprocal_rank",
            "chunk_id", "document_id", "page", "section", "retrieved_chunks_json"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        run_id = f"RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        for chunk_cfg in CHUNK_CONFIGS:
            for emb_model in EMBEDDING_MODELS:
                for k in TOP_K_CONFIGS:
                    print(f"Running: Config={chunk_cfg} | Model={emb_model} | Top-K={k}")
                    
                    for case in eval_cases:
                        is_failure = case.get("is_failure_case", False)
                        results = []
                        
                        try:
                            payload = {"text": case["query"], "limit": k}
                            resp = requests.post(API_URL, json=payload, timeout=30)
                            if resp.status_code == 200:
                                res_json = resp.json()
                                if res_json and "data" in res_json:
                                    search_data = res_json["data"].get("search_results", {})
                                    if isinstance(search_data, dict):
                                        results = search_data.get("points", [])
                                    elif isinstance(search_data, list):
                                        results = search_data
                        except Exception as e:
                            print(f"Error fetching API for query {case['query_id']}: {e}")

                        chunk_records = []
                        relevant_count = 0
                        first_relevant_rank = 0
                        first_chunk_id = "N/A"
                        first_doc_id = "N/A"

                        target_results = results[:k] if isinstance(results, list) else []

                        for rank, point in enumerate(target_results, start=1):
                            if not isinstance(point, dict):
                                continue
                                
                            p_load = point.get("payload", {})
                            c_id = point.get("id", "N/A")
                            doc_id = p_load.get("document_id", "N/A")
                            
                            doc_name = p_load.get("doc_name") or p_load.get("original_filename", "N/A")
                            page_nums = p_load.get("page_numbers", [1])
                            page_num = page_nums[0] if isinstance(page_nums, list) and page_nums else 1
                            section_headings = p_load.get("section_headings", [])
                            section_title = section_headings[0] if isinstance(section_headings, list) and section_headings else "General"
                            chunk_text = p_load.get("text", "")
                            score_val = round(point.get("score", 0.0), 4)

                            if rank == 1:
                                first_chunk_id = c_id
                                first_doc_id = doc_id

                            is_rel = evaluate_relevance(point, case)
                            
                            if is_rel:
                                relevant_count += 1
                                if first_relevant_rank == 0:
                                    first_relevant_rank = rank

                            chunk_records.append({
                                "rank": rank,
                                "chunk_id": c_id,
                                "document_id": doc_id,
                                "score": score_val,
                                "document": doc_name,
                                "page": page_num,
                                "section": section_title,
                                "text_preview": chunk_text[:100].replace("\n", " ") + "...",
                                "is_relevant": is_rel
                            })

                        prec_k = relevant_count / k if k > 0 else 0
                        recall = relevant_count / len(target_results) if len(target_results) > 0 else 0
                        mrr = 1.0 / first_relevant_rank if first_relevant_rank > 0 else 0.0


                        writer.writerow({
                            "run_id": run_id,
                            "timestamp": datetime.now().isoformat(),
                            "chunk_config": chunk_cfg,
                            "embedding_model": emb_model,
                            "top_k": k,
                            "query_id": case["query_id"],
                            "query": case["query"],
                            "is_failure_case": is_failure,
                            "failure_mode": case.get("failure_mode", "") or "",
                            "precision_at_k": round(prec_k, 4),
                            "recall": round(recall, 4),
                            "reciprocal_rank": round(mrr, 4),
                            "chunk_id": first_chunk_id,
                            "document_id": first_doc_id,
                            "page": chunk_records[0]["page"] if chunk_records else "N/A",
                            "section": chunk_records[0]["section"] if chunk_records else "N/A",
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
        total_relevant_retrieved=("recall", "count")
    ).reset_index()

    summary.to_csv(SUMMARY_CSV, index=False)
    print("Retrieval benchmark complete. Summary written to:", SUMMARY_CSV)

if __name__ == "__main__":
    run_evaluation_pipeline()