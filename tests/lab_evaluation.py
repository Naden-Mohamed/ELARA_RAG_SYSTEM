import asyncio
import httpx
import json
import os
import datetime
from pathlib import Path

API_URL = "http://127.0.0.1:8000/rag/query"
DATASET_PATH = Path(__file__).parent / "medical_benchmark_dataset.json"
HISTORY_FILE = Path(__file__).parent / "lab_evaluation_history.json"
HTML_REPORT_FILE = Path(__file__).parent / "lab_report.html"

FAILURE_MODES = {
    "1": "Semantic Gap (Vocabulary mismatch between query & text)",
    "2": "Context Fragmentation (Information split across chunk boundary)",
    "3": "Irrelevant High-Score Noise (Generic boilerplate text outranking answer)",
    "4": "Missing Context in Vector Store (Information not indexed / parsing error)",
    "5": "No Failure / Acceptable Retrieval"
}


def load_dataset():
    if not os.path.exists(DATASET_PATH):
        print(f"[!] Error: Dataset file not found at {DATASET_PATH}")
        return []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_history_record(record: dict):
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []
    
    history.append(record)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Successfully saved experiment data to: {HISTORY_FILE.name}")


def generate_html_report():
    if not os.path.exists(HISTORY_FILE):
        return

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    experiment_rows = ""
    for exp in history:
        p3 = f"{exp['metrics']['avg_precision_at_3']:.2%}" if exp['metrics'].get('avg_precision_at_3') is not None else "N/A"
        p5 = f"{exp['metrics']['avg_precision_at_5']:.2%}" if exp['metrics'].get('avg_precision_at_5') is not None else "N/A"
        p10 = f"{exp['metrics']['avg_precision_at_10']:.2%}" if exp['metrics'].get('avg_precision_at_10') is not None else "N/A"
        
        failure_summary = "; ".join(exp.get("failure_notes", [])) or "None Recorded"

        experiment_rows += f"""
        <tr>
            <td>
                <strong>{exp['experiment_name']}</strong><br>
                <span class="meta-date">{exp['timestamp']}</span>
            </td>
            <td><code>{exp['parameters']['chunk_size']} / {exp['parameters']['chunk_overlap']}</code></td>
            <td>{exp['parameters']['embedding_model']}</td>
            <td>{exp['parameters']['retrieval_strategy']}</td>
            <td><strong>Top-{exp['parameters']['top_k']}</strong></td>
            <td><span class="metric-badge">{p3}</span></td>
            <td><span class="metric-badge">{p5}</span></td>
            <td><span class="metric-badge">{p10}</span></td>
            <td>{exp['metrics']['avg_latency']}s</td>
            <td><span class="failure-text">{failure_summary}</span></td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ELARA RAG System - Experiment Evaluation Dashboard</title>
    <style>
        :root {{
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --primary: #0284c7;
            --text-dark: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --badge-bg: #e0f2fe;
            --badge-text: #0369a1;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-dark);
            margin: 0;
            padding: 30px;
        }}
        .header {{
            margin-bottom: 25px;
        }}
        .header h1 {{
            margin: 0 0 5px 0;
            font-size: 26px;
            color: var(--text-dark);
        }}
        .header p {{
            margin: 0;
            color: var(--text-muted);
            font-size: 15px;
        }}
        .table-container {{
            background: var(--card-bg);
            border-radius: 10px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
            border: 1px solid var(--border);
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 13.5px;
        }}
        th {{
            background-color: #f1f5f9;
            color: #334155;
            padding: 12px 16px;
            font-weight: 600;
            border-bottom: 1px solid var(--border);
            text-transform: uppercase;
            font-size: 11.5px;
            letter-spacing: 0.5px;
        }}
        td {{
            padding: 14px 16px;
            border-bottom: 1px solid var(--border);
            vertical-align: top;
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
        tr:hover {{
            background-color: #f8fafc;
        }}
        code {{
            background: #f1f5f9;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
            color: #0f172a;
        }}
        .meta-date {{
            color: var(--text-muted);
            font-size: 11.5px;
        }}
        .metric-badge {{
            display: inline-block;
            background: var(--badge-bg);
            color: var(--badge-text);
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: 600;
        }}
        .failure-text {{
            font-size: 12px;
            color: #b91c1c;
            line-height: 1.4;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ELARA RAG System - Lab Benchmark Leaderboard</h1>
        <p>Continuous Evaluation Tracking for Chunk Configurations, Retrieval Models & Top-K Sweeps</p>
    </div>
    
    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>Experiment Run</th>
                    <th>Chunk Config (Size/Overlap)</th>
                    <th>Embedding Model</th>
                    <th>Strategy</th>
                    <th>Top-K</th>
                    <th>Precision@3</th>
                    <th>Precision@5</th>
                    <th>Precision@10</th>
                    <th>Avg Latency</th>
                    <th>Failure Mode Documentation</th>
                </tr>
            </thead>
            <tbody>
                {experiment_rows}
            </tbody>
        </table>
    </div>
</body>
</html>"""

    with open(HTML_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[+] Updated visual dashboard: {HTML_REPORT_FILE.name}")


async def interactive_evaluation():
    print("\n" + "=" * 76)
    print("        ELARA LAB BENCHMARK & RETRIEVAL EVALUATION WORKFLOW")
    print("=" * 76)

    dataset = load_dataset()
    if not dataset:
        return

    # Configuration Prompts
    print("\n[STEP 1/3] Configure Current Experiment Parameters:")
    exp_name = input("  > Experiment / Run Name (e.g., Config_A_BGE_512): ").strip() or "Run_" + datetime.datetime.now().strftime("%H%M%S")
    chunk_size = input("  > Chunk Size used in ingestion (e.g., 512, 256): ").strip() or "512"
    chunk_overlap = input("  > Chunk Overlap used (e.g., 50, 30): ").strip() or "50"
    emb_model = input("  > Embedding Model (e.g., sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2): ").strip() or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    strategy = input("  > Retrieval Strategy [Dense / Hybrid / Re-ranked]: ").strip() or "Dense"
    
    top_k_input = input("  > Top-K to retrieve per query (3 / 5 / 10) [Default: 5]: ").strip() or "5"
    top_k = int(top_k_input)

    print("\n[STEP 2/3] Executing Retrieval & Ground Truth Evaluation...")
    print(f"Loaded {len(dataset)} verified benchmark questions from dataset.\n")

    eval_records = []
    latencies = []
    p3_scores = []
    p5_scores = []
    p10_scores = []
    failure_notes = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        for idx, item in enumerate(dataset, start=1):
            print("\n" + "#" * 76)
            print(f" QUESTION {idx}/{len(dataset)} | ID: {item['id']} | PERSONA: {item['persona'].upper()}")
            print("#" * 76)
            print(f" Query       : \"{item['query']}\"")
            print(f" Target Sec  : {item['target_section']}")
            print(f" Ground Truth: {item['ground_truth']}")
            print(f" Reference   : {item['source']}")
            print("-" * 76)

            payload = {
                "query": item["query"],
                "persona": item["persona"],
                "top_k": top_k
            }

            try:
                resp = await client.post(API_URL, json=payload)
                if resp.status_code != 200:
                    print(f"[!] Request failed with status {resp.status_code}: {resp.text}")
                    continue

                response_data = resp.json().get("data", {})
                retrieved_chunks = response_data.get("retrieved_chunks", [])
                latency = response_data.get("latency_seconds", 0.0)
                latencies.append(latency)
            except Exception as e:
                print(f"[!] Connection failed: {e}")
                continue

            print(f"\nRetrieved {len(retrieved_chunks)} chunks (Latency: {latency}s).")
            print("Please label each retrieved chunk against the Ground Truth:\n")

            labels = []
            for c_idx, chunk in enumerate(retrieved_chunks, start=1):
                print(f"  [Chunk #{c_idx}] Similarity Score: {chunk.get('score')} | ID: {chunk.get('chunk_id')}")
                print(f"  Pages: {chunk.get('page_numbers')} | Headings: {chunk.get('section_headings')}")
                print(f"  Snippet: \"{chunk.get('text', '')[:200]}...\"")

                while True:
                    verdict = input("  >> Relevant? (1 = Relevant, 0 = Irrelevant): ").strip()
                    if verdict in ["0", "1"]:
                        labels.append(int(verdict))
                        break
                    print("     Invalid input. Please enter 1 or 0.")

            # Metrics for this question
            k_len = len(labels)
            p_at_3 = sum(labels[:3]) / min(3, k_len) if k_len > 0 else 0
            p_at_5 = sum(labels[:5]) / min(5, k_len) if k_len > 0 else 0
            p_at_10 = sum(labels[:10]) / min(10, k_len) if k_len > 0 else 0

            p3_scores.append(p_at_3)
            p5_scores.append(p_at_5)
            p10_scores.append(p_at_10)

            print(f"\n  [Question Summary] -> Precision@3: {p_at_3:.2f} | Precision@5: {p_at_5:.2f} | Precision@10: {p_at_10:.2f}")

            # Failure Mode Diagnostic if performance drops
            if sum(labels) == 0 or p_at_3 < 0.34:
                print("\n  [!] Low relevance or failure detected for this query.")
                print("  Document the Failure Mode:")
                for k, v in FAILURE_MODES.items():
                    print(f"    [{k}] {v}")
                choice = input("  Select Mode (1-5): ").strip()
                selected_mode = FAILURE_MODES.get(choice, "Custom Failure Note")
                failure_notes.append(f"Q{item['id']} ({item['persona']}): {selected_mode}")

            eval_records.append({
                "question_id": item["id"],
                "query": item["query"],
                "persona": item["persona"],
                "ground_truth": item["ground_truth"],
                "chunks_retrieved": len(retrieved_chunks),
                "labels": labels,
                "precision_at_3": p_at_3,
                "precision_at_5": p_at_5,
                "precision_at_10": p_at_10
            })

    # Summary Calculations
    mean_p3 = sum(p3_scores) / len(p3_scores) if p3_scores else 0
    mean_p5 = sum(p5_scores) / len(p5_scores) if p5_scores else 0
    mean_p10 = sum(p10_scores) / len(p10_scores) if p10_scores else 0
    mean_latency = round(sum(latencies) / len(latencies), 3) if latencies else 0

    print("\n" + "=" * 76)
    print("                    FINAL BENCHMARK SUMMARY")
    print("=" * 76)
    print(f" Experiment Name     : {exp_name}")
    print(f" Evaluated Questions : {len(eval_records)}")
    print(f" Mean Precision@3    : {mean_p3:.2%}")
    print(f" Mean Precision@5    : {mean_p5:.2%}")
    print(f" Mean Precision@10   : {mean_p10:.2%}")
    print(f" Average Latency     : {mean_latency} seconds")
    print("=" * 76)

    # Save structured record
    experiment_entry = {
        "experiment_name": exp_name,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": emb_model,
            "retrieval_strategy": strategy,
            "top_k": top_k
        },
        "metrics": {
            "avg_precision_at_3": mean_p3,
            "avg_precision_at_5": mean_p5,
            "avg_precision_at_10": mean_p10,
            "avg_latency": mean_latency
        },
        "failure_notes": failure_notes,
        "questions_detail": eval_records
    }

    save_history_record(experiment_entry)
    generate_html_report()


if __name__ == "__main__":
    asyncio.run(interactive_evaluation())