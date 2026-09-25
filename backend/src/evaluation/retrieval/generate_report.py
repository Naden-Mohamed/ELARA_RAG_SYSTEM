import json
import os

import pandas as pd

METRICS_CSV = "evaluation/retrieval/results/eval_metrics.csv"
SUMMARY_CSV = "evaluation/retrieval/results/eval_summary.csv"
OUTPUT_HTML = "evaluation/retrieval/results/report.html"


def generate_html_dashboard():
    if not os.path.exists(METRICS_CSV) or not os.path.exists(SUMMARY_CSV):
        print("Required CSV files not found. Run run_retrieval_eval.py first.")
        return

    df_summary = pd.read_csv(SUMMARY_CSV)
    df_raw = pd.read_csv(METRICS_CSV)

    if df_raw.empty or "run_id" not in df_raw.columns:
        print(
            "Error: eval_metrics.csv is empty or missing 'run_id'. Run evaluation first."
        )
        return

    latest_run_id = df_raw["run_id"].iloc[-1]

    df_latest = df_raw[df_raw["run_id"] == latest_run_id]

    # 1. Summary Rows HTML
    summary_rows = ""
    for _, row in df_summary.iterrows():
        summary_rows += f"""
        <tr>
            <td><code>{row["chunk_config"]}</code></td>
            <td><code>{row["embedding_model"]}</code></td>
            <td><span class="badge-blue">Top-{row["top_k"]}</span></td>
            <td><strong>{(row["avg_precision"] * 100):.2f}%</strong></td>
            <td><strong>{row["mean_reciprocal_rank"]:.4f}</strong></td>
            <td>{row["total_queries"]}</td>
            <td>{row["total_relevant_retrieved"]}</td>
        </tr>
        """

    # 2. Detailed Cases Rows HTML
    detail_rows = ""
    for _, row in df_latest.iterrows():
        chunks = json.loads(row["retrieved_chunks_json"])

        chunks_html = ""
        for c in chunks:
            badge = (
                '<span class="tag-success">RELEVANT</span>'
                if c["is_relevant"]
                else '<span class="tag-fail">NON-RELEVANT</span>'
            )
            chunks_html += f"""
            <div class="chunk-box">
                <div class="chunk-header">
                    <span><strong>Rank #{c["rank"]}</strong> (Cosine Score: {c["score"]})</span>
                    {badge}
                </div>
                <div class="chunk-meta">
                    <strong>Doc:</strong> {c["document"]} | <strong>Page:</strong> {c["page"]} | <strong>Section:</strong> {c["section"]}
                </div>
                <div class="chunk-id">ID: {c["chunk_id"]}</div>
                <div class="chunk-text">"{c["text_preview"]}"</div>
            </div>
            """

        status_badge = (
            '<span class="badge-purple">Negative/Failure Case</span>'
            if row["is_failure_case"]
            else '<span class="badge-blue">Standard Query</span>'
        )

        detail_rows += f"""
        <tr>
            <td><strong>{row["query_id"]}</strong><br>{status_badge}</td>
            <td>
                <strong>{row["query"]}</strong>
                {f"<div class='failure-tag'>Mode: {row['failure_mode']}</div>" if row["is_failure_case"] else ""}
            </td>
            <td><code>{row["chunk_config"]}</code><br><span class="badge-blue">Top-{row["top_k"]}</span></td>
            <td>
                <strong>P@{row["top_k"]}:</strong> {(row["precision_at_k"] * 100):.1f}%<br>
                <strong>MRR:</strong> {row["reciprocal_rank"]}
            </td>
            <td>{chunks_html}</td>
        </tr>
        """

    html_page = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ELARA RAG — Retrieval Benchmark Dashboard</title>
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0b0f19; color: #e2e8f0; padding: 30px; }}
            h1 {{ color: #38bdf8; font-size: 24px; margin-bottom: 8px; }}
            h2 {{ color: #94a3b8; font-size: 18px; margin: 30px 0 12px 0; border-left: 4px solid #38bdf8; padding-left: 10px; }}
            .subtitle {{ color: #64748b; font-size: 14px; margin-bottom: 24px; }}

            table {{ width: 100%; border-collapse: separate; border-spacing: 0; background: #131b2e; border-radius: 8px; overflow: hidden; margin-bottom: 30px; border: 1px solid #1e293b; }}
            th, td {{ padding: 12px 16px; border-bottom: 1px solid #1e293b; text-align: left; vertical-align: top; font-size: 13px; }}
            th {{ background: #0f172a; color: #94a3b8; font-weight: 600; font-size: 12px; text-transform: uppercase; }}

            .badge-blue {{ background: #0369a1; color: #f0f9ff; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
            .badge-purple {{ background: #6b21a8; color: #faf5ff; padding: 2px 8px; border-radius: 4px; font-size: 11px; }}
            .tag-success {{ background: #065f46; color: #d1fae5; padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: bold; }}
            .tag-fail {{ background: #7f1d1d; color: #fee2e2; padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: bold; }}

            .chunk-box {{ background: #0b0f19; border: 1px solid #1e293b; border-radius: 6px; padding: 10px; margin-bottom: 8px; }}
            .chunk-header {{ display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 12px; }}
            .chunk-meta {{ font-size: 11px; color: #38bdf8; margin-bottom: 2px; }}
            .chunk-id {{ font-size: 10px; color: #64748b; font-family: monospace; }}
            .chunk-text {{ font-size: 12px; color: #cbd5e1; margin-top: 5px; font-style: italic; line-height: 1.4; }}
            .failure-tag {{ color: #f87171; font-size: 11px; margin-top: 4px; font-weight: 500; }}
            code {{ font-family: Consolas, monospace; color: #a5f3fc; background: #082f49; padding: 2px 5px; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <h1>ELARA Clinical RAG — Retrieval Evaluation Benchmark</h1>
        <div class="subtitle">Comparison of Top-K configurations, Chunking Strategies, Precision@K, and MRR.</div>

        <h2>Aggregated Metrics Grid (Combinations & Averages)</h2>
        <table>
            <thead>
                <tr>
                    <th>Chunking Strategy</th>
                    <th>Embedding Model</th>
                    <th>Top-K</th>
                    <th>Precision@K</th>
                    <th>MRR</th>
                    <th>Evaluated Cases</th>
                    <th>Total Relevant Retrieved</th>
                </tr>
            </thead>
            <tbody>{summary_rows}</tbody>
        </table>

        <h2>Detailed Runs & Individual Ground-Truth Matches</h2>
        <table>
            <thead>
                <tr>
                    <th style="width: 8%;">ID</th>
                    <th style="width: 25%;">Evaluation Query</th>
                    <th style="width: 15%;">Parameters</th>
                    <th style="width: 12%;">Performance</th>
                    <th style="width: 40%;">Retrieved Chunks & Citations (Doc, Page, Section)</th>
                </tr>
            </thead>
            <tbody>{detail_rows}</tbody>
        </table>
    </body>
    </html>
    """

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_page)
    print("Report generated successfully at:", OUTPUT_HTML)


if __name__ == "__main__":
    generate_html_dashboard()
