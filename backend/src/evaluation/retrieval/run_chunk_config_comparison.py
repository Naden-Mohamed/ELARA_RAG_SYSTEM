"""
src/evaluation/retrieval/run_chunk_config_comparison.py

Compares two chunking configurations end-to-end: chunk -> embed -> push to an
isolated Qdrant collection -> retrieve -> label relevance -> precision/recall/MRR.

IMPORTANT -- read before running:
DocumentParserService.get_chunks(chunk_size, chunk_overlap) accepts a
chunk_overlap parameter but never uses it (services/data_service.py --
HybridChunker has no overlap mechanism; it chunks by document structure, not
a sliding token window). So "chunk_size=512/overlap=50" vs
"chunk_size=512/overlap=100" is a no-op comparison against production code
today -- both configs produce byte-identical chunks.

This script does two things about that:
  1. Compares chunk_size honestly (that parameter IS real and does change
     chunking) across CHUNK_SIZE_CONFIGS.
  2. Implements a real overlap as a post-processing step (`apply_overlap`)
     so "does overlap actually help on this corpus" is answerable at all.
     If the answer is yes, the fix is to port `apply_overlap` (or an
     equivalent sliding-window pass) into get_chunks() for real; if no,
     you've saved yourself from shipping a parameter that does nothing.

Usage (needs the full ML deps -- docling, sentence-transformers, torch):
    uv sync
    uv run python src/evaluation/retrieval/run_chunk_config_comparison.py \\
        --pdf data/your_source.pdf

This does NOT touch your running app's Qdrant collection -- it creates and
tears down its own temporary collections (elara_eval_chunkcfg_*).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATASET_PATH, label_relevance, load_evaluation_cases

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
TOP_K_VALUES = [3, 5, 10]


@dataclass
class ChunkConfig:
    name: str
    chunk_size: int
    overlap_tokens: int  # tokens of adjacent-chunk context stitched in; 0 = none


CHUNK_CONFIGS = [
    ChunkConfig(name="size256_no_overlap", chunk_size=256, overlap_tokens=0),
    ChunkConfig(name="size512_no_overlap", chunk_size=512, overlap_tokens=0),
    ChunkConfig(name="size512_overlap64", chunk_size=512, overlap_tokens=64),
]


def apply_overlap(
    chunks: list[dict[str, Any]], overlap_tokens: int, tokenizer
) -> list[dict[str, Any]]:
    """Real sliding-window overlap, applied after HybridChunker's structural
    chunking: prepends the tail of the previous chunk and appends the head of
    the next chunk to each chunk's embedded text. This is what get_chunks()'s
    chunk_overlap parameter should be doing and currently is not."""
    if overlap_tokens <= 0 or len(chunks) < 2:
        return chunks

    def tail_tokens(text: str, n: int) -> str:
        ids = tokenizer.tokenizer(text)["input_ids"]
        if len(ids) <= n:
            return text
        return tokenizer.tokenizer.decode(ids[-n:])

    def head_tokens(text: str, n: int) -> str:
        ids = tokenizer.tokenizer(text)["input_ids"]
        if len(ids) <= n:
            return text
        return tokenizer.tokenizer.decode(ids[:n])

    out = []
    for i, chunk in enumerate(chunks):
        prefix = tail_tokens(chunks[i - 1]["raw_text"], overlap_tokens) if i > 0 else ""
        suffix = (
            head_tokens(chunks[i + 1]["raw_text"], overlap_tokens)
            if i < len(chunks) - 1
            else ""
        )
        stitched = f"{prefix} {chunk['text']} {suffix}".strip()
        new_chunk = dict(chunk)
        new_chunk["text"] = stitched
        new_chunk["metadata"] = {**chunk["metadata"], "overlap_tokens": overlap_tokens}
        out.append(new_chunk)
    return out


async def build_and_index(
    pdf_path: Path, config: ChunkConfig, collection_suffix: str
) -> tuple[str, int]:
    """Chunks pdf_path with `config`, embeds, and pushes into a fresh, isolated
    Qdrant collection. Returns (collection_name, chunk_count)."""
    from core.config import get_settings
    from db.qdrant_vectordb import Qdrant
    from services.data_service import DocumentParserService
    from services.embedding import EmbeddingService

    settings = get_settings()
    parser = DocumentParserService()

    document = parser.get_file_content(str(pdf_path))
    if document is None:
        raise RuntimeError(f"Docling could not parse {pdf_path}")

    base_chunks = parser.get_chunks(
        document, chunk_size=config.chunk_size, chunk_overlap=0
    )
    if not base_chunks:
        raise RuntimeError(f"No chunks produced for config {config.name}")

    # Real overlap post-processing (see apply_overlap docstring).
    from docling_core.transforms.chunker.tokenizer.huggingface import (
        HuggingFaceTokenizer,
    )
    from transformers import AutoTokenizer

    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(settings.TOKENIZER_MODEL_ID),
        max_tokens=config.chunk_size,
    )
    chunks = apply_overlap(base_chunks, config.overlap_tokens, tokenizer)

    embedder = EmbeddingService()
    embedder.set_embedding_model(
        settings.BGE_EMBEDDING_MODEL_ID, settings.BGE_EMBEDDING_MODEL_SIZE
    )

    texts = [c["text"] for c in chunks]
    vectors = embedder.embed_text(texts, document_type="document")
    if vectors is None:
        raise RuntimeError("Embedding failed.")

    collection_name = f"elara_eval_chunkcfg_{collection_suffix}_{uuid.uuid4().hex[:8]}"

    qdrant = Qdrant()
    await qdrant.connect()
    await qdrant.create_collection(
        collection_name, embedding_size=embedder.embedding_size, do_reset=1
    )

    metadatas = [
        {**c["metadata"], "text": c["text"], "original_filename": pdf_path.name}
        for c in chunks
    ]
    await qdrant.insert_many(
        collection_name,
        texts=texts,
        vectors=[v.tolist() if hasattr(v, "tolist") else v for v in vectors],
        metadatas=metadatas,
    )

    return collection_name, len(chunks), qdrant, embedder


async def search_collection(
    qdrant, embedder, collection_name: str, query: str, limit: int
) -> list[dict[str, Any]]:
    query_vec = embedder.embed_text(query, document_type="query")
    if query_vec is None:
        return []
    vec = query_vec[0]
    vec = vec.tolist() if hasattr(vec, "tolist") else vec
    results = await qdrant.search_by_vector(collection_name, vec, limit)
    if not results or not getattr(results, "points", None):
        return []
    return [
        {"id": str(p.id), "score": float(p.score), "payload": p.payload or {}}
        for p in results.points
    ]


def metrics_at_k(
    points: list[dict[str, Any]], case: dict[str, Any], k: int
) -> dict[str, float]:
    top_k = points[:k]
    labels = [label_relevance(p, case) for p in top_k]
    relevant_ranks = [i + 1 for i, l in enumerate(labels) if l["relevant"]]
    return {
        "precision_at_k": len(relevant_ranks) / k if k else 0.0,
        "recall_at_k": 1.0 if relevant_ranks else 0.0,
        "mrr_at_k": 1.0 / relevant_ranks[0] if relevant_ranks else 0.0,
        "hit_at_k": int(bool(relevant_ranks)),
    }


async def run_comparison(pdf_path: Path) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cases = [
        c
        for c in load_evaluation_cases(DATASET_PATH)
        if c.get("expected_status") == "answered"
    ]

    from db.qdrant_vectordb import Qdrant

    metric_rows = []
    chunk_label_rows = []
    collections_to_clean: list[tuple[Qdrant, str]] = []

    try:
        for config in CHUNK_CONFIGS:
            print(
                f"\n{'=' * 70}\nIndexing with config: {config.name} (size={config.chunk_size}, overlap={config.overlap_tokens})\n{'=' * 70}"
            )
            collection_name, chunk_count, qdrant, embedder = await build_and_index(
                pdf_path, config, config.name
            )
            collections_to_clean.append((qdrant, collection_name))
            print(f"  {chunk_count} chunks indexed into {collection_name}")

            for case in cases:
                points = await search_collection(
                    qdrant, embedder, collection_name, case["query"], max(TOP_K_VALUES)
                )

                for k in TOP_K_VALUES:
                    m = metrics_at_k(points, case, k)
                    metric_rows.append(
                        {
                            "config": config.name,
                            "chunk_size": config.chunk_size,
                            "overlap_tokens": config.overlap_tokens,
                            "id": case["id"],
                            "top_k": k,
                            **m,
                        }
                    )

                labels = [label_relevance(p, case) for p in points]
                for rank, (point, label) in enumerate(zip(points, labels), start=1):
                    chunk_label_rows.append(
                        {
                            "config": config.name,
                            "id": case["id"],
                            "rank": rank,
                            "chunk_id": point["id"],
                            "score": point["score"],
                            "relevant": label["relevant"],
                            "reason": label["reason"],
                            "text_preview": point["payload"].get("text", "")[:200],
                        }
                    )
    finally:
        for qdrant, collection_name in collections_to_clean:
            try:
                await qdrant.delete_collection(collection_name)
                print(f"Cleaned up temporary collection {collection_name}")
            except Exception as exc:
                print(f"WARNING: could not delete {collection_name}: {exc}")

    metrics_df = pd.DataFrame(metric_rows)
    chunks_df = pd.DataFrame(chunk_label_rows)

    metrics_path = RESULTS_DIR / "chunk_config_comparison_cases.csv"
    chunks_path = RESULTS_DIR / "chunk_config_chunk_labels.csv"
    metrics_df.to_csv(metrics_path, index=False)
    chunks_df.to_csv(chunks_path, index=False)

    summary = (
        metrics_df.groupby(["config", "top_k"])[
            ["precision_at_k", "recall_at_k", "mrr_at_k", "hit_at_k"]
        ]
        .mean()
        .reset_index()
    )
    summary_path = RESULTS_DIR / "chunk_config_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\n{'=' * 70}\nCHUNK CONFIG COMPARISON SUMMARY\n{'=' * 70}")
    print(summary.to_string(index=False))
    print(f"\nPer-case metrics: {metrics_path}")
    print(f"Per-chunk relevance labels: {chunks_path}")
    print(f"Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare chunking configs end-to-end.")
    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to the source PDF to chunk (e.g. data/who_guide.pdf)",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    asyncio.run(run_comparison(pdf_path))


if __name__ == "__main__":
    main()
