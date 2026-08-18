# services/reranker.py
from sentence_transformers import CrossEncoder
from models.data_chunk import DataChunk, RerankedChunk
import asyncio

class RerankerService:
    """Reorders a candidate set of retrieved chunks by query-relevance using a cross-encoder."""

    def __init__(self, model_id: str = "BAAI/bge-reranker-base", device: str = "cpu"):
        self.model = CrossEncoder(model_id, device=device, max_length=512)

    async def rerank(
        self,
        query: str,
        candidates: list[DataChunk],
        top_k: int = 5,
    ) -> list[RerankedChunk]:
        """Scores each candidate against the query and returns the top_k, re-sorted.

        Args:
            query: The user's search query.
            candidates: Documents to rerank, typically an over-fetched set
                (e.g. top-30) from a first-stage dense vector search.
            top_k: Number of top-scoring documents to return.

        Returns:
            Up to top_k RerankedChunk instances, sorted by cross-encoder
            score descending (higher = more relevant). The returned score
            replaces each document's original retrieval score. Returns []
            if candidates is empty.
        """        
        if not candidates:
            return []

        pairs = [(query, c.chunk_text) for c in candidates]

        scores = await asyncio.to_thread(self.model.predict, pairs)  # higher = more relevant

        reranked = sorted(
            zip(candidates, scores), key=lambda pair: pair[1], reverse=True
        )
        return [
            RerankedChunk(text=doc.chunk_text, score=float(score), metadata=doc.chunk_metadata)
            for doc, score in reranked[:top_k]
        ]