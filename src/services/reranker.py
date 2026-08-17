# services/reranker.py
from sentence_transformers import CrossEncoder
from models.data_chunk import RetrievedDocument

class RerankerService:

    def __init__(self, model_id: str = "BAAI/bge-reranker-base", device: str = "cpu"):
        self.model = CrossEncoder(model_id, device=device, max_length=512)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedDocument],
        top_k: int = 5,
    ) -> list[RetrievedDocument]:
        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(pairs)  # higher = more relevant

        reranked = sorted(
            zip(candidates, scores), key=lambda pair: pair[1], reverse=True
        )
        return [
            RetrievedDocument(text=doc.text, score=float(score), metadata=doc.metadata)
            for doc, score in reranked[:top_k]
        ]