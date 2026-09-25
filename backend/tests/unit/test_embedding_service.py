import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from src.models.enums.LLMEnums import DocumentTypeEnum
from src.services.embedding import EmbeddingService


def _mock_sentence_transformer(dim: int):
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = dim
    mock_model.encode.side_effect = lambda texts, normalize_embeddings=True: np.array(
        [[0.1] * dim for _ in texts]
    )
    return mock_model


class TestEmbeddingDimensionValidation:
    def test_actual_model_dimension_is_used(self):
        service = EmbeddingService()
        with patch(
            "services.embedding.SentenceTransformer",
            return_value=_mock_sentence_transformer(384),
        ):
            service.set_embedding_model(
                model_id="some/minilm-model", embedding_size=1024
            )
        assert service.embedding_size == 384  # not the wrong configured 1024

    def test_model_load_failure_leaves_client_none(self):
        service = EmbeddingService()
        with patch(
            "services.embedding.SentenceTransformer", side_effect=RuntimeError("boom")
        ):
            service.set_embedding_model(model_id="broken/model", embedding_size=384)
        assert service.client is None
        assert service.embed_text("hello") is None


class TestInstructionPrefixing:
    """Only BGE-family models should get instruction-style prefixes."""

    def test_bge_model_gets_query_instruction(self):
        service = EmbeddingService()
        mock_model = _mock_sentence_transformer(1024)
        with patch("services.embedding.SentenceTransformer", return_value=mock_model):
            service.set_embedding_model(
                model_id="BAAI/bge-base-en", embedding_size=1024
            )

        service.embed_text(
            "pregnancy symptoms", document_type=DocumentTypeEnum.QUERY.value
        )
        called_texts = mock_model.encode.call_args.args[0]
        assert called_texts[0].startswith(
            "Represent this query for searching relevant passages:"
        )

    def test_non_bge_model_gets_no_instruction_prefix(self):
        service = EmbeddingService()
        mock_model = _mock_sentence_transformer(384)
        with patch("services.embedding.SentenceTransformer", return_value=mock_model):
            service.set_embedding_model(
                model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                embedding_size=384,
            )

        service.embed_text(
            "pregnancy symptoms", document_type=DocumentTypeEnum.QUERY.value
        )
        called_texts = mock_model.encode.call_args.args[0]
        assert called_texts[0] == "pregnancy symptoms"  # no prefix added


class TestEmbedTextBehavior:
    def test_returns_none_when_model_not_loaded(self):
        service = EmbeddingService()
        assert service.embed_text("hello") is None

    def test_long_text_is_truncated(self):
        service = EmbeddingService(default_input_max_characters=10)
        mock_model = _mock_sentence_transformer(384)
        with patch("services.embedding.SentenceTransformer", return_value=mock_model):
            service.set_embedding_model(model_id="test/model", embedding_size=384)
        service.embed_text("this text is way longer than ten characters")
        called_texts = mock_model.encode.call_args.args[0]
        assert len(called_texts[0]) <= 10
