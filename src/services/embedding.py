from models.enums.LLMEnums import DocumentTypeEnum
from sentence_transformers import SentenceTransformer
import logging
from typing import List , Union, Optional

class EmbeddingService:
    def __init__(self,
                 default_input_max_characters: int = 1000,
                 default_generation_max_output_tokens: int = 1000,
                 default_generation_temperature: float = 0.1):

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.embedding_model_id = None
        self.embedding_size = None
        self.client = None  

        self.logger = logging.getLogger(__name__)

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size
        try:
            self.client = SentenceTransformer(
                model_id,
                trust_remote_code=True  # required for BAAI/bge-multilingual-gemma2
            )
            actual_dim = self.client.get_embedding_dimension()
            if actual_dim != embedding_size:
                self.logger.warning(
                    f"Configured EMBEDDING_MODEL_SIZE={embedding_size} does not match "
                    f"actual model output dim={actual_dim}. Using {actual_dim}."
                )
            self.embedding_size = actual_dim

            # Only BGE-family models were trained with instruction prefixes
            self.is_instruction_tuned = "bge" in model_id.lower()

            self.logger.info(f"Embedding model '{model_id}' loaded (dim={actual_dim}).")

            self.logger.info(f"BGE model '{model_id}' loaded successfully.")

            print(f"BGE model '{model_id}' loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load BGE model '{model_id}': {e}")
            self.client = None

    def process_text(self, text: str):
        if len(text) > self.default_input_max_characters:
            self.logger.warning(
                f"Input text exceeds maximum character limit of {self.default_input_max_characters}. Truncating."
            )
            return text[:self.default_input_max_characters]
        return text


    # Batch Embedding
    def embed_text(self, text: Union[str, List[str]], document_type: str = ""):
        if not self.client:
            self.logger.error("BGE model is not loaded. Call set_embedding_model() first.")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model for BGE was not set.")
            return None
        
        if isinstance(text, str):
            text = [text]

        try:
            text = [self.process_text(t) for t in text]
            print(text)

            # bge-multilingual-gemma2 uses instruction-based embedding
            # document_type differentiates query vs passage for better accuracy
            if document_type == DocumentTypeEnum.QUERY.value:
                instruction = "Represent this query for searching relevant passages: "
            else:
                instruction = "Represent this passage for retrieval: "

            embedding = self.client.encode(
                [instruction + t for t in text],
                normalize_embeddings=True  # recommended for BGE models
            )

            if embedding is None or len(embedding) == 0:
                self.logger.error("BGE embedding returned empty result.")
                return None

            return embedding

        except Exception as e:
            self.logger.error(f"BGE embedding error: {e}")
            raise

