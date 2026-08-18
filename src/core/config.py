from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / ".env"

class Settings(BaseSettings):
    APP_NAME: str = "ELARA"

    # MongoDB Atlas
    MONGODB_URI: str
    MONGODB_DB_NAME: str = "ai_projects"
    COLLECTION_NAME: str = "ELARA"

    # Qdrant Vector DB
    QDRANT_API_KEY: str = ""
    QDRANT_URL: str = ""
    DENSE_VECTOR_NAME: str = "dense_vector"
    SPARSE_VECTOR_NAME: str = "sparse_vector"

    # File & Chunking
    FILE_ALLOWED_TYPES: List[str] = ["text/plain", "application/pdf", ".docx"]
    FILE_MAX_SIZE_MB: int = 10
    FILE_DEFAULT_CHUNK_SIZE: int = 512000  # 512 KB
    USE_SIMPLE_CHUNKER: bool = False

        
    # LLM & Embedding Models
    GROQ_API_KEY: str = ""
    GENERATION_MODEL_ID: str = "openai/gpt-oss-120b"
    BGE_EMBEDDING_MODEL_ID="BAAI/bge-m3"
    BGE_EMBEDDING_MODEL_SIZE=1024
    MiniLM_EMBEDDING_MODEL_ID="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" 
    MiniLM_EMBEDDING_MODEL_SIZE=384
    GENERATION_BACKEND: str = "GROQ"
    EMBEDDING_BACKEND: str = "BGE"

    INPUT_DEFAULT_MAX_CHARACTERS: int = 6000
    GENERATION_DEFAULT_MAX_TOKENS: int = 1000
    GENERATION_DEFAULT_TEMPERATURE: float = 0.1
    TOKENIZER_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"

    CROSS_ENCODER_RERANKER: str = "BAAI/bge-reranker-base"
    # JWT Security
    JWT_SECRET_KEY: str = "super_secret_elara_key_change_in_prod"

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore"
    )

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()