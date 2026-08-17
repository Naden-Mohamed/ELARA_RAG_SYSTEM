from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from functools import lru_cache


class Settings(BaseSettings):
    APP_NAME: str = "ELARA"

    # MongoDB Atlas
    MONGODB_URI: str = "mongodb+srv://Naden_Mohamed:nadonano123@cluster0.bvpgugf.mongodb.net/?appName=Cluster0"
    MONGODB_DB_NAME: str = "ai_projects"
    COLLECTION_NAME: str = "ELARA"

    QDRANT_API_KEY: str = ""
    QDRANT_URL: str =""

    FILE_ALLOWED_TYPES: List[str] = ["text/plain", "application/pdf",".docx"]
    FILE_MAX_SIZE_MB: int = 10
    FILE_DEFAULT_CHUNK_SIZE: int = 512000 # 512 KB
     
        
    GROQ_API_KEY: str = ""
    GENERATION_MODEL_ID: str = "openai/gpt-oss-120b"
    EMBEDDING_MODEL_ID: str ="BAAI/bge-m3" # High-performance, multilingual, and supports hybrid retrieval.
    EMBEDDING_MODEL_SIZE: int = 1024
    GENERATION_BACKEND: str ="GROQ"
    EMBEDDING_BACKEND: str ="BGE"


    INPUT_DEFAULT_MAX_CHARACTERS: int = 6000
    GENERATION_DEFAULT_MAX_TOKENS: int = 1000
    GENERATION_DEFAULT_TEMPERATURE: float = 0.1
    TOKENIZER_MODEL_ID : str = "sentence-transformers/all-MiniLM-L6-v2"

    class Config:
        env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        env_encoding = "utf-8"

# re-reads and re-parses the .env file on every request that calls it
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()