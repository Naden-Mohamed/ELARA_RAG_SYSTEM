from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from db.chunk_model import DataChunk

class UserPersonaEnum(str, Enum):
    DOCTOR = "doctor"
    MOTHER = "mother"
    GENERAL = "general"

class LanguageEnum(str, Enum):
    AR = "ar"
    EN = "en"

class MockChunkInput(BaseModel):
    chunk_id: str
    doc_name: str
    page_number: int | str
    section: str
    text: str
    score:float

class QueryRequest(BaseModel):
    query: str
    persona: UserPersonaEnum = UserPersonaEnum.GENERAL
    language: LanguageEnum = LanguageEnum.EN
    top_k: int = 5

class DirectPromptTestRequest(BaseModel):
    query: str
    persona: UserPersonaEnum = UserPersonaEnum.GENERAL
    language: LanguageEnum = LanguageEnum.EN
    context_chunks: List[MockChunkInput] = []

class RetrievedChunkDTO(BaseModel):
    chunk_id: str
    doc_name: str
    text: str
    score: float
    page_numbers: List[int] = []
    section_headings: List[str] = []

class RAGResponseData(BaseModel):
    answer: str
    persona_applied: str
    retrieved_chunks: List[RetrievedChunkDTO]
    latency_seconds: float

class LLMTestResponse(BaseModel):
    answer: str
    persona_applied: str
    language_applied: str
    latency_seconds: float
    citations_detected: List[str] = []