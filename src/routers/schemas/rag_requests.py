from enum import Enum

from pydantic import BaseModel


class UserPersonaEnum(str, Enum):
    DOCTOR = "doctor"
    MOTHER = "mother"


class LanguageEnum(str, Enum):
    AR = "ar"
    EN = "en"


class MockChunkInput(BaseModel):
    chunk_id: str
    doc_name: str
    page_number: int | str
    section: str
    text: str
    score: float


class QueryRequest(BaseModel):
    query: str
    persona: UserPersonaEnum = UserPersonaEnum.MOTHER
    language: LanguageEnum = LanguageEnum.EN
    top_k: int = 5


class RetrievedChunkDTO(BaseModel):
    chunk_id: str
    doc_name: str
    text: str
    score: float
    page_numbers: list[int] = []
    section_headings: list[str] = []


class RAGResponseData(BaseModel):
    answer: str
    persona_applied: str
    retrieved_chunks: list[RetrievedChunkDTO]
    latency_seconds: float


class LLMTestResponse(BaseModel):
    answer: str
    persona_applied: str
    language_applied: str
    latency_seconds: float
    citations_detected: list[str] = []
