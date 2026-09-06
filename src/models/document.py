from datetime import datetime

from bson.objectid import ObjectId
from pydantic import BaseModel, Field

from models.enums.DocumentStatusEnum import DocumentStatusEnums


class Document(BaseModel):
    id: ObjectId | None = Field(None, alias="_id")
    doc_name: str | None = None
    doc_path: str | None = None
    doc_type: str | None = None
    doc_size: float | None = None
    doc_metadata: dict = {}
    status: str = DocumentStatusEnums.PENDING.value
    chunk_count: int | None = None
    error_message: str | None = None
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: datetime | None = None

    class Config:
        arbitrary_types_allowed = True
