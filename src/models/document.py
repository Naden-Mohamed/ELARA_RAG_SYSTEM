from pydantic import BaseModel, Field
from typing import Optional
from bson.objectid import ObjectId
from datetime import datetime
from models.enums.DocumentStatusEnum import DocumentStatusEnums

class Document(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    doc_name: str | None = None
    doc_path: str | None = None
    doc_type: str | None = None
    doc_size: int | None = None
    doc_metadata: dict = {}
    status: str = DocumentStatusEnums.PENDING.value
    chunk_count: int | None = None
    error_message: str | None = None
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None

    class Config:
        arbitrary_types_allowed = True
