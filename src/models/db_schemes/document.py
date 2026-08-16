from pydantic import BaseModel, Field, validator
from typing import Optional
from bson.objectid import ObjectId
from datetime import datetime

class Document(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id") 
    doc_name: str | None = None
    doc_path:str | None = None
    doc_type: str| None = None
    doc_size: int | None = None
    doc_metadata: dict  = {}
    # doc_pushed_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        arbitrary_types_allowed = True
