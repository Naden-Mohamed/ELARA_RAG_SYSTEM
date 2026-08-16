from pydantic import BaseModel, Field, validator
from typing import Optional
from bson.objectid import ObjectId
from datetime import datetime
from fastapi import UploadFile

class Document(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id") 
    document: UploadFile
    doc_name: str | None = None
    doc_path:str | None = None
    doc_type: str| None = None
    num_pages: int | None = None
    doc_metadata: dict  = {}
    # doc_pushed_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        arbitrary_types_allowed = True
