from pydantic import BaseModel, Field
from typing import Optional
from bson.objectid import ObjectId
from datetime import datetime

class DataChunk(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    chunk_text: str = Field(..., min_length=1)
    chunk_metadata: dict
    chunk_order: int = Field(..., gt=0)
    chunk_document_id: ObjectId

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def get_indexes(cls):
        return [
            {
                "key": [
                    ("chunk_document_id", 1)
                ],
                "name": "chunk_doc_id_index_1",
                "unique": False
            }
        ]
    
class RerankedChunk(BaseModel):
    text: str | None = None
    score: float | None = None
    metadata: dict |None = None

