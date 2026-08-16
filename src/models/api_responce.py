
from pydantic import BaseModel, Field
from typing import Generic, TypeVar, Literal
from datetime import datetime
from models.enums.ResponceStatusEnum import ResponseStatus


T = TypeVar("T")

class APIResponce(BaseModel, Generic[T]):
    status_code: int
    status: str
    data: T | None = None
    error: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    trace_id: str | None = None
