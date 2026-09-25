from datetime import datetime

from pydantic import BaseModel, Field


class APIResponce[T](BaseModel):
    status_code: int
    status: str
    data: T | None = None
    error: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    trace_id: str | None = None
