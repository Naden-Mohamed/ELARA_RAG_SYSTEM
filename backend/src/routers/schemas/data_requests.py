from pydantic import BaseModel


class PushRequest(BaseModel):
    document_id: str
    do_reset: int = 0


class SearchRequest(BaseModel):
    text: str
    limit: int = 5
