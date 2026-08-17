from pydantic import BaseModel

class PushRequest(BaseModel):
    do_reset: int = 0

class SearchRequest(BaseModel):
    text: str
    limit: int = 5