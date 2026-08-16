from pydantic import BaseModel

class ParsedDocument(BaseModel):
    text: str = ""
    tables : str = ""
    metadata: dict[str,str] = {}