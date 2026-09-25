from datetime import datetime

from pydantic import BaseModel


class MessageItemDTO(BaseModel):
    message_id: str
    role: str
    content: str
    citations: list[dict] = []
    created_at: datetime


class ChatHistoryResponse(BaseModel):
    chat_id: str
    title: str
    total_messages: int
    page: int
    page_size: int
    has_more: bool
    messages: list[MessageItemDTO]


class ChatSummaryDTO(BaseModel):
    chat_id: str
    title: str
    updated_at: datetime
    is_archived: bool
