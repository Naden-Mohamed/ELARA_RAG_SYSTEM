from fastapi import APIRouter, Request, Depends, status, Query
from pydantic import BaseModel

from models.api_responce import APIResponce
from routers.auth_router import get_current_user
from routers.schemas.rag_requests import UserPersonaEnum, LanguageEnum, MockChunkInput
from routers.schemas.chat_schemas import ChatHistoryResponse, ChatSummaryDTO
from db.chat_model import ChatModel

chat_router = APIRouter(tags=["Chat & Memory"], prefix="/chat")


class SendMessageRequest(BaseModel):
    query: str
    chat_id: str | None = None
    language: LanguageEnum = LanguageEnum.AR


@chat_router.post("/send", response_model=APIResponce)
async def send_chat_message(
    payload: SendMessageRequest,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    db = request.app.state.db_client
    llm_service = request.app.state.llm_service
    chat_model = ChatModel(db)
    user_id = str(current_user["_id"])
    
    # Resolve Chat & Short-Term Memory
    chat_id = payload.chat_id or await chat_model.get_or_create_chat(user_id)
    recent_history = await chat_model.get_recent_messages(chat_id, limit=6)
    
    # Resolve Long-Term Memory
    mother_profile = current_user.get("mother_profile")
    dynamic_memories = await chat_model.get_active_clinical_memory(user_id)
    
    # Dummy Chunks (or real Vector Search hits)
    chunks = [
        MockChunkInput(
            chunk_id="chk_1",
            doc_name="WHO_MNH_Care_2025.pdf",
            page_number=5,
            section="Hypertension in Pregnancy",
            text="Pre-eclampsia warning signs include severe persistent headache, visual disturbances, and epigastric pain."
        )
    ]

    # Generate LLM Response with Memory Injection
    answer, latency, citations = await llm_service.generate_chat_response(
        query=payload.query,
        chunks=chunks,
        persona=UserPersonaEnum(current_user.get("persona", "mother")),
        language=payload.language,
        history=recent_history,
        mother_profile=mother_profile,
        dynamic_memories=dynamic_memories
    )

    # Persist Interaction to MongoDB
    await chat_model.add_message(chat_id, user_id, "user", payload.query)
    await chat_model.add_message(chat_id, user_id, "assistant", answer, citations, latency)

    return APIResponce(
        status_code=status.HTTP_200_OK,
        status="success",
        data={
            "chat_id": chat_id,
            "answer": answer,
            "citations": citations,
            "latency_seconds": latency
        }
    )


@chat_router.get("/my-chats", response_model=APIResponce)
async def list_user_chats(
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Retrieves all active chat sessions belonging to the authenticated user."""
    db = request.app.state.db_client
    chat_model = ChatModel(db)
    user_id = str(current_user["_id"])
    
    chats = await chat_model.get_user_chats(user_id)
    return APIResponce(
        status_code=status.HTTP_200_OK,
        status="success",
        data=chats
    )


@chat_router.get("/{chat_id}/history", response_model=APIResponce)
async def get_chat_history(
    chat_id: str,
    request: Request,
    page: int = Query(default=1, ge=1, description="Page number starting from 1"),
    page_size: int = Query(default=20, ge=1, le=100, description="Number of messages per request"),
    current_user: dict = Depends(get_current_user)
):
    """Retrieves paginated message history for a specific chat session."""
    db = request.app.state.db_client
    chat_model = ChatModel(db)
    user_id = str(current_user["_id"])

    history_data = await chat_model.get_chat_history_paginated(
        chat_id=chat_id,
        user_id=user_id,
        page=page,
        page_size=page_size
    )

    if not history_data:
        return APIResponce(
            status_code=status.HTTP_404_NOT_FOUND,
            status="failed",
            error="Chat not found or unauthorized access."
        )

    return APIResponce(
        status_code=status.HTTP_200_OK,
        status="success",
        data=history_data
    )