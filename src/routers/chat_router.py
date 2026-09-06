import asyncio

from fastapi import APIRouter, Depends, Query, Request, status
from pydantic import BaseModel

from core.safety_gate import (
    build_safe_fallback_message,
    pre_generation_gate,
    validate_grounded_response,
)
from db.chat_model import ChatModel
from models.api_responce import APIResponce
from models.enums.DataBaseEnum import DataBaseEnums
from models.enums.LLMEnums import DocumentTypeEnum
from routers.auth_router import get_current_user
from routers.schemas.rag_requests import LanguageEnum, MockChunkInput, UserPersonaEnum

chat_router = APIRouter(tags=["Chat & Memory"], prefix="/chat")


class SendMessageRequest(BaseModel):
    query: str
    chat_id: str | None = None
    language: LanguageEnum = LanguageEnum.AR


async def _retrieve_chunks(
    request: Request, query: str, top_k: int = 5
) -> list[MockChunkInput]:
    """Direct retrieval -- no self-HTTP call, no event-loop blocking."""
    embedding_service = request.app.state.embedding_service
    vectordb = request.app.state.vectordb

    embeddings = await asyncio.to_thread(
        embedding_service.embed_text, query, DocumentTypeEnum.QUERY.value
    )
    if embeddings is None or len(embeddings) == 0:
        return []

    query_vector = embeddings[0]
    query_vector = (
        query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)
    )

    results = await vectordb.search_by_vector(
        DataBaseEnums.DOCUMENTS_COLLECTION.value, query_vector, top_k
    )
    if not results or not getattr(results, "points", None):
        return []

    chunks = []
    for point in results.points:
        p_load = point.payload or {}
        page_nums = p_load.get("page_numbers", [1])
        page_num = page_nums[0] if isinstance(page_nums, list) and page_nums else 1
        sections = p_load.get("section_headings", [])
        section_title = (
            sections[0]
            if isinstance(sections, list) and sections
            else "General Recommendations"
        )

        chunks.append(
            MockChunkInput(
                chunk_id=str(point.id),
                doc_name=p_load.get("doc_name")
                or p_load.get("original_filename", "WHO_Guidelines.pdf"),
                page_number=page_num,
                section=section_title,
                text=p_load.get("text", ""),
                score=point.score or 0.0,
            )
        )
    return chunks


@chat_router.post("/send", response_model=APIResponce)
async def send_chat_message(
    payload: SendMessageRequest,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    db = request.app.state.db_client
    llm_service = request.app.state.llm_service
    chat_model = ChatModel(db)
    user_id = str(current_user["_id"])

    chat_id = payload.chat_id or await chat_model.get_or_create_chat(user_id)
    recent_history = await chat_model.get_recent_messages(chat_id, limit=6)

    mother_profile = current_user.get("mother_profile")
    dynamic_memories = await chat_model.get_active_clinical_memory(user_id)

    chunks = await _retrieve_chunks(request, payload.query)

    gate_result = pre_generation_gate(payload.query, chunks)
    if not gate_result["allow"]:
        answer = build_safe_fallback_message(payload.language)
        citations, latency = [], 0.0
    else:
        answer, latency, citations = await llm_service.generate_chat_response(
            query=payload.query,
            chunks=chunks,
            persona=UserPersonaEnum(current_user.get("persona", "mother")),
            language=payload.language,
            history=recent_history,
            mother_profile=mother_profile,
            dynamic_memories=dynamic_memories,
        )
        validation = validate_grounded_response(answer, citations, chunks)
        if not validation["valid"]:
            answer = build_safe_fallback_message(payload.language)
            citations = []

    await chat_model.add_message(chat_id, user_id, "user", payload.query)
    await chat_model.add_message(
        chat_id, user_id, "assistant", answer, citations, latency
    )

    return APIResponce(
        status_code=status.HTTP_200_OK,
        status="success",
        data={
            "chat_id": chat_id,
            "answer": answer,
            "citations": citations,
            "latency_seconds": latency,
        },
    )


@chat_router.get("/my-chats", response_model=APIResponce)
async def list_user_chats(
    request: Request, current_user: dict = Depends(get_current_user)
):
    db = request.app.state.db_client
    chat_model = ChatModel(db)
    user_id = str(current_user["_id"])

    chats = await chat_model.get_user_chats(user_id)
    return APIResponce(status_code=status.HTTP_200_OK, status="success", data=chats)


@chat_router.get("/{chat_id}/history", response_model=APIResponce)
async def get_chat_history(
    chat_id: str,
    request: Request,
    page: int = Query(default=1, ge=1, description="Page number starting from 1"),
    page_size: int = Query(
        default=20, ge=1, le=100, description="Number of messages per request"
    ),
    current_user: dict = Depends(get_current_user),
):
    db = request.app.state.db_client
    chat_model = ChatModel(db)
    user_id = str(current_user["_id"])

    history_data = await chat_model.get_chat_history_paginated(
        chat_id=chat_id, user_id=user_id, page=page, page_size=page_size
    )

    if not history_data:
        return APIResponce(
            status_code=status.HTTP_404_NOT_FOUND,
            status="failed",
            error="Chat not found or unauthorized access.",
        )

    return APIResponce(
        status_code=status.HTTP_200_OK, status="success", data=history_data
    )
