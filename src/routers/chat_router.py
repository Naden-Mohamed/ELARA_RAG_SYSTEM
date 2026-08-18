from fastapi import APIRouter, Request, Depends, status, Query
from pydantic import BaseModel
import requests

from models.api_responce import APIResponce
from routers.auth_router import get_current_user
from routers.schemas.rag_requests import UserPersonaEnum, LanguageEnum, MockChunkInput
from db.chat_model import ChatModel
from services.llm_service import LLMService

chat_router = APIRouter(tags=["Chat & Memory"], prefix="/chat")
llm_service = LLMService()


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
    chat_model = ChatModel(db)
    user_id = str(current_user["_id"])
    
    chat_id = payload.chat_id or await chat_model.get_or_create_chat(user_id)
    recent_history = await chat_model.get_recent_messages(chat_id, limit=6)
    
    mother_profile = current_user.get("mother_profile")
    dynamic_memories = await chat_model.get_active_clinical_memory(user_id)
    
    chunks = []
    try:
        base_url = str(request.base_url).rstrip("/")
        search_payload = {
            "text": payload.query,
            "limit": 5
        }
        
        response = requests.post(f"{base_url}/rag/search", json=search_payload, timeout=30)
        
        if response.status_code == 200:
            res_json = response.json()
            if res_json and "data" in res_json:
                search_data = res_json["data"].get("search_results", {})
                points = search_data.get("points", []) if isinstance(search_data, dict) else search_data
                
                for point in points:
                    p_load = point.get("payload", {})
                    page_nums = p_load.get("page_numbers", [1])
                    page_num = page_nums[0] if isinstance(page_nums, list) and page_nums else 1
                    sections = p_load.get("section_headings", [])
                    section_title = sections[0] if isinstance(sections, list) and sections else "General Recommendations"

                    chunks.append(
                        MockChunkInput(
                            chunk_id=str(point.get("id", "chk_real")),
                            doc_name=p_load.get("doc_name") or p_load.get("original_filename", "WHO_Guidelines.pdf"),
                            page_number=page_num,
                            section=section_title,
                            text=p_load.get("text", "")
                        )
                    )
    except Exception as e:
        print(f"Error calling internal rag search endpoint: {e}")
        
    if not chunks:
        chunks = [
            MockChunkInput(
                chunk_id="chk_fallback",
                doc_name="WHO_Guidelines.pdf",
                page_number=1,
                section="General",
                text="No matching content was found in the available medical documents."
            )
        ]

    answer, latency, citations = await llm_service.generate_chat_response(
        query=payload.query,
        chunks=chunks,
        persona=UserPersonaEnum(current_user.get("persona", "mother")),
        language=payload.language,
        history=recent_history,
        mother_profile=mother_profile,
        dynamic_memories=dynamic_memories
    )

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