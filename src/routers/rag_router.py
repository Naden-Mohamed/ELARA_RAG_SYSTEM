from fastapi import APIRouter, Request, status, HTTPException
import logging

from core.config import get_settings
from models.api_responce import APIResponce
from models.enums.ResponceStatusEnum import ResponseStatusEnums
from models.enums.DocumentStatusEnum import DocumentStatusEnums
from models.enums.DataBaseEnum import DataBaseEnums
from models.enums.LLMEnums import DocumentTypeEnum
from db.document_model import DocumentModel
from db.chunk_model import ChunkModel
from routers.schemas.data_requests import SearchRequest, PushRequest
from models.schemas.rag_requests import (
    QueryRequest,
    DirectPromptTestRequest,
    RAGResponseData,
    RetrievedChunkDTO
)
from services.llm_service import LLMService

logger = logging.getLogger(__name__)
rag = APIRouter(tags=["api/rag"], prefix="/rag")
settings = get_settings()
llm_service = LLMService()


# -------------------------------------------------------------
# 1. Indexing & Vector Search Endpoints (Updated & Compatible)
# -------------------------------------------------------------

@rag.post("/index/push")
async def index_push(
    request: Request,
    push_request: PushRequest
):
    db_client = request.app.state.db_client
    vectordb = request.app.state.vectordb
    embedding_service = request.app.state.embedding_service
    chunk_model = await ChunkModel.get_instance(db_client=db_client)
    document_model = await DocumentModel.get_instance(db_client)
    
    doc = await document_model.get_document_by_id(push_request.document_id)
    file_chunks = await chunk_model.get_document_chunks(document_id=push_request.document_id)
    
    texts = [c.chunk_text if hasattr(c, "chunk_text") else c["chunk_text"] for c in file_chunks]

    embeddings = embedding_service.embed_text(texts, document_type=DocumentTypeEnum.DOCUMENT.value)

    if embeddings is None:
        await document_model.update_status(
            doc_id=push_request.document_id,
            status=DocumentStatusEnums.FAILED.value,
            error_message="Embedding step failed",
        )
        return APIResponce(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            status=ResponseStatusEnums.RAG_ANSWER_ERROR.value,
            error="Embedding failed"
        )

    metadatas = [
        {
            **(c.chunk_metadata if hasattr(c, "chunk_metadata") else c.get("metadata", {})),
            "document_id": push_request.document_id,
            "doc_name": doc.doc_name if doc else None
        }
        for c in file_chunks
    ]

    inserted = await vectordb.insert_many(
        collection_name=settings.COLLECTION_NAME,
        texts=texts,
        vectors=[vec.tolist() if hasattr(vec, "tolist") else vec for vec in embeddings],
        metadatas=metadatas,
    )

    if not inserted:
        await document_model.update_status(
            doc_id=push_request.document_id,
            status=DocumentStatusEnums.FAILED.value,
            error_message="Qdrant insert failed",
        )
        return APIResponce(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            status=ResponseStatusEnums.INSERT_INTO_VECTORDB_ERROR.value,
            error="Failed to store vectors"
        )

    await document_model.update_status(
        doc_id=push_request.document_id,
        status=DocumentStatusEnums.PROCESSED.value,
        chunk_count=len(file_chunks),
    )

    return APIResponce(
        status_code=status.HTTP_200_OK,
        status=ResponseStatusEnums.FILE_PROCESSED_SUCCESSFULLY.value,
        data={"document_id": push_request.document_id, "chunk_count": len(file_chunks)}
    )


@rag.post("/index/info")
async def get_index_info(
    request: Request,
    document_id: str
):
    db_client = request.app.state.db_client
    vectordb = request.app.state.vectordb
    document_model = await DocumentModel.get_instance(db_client)

    info = await vectordb.get_collection_info(settings.COLLECTION_NAME)

    if not info:
        await document_model.update_status(
            doc_id=document_id,
            status=DocumentStatusEnums.FAILED.value,
            error_message="no info retrieved",
        )
        return APIResponce(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            status=ResponseStatusEnums.INSERT_INTO_VECTORDB_ERROR.value,
            error="Failed to get_collection_info"
        )

    await document_model.update_status(
        doc_id=document_id,
        status=DocumentStatusEnums.PROCESSED.value,
    )

    return APIResponce(
        status_code=status.HTTP_200_OK,
        status=ResponseStatusEnums.FILE_PROCESSED_SUCCESSFULLY.value,
        data={"document_id": document_id, "index_info": str(info)}
    )


@rag.post("/index/search")
async def search_by_vector(
    request: Request,
    search_request: SearchRequest
):
    vectordb = request.app.state.vectordb
    embedding_service = request.app.state.embedding_service
    query_embeddings = embedding_service.embed_text(search_request.text, document_type=DocumentTypeEnum.QUERY.value)

    results = await vectordb.search_by_vector(
        collection_name=settings.COLLECTION_NAME,
        vector=query_embeddings[0].tolist() if hasattr(query_embeddings[0], "tolist") else query_embeddings[0],
        top_k=search_request.limit if hasattr(search_request, "limit") else 5
    )

    if results is None:
        return APIResponce(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            status=ResponseStatusEnums.VECTORDB_SEARCH_ERROR.value,
            error="No search results"
        )

    return APIResponce(
        status_code=status.HTTP_200_OK,
        status=ResponseStatusEnums.VECTORDB_SEARCH_SUCCESS.value,
        data={"search_results": results}
    )


# -------------------------------------------------------------
# 2. LLM Direct Testing & End-to-End Generation Endpoints
# -------------------------------------------------------------

@rag.post("/direct-generate", response_model=APIResponce)
async def test_llm_direct_prompt(request_payload: DirectPromptTestRequest):
    """Directly tests LLM generation & persona prompt without running vector search."""
    try:
        messages = llm_service.build_rag_messages(
            query=request_payload.query,
            context_chunks=request_payload.context_chunks or ["No context passed for direct prompt evaluation."],
            persona=request_payload.persona
        )
        answer, latency = await llm_service.generate_response(messages)

        return APIResponce(
            status_code=status.HTTP_200_OK,
            status="success",
            data={
                "persona": request_payload.persona.value,
                "answer": answer,
                "latency_seconds": latency,
                "raw_messages": messages
            }
        )
    except Exception as e:
        return APIResponce(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            status="failed",
            error=str(e)
        )


@rag.post("/query", response_model=APIResponce)
async def rag_query(request: Request, payload: QueryRequest):
    """End-to-End RAG: Vector search in Qdrant -> Prompt Construction -> LLM Generation."""
    vectordb = request.app.state.vectordb
    embedding_service = request.app.state.embedding_service

    # 1. Embed query
    query_vector = embedding_service.embed_text(
        text=payload.query,
        document_type=DocumentTypeEnum.QUERY.value
    )
    if query_vector is None:
        raise HTTPException(status_code=500, detail="Failed to generate query embedding.")

    # 2. Retrieve from Qdrant
    hits = await vectordb.search_by_vector(
        collection_name=settings.COLLECTION_NAME,
        vector=query_vector[0].tolist() if hasattr(query_vector[0], "tolist") else query_vector[0],
        top_k=payload.top_k
    )

    if not hits:
        return APIResponce(
            status_code=status.HTTP_200_OK,
            status="success",
            data={"answer": "No relevant documents found.", "retrieved_chunks": []}
        )

    retrieved_dtos: list[RetrievedChunkDTO] = []
    context_chunks: list[str] = []

    # Map Qdrant points to DTOs
    for hit in hits:
        payload_dict = hit.payload or {}
        chunk_text = payload_dict.get("text", "")
        context_chunks.append(chunk_text)

        meta = payload_dict.get("metadata", {})
        retrieved_dtos.append(
            RetrievedChunkDTO(
                chunk_id=str(hit.id),
                doc_name=payload_dict.get("doc_name", "Unknown"),
                text=chunk_text,
                score=round(hit.score, 4) if hasattr(hit, "score") else 0.0,
                page_numbers=meta.get("page_numbers", []),
                section_headings=meta.get("section_headings", [])
            )
        )

    # 3. Generate Answer
    messages = llm_service.build_rag_messages(
        query=payload.query,
        context_chunks=context_chunks,
        persona=payload.persona
    )
    answer, latency = await llm_service.generate_response(messages)

    response_payload = RAGResponseData(
        answer=answer,
        persona_applied=payload.persona.value,
        retrieved_chunks=retrieved_dtos,
        latency_seconds=latency
    )

    return APIResponce(
        status_code=status.HTTP_200_OK,
        status="success",
        data=response_payload.dict()
    )