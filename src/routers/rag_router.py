from fastapi import APIRouter, Request, status
from models.enums.ResponceStatusEnum import ResponseStatusEnums
from models.enums.DocumentStatusEnum import DocumentStatusEnums
from models.enums.DataBaseEnum import DataBaseEnums
from models.enums.LLMEnums import DocumentTypeEnum
from db.document_model import DocumentModel
from models.api_responce import APIResponce
from db.chunk_model import ChunkModel
from routers.schemas.data_requests import SearchRequest, PushRequest
import logging
from core.config import get_settings
import os

logger = logging.getLogger(__name__)
rag = APIRouter(tags=["api/rag"], prefix="/rag")
settings = get_settings()
@rag.post("/push")                    
async def index_push(                   
    request: Request,
    push_request: PushRequest):

    db_client = request.app.state.db_client
    vectordb = request.app.state.vectordb
    embedding_service = request.app.state.embedding_service 
    chunk_model = await ChunkModel.get_instance(db_client=db_client)
    document_model = await DocumentModel.get_instance(db_client)
    doc = await document_model.get_document_by_id(push_request.document_id)
    print(f"doc{doc}")
    
    file_chunks = await chunk_model.get_document_chunks(document_id=push_request.document_id)
    print(f"file_chunks {file_chunks}")
    texts = [c.chunk_text for c in file_chunks]
    print(f"return {texts}")
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

    # every chunk carries document_id back to Mongo, so a document can be
    # looked up, deleted, or re-ingested without orphaning vectors in Qdrant
    metadatas = [
        {**c.chunk_metadata, "document_id": push_request.document_id, "doc_name": doc.doc_name if doc else None}
        for c in file_chunks
    ]

    inserted = await vectordb.insert_many(
        collection_name=DataBaseEnums.DOCUMENTS_COLLECTION.value,
        texts=texts,
        vectors=[vec.tolist() for vec in embeddings],
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

@rag.post("/info")                    
async def get_index_info(                   
    request: Request,
    document_id:str):

    db_client = request.app.state.db_client
    vectordb = request.app.state.vectordb

    document_model = await DocumentModel.get_instance(db_client)
    doc = await document_model.get_document_by_id(document_id)
    # collection vs doc_id
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
        data={"document_id": document_id, "index_info": len(info)}
    )


@rag.post("/search")                    
async def search_by_vector(                   
    request: Request,
     search_request: SearchRequest):

    vectordb = request.app.state.vectordb
    embedding_service = request.app.state.embedding_service 
    query_embeddings = embedding_service.embed_text(search_request.text, document_type=DocumentTypeEnum.QUERY.value)

    results = await vectordb.search_by_vector(DataBaseEnums.DATA_CHUNKS_COLLECTION.value,query_embeddings[0],5 )

    if not results:
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



