from fastapi import APIRouter, UploadFile, Request, status
from models.enums.ResponceStatusEnum import ResponseStatusEnums
from models.enums.DocumentStatusEnum import DocumentStatusEnums
from models.enums.LLMEnums import DocumentTypeEnum
from db.document_model import DocumentModel
from db.chunk_model import ChunkModel
from models.api_responce import APIResponce
from models.document import Document
from models.data_chunk import DataChunk
from services.data_service import DocumentParserService
from bson import ObjectId
import logging
import aiofiles
from core.config import get_settings
import os

logger = logging.getLogger(__name__)
data = APIRouter(tags=["api/data"], prefix="/data")
settings = get_settings()


@data.post("/upload")                    
async def upload_file(                   
    request: Request,
    file: UploadFile
) -> APIResponce:
    db_client = request.app.state.db_client
    document_model = await DocumentModel.get_instance(db_client)

    data_service = DocumentParserService()
    is_valid, result_signal = data_service.validate_uploaded_file(file=file)

    if not is_valid:
        return APIResponce(
            status_code=status.HTTP_400_BAD_REQUEST,
            status=result_signal,
            error="File validation failed"
        )

    unique_file_name = data_service.generate_unique_filename(original_filename=file.filename)
    file_path = os.path.join(data_service.files_path, unique_file_name)

    try:
        async with aiofiles.open(file_path, "wb") as out_file:
            while True:
                chunk = await file.read(settings.FILE_DEFAULT_CHUNK_SIZE)
                if not chunk:
                    break
                await out_file.write(chunk)
    except Exception as e:
        logger.exception("Failed to write uploaded file to disk")
        raise e

    try:
        doc = Document(
            _id=ObjectId(),
            doc_name=file.filename,
            doc_path=file_path,          
            doc_type=file.content_type,
            doc_metadata={},
            doc_size=file.size,
            status=DocumentStatusEnums.PENDING.value,       
        )
        doc_id = await document_model.upload_document(doc=doc)
    except Exception:
        logger.exception("File upload DB insertion failed")
        return APIResponce(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            status=ResponseStatusEnums.FILE_UPLOAD_FAILED.value,
            error="File upload failed"
        )

    return APIResponce(
        status_code=status.HTTP_200_OK,
        status=ResponseStatusEnums.FILE_UPLOADED_SUCCESSFULLY.value,
        data={"document_id": str(doc_id)}
    )


@data.post("/ingest")                    
async def ingest_file(                   
    request: Request,
    document_id: str
) -> APIResponce:
    db_client = request.app.state.db_client
    document_model = await DocumentModel.get_instance(db_client)

    doc = await document_model.get_document_by_id(document_id)
    if not doc:
        return APIResponce(
            status_code=status.HTTP_404_NOT_FOUND,
            status=ResponseStatusEnums.FILE_ID_ERROR.value,
            error="Document not found"
        )

    if not doc.doc_path or not os.path.exists(doc.doc_path):
        await document_model.update_status(
            doc_id=document_id,
            status=DocumentStatusEnums.FAILED.value,
            error_message="Stored file missing on disk",
        )
        return APIResponce(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            status=ResponseStatusEnums.NO_FILES_FOUNDED_TO_PROCESS.value,
            error="File doesn't exist"
        )

    await document_model.update_status(
        doc_id=document_id,
        status=DocumentStatusEnums.PROCESSING.value,
    )

    if settings.USE_SIMPLE_CHUNKER:
        from services.simple_chunker import SimpleChunker
        file_chunks = SimpleChunker().chunk(doc.doc_path)
    else:
        document_parser = DocumentParserService()
        file_content = document_parser.get_file_content(doc.doc_path)
        
        if not file_content:
            await document_model.update_status(
                doc_id=document_id,
                status=DocumentStatusEnums.FAILED.value,
                error_message="Docling parsing returned no content",
            )
            return APIResponce(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                status=ResponseStatusEnums.FILE_PROCESSING_FAILED.value,
                error="No file content parsed"
            )
            
        file_chunks = document_parser.get_chunks(file_content)

    if not file_chunks:
        await document_model.update_status(
            doc_id=document_id,
            status=DocumentStatusEnums.FAILED.value,
            error_message="Chunking produced 0 chunks",
        )
        return APIResponce(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            status=ResponseStatusEnums.FILE_PROCESSING_FAILED.value,
            error="No chunks produced"
        )

    file_chunks_records = [
        DataChunk(
            _id=ObjectId(),
            chunk_text=chunk["text"],            
            chunk_metadata={
                **chunk["metadata"],
                # "raw_text": chunk["raw_text"],    
                "original_filename": doc.doc_name,
                "page_numbers": chunk.get("metadata", {}).get("page_numbers", []),
                "section_headings": chunk.get("metadata", {}).get("section_headings", []),
            },
            chunk_order=chunk["metadata"]["chunk_order"] + 1,
            chunk_document_id =  ObjectId(document_id)

        )
        for chunk in file_chunks
        ]

    chunk_model = await ChunkModel.get_instance(db_client=db_client)
    chunks_count = await chunk_model.insert_many_chunks(chunks = file_chunks_records)

    if chunks_count == 0:
        return APIResponce(
            status_code=status.HTTP_204_NO_CONTENT,
            data=0,
            status=ResponseStatusEnums.INSERT_INTO_VECTORDB_ERROR.value,
            error="No chunks inserted"
        )

    await document_model.update_status(
        doc_id=document_id,
        status=DocumentStatusEnums.PROCESSED.value,
        chunk_count=chunks_count
    )

    return APIResponce(
        status_code=status.HTTP_200_OK,
        data={"inserted_chunks_count": chunks_count},
        status=ResponseStatusEnums.FILE_PROCESSED_SUCCESSFULLY.value
    )

@data.get("/documents")
async def list_documents(request: Request) -> APIResponce:
    db_client = request.app.state.db_client
    document_model = await DocumentModel.get_instance(db_client)
    
 
    cursor = document_model.collection.find({})
    docs = await cursor.to_list(length=100)
    
    result = [
        {
            "document_id": str(d["_id"]),
            "doc_name": d.get("doc_name"),
            "status": d.get("status"),
            "chunk_count": d.get("chunk_count", 0),
            "created_at": d.get("created_at")
        }
        for d in docs
    ]
    
    return APIResponce(
        status_code=status.HTTP_200_OK,
        status=ResponseStatusEnums.FILE_PROCESSED_SUCCESSFULLY.value,
        data={"documents": result}
    )


@data.get("/ingested")
async def get_ingested_documents(request: Request) -> APIResponce:
    db_client = request.app.state.db_client
    
    pipeline = [
        {
            "$group": {
                "_id": "$chunk_document_id",
                "original_filename": {"$first": "$chunk_metadata.original_filename"},
                "chunks_count": {"$sum": 1}
            }
        }
    ]
    
    ingested_docs = await db_client["chunks"].aggregate(pipeline).to_list(length=100)
    
    data_list = [
        {
            "document_id": str(item["_id"]),
            "filename": item.get("original_filename", "Unknown"),
            "chunks_count": item.get("chunks_count", 0)
        }
        for item in ingested_docs
    ]
    
    return APIResponce(
        status_code=status.HTTP_200_OK,
        status=ResponseStatusEnums.FILE_PROCESSED_SUCCESSFULLY.value,
        data={"ingested_documents": data_list}
    )