from fastapi import APIRouter, UploadFile, Request, status
from models.enums.ResponceStatusEnum import ResponseStatus
from db.document_model import DocumentModel
from models.api_responce import APIResponce
from models.db_schemes.document import Document
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
    document_model = await DocumentModel.get_instance(db_client)  # one call, correct

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
        raise e
    try:
        doc = Document(
            _id=ObjectId(),
            doc_name=file.filename,
            doc_type=file.content_type,
            doc_metadata={},
            doc_size=file.size           
        )
        doc_id = await document_model.upload_document(doc=doc)
    except Exception:
        logger.exception("File upload failed")
        return APIResponce(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            status=ResponseStatus.FILE_UPLOAD_FAILED.value,
            error="File upload failed"
        )

    return APIResponce(
        status_code=status.HTTP_200_OK,
        status=ResponseStatus.FILE_UPLOADED_SUCCESSFULLY.value,
        data={"document_id": doc_id}    # return the created ID
    )

@data.post("/ingest")                    
async def ingest_file(                   
    request: Request,
    file_path: str
) -> APIResponce:

    if not os.path.exists(file_path):
        return APIResponce(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                status=ResponseStatus.NO_FILES_FOUNDED_TO_PROCESS.value,
                error="File doesn't exist"
                )
    document_parser = DocumentParserService()
    file_content = document_parser.get_file_content(file_path)

    if not file_content:
        return APIResponce(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                status=ResponseStatus.NO_FILES_FOUNDED_TO_PROCESS.value,
                error="No file content parsed"
                        )
    file_chunks = document_parser.get_chunks(document = file_content)






