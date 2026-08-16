from fastapi import APIRouter, UploadFile, Request, status
from models.enums.responce_status import ResponseStatus
from db.document_model import DocumentModel
from models.api_responce import APIResponce
from models.db_schemes.document import Document
from services.data_service import DocumentParserService
from bson import ObjectId
import logging

logger = logging.getLogger(__name__)
data = APIRouter(tags=["api/data"], prefix="/data")

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

    try:
        doc = Document(
            _id=ObjectId(),
            document = file,
            doc_name=file.filename,
            doc_type=file.content_type,
            doc_metadata={},
            num_pages=0             # set after parsing
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
        status_code=status.HTTP_201_CREATED,
        status=ResponseStatus.FILE_UPLOADED_SUCCESSFULLY.value,
        data={"document_id": doc_id}    # return the created ID
    )