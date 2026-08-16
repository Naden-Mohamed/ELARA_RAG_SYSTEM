from core.config import get_settings
from models.enums.responce_status import ResponseStatus
import os
from pathlib import Path
# from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
# from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
# from docling.document_converter import DocumentConverter, PdfFormatOption
# from docling.datamodel.pipeline_options import PdfPipelineOptions
# from docling.datamodel.base_models import InputFormat
# from transformers import AutoTokenizer
from fastapi import UploadFile
import logging

logger = logging.getLogger(__name__)

class DocumentParserService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def validate_uploaded_file(self, file: UploadFile):
        if file.content_type not in self.settings.FILE_ALLOWED_TYPES:
            return False, ResponseStatus.FILE_TYPE_NOT_SUPPORTED.value

        if file.size is None or file.size > self.settings.FILE_MAX_SIZE_MB* 1024 * 1024 :
            return False, ResponseStatus.FILE_SIZE_EXCEEDED.value
        
        return True, ResponseStatus.FILE_VALIDATED_SUCCESSFULLY.value
    def file_parse():
        pass