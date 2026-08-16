from core.config import get_settings
from models.enums.ResponceStatusEnum import ResponseStatus
import os
import random
import string
from pathlib import Path
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from transformers import AutoTokenizer
from fastapi import UploadFile
import logging

logger = logging.getLogger(__name__)

class DocumentParserService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_dir = os.path.dirname(os.path.dirname(__name__))
        self.files_path = os.path.join(self.base_dir, "data")

    def generate_unique_filename(self, original_filename:str | None, lenght:int = 5):
        characters = string.ascii_letters + string.digits
        random_prefix = ''.join(random.choices(characters, k=lenght))
        return f"{random_prefix}_{original_filename}"

    

    def validate_uploaded_file(self, file: UploadFile):
        if file.content_type not in self.settings.FILE_ALLOWED_TYPES:
            return False, ResponseStatus.FILE_TYPE_NOT_SUPPORTED.value

        if file.size is None or file.size > self.settings.FILE_MAX_SIZE_MB* 1024 * 1024 :
            return False, ResponseStatus.FILE_SIZE_EXCEEDED.value
        
        return True, ResponseStatus.FILE_VALIDATED_SUCCESSFULLY.value
    def get_file_content(self, file_path: str):
        """
        Accepts a full absolute path. Returns a Docling DoclingDocument or None.
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None

        try:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True          
            pipeline_options.do_table_structure = True

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            result = converter.convert(Path(file_path))
            return result.document
        except Exception as e:
            logger.error(f"Docling conversion failed for {file_path}: {e}")
            return None

    def get_chunks(self, document, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Returns a list of dicts — each dict is a self-contained chunk record
        ready to be stored, with text and all metadata extracted.
        """
        settings = get_settings()
        try:
            tokenizer = HuggingFaceTokenizer(
                tokenizer=AutoTokenizer.from_pretrained(settings.TOKENIZER_MODEL_ID),
                max_tokens=chunk_size,
            )
            chunker = HybridChunker(tokenizer=tokenizer)
            chunk_iter = chunker.chunk(dl_doc=document)
        except Exception as e:
            logger.error(f"Chunking failed: {e}")
            return []

        processed_chunks = []

        for idx, chunk in enumerate(chunk_iter):

            contextualized_text = chunker.contextualize(chunk=chunk)

            # Extract Docling metadata from chunk.meta
            meta = chunk.meta if hasattr(chunk, "meta") else {}
            headings = getattr(meta, "headings", []) or []
            doc_items = getattr(meta, "doc_items", []) or []

            # Pull page numbers from the doc_items provenance
            page_numbers = []
            element_types = []
            for item in doc_items:
                prov = getattr(item, "prov", []) or []
                for p in prov:
                    page_no = getattr(p, "page_no", None)
                    if page_no is not None:
                        page_numbers.append(page_no)
                element_types.append(type(item).__name__)

            page_numbers = sorted(set(page_numbers))

            processed_chunks.append({
                # What gets embedded and stored as searchable text
                "text": contextualized_text,
                # Raw text without heading context (useful for display)
                "raw_text": chunk.text,
                "metadata": {
                    "chunk_index": idx,
                    "page_numbers": page_numbers,               # [3, 4]
                    "section_headings": headings,               # ["Chapter 2", "Newton's Laws"]
                    "element_types": list(set(element_types)),  # ["TextItem", "TableItem"]
                    "token_count": tokenizer.count_tokens(contextualized_text),
                    "has_table": any("Table" in t for t in element_types),
                    "has_figure": any("Figure" in t or "Picture" in t for t in element_types),
                    "chunk_type": self._classify_chunk_type(element_types),
                }
            })

        return processed_chunks

    def _classify_chunk_type(self, element_types: list) -> str:
        """Classify the dominant element type for this chunk."""
        if not element_types:
            return "text"
        types_str = " ".join(element_types).lower()
        if "table" in types_str:
            return "table"
        if "figure" in types_str or "picture" in types_str:
            return "figure_caption"
        if "formula" in types_str or "equation" in types_str:
            return "equation"
        return "text"