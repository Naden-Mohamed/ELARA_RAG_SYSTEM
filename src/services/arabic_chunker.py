import logging
import re
from typing import Any

import fitz  # PyMuPDF
from camel_tools.utils.normalize import (
    normalize_alef_ar,
    normalize_alef_maksura_ar,
    normalize_teh_marbuta_ar,
)

# You will need: pip install pymupdf pyarabic camel-tools
from pyarabic import araby

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArabicTextPreprocessor:
    """
    Production-ready Arabic text normalization for RAG/Embeddings.
    """

    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""

        # 1. Remove Tashkeel (Vowel marks) and Tatweel (Elongation)
        text = araby.strip_tashkeel(text)
        text = araby.strip_tatweel(text)

        # 2. Normalize orthography (Crucial for semantic search)
        text = normalize_alef_ar(text)  # أ, إ, آ -> ا
        text = normalize_alef_maksura_ar(text)  # ى -> ي
        text = normalize_teh_marbuta_ar(
            text
        )  # ة -> ه (Optional, but highly recommended for retrieval)

        # 3. Clean up formatting artifacts, keeping Arabic punctuation
        # Replace multiple newlines, spaces, and tabs
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 4. Handle broken Bi-directional text (English inside Arabic)
        # Sometimes PDF parsers drop spaces around English words in Arabic text
        text = re.sub(r"([a-zA-Z0-9])([\u0600-\u06FF])", r"\1 \2", text)
        text = re.sub(r"([\u0600-\u06FF])([a-zA-Z0-9])", r"\1 \2", text)

        return text.strip()


class ArabicPdfChunker:
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text(self, file_path: str) -> list[dict[str, Any]]:
        pages = []
        try:
            doc = fitz.open(file_path)
            for i, page in enumerate(doc):
                raw_text = page.get_text(
                    "text",
                    flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE,
                )

                cleaned_text = ArabicTextPreprocessor.clean_text(raw_text)
                pages.append({"page_number": i + 1, "text": cleaned_text})

            logger.info(f"Extracted {len(pages)} pages from {file_path}")
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e!s}")
            raise
        finally:
            if "doc" in locals():
                doc.close()

        return pages

    def chunk(self, file_path: str) -> list[dict[str, Any]]:
        pages = self.extract_text(file_path)

        full_text = ""
        char_to_page = []

        for p in pages:
            page_text = p["text"] + "\n"
            full_text += page_text
            char_to_page.extend([p["page_number"]] * len(page_text))

        chunks = []
        start = 0
        order = 1

        while start < len(full_text):
            end = min(start + self.chunk_size, len(full_text))

            if end < len(full_text):
                window = full_text[start:end]
                match = list(re.finditer(r"[\.،؟]\s", window))

                if match:
                    # Get the last boundary found in this window
                    last_boundary = match[-1].end()
                    # Only split if boundary is past the 50% mark of the chunk size to avoid tiny chunks
                    if last_boundary > self.chunk_size * 0.5:
                        end = start + last_boundary

            chunk_text = full_text[start:end].strip()

            if chunk_text:
                start_page = (
                    char_to_page[start]
                    if start < len(char_to_page)
                    else pages[-1]["page_number"]
                )
                end_idx = min(end - 1, len(char_to_page) - 1)
                end_page = char_to_page[end_idx] if end_idx >= 0 else start_page

                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": {
                            "chunk_order": order,
                            "page_numbers": sorted({start_page, end_page}),
                            "language": "ar",
                        },
                    }
                )
                order += 1

            start = max(end - self.chunk_overlap, start + 1)

        return chunks
