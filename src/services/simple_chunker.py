import re
import unicodedata
import logging
from typing import List, Dict, Any
from pypdf import PdfReader

# Configure logging for production monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    A production-ready text preprocessing pipeline for raw PDF text.
    """
    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        
        # 1. Unicode Normalization (NFKC)
        # Converts ligatures (e.g., 'ﬁ' -> 'fi') and normalizes special characters
        text = unicodedata.normalize("NFKC", text)
        
        # 2. Remove non-printable/control characters (keeping newlines and tabs)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # 3. De-hyphenation (Fix words split across lines)
        # Matches word characters, a hyphen, optional whitespace/newlines, and more word characters
        text = re.sub(r'([a-zA-Z]+)-\s*\n\s*([a-zA-Z]+)', r'\1\2', text)
        
        # 4. Whitespace Normalization
        # Replace multiple spaces/tabs with a single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Standardize excessive newlines (reduce 3+ newlines to exactly 2 to preserve paragraphs)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()


class SimpleChunker:
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_by_page(self, file_path: str) -> List[Dict[str, Any]]:
        """Extracts and preprocesses text from a PDF page by page."""
        pages = []
        try:
            reader = PdfReader(file_path)
            for i, page in enumerate(reader.pages):
                raw_text = page.extract_text() or ""
                cleaned_text = TextPreprocessor.clean_text(raw_text)
                
                pages.append({"page_number": i + 1, "text": cleaned_text})
                
            logger.info(f"Successfully extracted and cleaned {len(pages)} pages from {file_path}")
        except Exception as e:
            logger.error(f"Failed to process PDF {file_path}: {str(e)}")
            raise

        return pages

    def chunk(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Returns the same shape your pipeline already expects from
        DocumentParserService.get_chunks(): a list of
        {"text": str, "metadata": {...}} dicts.
        """
        pages = self.extract_text_by_page(file_path)
        if not any(p["text"].strip() for p in pages):
            logger.warning(f"No extractable text found in {file_path}. Possibly a scanned document.")
            return []  

        full_text = ""
        char_to_page = [] 
        
        for p in pages:
            # We add a newline to separate pages cleanly
            page_text = p["text"] + "\n"
            full_text += page_text
            char_to_page.extend([p["page_number"]] * len(page_text))

        chunks = []
        start = 0
        order = 1

        while start < len(full_text):
            end = min(start + self.chunk_size, len(full_text))

            # Attempt a clean sentence break if we are not at the very end
            if end < len(full_text):
                boundary = full_text.rfind(". ", start, end)
                if boundary != -1 and boundary > start + self.chunk_size * 0.5:
                    end = boundary + 1

            chunk_text = full_text[start:end].strip()
            
            if chunk_text:
                start_page = char_to_page[start] if start < len(char_to_page) else pages[-1]["page_number"]
                end_idx = min(end - 1, len(char_to_page) - 1)
                end_page = char_to_page[end_idx] if end_idx >= 0 else start_page
                
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "chunk_order": order,
                        "page_numbers": sorted(set([start_page, end_page])),
                        "chunk_type": "text",
                        "has_table": False,
                        "section_headings": [],
                        "source": "simple_chunker",
                    },
                })
                order += 1

            # Advance the window, ensuring we always step forward
            start = max(end - self.chunk_overlap, start + 1)

        logger.info(f"Generated {len(chunks)} chunks from {file_path}")
        return chunks