from pypdf import PdfReader
 
 
class SimpleChunker:
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
 
    def extract_text_by_page(self, file_path: str) -> list[dict]:
        reader = PdfReader(file_path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append({"page_number": i + 1, "text": text})
        return pages
 
    def chunk(self, file_path: str) -> list[dict]:
        """Returns the same shape your pipeline already expects from
        DocumentParserService.get_chunks(): a list of
        {"text": str, "metadata": {...}} dicts."""
        pages = self.extract_text_by_page(file_path)
        if not any(p["text"].strip() for p in pages):
            return []  # e.g. a scanned PDF with no extractable text layer
 
        full_text = ""
        char_to_page = []  # char_to_page[i] -> page number for char i in full_text
        for p in pages:
            full_text += p["text"] + "\n"
            char_to_page.extend([p["page_number"]] * (len(p["text"]) + 1))
 
        chunks = []
        start = 0
        order = 1
 
        while start < len(full_text):
            end = min(start + self.chunk_size, len(full_text))
 

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
                        "source": "simple_chunker",  # tag it so you can tell
                                                       # which chunks came from
                                                       # the fallback vs. Docling
                    },
                })
                order += 1
 
            # guard against an infinite loop if chunk_overlap >= chunk_size
            start = max(end - self.chunk_overlap, start + 1)
 
        return chunks