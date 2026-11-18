"""
PDF Loader for Multi-Modal RAG.
Extracts page-by-page text and metadata from PDF files using pypdf / pymupdf.
"""

import os
from typing import List, Dict, Any

class PDFLoader:
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load(self, file_path: str) -> List[Dict[str, Any]]:
        """Parses a PDF file and returns chunked documents with page-level metadata."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        file_name = os.path.basename(file_path)
        pages_text = self._extract_pages(file_path)
        if not pages_text:
            return []

        documents = []
        total_pages = len(pages_text)

        for page_num, text in enumerate(pages_text, start=1):
            if not text.strip():
                continue

            chunks = self._chunk_text(text)
            for chunk_idx, chunk in enumerate(chunks):
                doc_id = f"pdf_{file_name}_p{page_num}_{chunk_idx}"
                content = (
                    f"[Document: {file_name} | Page {page_num} of {total_pages}]\n"
                    f"{chunk}"
                )
                documents.append({
                    "id": doc_id,
                    "modality": "pdf_text",
                    "content": content,
                    "metadata": {
                        "source": file_name,
                        "file_path": os.path.abspath(file_path),
                        "type": "pdf_document",
                        "page": page_num,
                        "total_pages": total_pages,
                        "chunk_index": chunk_idx
                    }
                })

        return documents

    def _extract_pages(self, file_path: str) -> List[str]:
        """Extracts text from each page using pymupdf, fitz, pypdf, pdfplumber, or PyPDF2."""
        # 1. Try PyMuPDF (pymupdf / fitz)
        try:
            import pymupdf
            doc = pymupdf.open(file_path)
            return [page.get_text() or "" for page in doc]
        except ImportError:
            pass

        try:
            import fitz
            doc = fitz.open(file_path)
            return [page.get_text() or "" for page in doc]
        except ImportError:
            pass

        # 2. Try modern pypdf
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            return [page.extract_text() or "" for page in reader.pages]
        except ImportError:
            pass

        # 3. Try pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                return [page.extract_text() or "" for page in pdf.pages]
        except ImportError:
            pass

        # 4. Try legacy PyPDF2
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(file_path)
            return [page.extract_text() or "" for page in reader.pages]
        except ImportError:
            pass

        raise ImportError(
            "No PDF parsing library found! Please install pypdf or pymupdf: pip install pypdf pymupdf"
        )

    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return []

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end == len(words):
                break
            start += self.chunk_size - self.chunk_overlap
        return chunks
