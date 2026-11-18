"""
Core Ingestion Manager.
Orchestrates multi-modal document extraction, configurable chunking,
vector embedding computation, and database storage with index metadata persistence.
"""

import os
import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union

from src.chunking.chunking_engine import ChunkingEngine, ChunkingConfig
from src.embeddings.embedding_manager import EmbeddingManager, EmbeddingConfig
from src.vectordb.base_adapter import VectorDBAdapter
from src.vectordb.chroma_adapter import ChromaAdapter

logger = logging.getLogger(__name__)

@dataclass
class IngestionResult:
    source: str
    modality: str
    num_raw_docs: int
    num_chunks: int
    num_embedded: int
    index_metadata: Dict[str, Any]
    errors: List[str] = field(default_factory=list)

class IngestionManager:
    def __init__(self, metadata_path: str = "index_metadata.json"):
        self.metadata_path = metadata_path

    def _save_index_metadata(self, metadata: Dict[str, Any]) -> None:
        """Persists session metadata to disk for compatibility and audit tracking."""
        try:
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved index metadata to {self.metadata_path}")
        except Exception as e:
            logger.error(f"Failed to persist index metadata: {e}")

    def ingest(
        self,
        source: Any,
        source_type: str,
        chunking_config: Optional[ChunkingConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        vector_adapter: Optional[VectorDBAdapter] = None
    ) -> IngestionResult:
        """
        Orchestrates end-to-end ingestion pipeline:
        1. Parse source data based on source_type (pdf, excel, web, image, video, text)
        2. Chunk data using ChunkingEngine
        3. Embed chunks using EmbeddingManager
        4. Upsert into active VectorDBAdapter
        5. Persist index_metadata.json
        """
        c_config = chunking_config or ChunkingConfig()
        e_config = embedding_config or EmbeddingConfig()
        v_adapter = vector_adapter or ChromaAdapter()

        chunking_engine = ChunkingEngine(config=c_config)
        embedding_manager = EmbeddingManager(config=e_config)

        errors: List[str] = []
        raw_items: List[Dict[str, Any]] = []
        chunks: List[Dict[str, Any]] = []
        modality = source_type.lower()
        source_name = str(source) if isinstance(source, (str, int)) else getattr(source, "name", "memory_source")

        # Step 1: Extraction & Modality Routing
        try:
            if modality in ["pdf", "pdf_document"]:
                chunks = self._process_pdf(source, chunking_engine)
                raw_items = chunks
            elif modality in ["excel", "spreadsheet", "csv"]:
                chunks = self._process_excel(source, chunking_engine)
                raw_items = chunks
            elif modality in ["web", "url", "web_content"]:
                chunks = self._process_web(source, chunking_engine)
                raw_items = chunks
            elif modality in ["image", "image_caption"]:
                chunks = self._process_image(source, chunking_engine)
                raw_items = chunks
            elif modality in ["video", "video_transcript"]:
                chunks = self._process_video(source, chunking_engine)
                raw_items = chunks
            elif modality in ["text", "raw_text"]:
                text_content = source if isinstance(source, str) else str(source)
                chunks = chunking_engine.chunk_text(text_content, metadata={"source": "direct_text", "modality": "text"})
                raw_items = [{"text": text_content}]
            else:
                text_content = str(source)
                chunks = chunking_engine.chunk_text(text_content, metadata={"source": source_name, "modality": modality})
                raw_items = [{"text": text_content}]
        except Exception as e:
            err_msg = f"Document extraction/chunking failed for {source_type}: {str(e)}"
            logger.error(err_msg)
            errors.append(err_msg)

        if not chunks:
            return IngestionResult(
                source=source_name,
                modality=modality,
                num_raw_docs=len(raw_items),
                num_chunks=0,
                num_embedded=0,
                index_metadata={},
                errors=errors or ["No chunks produced from source."]
            )

        # Step 2: Vector Embedding Generation
        embedded_count = 0
        embeddings: Optional[List[List[float]]] = None

        texts_to_embed = [c.get("content", c.get("text", "")) for c in chunks]
        try:
            embeddings = embedding_manager.embed_texts(texts_to_embed)
            embedded_count = len(embeddings)
        except Exception as e:
            err_msg = f"Embedding computation failed: {str(e)}"
            logger.warning(err_msg)
            errors.append(err_msg)
            # If embedding computation fails (e.g. model not installed), adapter may handle default or fail
            embeddings = None

        # Step 3: Vector Storage Ingestion
        try:
            if not getattr(v_adapter, "_is_connected", False):
                v_adapter.connect()
            v_adapter.add_documents(docs=chunks, embeddings=embeddings)
        except Exception as e:
            err_msg = f"Vector database upsert failed: {str(e)}"
            logger.error(err_msg)
            errors.append(err_msg)

        # Step 4: Construct and Save Index Metadata
        collection_info = {}
        try:
            collection_info = v_adapter.get_collection_info()
        except Exception:
            pass

        index_metadata = {
            "source": source_name,
            "modality": modality,
            "strategy": c_config.strategy,
            "chunk_size": c_config.chunk_size,
            "chunk_overlap": c_config.chunk_overlap,
            "embedding_model": e_config.model_name,
            "dimensions": e_config.dimensions,
            "vector_db": collection_info.get("name", "active_db"),
            "num_chunks_indexed": len(chunks),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0"
        }

        self._save_index_metadata(index_metadata)

        return IngestionResult(
            source=source_name,
            modality=modality,
            num_raw_docs=len(raw_items),
            num_chunks=len(chunks),
            num_embedded=embedded_count,
            index_metadata=index_metadata,
            errors=errors
        )

    # -------------------------------------------------------------------------
    # Modality Processors
    # -------------------------------------------------------------------------
    def _process_pdf(self, source: Any, chunking_engine: ChunkingEngine) -> List[Dict[str, Any]]:
        """Extracts text from PDF pages and chunks with page-level tracking."""
        pages = []
        if isinstance(source, list):
            # Already extracted pages
            return chunking_engine.chunk_pdf(source)

        file_path = str(source)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        file_name = os.path.basename(file_path)

        # Try pypdf
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages, start=1):
                txt = page.extract_text() or ""
                if txt.strip():
                    pages.append({"page_number": page_num, "text": txt, "source": file_name})
        except ImportError:
            # Try pymupdf / fitz
            try:
                import fitz
                doc = fitz.open(file_path)
                for page_num in range(len(doc)):
                    txt = doc[page_num].get_text() or ""
                    if txt.strip():
                        pages.append({"page_number": page_num + 1, "text": txt, "source": file_name})
            except ImportError:
                # Fallback: read raw
                with open(file_path, "rb") as f:
                    raw = f.read().decode("latin-1", errors="ignore")
                    pages.append({"page_number": 1, "text": raw[:5000], "source": file_name})

        return chunking_engine.chunk_pdf(pages)

    def _process_excel(self, source: Any, chunking_engine: ChunkingEngine) -> List[Dict[str, Any]]:
        """Extracts records from Excel/CSV files and chunks them."""
        if isinstance(source, (dict, list)):
            return chunking_engine.chunk_excel(source)

        file_path = str(source)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Spreadsheet file not found: {file_path}")

        sheets_data = {}
        try:
            import pandas as pd
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
                sheets_data["CSV_Data"] = df.to_string(index=False)
            else:
                xl = pd.ExcelFile(file_path)
                for s_name in xl.sheet_names:
                    df = xl.parse(s_name)
                    sheets_data[s_name] = df.to_string(index=False)
        except Exception as e:
            logger.warning(f"Failed to parse with pandas: {e}. Reading text.")
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                sheets_data["Sheet1"] = f.read()

        return chunking_engine.chunk_excel(sheets_data)

    def _process_web(self, source: Any, chunking_engine: ChunkingEngine) -> List[Dict[str, Any]]:
        """Scrapes or formats web content."""
        if isinstance(source, dict):
            url = source.get("url", "web_page")
            content = source.get("content", source.get("text", ""))
            return chunking_engine.chunk_web(content, url)

        source_str = str(source)
        if source_str.startswith("http://") or source_str.startswith("https://"):
            url = source_str
            try:
                import requests
                from bs4 import BeautifulSoup
                resp = requests.get(url, timeout=10)
                soup = BeautifulSoup(resp.text, "html.parser")
                content = soup.get_text(separator="\n", strip=True)
            except Exception as e:
                content = f"Web content from {url} (Extraction failed: {e})"
            return chunking_engine.chunk_web(content, url)
        else:
            return chunking_engine.chunk_web(source_str, "web_snippet")

    def _process_image(self, source: Any, chunking_engine: ChunkingEngine) -> List[Dict[str, Any]]:
        """Processes image caption or description."""
        if isinstance(source, dict):
            caption = source.get("caption", source.get("description", ""))
            path = source.get("image_path", source.get("path"))
            return chunking_engine.chunk_image_caption(caption, path)
        return chunking_engine.chunk_image_caption(str(source))

    def _process_video(self, source: Any, chunking_engine: ChunkingEngine) -> List[Dict[str, Any]]:
        """Processes video transcripts with timestamps."""
        if isinstance(source, (list, dict)):
            transcript = source if isinstance(source, list) else source.get("transcript", [])
            timestamps = source.get("timestamps") if isinstance(source, dict) else None
            return chunking_engine.chunk_video_transcript(transcript, timestamps)
        return chunking_engine.chunk_video_transcript(str(source))
