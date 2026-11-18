"""
Central Platform Engine for Advanced Multi-Modal & Multi-Vector RAG.
Orchestrates configurable chunking, embeddings, vector database adapters,
retrieval + reranking, multi-provider LLM generation, and evaluation tracking.
"""

import os
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union
from PIL import Image

from src.chunking.chunking_engine import ChunkingEngine, ChunkingConfig
from src.embeddings.embedding_manager import EmbeddingManager, EmbeddingConfig
from src.vectordb.vector_db_manager import VectorDBManager
from src.vectordb.base_adapter import VectorDBAdapter
from src.llm.llm_manager import LLMManager, LLMConfig
from src.llm.generation_config import GenerationConfig
from src.retrieval.retrieval_engine import RetrievalEngine, RetrievalConfig, RerankingConfig
from src.evaluation.rag_evaluator import RAGEvaluator, EvalResult
from src.evaluation.eval_history import EvalHistoryManager
from src.core.ingestion_manager import IngestionManager, IngestionResult

try:
    from src.loaders.web_scraper import WebScraperLoader
except Exception:
    WebScraperLoader = None

try:
    from src.loaders.excel_loader import ExcelDatasetLoader
except Exception:
    ExcelDatasetLoader = None

try:
    from src.loaders.dataset_loader import DatasetLoader
except Exception:
    DatasetLoader = None

try:
    from src.loaders.image_loader import ImageLoader
except Exception:
    ImageLoader = None

try:
    from src.loaders.video_loader import VideoLoader
except Exception:
    VideoLoader = None

try:
    from src.loaders.pdf_loader import PDFLoader
except Exception:
    PDFLoader = None


logger = logging.getLogger(__name__)

@dataclass
class PlatformConfig:
    chunking_config: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_db_name: str = "chroma"
    vector_db_config: Dict[str, Any] = field(default_factory=dict)
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    retrieval_config: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranking_config: RerankingConfig = field(default_factory=RerankingConfig)

@dataclass
class QueryResult:
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    visual_evidence: List[str]
    retrieval_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float
    tokens_used: int
    num_retrieved: int
    num_after_rerank: int
    model_used: str
    provider_used: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "visual_evidence": self.visual_evidence,
            "latency_seconds": round(self.total_latency_ms / 1000.0, 3),
            "retrieval_latency_ms": round(self.retrieval_latency_ms, 1),
            "llm_latency_ms": round(self.llm_latency_ms, 1),
            "tokens_used": self.tokens_used,
            "num_retrieved": self.num_retrieved,
            "num_after_rerank": self.num_after_rerank,
            "model_used": self.model_used,
            "provider_used": self.provider_used,
            "timestamp": self.timestamp
        }

class PlatformEngine:
    def __init__(self, config: Optional[PlatformConfig] = None):
        self.config = config or PlatformConfig()

        # Initialize sub-systems
        self.chunking_engine = ChunkingEngine(self.config.chunking_config)
        self.embedding_manager = EmbeddingManager(self.config.embedding_config)
        self.vector_db_manager = VectorDBManager(default_adapter=self.config.vector_db_name)
        
        # Configure active adapter if specified
        if self.config.vector_db_name:
            self.vector_db_manager.set_active_adapter(self.config.vector_db_name)

        self.llm_manager = LLMManager(self.config.llm_config)
        self.retrieval_engine = RetrievalEngine()
        self.eval_history = EvalHistoryManager()
        self.evaluator = RAGEvaluator(llm_manager=self.llm_manager)
        self.ingestion_manager = IngestionManager()

        # Loaders for multi-modal sources
        self.web_loader = WebScraperLoader() if WebScraperLoader else None
        self.excel_loader = ExcelDatasetLoader() if ExcelDatasetLoader else None
        self.dataset_loader = DatasetLoader() if DatasetLoader else None
        self.image_loader = ImageLoader() if ImageLoader else None
        self.video_loader = VideoLoader() if VideoLoader else None
        self.pdf_loader = PDFLoader() if PDFLoader else None


    # -------------------------------------------------------------
    # Dynamic Configuration Updaters
    # -------------------------------------------------------------
    def update_chunking_config(self, cfg: ChunkingConfig):
        self.config.chunking_config = cfg
        self.chunking_engine = ChunkingEngine(cfg)

    def update_embedding_config(self, cfg: EmbeddingConfig):
        self.config.embedding_config = cfg
        self.embedding_manager = EmbeddingManager(cfg)

    def update_llm_config(self, cfg: LLMConfig):
        self.config.llm_config = cfg
        self.llm_manager = LLMManager(cfg)
        self.evaluator = RAGEvaluator(llm_manager=self.llm_manager)

    def update_vector_db(self, db_name: str) -> Dict[str, Any]:
        result = self.vector_db_manager.set_active_adapter(db_name)
        if result.get("success", False):
            self.config.vector_db_name = db_name
        return result

    # -------------------------------------------------------------
    # Query Execution with Full Pipeline (Retrieval, Rerank, LLM)
    # -------------------------------------------------------------
    def query(
        self,
        question: str,
        retrieval_config: Optional[RetrievalConfig] = None,
        reranking_config: Optional[RerankingConfig] = None,
        gen_config: Optional[GenerationConfig] = None,
        source_filter: Optional[str] = None,
        modality_filter: Optional[str] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        query_image_path: Optional[str] = None
    ) -> QueryResult:
        start_total = time.time()
        r_cfg = retrieval_config or self.config.retrieval_config
        rerank_cfg = reranking_config or self.config.reranking_config
        g_cfg = gen_config or self.config.generation_config

        # Contextual query enrichment for follow-ups
        search_query = question
        target_source = source_filter if (source_filter and not source_filter.startswith("All Sources")) else None
        if target_source:
            search_query = f"{question} [Source: {target_source}]"

        # 1. Retrieval
        start_retrieval = time.time()
        active_adapter = self.vector_db_manager.get_active_adapter()

        # Generate query embedding
        try:
            query_embedding = self.embedding_manager.embed_query(search_query)
        except Exception as e:
            logger.warning(f"Embedding query failed, falling back to empty vector: {e}")
            query_embedding = [0.0] * self.embedding_manager.config.dimensions

        # Retrieve documents
        filter_dict = {}
        if target_source:
            filter_dict["source"] = target_source
        if modality_filter:
            filter_dict["modality"] = modality_filter

        retrieved_docs = self.retrieval_engine.retrieve(
            query=search_query,
            query_embedding=query_embedding,
            vector_adapter=active_adapter,
            retrieval_config=r_cfg,
            source_filter=target_source,
            modality_filter=modality_filter
        )
        retrieval_latency_ms = (time.time() - start_retrieval) * 1000.0
        num_retrieved = len(retrieved_docs)

        # 2. Reranking
        num_after_rerank = num_retrieved
        if rerank_cfg and rerank_cfg.enabled and retrieved_docs:
            retrieved_docs = self.retrieval_engine.rerank(
                query=question,
                docs=retrieved_docs,
                reranking_config=rerank_cfg
            )
            num_after_rerank = len(retrieved_docs)

        if not retrieved_docs:
            total_latency_ms = (time.time() - start_total) * 1000.0
            return QueryResult(
                question=question,
                answer="No relevant documents found matching your query criteria and similarity threshold.",
                sources=[],
                visual_evidence=[],
                retrieval_latency_ms=retrieval_latency_ms,
                llm_latency_ms=0.0,
                total_latency_ms=total_latency_ms,
                tokens_used=0,
                num_retrieved=0,
                num_after_rerank=0,
                model_used=self.config.llm_config.model_name,
                provider_used=self.config.llm_config.provider
            )

        # 3. Assemble Context & Visual Evidence
        context_blocks = []
        visual_evidence_paths = []
        visual_images = []

        image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

        for idx, item in enumerate(retrieved_docs, start=1):
            meta = item.get("metadata", {})
            modality = meta.get("modality", "text")
            src_name = meta.get("source", "unknown")
            timestamp = meta.get("timestamp", "")
            page_num = meta.get("page_number", "")
            sheet_name = meta.get("sheet_name", "")

            header = f"[Source #{idx} | Modality: {modality} | Source: {src_name}"
            if page_num:
                header += f" | Page: {page_num}"
            if sheet_name:
                header += f" | Sheet: {sheet_name}"
            if timestamp:
                header += f" | Timestamp: {timestamp}"
            header += "]"

            content_text = item.get("content", "")
            context_blocks.append(f"{header}\n{content_text}\n")

            # Check for image or video frame attachment
            img_path = meta.get("frame_path")
            if not img_path and (modality == "image" or meta.get("type") == "image"):
                img_path = meta.get("file_path")

            if img_path and os.path.exists(img_path) and os.path.splitext(img_path)[1].lower() in image_extensions:
                if img_path not in visual_evidence_paths:
                    visual_evidence_paths.append(img_path)
                    try:
                        visual_images.append(Image.open(img_path))
                    except Exception:
                        pass

        combined_context = "\n----------------------------------------\n".join(context_blocks)

        # 4. Formulate Prompt & Generate via LLM Manager
        start_llm = time.time()
        system_instructions = (
            "You are an expert Multi-Modal Retrieval-Augmented Generation (RAG) assistant. "
            "You have access to retrieved multimodal context containing scraped web pages, "
            "Excel spreadsheets, datasets, images, and timestamped video keyframes.\n\n"
            "INSTRUCTIONS:\n"
            "1. Answer the user's question accurately using ONLY the provided multimodal context.\n"
            "2. When citing videos, always mention the specific filename and TIMESTAMP.\n"
            "3. When citing Excel/tabular data, cite sheet name, columns, and row details.\n"
            "4. When citing images, reference the image file name and visual features.\n"
            "5. If the context does not contain sufficient information, state what is missing honestly.\n"
            "6. Answer clearly, professionally, and factually."
        )

        prompt_text = (
            f"{system_instructions}\n\n"
            f"=== RETRIEVED MULTIMODAL CONTEXT ===\n"
            f"{combined_context}\n\n"
            f"=== USER QUESTION ===\n"
            f"{question}\n\n"
            f"=== YOUR DETAILED MULTIMODAL ANSWER ==="
        )

        gen_resp = self.llm_manager.generate(
            prompt=prompt_text,
            context=combined_context,
            gen_config=g_cfg
        )

        llm_latency_ms = gen_resp.get("latency_ms", (time.time() - start_llm) * 1000.0)
        total_latency_ms = (time.time() - start_total) * 1000.0

        return QueryResult(
            question=question,
            answer=gen_resp.get("text", "Error generating response."),
            sources=retrieved_docs,
            visual_evidence=visual_evidence_paths,
            retrieval_latency_ms=retrieval_latency_ms,
            llm_latency_ms=llm_latency_ms,
            total_latency_ms=total_latency_ms,
            tokens_used=gen_resp.get("tokens_used", 0),
            num_retrieved=num_retrieved,
            num_after_rerank=num_after_rerank,
            model_used=self.config.llm_config.model_name,
            provider_used=self.config.llm_config.provider
        )

    # -------------------------------------------------------------
    # Ingestion Wrappers with Chunking Engine Integration
    # -------------------------------------------------------------
    def ingest_url(self, url: str) -> tuple[int, str]:
        """Scrapes web page, chunks with ChunkingEngine, and embeds into active DB."""
        docs, readable_source = self.web_loader.scrape_url(url)
        # Apply ChunkingEngine to docs
        chunked_docs = []
        for d in docs:
            text = d.get("content", "")
            meta = d.get("metadata", {})
            sub_chunks = self.chunking_engine.chunk_text(text, meta)
            chunked_docs.extend(sub_chunks)

        self._store_documents(chunked_docs)
        return len(chunked_docs), readable_source

    def ingest_excel(self, file_path: str) -> int:
        """Loads spreadsheet, applies chunking config, and stores into active DB."""
        docs = self.excel_loader.load(file_path)
        chunked_docs = []
        for d in docs:
            text = d.get("content", "")
            meta = d.get("metadata", {})
            sub_chunks = self.chunking_engine.chunk_text(text, meta)
            chunked_docs.extend(sub_chunks)

        self._store_documents(chunked_docs)
        return len(chunked_docs)

    def ingest_pdf(self, file_path: str) -> int:
        """Parses PDF, chunks with ChunkingEngine, and indexes into active DB."""
        docs = self.pdf_loader.load(file_path)
        chunked_docs = []
        for d in docs:
            text = d.get("content", "")
            meta = d.get("metadata", {})
            sub_chunks = self.chunking_engine.chunk_text(text, meta)
            chunked_docs.extend(sub_chunks)

        self._store_documents(chunked_docs)
        return len(chunked_docs)

    def ingest_dataset(self, source: Union[str, List[Dict[str, Any]]], name: str = "dataset") -> int:
        """Loads dataset records and indexes into active DB."""
        if isinstance(source, str):
            docs = self.dataset_loader.load_from_file(source)
        else:
            docs = self.dataset_loader.load_records(source, dataset_name=name)

        chunked_docs = []
        for d in docs:
            text = d.get("content", "")
            meta = d.get("metadata", {})
            sub_chunks = self.chunking_engine.chunk_text(text, meta)
            chunked_docs.extend(sub_chunks)

        self._store_documents(chunked_docs)
        return len(chunked_docs)

    def ingest_image(self, path: str) -> int:
        """Extracts OCR/visual features and indexes image."""
        if os.path.isdir(path):
            docs = self.image_loader.load_directory(path)
        else:
            docs = [self.image_loader.load_image(path)]

        self._store_documents(docs)
        return len(docs)

    def ingest_video(self, path: str, max_frames: int = 8) -> int:
        """Extracts keyframes and captions from video."""
        docs = self.video_loader.load_video(path, max_frames=max_frames)
        self._store_documents(docs)
        return len(docs)

    def _store_documents(self, documents: List[Dict[str, Any]]):
        """Helper to embed and store docs in active vector db adapter."""
        if not documents:
            return

        active_adapter = self.vector_db_manager.get_active_adapter()

        # Compute embeddings via EmbeddingManager
        texts = [doc.get("content", "") for doc in documents]
        embeddings = self.embedding_manager.embed_texts(texts)

        # Upsert into adapter
        active_adapter.add_documents(documents, embeddings)

        # Persist index metadata
        index_meta = {
            "chunking_strategy": self.config.chunking_config.strategy,
            "chunk_size": self.config.chunking_config.chunk_size,
            "chunk_overlap": self.config.chunking_config.chunk_overlap,
            "embedding_model": self.config.embedding_config.model_name,
            "embedding_dimensions": self.config.embedding_config.dimensions,
            "vector_db": self.vector_db_manager.get_active_adapter_name(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_documents": len(documents)
        }
        self.ingestion_manager._save_index_metadata(index_meta)

    # -------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------
    def run_evaluation(
        self,
        questions: List[str],
        expected_answers: Optional[List[str]] = None,
        source_docs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Runs batch evaluation over provided questions and saves run to history."""
        test_cases = []
        for i, q in enumerate(questions):
            # Run query through RAG pipeline
            q_res = self.query(q)
            exp = expected_answers[i] if (expected_answers and i < len(expected_answers)) else None
            contexts = [s.get("content", "") for s in q_res.sources]

            test_cases.append({
                "question": q,
                "generated_answer": q_res.answer,
                "expected_answer": exp,
                "retrieved_contexts": contexts,
                "latency_ms": {
                    "retrieval": q_res.retrieval_latency_ms,
                    "llm_generation": q_res.llm_latency_ms,
                    "total": q_res.total_latency_ms
                }
            })

        results = self.evaluator.evaluate_batch(test_cases)
        aggregate = self.evaluator.compute_aggregate_metrics(results)

        # Save session to history
        session_data = {
            "llm_provider": self.config.llm_config.provider,
            "llm_model": self.config.llm_config.model_name,
            "embedding_model": self.config.embedding_config.model_name,
            "vector_db": self.vector_db_manager.get_active_adapter_name(),
            "chunking_strategy": self.config.chunking_config.strategy,
            "chunk_size": self.config.chunking_config.chunk_size,
            "num_test_questions": len(questions),
            "aggregate_metrics": aggregate,
            "results": [r.to_dict() if hasattr(r, "to_dict") else r.__dict__ for r in results]
        }
        session_id = self.eval_history.save_session(session_data)

        return {
            "session_id": session_id,
            "aggregate_metrics": aggregate,
            "results": results
        }
