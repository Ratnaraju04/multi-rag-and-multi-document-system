"""
FastAPI Server for Multi-Modal RAG Platform.
Provides high-performance REST APIs for React frontend:
- Multi-modal ingestion (Web, Excel, PDF, JSON/JSONL, Images, Videos)
- Real-time retrieval & generation with latency profiling
- 7 Advanced configuration modules management
- Side-by-side comparison lab & evaluation suite
- Static media serving for visual evidence & video keyframes
"""

import os
import sys

# Ensure Anaconda packages (chromadb, sentence-transformers, torch, cv2) are discoverable
for _pkg_path in [
    r"C:\Users\tanee\anaconda3\lib\site-packages",
    r"C:\ProgramData\anaconda3\lib\site-packages"
]:
    if os.path.exists(_pkg_path) and _pkg_path not in sys.path:
        sys.path.append(_pkg_path)

import time
import json
import uuid
import shutil
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Ensure upload and media directories exist
UPLOAD_DIR = os.path.abspath("./uploaded_user_data")
FRAMES_DIR = os.path.abspath("./extracted_frames")
SAMPLE_DIR = os.path.abspath("./colab_sample_data")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

# Import backend modules
from multimodal_rag_pipeline import MultimodalRAGEngine, RAGConfig, create_synthetic_multimodal_data
from src.chunking.chunking_engine import ChunkingEngine, ChunkingConfig
from src.embeddings.embedding_manager import EmbeddingManager, EmbeddingConfig, MODEL_REGISTRY
from src.vectordb.vector_db_manager import VectorDBManager
from src.vectordb.chroma_adapter import ChromaAdapter
from src.vectordb.faiss_adapter import FAISSAdapter
from src.vectordb.qdrant_adapter import QdrantAdapter
from src.llm.llm_manager import LLMManager, LLMConfig, CATALOG_MODELS
from src.llm.generation_config import GenerationConfig, PROVIDER_SUPPORT_MATRIX
from src.retrieval.retrieval_engine import RetrievalEngine, RetrievalConfig, RerankingConfig
from src.evaluation.eval_history import EvalHistoryManager
from src.evaluation.rag_evaluator import RAGEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_api")

app = FastAPI(title="Multi-Modal RAG Platform API", version="2.0.0")

# Enable CORS for local Vite development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------
class AppState:
    def __init__(self):
        self.rag_engine: Optional[MultimodalRAGEngine] = None
        self.vector_db_name = "ChromaDB"
        self.chunking_config = ChunkingConfig()
        self.embedding_config = EmbeddingConfig()
        self.llm_config = LLMConfig(
            provider="gemini",
            model_name="gemini-1.5-flash",
            api_key=os.environ.get("GEMINI_API_KEY", "")
        )
        self.gen_config = GenerationConfig()
        self.retrieval_config = RetrievalConfig()
        self.reranking_config = RerankingConfig()
        self.eval_history = EvalHistoryManager()

    def get_or_create_engine(self) -> MultimodalRAGEngine:
        if self.rag_engine is None:
            config = RAGConfig(api_key=os.environ.get("GEMINI_API_KEY", ""))
            self.rag_engine = MultimodalRAGEngine(config)
        return self.rag_engine

state = AppState()

# ---------------------------------------------------------------------------
# Media Serving Endpoint
# ---------------------------------------------------------------------------
@app.get("/media/{file_path:path}")
async def serve_media(file_path: str):
    """Safely serves media files (visual evidence, frames, samples)."""
    # Try direct absolute path, or relative to current workspace
    candidates = [
        file_path,
        os.path.abspath(file_path),
        os.path.join(FRAMES_DIR, os.path.basename(file_path)),
        os.path.join(SAMPLE_DIR, os.path.basename(file_path)),
        os.path.join(UPLOAD_DIR, os.path.basename(file_path)),
    ]
    for p in candidates:
        if os.path.exists(p) and os.path.isfile(p):
            return FileResponse(p)
    raise HTTPException(status_code=404, detail="Media file not found")

@app.get("/api/image")
async def get_image(path: str = Query(..., description="File path to image")):
    if os.path.exists(path) and os.path.isfile(path):
        return FileResponse(path)
    # Check by filename in known directories
    fname = os.path.basename(path)
    for d in [FRAMES_DIR, SAMPLE_DIR, UPLOAD_DIR]:
        check_p = os.path.join(d, fname)
        if os.path.exists(check_p):
            return FileResponse(check_p)
    raise HTTPException(status_code=404, detail=f"Image not found: {path}")

# ---------------------------------------------------------------------------
# Status & Pipeline Overview
# ---------------------------------------------------------------------------
@app.get("/api/status")
async def get_status():
    eng = state.get_or_create_engine()
    count = 0
    sources = []
    try:
        count = eng.vector_store.collection.count()
        sources = eng.vector_store.get_all_sources()
    except Exception as e:
        logger.warning(f"Error fetching vector store counts: {e}")

    return {
        "status": "online",
        "active_db": state.vector_db_name,
        "total_chunks": count,
        "sources": sources,
        "chunking_strategy": state.chunking_config.strategy,
        "chunk_size": state.chunking_config.chunk_size,
        "chunk_overlap": state.chunking_config.chunk_overlap,
        "embedding_model": state.embedding_config.model_name,
        "llm_provider": state.llm_config.provider,
        "llm_model": state.llm_config.model_name,
        "retrieval_top_k": state.retrieval_config.top_k,
        "reranker_enabled": state.reranking_config.enabled,
        "gemini_api_key_set": bool(os.environ.get("GEMINI_API_KEY"))
    }

# ---------------------------------------------------------------------------
# API Key Configuration
# ---------------------------------------------------------------------------
class APIKeyRequest(BaseModel):
    provider: str = "gemini"
    api_key: str

@app.post("/api/config/api-key")
async def set_api_key(req: APIKeyRequest):
    key = req.api_key.strip()
    if req.provider == "gemini":
        os.environ["GEMINI_API_KEY"] = key
        state.llm_config.api_key = key
        if state.rag_engine:
            state.rag_engine.config.api_key = key
            state.rag_engine.gemini_client = state.rag_engine._init_gemini()
    elif req.provider == "groq":
        os.environ["GROQ_API_KEY"] = key
    elif req.provider == "huggingface":
        os.environ["HF_TOKEN"] = key
    return {"success": True, "message": f"{req.provider.capitalize()} API Key updated"}

# ---------------------------------------------------------------------------
# Quick Demo: Load Synthetic Multimodal Data
# ---------------------------------------------------------------------------
@app.post("/api/load-sample-data")
async def load_sample_data():
    try:
        t0 = time.time()
        eng = state.get_or_create_engine()
        sample_data = create_synthetic_multimodal_data(SAMPLE_DIR)
        
        counts = {}
        if "excel" in sample_data and os.path.exists(sample_data["excel"]):
            counts["excel"] = eng.ingest_excel(sample_data["excel"])
        if "pdf" in sample_data and os.path.exists(sample_data["pdf"]):
            counts["pdf"] = eng.ingest_pdf(sample_data["pdf"])
        if "dataset" in sample_data and os.path.exists(sample_data["dataset"]):
            counts["dataset"] = eng.ingest_dataset(sample_data["dataset"], name="safety_benchmark")
        if "image" in sample_data and os.path.exists(sample_data["image"]):
            counts["image"] = eng.ingest_image(sample_data["image"])
        if "video" in sample_data and os.path.exists(sample_data["video"]):
            counts["video"] = eng.ingest_video(sample_data["video"], max_frames=6)
        if "web_url" in sample_data:
            counts["web"], _ = eng.ingest_url(sample_data["web_url"])

        total_indexed = eng.vector_store.collection.count()
        duration = round(time.time() - t0, 2)

        return {
            "success": True,
            "message": f"Successfully loaded sample demo data across all 5 modalities in {duration}s!",
            "counts": counts,
            "total_chunks": total_indexed,
            "sources": eng.vector_store.get_all_sources()
        }
    except Exception as e:
        logger.exception("Sample data ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# Custom Ingestion: Upload File (Excel, PDF, Dataset, Image, Video)
# ---------------------------------------------------------------------------
@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    file_type: str = Form("auto"),
    video_frames: int = Form(8)
):
    try:
        eng = state.get_or_create_engine()
        save_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        ext = os.path.splitext(file.filename)[1].lower()
        chunk_count = 0
        source_name = file.filename

        # Determine type by extension or file_type parameter
        if ext in [".xlsx", ".xls", ".csv"] or file_type == "excel":
            chunk_count = eng.ingest_excel(save_path)
        elif ext == ".pdf" or file_type == "pdf":
            chunk_count = eng.ingest_pdf(save_path)
        elif ext in [".json", ".jsonl"] or file_type == "dataset":
            chunk_count = eng.ingest_dataset(save_path, name=file.filename)
        elif ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"] or file_type == "image":
            chunk_count = eng.ingest_image(save_path)
        elif ext in [".mp4", ".avi", ".mov", ".mkv"] or file_type == "video":
            chunk_count = eng.ingest_video(save_path, max_frames=video_frames)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {ext}")

        total_chunks = eng.vector_store.collection.count()
        return {
            "success": True,
            "filename": file.filename,
            "indexed_chunks": chunk_count,
            "total_chunks": total_chunks,
            "sources": eng.vector_store.get_all_sources(),
            "message": f"Indexed {chunk_count} chunks from {file.filename}!"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload processing failed")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# Custom Ingestion: Web Scraper
# ---------------------------------------------------------------------------
class ScrapeRequest(BaseModel):
    url: str

@app.post("/api/scrape-url")
async def scrape_url(req: ScrapeRequest):
    if not req.url or not req.url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid URL format. Must start with http:// or https://")
    try:
        eng = state.get_or_create_engine()
        count, source_label = eng.ingest_url(req.url)
        return {
            "success": True,
            "source_label": source_label,
            "indexed_chunks": count,
            "total_chunks": eng.vector_store.collection.count(),
            "sources": eng.vector_store.get_all_sources(),
            "message": f"Indexed {count} chunks from {source_label}!"
        }
    except Exception as e:
        logger.exception("Web scraping failed")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# Reset Database Collection
# ---------------------------------------------------------------------------
@app.post("/api/reset-db")
async def reset_database():
    try:
        eng = state.get_or_create_engine()
        eng.vector_store.reset_collection()
        return {
            "success": True,
            "total_chunks": 0,
            "sources": [],
            "message": "Vector database collection successfully cleared."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# Chat Query Execution
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str
    source_filter: Optional[str] = None
    chat_history: Optional[List[Dict[str, Any]]] = None

@app.post("/api/chat")
async def execute_chat(req: ChatRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    eng = state.get_or_create_engine()
    total_docs = 0
    try:
        total_docs = eng.vector_store.collection.count()
    except Exception:
        pass

    if total_docs == 0:
        return {
            "answer": "⚠️ Vector store is currently empty! Please load the 1-click sample demo data or upload your documents/media from the Ingestion Hub first.",
            "visual_evidence": [],
            "sources": [],
            "latency_seconds": 0.0,
            "retrieval_latency_ms": 0.0,
            "llm_latency_ms": 0.0,
            "model_used": state.llm_config.model_name,
            "vector_db_used": state.vector_db_name
        }

    try:
        filter_src = None
        if req.source_filter and not req.source_filter.startswith("All Sources"):
            filter_src = req.source_filter

        logger.info(f"[API /chat] User query: '{req.question}' | Focus: '{filter_src}'")

        active_llm_mgr = None
        if state.llm_config.provider != "gemini" or state.llm_config.model_name != "gemini-1.5-flash":
            active_llm_mgr = LLMManager(state.llm_config)

        if hasattr(eng, "query_with_config"):
            result = eng.query_with_config(
                question=req.question,
                retrieval_config=state.retrieval_config,
                gen_config=state.gen_config,
                source_filter=filter_src,
                chat_history=req.chat_history or [],
                llm_manager=active_llm_mgr,
                reranking_config=state.reranking_config
            )
            # Smart fallback: If source_filter returned 0 results, fall back to searching all sources
            if filter_src and not result.get("sources"):
                fallback_res = eng.query_with_config(
                    question=req.question,
                    retrieval_config=state.retrieval_config,
                    gen_config=state.gen_config,
                    source_filter=None,
                    chat_history=req.chat_history or [],
                    llm_manager=active_llm_mgr,
                    reranking_config=state.reranking_config
                )
                if fallback_res.get("sources"):
                    result = fallback_res
                    result["answer"] = f"> ℹ️ *No matches found inside locked focus '{filter_src}'. Showing best answer from all indexed sources:*\n\n" + result.get("answer", "")
        else:
            result = eng.query(
                question=req.question,
                top_k=state.retrieval_config.top_k,
                source_filter=filter_src,
                chat_history=req.chat_history or []
            )
            if filter_src and not result.get("sources"):
                fallback_res = eng.query(
                    question=req.question,
                    top_k=state.retrieval_config.top_k,
                    source_filter=None,
                    chat_history=req.chat_history or []
                )
                if fallback_res.get("sources"):
                    result = fallback_res
                    result["answer"] = f"> ℹ️ *No matches found inside locked focus '{filter_src}'. Showing best answer from all indexed sources:*\n\n" + result.get("answer", "")

        answer = result.get("answer", "No response generated.")
        raw_visuals = result.get("visual_evidence", [])
        sources = result.get("sources", [])
        latency = result.get("latency_seconds", 0.0)
        ret_latency = result.get("retrieval_latency_ms", 0.0)
        llm_latency = result.get("llm_latency_ms", 0.0)
        model_used = result.get("model_used", state.llm_config.model_name)

        # Normalize visual paths to accessible URLs
        visual_urls = []
        image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
        for v in raw_visuals:
            if isinstance(v, str) and os.path.exists(v) and os.path.splitext(v)[1].lower() in image_extensions:
                # Provide direct API image URL
                visual_urls.append({
                    "path": v,
                    "filename": os.path.basename(v),
                    "url": f"/api/image?path={v}"
                })

        return {
            "answer": answer,
            "visual_evidence": visual_urls,
            "sources": sources,
            "latency_seconds": latency,
            "retrieval_latency_ms": ret_latency,
            "llm_latency_ms": llm_latency,
            "model_used": model_used,
            "vector_db_used": state.vector_db_name
        }
    except Exception as e:
        logger.exception("Error processing chat query")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# Module 1: Chunking Settings & Live Preview
# ---------------------------------------------------------------------------
class ChunkingSettingsUpdate(BaseModel):
    strategy: str = "RecursiveCharacterTextSplitter"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    min_chunk_size: int = 100
    preserve_separators: bool = True
    similarity_threshold: float = 0.5

@app.post("/api/config/chunking")
async def update_chunking_config(cfg: ChunkingSettingsUpdate):
    state.chunking_config = ChunkingConfig(
        strategy=cfg.strategy,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        min_chunk_size=cfg.min_chunk_size,
        preserve_separators=cfg.preserve_separators,
        similarity_threshold=cfg.similarity_threshold
    )
    return {"success": True, "config": state.chunking_config.__dict__}

class ChunkPreviewRequest(BaseModel):
    text: str
    strategy: str = "RecursiveCharacterTextSplitter"
    chunk_size: int = 1000
    chunk_overlap: int = 150

@app.post("/api/preview-chunks")
async def preview_chunks(req: ChunkPreviewRequest):
    cfg = ChunkingConfig(
        strategy=req.strategy,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap
    )
    engine = ChunkingEngine(cfg)
    chunks = engine.preview_chunks(req.text)
    return {
        "strategy": req.strategy,
        "total_chunks": len(chunks),
        "chunks": [{"index": i + 1, "chars": len(c), "text": c} for i, c in enumerate(chunks)]
    }

# ---------------------------------------------------------------------------
# Module 2: Vector DB Hub
# ---------------------------------------------------------------------------
class VectorDBUpdate(BaseModel):
    db_name: str
    cloud_url: Optional[str] = None
    cloud_key: Optional[str] = None

@app.post("/api/config/vector-db")
async def set_vector_db(req: VectorDBUpdate):
    state.vector_db_name = req.db_name
    return {"success": True, "active_db": req.db_name}

@app.post("/api/test/db")
async def test_db_connection(req: VectorDBUpdate):
    db = req.db_name
    try:
        if db == "ChromaDB":
            ad = ChromaAdapter()
            res = ad.test_connection()
            return {"success": res.get("success", True), "message": f"ChromaDB ready: {res.get('document_count', 0)} chunks"}
        elif db == "FAISS":
            ad = FAISSAdapter()
            res = ad.test_connection()
            return {"success": True, "message": f"FAISS Local Flat L2 Index Ready — {res.get('message', '')}"}
        elif db == "Qdrant":
            ad = QdrantAdapter(url=req.cloud_url or None, api_key=req.cloud_key or None)
            res = ad.test_connection()
            return {"success": res.get("success", False), "message": f"Qdrant ({res.get('mode', 'local')}) — {res.get('message', '')}"}
        else:
            return {"success": False, "message": f"{db} requires remote cloud credentials and deployment."}
    except Exception as e:
        return {"success": False, "message": str(e)}

# ---------------------------------------------------------------------------
# Module 3: Embedding Model Hub
# ---------------------------------------------------------------------------
@app.get("/api/models/embeddings")
async def get_embedding_models():
    return {"registry": MODEL_REGISTRY}

class EmbeddingTestRequest(BaseModel):
    model_name: str

@app.post("/api/test/embedding")
async def test_embedding(req: EmbeddingTestRequest):
    try:
        t0 = time.time()
        meta = MODEL_REGISTRY.get(req.model_name, {})
        cfg = EmbeddingConfig(
            model_name=req.model_name,
            provider=meta.get("provider", "local"),
            dimensions=meta.get("dimensions", 384),
            execution_mode=meta.get("execution_mode", "local")
        )
        state.embedding_config = cfg
        mgr = EmbeddingManager(cfg)
        vec = mgr.embed_query("Multi-Modal RAG Platform validation")
        latency = round((time.time() - t0) * 1000.0, 1)
        return {
            "success": True,
            "dimensions": len(vec),
            "latency_ms": latency,
            "message": f"Generated {len(vec)}d vector in {latency}ms"
        }
    except Exception as e:
        return {"success": False, "message": str(e)}

# ---------------------------------------------------------------------------
# Module 4: Multi-LLM Providers
# ---------------------------------------------------------------------------
@app.get("/api/models/llms")
async def get_catalog_models():
    return {"catalog": CATALOG_MODELS, "support_matrix": PROVIDER_SUPPORT_MATRIX}

class LLMUpdate(BaseModel):
    provider: str
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = "http://localhost:11434"

@app.post("/api/config/llm")
async def set_llm_config(req: LLMUpdate):
    state.llm_config = LLMConfig(
        provider=req.provider,
        model_name=req.model_name,
        api_key=req.api_key or os.environ.get(f"{req.provider.upper()}_API_KEY", ""),
        base_url=req.base_url or "http://localhost:11434",
        is_local=(req.provider == "ollama")
    )
    return {"success": True, "llm": state.llm_config.__dict__}

@app.post("/api/test/llm")
async def test_llm_connection(req: LLMUpdate):
    try:
        cfg = LLMConfig(
            provider=req.provider,
            model_name=req.model_name,
            api_key=req.api_key or os.environ.get(f"{req.provider.upper()}_API_KEY", ""),
            base_url=req.base_url or "http://localhost:11434",
            is_local=(req.provider == "ollama")
        )
        mgr = LLMManager(cfg)
        res = mgr.test_connection()
        return res
    except Exception as e:
        return {"success": False, "message": str(e)}

# ---------------------------------------------------------------------------
# Module 5 & 6: Generation, Retrieval, Reranking
# ---------------------------------------------------------------------------
class GenParamsUpdate(BaseModel):
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: Optional[int] = 40
    max_output_tokens: int = 2048
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

@app.post("/api/config/generation")
async def set_gen_params(req: GenParamsUpdate):
    state.gen_config = GenerationConfig(
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        max_output_tokens=req.max_output_tokens,
        frequency_penalty=req.frequency_penalty,
        presence_penalty=req.presence_penalty
    )
    return {"success": True, "generation_config": state.gen_config.__dict__}

class RetrievalParamsUpdate(BaseModel):
    top_k: int = 5
    similarity_threshold: float = 0.5
    search_type: str = "similarity"
    mmr_lambda: float = 0.5
    max_context_tokens: int = 4096
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = 3

@app.post("/api/config/retrieval")
async def set_retrieval_params(req: RetrievalParamsUpdate):
    state.retrieval_config = RetrievalConfig(
        top_k=req.top_k,
        similarity_threshold=req.similarity_threshold,
        search_type=req.search_type,
        mmr_lambda=req.mmr_lambda,
        max_context_tokens=req.max_context_tokens
    )
    state.reranking_config = RerankingConfig(
        enabled=req.reranker_enabled,
        model=req.reranker_model,
        top_n=req.reranker_top_n
    )
    return {
        "success": True,
        "retrieval_config": state.retrieval_config.__dict__,
        "reranking_config": state.reranking_config.__dict__
    }

# ---------------------------------------------------------------------------
# Comparison Lab
# ---------------------------------------------------------------------------
class CompareConfigRequest(BaseModel):
    label: str
    provider: str
    model_name: str
    top_k: int = 5
    reranker_enabled: bool = False

class CompareLabRequest(BaseModel):
    questions: List[str]
    configs: List[CompareConfigRequest]

@app.post("/api/compare")
async def run_comparison(req: CompareLabRequest):
    eng = state.get_or_create_engine()
    results = []

    for q in req.questions:
        q_entry = {"question": q, "runs": []}
        for cfg in req.configs:
            t0 = time.time()
            llm_cfg = LLMConfig(
                provider=cfg.provider,
                model_name=cfg.model_name,
                api_key=os.environ.get(f"{cfg.provider.upper()}_API_KEY", "")
            )
            llm_mgr = LLMManager(llm_cfg)
            ret_cfg = RetrievalConfig(top_k=cfg.top_k)
            rerank_cfg = RerankingConfig(enabled=cfg.reranker_enabled)

            try:
                res = eng.query_with_config(
                    question=q,
                    retrieval_config=ret_cfg,
                    gen_config=state.gen_config,
                    llm_manager=llm_mgr,
                    reranking_config=rerank_cfg
                )
                q_entry["runs"].append({
                    "config_label": cfg.label,
                    "model": cfg.model_name,
                    "answer": res.get("answer", ""),
                    "latency_seconds": res.get("latency_seconds", round(time.time() - t0, 2)),
                    "num_sources": len(res.get("sources", []))
                })
            except Exception as e:
                q_entry["runs"].append({
                    "config_label": cfg.label,
                    "model": cfg.model_name,
                    "answer": f"Error: {str(e)}",
                    "latency_seconds": 0.0,
                    "num_sources": 0
                })
        results.append(q_entry)

    return {"benchmark_results": results}

# ---------------------------------------------------------------------------
# Evaluation Suite
# ---------------------------------------------------------------------------
@app.get("/api/eval/history")
async def get_eval_history():
    try:
        hist = state.eval_history.load_history()
        return {"history": hist}
    except Exception as e:
        return {"history": [], "error": str(e)}

class EvalRunRequest(BaseModel):
    mode: str = "Ground-Truth"
    questions: List[str] = []
    expected_answers: List[str] = []

@app.post("/api/eval/run")
async def run_evaluation(req: EvalRunRequest):
    try:
        evaluator = RAGEvaluator()
        eng = state.get_or_create_engine()
        results = []

        total_corr = 0.0
        total_prec = 0.0
        total_faith = 0.0

        for i, q in enumerate(req.questions):
            ans_res = eng.query(q, top_k=state.retrieval_config.top_k)
            ans = ans_res.get("answer", "")
            sources = ans_res.get("sources", [])
            context = "\n".join([s.get("content", "") for s in sources])
            expected = req.expected_answers[i] if i < len(req.expected_answers) else ""

            eval_res = evaluator.evaluate(
                query=q,
                response=ans,
                retrieved_context=context,
                ground_truth=expected if req.mode == "Ground-Truth" else None
            )

            results.append({
                "question": q,
                "answer": ans,
                "expected": expected,
                "correctness": eval_res.correctness_score,
                "faithfulness": eval_res.faithfulness_score,
                "context_precision": eval_res.context_precision_score,
                "latency_ms": eval_res.latency_ms
            })
            total_corr += eval_res.correctness_score
            total_prec += eval_res.context_precision_score
            total_faith += eval_res.faithfulness_score

        n = max(len(results), 1)
        aggregate = {
            "avg_answer_correctness": round(total_corr / n, 2),
            "avg_context_precision": round(total_prec / n, 2),
            "avg_faithfulness": round(total_faith / n, 2),
            "total_evaluated": len(results)
        }

        # Save to eval history
        run_record = {
            "run_id": str(uuid.uuid4())[:8],
            "mode": req.mode,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "aggregate_metrics": aggregate,
            "detailed_results": results
        }
        state.eval_history.save_run(run_record)

        return {
            "success": True,
            "aggregate_metrics": aggregate,
            "results": results
        }
    except Exception as e:
        logger.exception("Evaluation run failed")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# Mount React Production Build (if built)
# ---------------------------------------------------------------------------
FRONTEND_DIST = os.path.abspath("./frontend/dist")
if os.path.exists(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
