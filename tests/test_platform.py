"""
Automated Comprehensive Test Suite for Multi-Modal RAG Platform.
Validates Chunking, Embeddings, Vector DBs, LLM Management, Retrieval/Reranking, Evaluation, and Pipeline Orchestration.
"""

import pytest
import os
import shutil

from src.chunking.chunking_engine import ChunkingEngine, ChunkingConfig
from src.embeddings.embedding_manager import EmbeddingManager, EmbeddingConfig, MODEL_REGISTRY
from src.vectordb.chroma_adapter import ChromaAdapter
from src.vectordb.faiss_adapter import FAISSAdapter
from src.vectordb.vector_db_manager import VectorDBManager
from src.llm.llm_manager import LLMManager, LLMConfig, CATALOG_MODELS
from src.llm.generation_config import GenerationConfig, PROVIDER_SUPPORT_MATRIX
from src.retrieval.retrieval_engine import RetrievalEngine, RetrievalConfig, RerankingConfig
from src.evaluation.rag_evaluator import RAGEvaluator, EvalResult
from src.evaluation.eval_history import EvalHistoryManager
from src.core.platform_engine import PlatformEngine, PlatformConfig

# =========================================================================
# MODULE 1: CHUNKING ENGINE TESTS
# =========================================================================

def test_recursive_chunking():
    cfg = ChunkingConfig(strategy="RecursiveCharacterTextSplitter", chunk_size=300, chunk_overlap=30)
    engine = ChunkingEngine(cfg)
    sample_text = "RAG architecture unites retrieval systems with LLMs. " * 30
    chunks = engine.chunk_text(sample_text, {"source": "test_doc.txt", "modality": "text"})
    assert len(chunks) > 1
    assert all("chunk_index" in c["metadata"] for c in chunks)
    assert all("total_chunks" in c["metadata"] for c in chunks)
    assert chunks[0]["metadata"]["strategy"] == "RecursiveCharacterTextSplitter"

def test_character_chunking():
    cfg = ChunkingConfig(strategy="CharacterTextSplitter", chunk_size=200, chunk_overlap=20)
    engine = ChunkingEngine(cfg)
    text = "Paragraph one with some information.\n\nParagraph two with additional facts.\n\nParagraph three."
    chunks = engine.chunk_text(text, {"source": "notes.md"})
    assert len(chunks) >= 1

def test_token_chunking():
    cfg = ChunkingConfig(strategy="TokenTextSplitter", chunk_size=50, chunk_overlap=10)
    engine = ChunkingEngine(cfg)
    text = "Artificial intelligence models require precise prompt engineering and structured context windows." * 10
    chunks = engine.chunk_text(text, {"source": "ai.txt"})
    assert len(chunks) > 1

def test_markdown_chunking():
    cfg = ChunkingConfig(strategy="MarkdownHeaderTextSplitter", chunk_size=400, chunk_overlap=50)
    engine = ChunkingEngine(cfg)
    md_text = "# Header 1\nSection 1 text here.\n\n## Header 2\nSection 2 detailed information."
    chunks = engine.chunk_text(md_text, {"source": "readme.md"})
    assert len(chunks) >= 1

def test_chunking_preview_and_stats():
    cfg = ChunkingConfig(strategy="RecursiveCharacterTextSplitter", chunk_size=250, chunk_overlap=25)
    engine = ChunkingEngine(cfg)
    text = "Multi-modal retrieval processes text, tables, images, and videos concurrently." * 15
    preview = engine.preview_chunks(text)
    assert len(preview) > 0
    chunks = engine.chunk_text(text, {"source": "demo"})
    stats = engine.get_chunk_stats(chunks)
    assert stats["total_chunks"] == len(chunks)
    assert stats["avg_chunk_size"] > 0

# =========================================================================
# MODULE 2 & 3: VECTOR DB & EMBEDDING MANAGER TESTS
# =========================================================================

def test_embedding_model_registry():
    assert "sentence-transformers/all-MiniLM-L6-v2" in MODEL_REGISTRY
    assert MODEL_REGISTRY["sentence-transformers/all-MiniLM-L6-v2"]["dimensions"] == 384
    assert MODEL_REGISTRY["BAAI/bge-base-en-v1.5"]["dimensions"] == 768

def test_embedding_manager_dimension_validation():
    cfg = EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2", dimensions=384)
    mgr = EmbeddingManager(cfg)
    vec = mgr.embed_query("Hello world")
    assert len(vec) == 384

def test_faiss_adapter_lifecycle():
    test_dir = "./test_faiss_store"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    adapter = FAISSAdapter(index_dir=test_dir)
    test_docs = [
        {"id": "doc1", "content": "Financial statement for Q2", "metadata": {"source": "fin.csv"}},
        {"id": "doc2", "content": "System architecture guidelines", "metadata": {"source": "arch.pdf"}}
    ]

    # Synthetic 384d normalized vectors
    import numpy as np
    v1 = np.ones(384, dtype=np.float32) / np.sqrt(384)
    v2 = np.zeros(384, dtype=np.float32)
    v2[0] = 1.0
    embeddings = [v1.tolist(), v2.tolist()]

    adapter.add_documents(test_docs, embeddings)
    info = adapter.get_collection_info()
    assert info["count"] == 2
    assert info["dimensions"] == 384

    # Search with v1
    res = adapter.search(v1.tolist(), top_k=1)
    assert len(res) == 1
    assert res[0]["id"] == "doc1"

    # Clean up test store
    if os.path.exists("./test_faiss_store"):
        shutil.rmtree("./test_faiss_store")

def test_vector_db_manager_switching():
    v_mgr = VectorDBManager(default_adapter="chroma")
    assert v_mgr.get_active_adapter_name() == "chroma"
    res = v_mgr.set_active_adapter("faiss")
    assert res["success"] is True
    assert v_mgr.get_active_adapter_name() == "faiss"

# =========================================================================
# MODULE 4 & 5: LLM MANAGER & GENERATION PARAMETERS
# =========================================================================

def test_generation_config_validation():
    cfg = GenerationConfig(temperature=0.2, top_p=0.9, max_output_tokens=2048)
    errors = cfg.validate("gemini")
    assert len(errors) == 0

    # Test invalid parameter range
    bad_cfg = GenerationConfig(temperature=3.5)
    bad_errors = bad_cfg.validate("gemini")
    assert any("temperature" in e.lower() for e in bad_errors)

def test_generation_config_provider_translation():
    cfg = GenerationConfig(temperature=0.3, top_p=0.85, max_output_tokens=1024, frequency_penalty=0.5)
    gemini_dict = cfg.to_provider_dict("gemini")
    assert "temperature" in gemini_dict
    assert "top_p" in gemini_dict
    assert "max_output_tokens" in gemini_dict

def test_llm_catalog_and_classification():
    assert len(CATALOG_MODELS) >= 5
    # Verify accurate tier labeling
    gemini_models = [m for m in CATALOG_MODELS if m["provider"] == "gemini"]
    assert any(m["pricing_model"] == "free-tier-limited" for m in gemini_models)
    ollama_models = [m for m in CATALOG_MODELS if m["provider"] == "ollama"]
    assert any(m["hosting"] == "local" for m in ollama_models)

# =========================================================================
# MODULE 6: RETRIEVAL & RERANKING
# =========================================================================

def test_retrieval_threshold_filtering():
    engine = RetrievalEngine()
    docs = [
        {"id": "1", "content": "Highly relevant", "score": 0.85},
        {"id": "2", "content": "Irrelevant noise", "score": 0.25}
    ]
    filtered = engine.filter_by_threshold(docs, threshold=0.50)
    assert len(filtered) == 1
    assert filtered[0]["id"] == "1"

def test_retrieval_token_budget():
    engine = RetrievalEngine()
    long_doc = {"id": "1", "content": "Word " * 2000, "score": 0.9}
    budgeted = engine._apply_token_budget([long_doc], max_tokens=500)
    assert len(budgeted) <= 1
    assert len(budgeted[0]["content"]) <= 2100

# =========================================================================
# MODULE 7: RAG EVALUATOR & HISTORY
# =========================================================================

def test_eval_history_sqlite_lifecycle():
    hist = EvalHistoryManager(db_path=":memory:")
    sample_session = {
        "llm_provider": "gemini",
        "llm_model": "gemini-1.5-flash",
        "embedding_model": "all-MiniLM-L6-v2",
        "vector_db": "chromadb",
        "chunking_strategy": "RecursiveCharacterTextSplitter",
        "chunk_size": 1000,
        "num_test_questions": 2,
        "aggregate_metrics": {"avg_answer_correctness": 0.92, "avg_faithfulness": 0.95},
        "individual_results": []
    }
    s_id = hist.save_session(sample_session)
    assert s_id is not None
    loaded = hist.load_history()
    assert len(loaded) == 1
    assert loaded[0]["session_id"] == s_id
    assert loaded[0]["aggregate_metrics"]["avg_answer_correctness"] == 0.92

def test_rag_evaluator_ground_truth_matching():
    evaluator = RAGEvaluator()
    f1 = evaluator._compute_f1_token_overlap(
        pred="Revenue grew by 24% in Q2",
        target="In Q2, revenue grew by 24 percent"
    )
    assert f1 > 0.60

def test_platform_engine_initialization():
    p_config = PlatformConfig()
    engine = PlatformEngine(p_config)
    assert engine.chunking_engine is not None
    assert engine.embedding_manager is not None
    assert engine.vector_db_manager is not None
    assert engine.llm_manager is not None
    assert engine.retrieval_engine is not None
