"""
Multi-Modal RAG Platform Frontend (Streamlit)
Supports Web URLs, Excel Sheets, Datasets, Images, and Videos with Chat Interface & Visual Keyframe Display.
Extended with Advanced Chunking, Multi-Vector DB, Embedding Hub, Multi-LLM, Reranking & Evaluation.
Run with: streamlit run app.py
"""

import os
import time
import json
import streamlit as st
from PIL import Image

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

# Page Configuration
st.set_page_config(
    page_title="Multi-Modal RAG Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px 15px;
        border-left: 4px solid #1E88E5;
        margin-bottom: 10px;
    }
    .badge-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 5px;
    }
    .badge-local { background-color: #E8F5E9; color: #2E7D32; }
    .badge-cloud { background-color: #E3F2FD; color: #1565C0; }
    .badge-free { background-color: #FFF3E0; color: #E65100; }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hello! I am your **Multi-Modal RAG Assistant**. Ingest data from the sidebar (Web, Excel, Datasets, Images, Video) or configure advanced chunking, vector DBs, and LLMs below, then ask me anything!"}
    ]

# Initialize default configuration states if not present
if "chunking_config" not in st.session_state:
    st.session_state.chunking_config = ChunkingConfig()

if "embedding_config" not in st.session_state:
    st.session_state.embedding_config = EmbeddingConfig()

if "vector_db_name" not in st.session_state:
    st.session_state.vector_db_name = "ChromaDB"

if "llm_config" not in st.session_state:
    st.session_state.llm_config = LLMConfig(
        provider="gemini",
        model_name="gemini-1.5-flash",
        api_key=os.environ.get("GEMINI_API_KEY", "")
    )

if "gen_config" not in st.session_state:
    st.session_state.gen_config = GenerationConfig()

if "retrieval_config" not in st.session_state:
    st.session_state.retrieval_config = RetrievalConfig()

if "reranking_config" not in st.session_state:
    st.session_state.reranking_config = RerankingConfig()

# Sidebar: Navigation & Configuration Hub
with st.sidebar:
    st.title("🧠 RAG Platform Hub")
    
    # ── Platform Navigation Shortcuts ──
    nav_c1, nav_c2 = st.columns(2)
    with nav_c1:
        st.page_link("pages/evaluation_dashboard.py", label="📊 Evaluation", icon="📊")
    with nav_c2:
        st.page_link("pages/comparison_lab.py", label="🔬 Compare Lab", icon="🔬")

    st.divider()

    st.subheader("📥 Data Ingestion Hub")
    st.caption("Upload or load your multi-modal sources into the vector store.")

    # API Key Configuration
    with st.expander("🔑 API Key Settings", expanded=False):
        api_key_input = st.text_input(
            "Gemini API Key",
            type="password",
            value=os.environ.get("GEMINI_API_KEY", ""),
            help="Get your free key at https://aistudio.google.com/app/apikey"
        )
        if api_key_input:
            api_key_input = api_key_input.strip()
            os.environ["GEMINI_API_KEY"] = api_key_input
            st.session_state.llm_config.api_key = api_key_input
            st.success("✅ Gemini API Key registered.")

            if st.session_state.rag_engine is not None:
                st.session_state.rag_engine.config.api_key = api_key_input
                st.session_state.rag_engine.gemini_client = st.session_state.rag_engine._init_gemini()

    # 1-Click Synthetic Sample Data
    st.subheader("⚡ Quick Demo")
    if st.button("Load Sample Demo Data (All 5 Modalities)", use_container_width=True, type="primary"):
        with st.spinner("Generating and indexing sample data (Excel, Video, Image, Dataset, Web)..."):
            config = RAGConfig(api_key=os.environ.get("GEMINI_API_KEY", ""))
            engine = MultimodalRAGEngine(config)
            
            sample_data = create_synthetic_multimodal_data("./colab_sample_data")
            n_xl = engine.ingest_excel(sample_data["excel"])
            if "pdf" in sample_data and os.path.exists(sample_data["pdf"]):
                engine.ingest_pdf(sample_data["pdf"])
            n_ds = engine.ingest_dataset(sample_data["dataset"], name="safety_benchmark")
            n_img = engine.ingest_image(sample_data["image"])
            n_vid = engine.ingest_video(sample_data["video"], max_frames=6)
            n_web, _ = engine.ingest_url(sample_data["web_url"])
            
            st.session_state.rag_engine = engine
            st.success(f"Indexed successfully! Total chunks: {engine.vector_store.collection.count()}")

    st.divider()
    st.subheader("📂 Custom Data Upload")

    # Ingestion Tabs
    tab_web, tab_excel, tab_pdf, tab_dataset, tab_image, tab_video = st.tabs([
        "🌐 Web", "📊 Excel", "📑 PDF", "📁 Dataset", "🖼️ Image", "🎥 Video"
    ])

    upload_dir = "./uploaded_user_data"
    os.makedirs(upload_dir, exist_ok=True)

    def get_or_create_engine():
        if st.session_state.rag_engine is None:
            config = RAGConfig(api_key=os.environ.get("GEMINI_API_KEY", ""))
            st.session_state.rag_engine = MultimodalRAGEngine(config)
        return st.session_state.rag_engine

    # 1. Web Scraper Tab
    with tab_web:
        web_url = st.text_input("Enter Web URL:", placeholder="https://en.wikipedia.org/wiki/...")
        if st.button("Scrape & Index Web Page"):
            if web_url:
                with st.spinner(f"Scraping {web_url}..."):
                    eng = get_or_create_engine()
                    count, source_label = eng.ingest_url(web_url)
                    st.session_state.active_source = source_label
                    st.success(f"✅ Indexed {count} chunks from **{source_label}**!")

    # 2. Excel Tab
    with tab_excel:
        excel_file = st.file_uploader("Upload Excel / CSV", type=["xlsx", "xls", "csv"])
        if excel_file and st.button("Index Spreadsheet"):
            save_path = os.path.join(upload_dir, excel_file.name)
            with open(save_path, "wb") as f:
                f.write(excel_file.getbuffer())
            with st.spinner("Processing spreadsheet rows..."):
                eng = get_or_create_engine()
                count = eng.ingest_excel(save_path)
                st.session_state.active_source = excel_file.name
                st.success(f"✅ Indexed {count} tabular chunks from {excel_file.name}!")

    # 3. PDF Tab
    with tab_pdf:
        pdf_file = st.file_uploader("Upload PDF Document", type=["pdf"])
        if pdf_file and st.button("Index PDF Document"):
            save_path = os.path.join(upload_dir, pdf_file.name)
            with open(save_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            with st.spinner("Extracting pages & indexing PDF text..."):
                eng = get_or_create_engine()
                count = eng.ingest_pdf(save_path)
                st.session_state.active_source = pdf_file.name
                st.success(f"✅ Indexed {count} chunks from {pdf_file.name}!")

    # 4. Dataset Tab
    with tab_dataset:
        dataset_file = st.file_uploader("Upload JSON / JSONL Dataset", type=["json", "jsonl"])
        if dataset_file and st.button("Index Dataset"):
            save_path = os.path.join(upload_dir, dataset_file.name)
            with open(save_path, "wb") as f:
                f.write(dataset_file.getbuffer())
            with st.spinner("Indexing dataset records..."):
                eng = get_or_create_engine()
                count = eng.ingest_dataset(save_path, name=dataset_file.name)
                st.session_state.active_source = dataset_file.name
                st.success(f"✅ Indexed {count} dataset chunks from {dataset_file.name}!")

    # 5. Image Tab
    with tab_image:
        image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg", "webp"])
        if image_file and st.button("Index Image"):
            save_path = os.path.join(upload_dir, image_file.name)
            with open(save_path, "wb") as f:
                f.write(image_file.getbuffer())
            with st.spinner("Extracting visual features & OCR..."):
                eng = get_or_create_engine()
                count = eng.ingest_image(save_path)
                st.session_state.active_source = image_file.name
                st.success(f"✅ Indexed image: {image_file.name}")

    # 6. Video Tab
    with tab_video:
        video_file = st.file_uploader("Upload MP4 / AVI / MOV", type=["mp4", "avi", "mov"])
        video_frames_count = st.slider("Keyframes to extract & caption", min_value=4, max_value=16, value=8)
        if video_file and st.button("Extract Frames & Index Video"):
            save_path = os.path.join(upload_dir, video_file.name)
            with open(save_path, "wb") as f:
                f.write(video_file.getbuffer())
            with st.spinner("Extracting & captioning keyframes..."):
                eng = get_or_create_engine()
                count = eng.ingest_video(save_path, max_frames=video_frames_count)
                st.session_state.active_source = os.path.basename(video_file.name)
                st.success(f"✅ Indexed {count} keyframes!")

    # ═════════════════════════════════════════════════════════════════════════
    # ADVANCED CONFIGURATION MODULES (Modules 1 - 7)
    # ═════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("⚙️ Advanced Configurations")

    # MODULE 1: Advanced Chunking Settings
    with st.expander("📐 Module 1: Chunking Settings", expanded=False):
        c_strategy = st.selectbox(
            "Chunking Strategy",
            options=ChunkingEngine.SUPPORTED_STRATEGIES,
            index=0,
            help="Choose text splitting algorithm for incoming documents."
        )
        c_size = st.slider("Chunk Size (characters)", min_value=100, max_value=4000, value=1000, step=50)
        c_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=500, value=150, step=10)
        c_min_size = st.number_input("Minimum Chunk Size", min_value=20, max_value=500, value=100)
        c_preserve = st.checkbox("Preserve Separators", value=True)
        
        c_similarity = 0.5
        if c_strategy == "SemanticChunker":
            c_similarity = st.slider("Semantic Similarity Threshold", 0.0, 1.0, 0.5, step=0.05)

        st.session_state.chunking_config = ChunkingConfig(
            strategy=c_strategy,
            chunk_size=c_size,
            chunk_overlap=c_overlap,
            min_chunk_size=c_min_size,
            preserve_separators=c_preserve,
            similarity_threshold=c_similarity
        )

        sample_preview_text = st.text_area(
            "Preview Sample Text",
            value="Multi-Modal Retrieval-Augmented Generation enhances LLM factuality. "
                  "It splits documents into discrete chunks, encodes them into high-dimensional vectors, "
                  "and retrieves nearest neighbours during query time. Modalities include text, tables, images, and videos.",
            height=70
        )
        if st.button("Preview Chunks"):
            c_engine = ChunkingEngine(st.session_state.chunking_config)
            preview_chunks = c_engine.preview_chunks(sample_preview_text)
            st.info(f"Generated {len(preview_chunks)} chunks with strategy `{c_strategy}`:")
            for idx, c in enumerate(preview_chunks[:3], 1):
                st.text_area(f"Chunk #{idx} ({len(c)} chars)", c, height=65)

    # MODULE 2: Vector Database Management
    with st.expander("🗄️ Module 2: Vector Database", expanded=False):
        db_choices = ["ChromaDB", "FAISS", "Qdrant", "Milvus", "Pinecone", "Weaviate"]
        chosen_db = st.selectbox("Select Vector Database", options=db_choices, index=0)
        st.session_state.vector_db_name = chosen_db

        cloud_url = ""
        cloud_key = ""
        if chosen_db in ["Qdrant", "Milvus", "Pinecone", "Weaviate"]:
            st.caption("Cloud Provider Settings:")
            cloud_url = st.text_input(f"{chosen_db} Endpoint URL", placeholder="https://xyz.cloud.provider:6333")
            cloud_key = st.text_input(f"{chosen_db} API Key", type="password")

        if st.button("🔌 Test DB Connection"):
            with st.spinner(f"Verifying {chosen_db} connectivity..."):
                if chosen_db == "ChromaDB":
                    chroma_ad = ChromaAdapter()
                    status = chroma_ad.test_connection()
                    if status.get("success"):
                        st.success(f"✅ ChromaDB Connected — Collection: `{status.get('collection')}`, Count: {status.get('document_count')} chunks")
                    else:
                        st.error(f"❌ Connection failed: {status.get('message')}")
                elif chosen_db == "FAISS":
                    faiss_ad = FAISSAdapter()
                    status = faiss_ad.test_connection()
                    st.success(f"✅ FAISS Local Flat L2 Index Ready — {status.get('message')}")
                elif chosen_db == "Qdrant":
                    qdrant_ad = QdrantAdapter(url=cloud_url if cloud_url else None, api_key=cloud_key if cloud_key else None)
                    status = qdrant_ad.test_connection()
                    if status.get("success"):
                        st.success(f"✅ Qdrant Connected ({status.get('mode')}) — {status.get('message')}")
                    else:
                        st.warning(f"⚠️ Qdrant Status: {status.get('message')}")
                else:
                    st.warning(f"⚠️ {chosen_db} requires active cloud service deployment and credentials.")

        if st.session_state.rag_engine:
            cnt = st.session_state.rag_engine.vector_store.collection.count()
            st.metric("Total Indexed Chunks (Active DB)", cnt)
            confirm_del = st.checkbox("Confirm database wipe", key="del_db_confirm")
            if st.button("🗑️ Reset Active Database", type="secondary"):
                if confirm_del:
                    st.session_state.rag_engine.vector_store.reset_collection()
                    st.session_state.messages = [
                        {"role": "assistant", "content": "🧹 Database reset completed."}
                    ]
                    st.success("Database cleared!")
                    st.rerun()
                else:
                    st.warning("Check 'Confirm database wipe' before clearing.")

    # MODULE 3: Embedding Model Hub
    with st.expander("🧬 Module 3: Embedding Model Hub", expanded=False):
        embed_keys = list(MODEL_REGISTRY.keys())
        selected_embed = st.selectbox(
            "Embedding Model",
            options=embed_keys,
            index=0,
            help="Choose model for semantic vector encoding."
        )
        meta_emb = MODEL_REGISTRY.get(selected_embed, {})
        st.markdown(f"**Provider**: `{meta_emb.get('provider')}` | **Dims**: `{meta_emb.get('dimensions')}`")
        st.markdown(f"**Execution**: `{meta_emb.get('execution_mode')}` | **Version**: `{meta_emb.get('version')}`")

        st.session_state.embedding_config = EmbeddingConfig(
            model_name=selected_embed,
            provider=meta_emb.get("provider", "local"),
            dimensions=meta_emb.get("dimensions", 384),
            execution_mode=meta_emb.get("execution_mode", "local")
        )

        if st.button("⚡ Test Embedding Function"):
            with st.spinner("Encoding sample sentence..."):
                t0 = time.time()
                try:
                    e_mgr = EmbeddingManager(st.session_state.embedding_config)
                    test_vec = e_mgr.embed_query("Multi-Modal RAG Platform validation")
                    latency = round((time.time() - t0) * 1000.0, 1)
                    st.success(f"✅ Generated {len(test_vec)}d embedding in {latency} ms!")
                except Exception as ex:
                    st.error(f"Embedding error: {ex}")

    # MODULE 4: Multiple LLM Providers
    with st.expander("🤖 Module 4: LLM Provider & Model", expanded=False):
        provider_choices = ["gemini", "ollama", "groq", "huggingface"]
        selected_provider = st.selectbox("LLM Provider", options=provider_choices, index=0)

        matching_models = [m for m in CATALOG_MODELS if m["provider"] == selected_provider]
        model_names = [m["model_name"] for m in matching_models] or ["default"]
        selected_model_name = st.selectbox("Select Model", options=model_names, index=0)

        # Show model characteristics
        m_info = next((m for m in matching_models if m["model_name"] == selected_model_name), {})
        hosting = m_info.get("hosting", "cloud")
        license_t = m_info.get("license_type", "open-weight")
        pricing = m_info.get("pricing_model", "cloud-api")

        badge_cls = "badge-local" if hosting == "local" else "badge-cloud"
        st.markdown(f"""
        <span class="badge-pill {badge_cls}">{hosting.upper()}</span>
        <span class="badge-pill badge-free">{license_t.upper()}</span>
        <span class="badge-pill badge-cloud">{pricing}</span>
        """, unsafe_allow_html=True)
        st.caption(m_info.get("description", ""))

        provider_api_key = ""
        ollama_url = "http://localhost:11434"
        if selected_provider == "gemini":
            provider_api_key = os.environ.get("GEMINI_API_KEY", "")
        elif selected_provider == "groq":
            provider_api_key = st.text_input("Groq API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""))
        elif selected_provider == "huggingface":
            provider_api_key = st.text_input("HuggingFace Token", type="password", value=os.environ.get("HF_TOKEN", ""))
        elif selected_provider == "ollama":
            ollama_url = st.text_input("Ollama Base URL", value="http://localhost:11434")

        st.session_state.llm_config = LLMConfig(
            provider=selected_provider,
            model_name=selected_model_name,
            api_key=provider_api_key,
            base_url=ollama_url,
            is_local=(hosting == "local")
        )

        if st.button("🔌 Test LLM Connection"):
            with st.spinner("Testing connection to model API..."):
                mgr = LLMManager(st.session_state.llm_config)
                t_res = mgr.test_connection()
                if t_res.get("success"):
                    st.success(f"✅ {t_res.get('message')}")
                else:
                    st.error(f"❌ {t_res.get('message')}")

    # MODULE 5: LLM Generation Parameters
    with st.expander("🎛️ Module 5: Generation Parameters", expanded=False):
        g_temp = st.slider("Temperature", 0.0, 2.0, 0.2, step=0.05, help="Lower values give more factual answers.")
        g_top_p = st.slider("Top-P (Nucleus Sampling)", 0.0, 1.0, 0.9, step=0.05)
        g_max_tokens = st.number_input("Max Output Tokens", min_value=128, max_value=8192, value=2048, step=128)

        # Check supported parameters for active provider
        supported_p = PROVIDER_SUPPORT_MATRIX.get(selected_provider, [])

        g_top_k = None
        if "top_k" in supported_p:
            g_top_k = st.slider("Generation Top-K (Token Sampling)", 1, 100, 40)
        else:
            st.caption("ℹ️ *Generation Top-K is not supported by selected provider API.*")

        g_freq = st.slider("Frequency Penalty", -2.0, 2.0, 0.0, step=0.1) if "frequency_penalty" in supported_p else 0.0
        g_pres = st.slider("Presence Penalty", -2.0, 2.0, 0.0, step=0.1) if "presence_penalty" in supported_p else 0.0

        if st.button("Reset to Factual RAG Defaults"):
            st.session_state.gen_config = GenerationConfig(temperature=0.2, top_p=0.9, max_output_tokens=2048)
            st.success("Reset to factual defaults (Temp: 0.2, Top-P: 0.9, Max: 2048).")
        else:
            st.session_state.gen_config = GenerationConfig(
                temperature=g_temp,
                top_p=g_top_p,
                top_k=g_top_k,
                max_output_tokens=g_max_tokens,
                frequency_penalty=g_freq,
                presence_penalty=g_pres
            )

    # MODULE 6: Retrieval and Reranking Configuration
    with st.expander("🔍 Module 6: Retrieval & Reranking", expanded=False):
        st.info("💡 **Retrieval Top-K** determines how many context chunks are retrieved from the Vector DB. This is completely distinct from LLM token sampling Top-K.")
        
        r_top_k = st.slider("Retrieval Top-K (Documents fetched)", min_value=1, max_value=20, value=5)
        r_threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.05)
        r_type = st.radio("Search Strategy", options=["similarity", "mmr"], horizontal=True)
        r_lambda = 0.5
        if r_type == "mmr":
            r_lambda = st.slider("MMR Diversity Lambda (1=sim, 0=diversity)", 0.0, 1.0, 0.5, step=0.05)
        r_max_tokens = st.number_input("Max Context Tokens Budget", 500, 16000, 4096, step=500)

        st.session_state.retrieval_config = RetrievalConfig(
            top_k=r_top_k,
            similarity_threshold=r_threshold,
            search_type=r_type,
            mmr_lambda=r_lambda,
            max_context_tokens=r_max_tokens
        )

        st.markdown("**Cross-Encoder Reranking**")
        rerank_enable = st.toggle("Enable Reranker", value=False)
        rerank_model = st.selectbox(
            "Reranker Model",
            options=["cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-TinyBERT-L-2-v2"],
            index=0,
            disabled=not rerank_enable
        )
        rerank_top_n = st.slider("Reranker Top-N Chunks Passed to LLM", 1, 10, 3, disabled=not rerank_enable)

        st.session_state.reranking_config = RerankingConfig(
            enabled=rerank_enable,
            model=rerank_model,
            top_n=rerank_top_n
        )

    # MODULE 7: Quick Evaluation Status
    with st.expander("📊 Module 7: Evaluation Summary", expanded=False):
        try:
            hist_mgr = EvalHistoryManager()
            history_recs = hist_mgr.load_history()
            total_runs = len(history_recs)
            st.metric("Total Evaluation Runs", total_runs)
            if total_runs > 0:
                latest = history_recs[0]
                agg = latest.get("aggregate_metrics", {})
                corr = agg.get("avg_answer_correctness")
                st.write(f"Latest Correctness: **{corr*100:.1f}%**" if corr else "Latest: Not Evaluated")
            st.page_link("pages/evaluation_dashboard.py", label="Open Full Dashboard 📊", icon="📊")
        except Exception:
            st.write("Evaluation history not yet initialized.")


# ═════════════════════════════════════════════════════════════════════════════
# Main Panel: Multi-Modal Chat & Visual Retrieval Display
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("<div class=\"main-header\">🧠 Multi-Modal RAG Platform</div>", unsafe_allow_html=True)
st.markdown("<div class=\"sub-header\">Retrieval-Augmented Generation across PDF Documents, Text, Spreadsheets, Datasets, Images, and Timestamped Videos</div>", unsafe_allow_html=True)

# Active Configuration Status Pill Banner
active_provider = st.session_state.llm_config.provider.upper()
active_model = st.session_state.llm_config.model_name
active_db = st.session_state.vector_db_name
active_chunk = st.session_state.chunking_config.strategy
active_ret_k = st.session_state.retrieval_config.top_k

st.caption(
    f"Active Pipeline: **LLM**: `{active_provider} : {active_model}` | "
    f"**Vector DB**: `{active_db}` | "
    f"**Chunker**: `{active_chunk}` | "
    f"**Retrieval Top-K**: `{active_ret_k}`"
)

# Suggested Questions Pills
st.write("**💡 Suggested questions to try:**")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    if st.button("📊 Excel Revenue", use_container_width=True):
        st.session_state.prompt_trigger = "Which product had the highest Q2 revenue in the financial report and what was its growth rate?"
with col2:
    if st.button("📑 PDF Guidelines", use_container_width=True):
        st.session_state.prompt_trigger = "What are the recommended chunk sizes and safety protocols in the AI guidelines PDF?"
with col3:
    if st.button("🎥 Video Keyframe", use_container_width=True):
        st.session_state.prompt_trigger = "What occurred at timestamp 00:02 or 00:04 in the product demo clip?"
with col4:
    if st.button("🖼️ Architecture", use_container_width=True):
        st.session_state.prompt_trigger = "What components are shown in the system architecture diagram?"
with col5:
    if st.button("📁 Dataset Score", use_container_width=True):
        st.session_state.prompt_trigger = "What accuracy did Gemini-2.5 achieve on adversarial robustness?"

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "visual_evidence" in msg and msg["visual_evidence"]:
            image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
            valid_visuals = [
                p for p in msg["visual_evidence"]
                if isinstance(p, str) and os.path.exists(p) and os.path.splitext(p)[1].lower() in image_extensions
            ]
            if valid_visuals:
                st.caption("🖼️ Attached Visual Evidence / Video Keyframes:")
                cols = st.columns(min(len(valid_visuals), 3))
                for i, p in enumerate(valid_visuals[:3]):
                    try:
                        with cols[i]:
                            st.image(Image.open(p), caption=os.path.basename(p), use_container_width=True)
                    except Exception:
                        pass

# Document Focus Selector
all_sources = ["All Sources (Search Everything)"]
if st.session_state.rag_engine:
    all_sources += [s for s in st.session_state.rag_engine.vector_store.get_all_sources() if s not in all_sources]

default_idx = 0
if "active_source" in st.session_state and st.session_state.active_source in all_sources:
    default_idx = all_sources.index(st.session_state.active_source)

col_focus, col_clear = st.columns([4, 1])
with col_focus:
    selected_source = st.selectbox(
        "🎯 Document Focus (Lock search to a specific file):",
        options=all_sources,
        index=default_idx,
        help="Select your file (e.g. LoanData.csv) to guarantee 100% relevant answers without interference from other files."
    )
with col_clear:
    st.write("")
    st.write("")
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat cleared! Ask me anything about your documents."}
        ]
        st.rerun()

# Handle Input (Chat bar or Suggested buttons)
user_query = st.chat_input("Ask a question about your uploaded documents, images, or videos...")
if "prompt_trigger" in st.session_state and st.session_state.prompt_trigger:
    user_query = st.session_state.prompt_trigger
    st.session_state.prompt_trigger = None

if user_query:
    # Append user question
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Process via RAG Engine
    with st.chat_message("assistant"):
        if st.session_state.rag_engine is None:
            config = RAGConfig(api_key=os.environ.get("GEMINI_API_KEY", ""))
            st.session_state.rag_engine = MultimodalRAGEngine(config)

        eng = st.session_state.rag_engine
        if eng.vector_store.collection.count() == 0:
            msg_reply = "⚠️ Vector store is currently empty! Please load the sample data or upload files from the left sidebar first."
            st.warning(msg_reply)
            st.session_state.messages.append({"role": "assistant", "content": msg_reply})
        else:
            with st.spinner(f"Retrieving from {st.session_state.vector_db_name} & generating via {active_provider} ({active_model})..."):
                filter_src = None if selected_source.startswith("All Sources") else selected_source
                
                # Create LLM Manager if alternate provider selected
                active_llm_mgr = None
                if st.session_state.llm_config.provider != "gemini" or st.session_state.llm_config.model_name != "gemini-1.5-flash":
                    active_llm_mgr = LLMManager(st.session_state.llm_config)

                if hasattr(eng, "query_with_config"):
                    result = eng.query_with_config(
                        question=user_query,
                        retrieval_config=st.session_state.retrieval_config,
                        gen_config=st.session_state.gen_config,
                        source_filter=filter_src,
                        chat_history=st.session_state.messages,
                        llm_manager=active_llm_mgr,
                        reranking_config=st.session_state.reranking_config
                    )
                else:
                    top_k_val = getattr(st.session_state.retrieval_config, "top_k", 5)
                    result = eng.query(
                        question=user_query,
                        top_k=top_k_val,
                        source_filter=filter_src,
                        chat_history=st.session_state.messages
                    )


                answer = result.get("answer", "No answer generated.")
                visuals = result.get("visual_evidence", [])
                sources = result.get("sources", [])
                latency = result.get("latency_seconds")
                ret_latency = result.get("retrieval_latency_ms", 0.0)
                llm_latency = result.get("llm_latency_ms", 0.0)
                model_used = result.get("model_used", active_model)

                st.markdown(answer)

                if latency is not None:
                    st.caption(
                        f"⚡ *Response generated in {latency}s "
                        f"(Retrieval: {ret_latency}ms | LLM: {llm_latency}ms) "
                        f"via `{model_used}` on `{st.session_state.vector_db_name}`*"
                    )

                image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
                valid_visuals = [
                    p for p in visuals
                    if isinstance(p, str) and os.path.exists(p) and os.path.splitext(p)[1].lower() in image_extensions
                ]
                if valid_visuals:
                    st.caption("🖼️ Attached Visual Evidence / Video Keyframes:")
                    cols = st.columns(min(len(valid_visuals), 3))
                    for i, p in enumerate(valid_visuals[:3]):
                        try:
                            with cols[i]:
                                st.image(Image.open(p), caption=os.path.basename(p), use_container_width=True)
                        except Exception:
                            pass

                with st.expander("🔍 View Retrieved Sources & Metadata"):
                    st.caption(f"Showing {len(sources)} retrieved chunks (Top-K: {st.session_state.retrieval_config.top_k}):")
                    for idx, s in enumerate(sources, 1):
                        m = s.get("metadata", {})
                        distance = s.get("distance", 0.0)
                        score_disp = f" | Sim Distance: {distance:.3f}" if distance else ""
                        st.markdown(f"**Source #{idx}** (`{m.get('modality', 'text')}` | `{m.get('source', '')}`{score_disp})")
                        st.text(s.get("content", ""))

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "visual_evidence": valid_visuals,
                    "latency": latency,
                    "model": model_used
                })
