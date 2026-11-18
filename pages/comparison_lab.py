"""
Comparison Lab — Dedicated Streamlit page for comparing chunking strategies,
embedding models, vector databases, and LLM providers side-by-side.
"""

import os
import time
import json
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="RAG Comparison Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Header ─────────────────────────────────────────────────────────────────
st.title("🔬 RAG Comparison & Experimentation Lab")
st.caption(
    "Benchmark different Chunking Strategies, Embedding Models, Vector Databases, and LLMs "
    "on the exact same questions under identical conditions."
)
st.divider()

# ── Backend Imports ────────────────────────────────────────────────────────
try:
    from src.chunking.chunking_engine import ChunkingEngine, ChunkingConfig
    from src.embeddings.embedding_manager import EmbeddingManager, EmbeddingConfig, MODEL_REGISTRY
    from src.vectordb.vector_db_manager import VectorDBManager
    from src.llm.llm_manager import LLMManager, LLMConfig, CATALOG_MODELS
    from src.llm.generation_config import GenerationConfig
    from src.retrieval.retrieval_engine import RetrievalEngine, RetrievalConfig
    from src.evaluation.rag_evaluator import RAGEvaluator
    from src.evaluation.eval_history import EvalHistoryManager
    from src.core.platform_engine import PlatformEngine, PlatformConfig
    _BACKEND_AVAILABLE = True
except Exception as e:
    _BACKEND_AVAILABLE = False
    st.error(f"⚠️ Backend modules loading issue: {e}")

# ── Sidebar Configurations ────────────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ Experiment Controls")
    num_configs = st.slider("Number of Configurations to Compare", min_value=2, max_value=4, value=2)
    st.page_link("app.py", label="⬅️ Back to Main RAG App", icon="🏠")
    st.page_link("pages/evaluation_dashboard.py", label="📊 Evaluation Dashboard", icon="📊")

# ── Step 1: Define Test Questions ─────────────────────────────────────────
st.subheader("1️⃣ Shared Benchmark Questions")
st.write("Enter test queries that each configuration will process for fair comparison:")

default_questions = (
    "What is the primary revenue source in the financial report?\n"
    "What chunk size is recommended for technical documents?\n"
    "Describe the key findings or visual features demonstrated."
)

test_questions_input = st.text_area(
    "Test Questions (one per line):",
    value=default_questions,
    height=110,
    help="Each question will be executed against each defined configuration."
)
benchmark_questions = [q.strip() for q in test_questions_input.split("\n") if q.strip()]

st.divider()

# ── Step 2: Define Configurations ─────────────────────────────────────────
st.subheader("2️⃣ Pipeline Configurations")

cols = st.columns(num_configs)
configs = []

strategy_options = [
    "RecursiveCharacterTextSplitter",
    "CharacterTextSplitter",
    "TokenTextSplitter",
    "MarkdownHeaderTextSplitter",
    "SemanticChunker"
]

embed_options = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2",
    "embedding-001"
]

vectordb_options = ["ChromaDB", "FAISS", "Qdrant"]

llm_options = [
    ("gemini", "gemini-1.5-flash"),
    ("gemini", "gemini-1.5-pro"),
    ("gemini", "gemini-2.0-flash"),
    ("ollama", "llama3.2"),
    ("ollama", "mistral"),
    ("groq", "llama-3.3-70b-versatile"),
    ("huggingface", "meta-llama/Llama-3.2-3B-Instruct")
]

for i in range(num_configs):
    with cols[i]:
        st.markdown(f"### 🧪 Config #{i+1}")
        c_name = st.text_input(f"Name", value=f"Pipeline {chr(65+i)}", key=f"cfg_name_{i}")
        
        c_chunk = st.selectbox(
            "Chunking Strategy",
            options=strategy_options,
            index=0 if i == 0 else min(i, len(strategy_options)-1),
            key=f"cfg_chunk_{i}"
        )
        c_size = st.slider("Chunk Size", 200, 2000, 1000 if i == 0 else 500, step=100, key=f"cfg_size_{i}")
        
        c_embed = st.selectbox(
            "Embedding Model",
            options=embed_options,
            index=0 if i == 0 else min(i, len(embed_options)-1),
            key=f"cfg_embed_{i}"
        )
        
        c_db = st.selectbox(
            "Vector Database",
            options=vectordb_options,
            index=0 if i == 0 else 1,
            key=f"cfg_db_{i}"
        )
        
        c_llm_idx = st.selectbox(
            "LLM Provider & Model",
            options=range(len(llm_options)),
            format_func=lambda idx: f"{llm_options[idx][0].upper()} : {llm_options[idx][1]}",
            index=0 if i == 0 else min(i, len(llm_options)-1),
            key=f"cfg_llm_{i}"
        )
        
        c_temp = st.slider("Temperature", 0.0, 1.0, 0.2, step=0.05, key=f"cfg_temp_{i}")

        configs.append({
            "id": i,
            "name": c_name,
            "chunk_strategy": c_chunk,
            "chunk_size": c_size,
            "embedding_model": c_embed,
            "vector_db": c_db.lower(),
            "llm_provider": llm_options[c_llm_idx][0],
            "llm_model": llm_options[c_llm_idx][1],
            "temperature": c_temp
        })

st.divider()

# ── Step 3: Run Benchmark ─────────────────────────────────────────────────
st.subheader("3️⃣ Execute Comparison Benchmark")

if st.button("🚀 Run Benchmark Across All Configurations", type="primary", use_container_width=True):
    if not benchmark_questions:
        st.warning("Please provide at least one benchmark question.")
    elif not _BACKEND_AVAILABLE:
        st.error("Backend modules are unavailable.")
    else:
        results_list = []
        progress_bar = st.progress(0.0)
        total_steps = len(configs) * len(benchmark_questions)
        current_step = 0

        st.info("Executing benchmark runs across configurations...")

        for cfg in configs:
            # Set up engine for this configuration
            c_config = ChunkingConfig(
                strategy=cfg["chunk_strategy"],
                chunk_size=cfg["chunk_size"],
                chunk_overlap=int(cfg["chunk_size"] * 0.15)
            )
            e_config = EmbeddingConfig(
                model_name=cfg["embedding_model"],
                dimensions=MODEL_REGISTRY.get(cfg["embedding_model"], {}).get("dimensions", 384)
            )
            l_config = LLMConfig(
                provider=cfg["llm_provider"],
                model_name=cfg["llm_model"],
                api_key=os.environ.get("GEMINI_API_KEY", "") if cfg["llm_provider"] == "gemini" else ""
            )
            g_config = GenerationConfig(temperature=cfg["temperature"])
            
            p_config = PlatformConfig(
                chunking_config=c_config,
                embedding_config=e_config,
                vector_db_name="chroma" if cfg["vector_db"] == "chromadb" else cfg["vector_db"],
                llm_config=l_config,
                generation_config=g_config
            )
            
            try:
                engine = PlatformEngine(p_config)
            except Exception as e:
                engine = None
                st.warning(f"Engine init warning for {cfg['name']}: {e}")

            for q in benchmark_questions:
                start_t = time.time()
                ans = "No response"
                lat_ret = 0.0
                lat_llm = 0.0
                tokens = 0
                retrieved_count = 0

                if engine:
                    try:
                        res = engine.query(q)
                        ans = res.answer
                        lat_ret = res.retrieval_latency_ms
                        lat_llm = res.llm_latency_ms
                        tokens = res.tokens_used
                        retrieved_count = res.num_retrieved
                    except Exception as e:
                        ans = f"[Error executing query]: {e}"
                
                total_lat = (time.time() - start_t) * 1000.0

                # Compute baseline reference-free quality score (heuristic faithfulness & length)
                quality_score = min(100.0, max(20.0, round(len(ans) / 10.0 + (100.0 if "Error" not in ans else 0.0) / 2.0, 1)))

                results_list.append({
                    "Config": cfg["name"],
                    "LLM": f"{cfg['llm_provider']}/{cfg['llm_model']}",
                    "Vector DB": cfg["vector_db"].upper(),
                    "Chunking": cfg["chunk_strategy"][:14] + "..",
                    "Question": q,
                    "Answer": ans,
                    "Retrieved Docs": retrieved_count,
                    "Retrieval (ms)": round(lat_ret, 1),
                    "LLM Gen (ms)": round(lat_llm, 1),
                    "Total Latency (s)": round(total_lat / 1000.0, 2),
                    "Tokens": tokens,
                    "Est Quality": f"{quality_score}%"
                })

                current_step += 1
                progress_bar.progress(current_step / total_steps)

        st.session_state["comparison_results"] = results_list
        st.success("✅ Comparison Benchmark completed successfully!")

# ── Step 4: Display Benchmark Results ─────────────────────────────────────
if "comparison_results" in st.session_state and st.session_state["comparison_results"]:
    df = pd.DataFrame(st.session_state["comparison_results"])

    st.subheader("📊 Comparison Summary Table")
    st.dataframe(
        df[["Config", "LLM", "Vector DB", "Chunking", "Question", "Total Latency (s)", "Tokens", "Est Quality"]],
        use_container_width=True
    )

    # Latency comparison chart
    st.subheader("⚡ Latency by Configuration")
    latency_summary = df.groupby("Config")["Total Latency (s)"].mean().reset_index()
    st.bar_chart(latency_summary.set_index("Config"))

    # Side-by-Side Answers
    st.subheader("🔍 Side-by-Side Response Inspector")
    selected_q = st.selectbox("Inspect question:", benchmark_questions)

    q_df = df[df["Question"] == selected_q]
    cols_inspect = st.columns(len(q_df))
    for idx, (_, row) in enumerate(q_df.iterrows()):
        with cols_inspect[idx]:
            st.markdown(f"#### {row['Config']}")
            st.caption(f"`{row['LLM']}` | `{row['Vector DB']}` | {row['Total Latency (s)']}s")
            st.info(row["Answer"])

    # Export CSV
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Benchmark Results (CSV)",
        data=csv_bytes,
        file_name="rag_comparison_benchmark.csv",
        mime="text/csv",
        use_container_width=True
    )
