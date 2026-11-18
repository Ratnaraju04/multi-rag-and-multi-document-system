"""
RAG Evaluation Dashboard — Standalone Streamlit page.
Provides Ground-Truth and Reference-Free evaluation modes with metric cards,
history table, model comparison, and accuracy trend charts.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Backend imports (graceful fallback) ──────────────────────────────────────
try:
    from src.evaluation.rag_evaluator import RAGEvaluator, EvalResult
    _EVALUATOR_AVAILABLE = True
except ImportError:
    _EVALUATOR_AVAILABLE = False
    RAGEvaluator = None
    EvalResult = None

try:
    from src.evaluation.eval_history import EvalHistoryManager
    _HISTORY_AVAILABLE = True
except ImportError:
    _HISTORY_AVAILABLE = False
    EvalHistoryManager = None

# ── Page Header ───────────────────────────────────────────────────────────────
st.title("📊 RAG Evaluation Dashboard")
st.caption("Assess your RAG pipeline quality. Metrics are estimates — always verify with human review.")

if not _EVALUATOR_AVAILABLE:
    st.info(
        "ℹ️ Evaluation backend not yet available. "
        "Metric cards show placeholder state. "
        "Run the full pipeline setup to enable live evaluation."
    )

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# SECTION A — Evaluation Mode
# ════════════════════════════════════════════════════════════════════════════
st.subheader("🎛️ Section A: Evaluation Mode")

eval_mode = st.radio(
    "Select Evaluation Mode",
    options=["Ground-Truth", "Reference-Free"],
    horizontal=True,
    help=(
        "**Ground-Truth**: Compare answers against known correct answers. "
        "**Reference-Free**: Estimate quality using retrieved context only."
    )
)

eval_data = {}

if eval_mode == "Ground-Truth":
    st.markdown("**Provide test questions and expected answers (one per line):**")
    col_q, col_a = st.columns(2)
    with col_q:
        questions_input = st.text_area(
            "Questions (one per line)",
            placeholder="What is RAG?\nHow does chunking work?\nWhat is vector search?",
            height=150,
            key="gt_questions"
        )
    with col_a:
        expected_input = st.text_area(
            "Expected Answers (one per line)",
            placeholder="RAG stands for Retrieval-Augmented Generation.\nChunking splits text into segments.\nVector search finds similar embeddings.",
            height=150,
            key="gt_expected"
        )
    source_docs = st.text_area(
        "Optional: Source Document Snippets (one per line, same order as questions)",
        placeholder="Leave blank to use the indexed vector store.",
        height=80,
        key="gt_source_docs"
    )

    questions = [q.strip() for q in questions_input.strip().split("\n") if q.strip()]
    expected  = [a.strip() for a in expected_input.strip().split("\n") if a.strip()]
    eval_data = {"mode": "Ground-Truth", "questions": questions, "expected": expected}

else:  # Reference-Free
    st.info(
        "🔍 **Reference-Free mode** runs evaluation over your existing conversation history "
        "using retrieved context as the reference. No ground-truth answers needed."
    )
    history = st.session_state.get("messages", [])
    st.metric("Conversation turns available", len([m for m in history if m.get("role") == "user"]))
    eval_data = {"mode": "Reference-Free", "history": history}

run_eval = st.button(
    "▶️ Run Evaluation",
    type="primary",
    disabled=not _EVALUATOR_AVAILABLE,
    help="Evaluation backend must be installed to run." if not _EVALUATOR_AVAILABLE else ""
)

# Store results in session state
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

if run_eval and _EVALUATOR_AVAILABLE:
    with st.spinner("Running evaluation… this may take a minute."):
        try:
            evaluator = RAGEvaluator()
            results = evaluator.evaluate(eval_data)
            st.session_state.eval_results = results
            st.success("✅ Evaluation complete.")
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")
            st.session_state.eval_results = None

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# SECTION B — Metric Cards
# ════════════════════════════════════════════════════════════════════════════
st.subheader("📈 Section B: Evaluation Metrics")

results = st.session_state.get("eval_results")

def _fmt_pct(val, decimals=1):
    """Format a 0-1 float as percentage string, or '--' if None."""
    if val is None:
        return "--%"
    return f"{val * 100:.{decimals}f}%"

def _fmt_sec(val):
    if val is None:
        return "--s"
    return f"{val:.2f}s"

# Pull metric values
ans_correctness  = getattr(results, "answer_correctness",  None) if results else None
faithfulness     = getattr(results, "faithfulness",        None) if results else None
ctx_precision    = getattr(results, "context_precision",   None) if results else None
ctx_recall       = getattr(results, "context_recall",      None) if results else None
ret_precision_k  = getattr(results, "retrieval_precision_at_k", None) if results else None
avg_resp_time    = getattr(results, "avg_response_time",   None) if results else None

row1 = st.columns(4)
row2 = st.columns(4)  # extra 2 metrics + 2 spacer cols

with row1[0]:
    st.metric(
        label="✅ Answer Correctness",
        value=_fmt_pct(ans_correctness),
        help="How closely the generated answer matches the expected answer."
    )

with row1[1]:
    st.metric(
        label="🏛️ Faithfulness",
        value=_fmt_pct(faithfulness),
        help="Grounded in retrieved context — does the answer only use retrieved information?"
    )

with row1[2]:
    st.metric(
        label="🎯 Context Precision",
        value=_fmt_pct(ctx_precision),
        help="Relevance of retrieved chunks — are the retrieved chunks actually useful?"
    )

with row1[3]:
    st.metric(
        label="🔄 Context Recall",
        value=_fmt_pct(ctx_recall),
        help="Relevant information retrieved — did we fetch all the needed context?"
    )

with row2[0]:
    st.metric(
        label="📌 Retrieval Precision@K",
        value=_fmt_pct(ret_precision_k),
        help="Fraction of top-K retrieved chunks that are relevant."
    )

with row2[1]:
    st.metric(
        label="⚡ Avg Response Time",
        value=_fmt_sec(avg_resp_time),
        help="Average end-to-end latency per query."
    )

# Empty placeholders for alignment
with row2[2]:
    st.empty()
with row2[3]:
    st.empty()

if results is None:
    st.caption("ℹ️ Metrics show **Not Evaluated** state (--). Run evaluation above to populate.")

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# SECTION C — Evaluation History
# ════════════════════════════════════════════════════════════════════════════
st.subheader("🗂️ Section C: Evaluation History")

history_df = None

if _HISTORY_AVAILABLE:
    try:
        mgr = EvalHistoryManager()
        history_records = mgr.load_all()
        if history_records:
            history_df = pd.DataFrame(history_records)
    except Exception as exc:
        st.warning(f"Could not load history: {exc}")
else:
    # Try loading from SQLite directly if file exists
    import os, sqlite3
    db_path = "eval_history.db"
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            history_df = pd.read_sql("SELECT * FROM eval_history ORDER BY timestamp DESC LIMIT 100", conn)
            conn.close()
        except Exception:
            pass

if history_df is not None and not history_df.empty:
    st.dataframe(history_df, use_container_width=True)
else:
    st.info("No evaluation history found. Run an evaluation to populate the history.")

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# SECTION D — Model Comparison Chart
# ════════════════════════════════════════════════════════════════════════════
st.subheader("🔬 Section D: Model Comparison")

if history_df is not None and not history_df.empty:
    # Try to aggregate by model/config columns if present
    group_col = None
    for candidate in ["model", "config_name", "llm_model", "embedding_model"]:
        if candidate in history_df.columns:
            group_col = candidate
            break

    metric_col = None
    for candidate in ["answer_correctness", "faithfulness", "context_precision"]:
        if candidate in history_df.columns:
            metric_col = candidate
            break

    if group_col and metric_col:
        chart_data = (
            history_df.groupby(group_col)[metric_col]
            .mean()
            .reset_index()
            .rename(columns={group_col: "Config", metric_col: "Score"})
            .set_index("Config")
        )
        st.bar_chart(chart_data)
    else:
        st.info("History data does not contain model/config columns needed for comparison chart.")
else:
    _demo_comparison = pd.DataFrame(
        {"Score": [0.0, 0.0, 0.0]},
        index=["Config A", "Config B", "Config C"]
    )
    st.caption("Sample placeholder chart — no real data yet.")
    st.bar_chart(_demo_comparison)

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# SECTION E — Accuracy Trend
# ════════════════════════════════════════════════════════════════════════════
st.subheader("📉 Section E: Accuracy Trend Over Time")

if history_df is not None and not history_df.empty:
    ts_col = None
    for candidate in ["timestamp", "created_at", "date"]:
        if candidate in history_df.columns:
            ts_col = candidate
            break

    metric_col2 = None
    for candidate in ["answer_correctness", "faithfulness", "context_recall"]:
        if candidate in history_df.columns:
            metric_col2 = candidate
            break

    if ts_col and metric_col2:
        trend_data = (
            history_df[[ts_col, metric_col2]]
            .dropna()
            .rename(columns={ts_col: "Date", metric_col2: "Score"})
            .set_index("Date")
            .sort_index()
        )
        st.line_chart(trend_data)
    else:
        st.info("History does not contain timestamp/metric columns for a trend chart.")
else:
    _demo_trend = pd.DataFrame({"Score": [0.0]}, index=["No data"])
    st.caption("Sample placeholder trend — no real data yet.")
    st.line_chart(_demo_trend)

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# SECTION F — Disclaimer
# ════════════════════════════════════════════════════════════════════════════
st.warning(
    "⚠️ **Disclaimer:** LLM-based evaluation provides *estimated* metrics. "
    "These are **NOT** a substitute for human verification. "
    "Reference-free metrics estimate faithfulness against retrieved context only. "
    "Ground-truth metrics depend entirely on the quality of the provided expected answers."
)
