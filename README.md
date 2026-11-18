# 🌟 Multi-Modal RAG System (Resume & Portfolio Ready)

A complete **Multi-Modal Retrieval-Augmented Generation (RAG)** pipeline designed for **Google Colab** and formatted cell-by-cell for deep understanding, code interviews, and resume presentation.

Supports 6 heterogeneous data sources:
1. 🌐 **Web Scraping** (HTML clean extraction & paragraph chunking)
2. 📊 **Excel Spreadsheets & CSV** (`.xlsx` with schema summaries & row-by-row semantic chunks)
3. 📑 **PDF Documents** (Page-by-page text extraction with sliding window overlap and page metadata)
4. 📁 **Structured Datasets** (JSON records & benchmark tables)
5. 🖼️ **Images** (System architecture diagrams & visual descriptions)
6. 🎥 **Videos** (OpenCV keyframe extraction with exact `MM:SS` timestamps)

---

## 📋 Step-by-Step Architecture (Notebook Flow)

| Step | Section in Colab | Purpose & Recruiter Value |
|------|------------------|---------------------------|
| **Step 1** | Dependencies Installation | Installs LangChain, ChromaDB, HuggingFace embeddings, OpenCV, and Gemini/Groq SDKs. |
| **Step 2** | API Keys Setup | Configures Gemini API key (or optional Groq API key). |
| **Step 3** | Multi-Modal Data Ingestion | Creates and ingests sample web page, Excel spreadsheet, JSON dataset, image diagram, and 6-second video. |
| **Step 4** | Modality-Aware Chunking | **Core Innovation**: Chunks text, formats Excel into Markdown rows, and extracts video frames at 2-second timestamps using OpenCV. |
| **Step 5** | Vector Embeddings | Uses local `sentence-transformers/all-MiniLM-L6-v2` (Zero rate limits, zero API cost, 384-dim vectors). |
| **Step 6** | ChromaDB Vector Store | Stores all multimodal chunks in ChromaDB with metadata (`modality`, `timestamp`, `frame_path`). |
| **Step 7** | Similarity Search & Retriever | Demonstrates pure semantic retrieval before calling the LLM; displays matching video keyframes inline! |
| **Step 8** | LLM Synthesis & QA Chain | Connects the retriever to Gemini / Groq to synthesize citations and grounded answers. |
| **Step 9** | Multi-Modal RAG Queries | Tests queries across all 5 modalities, displaying answers + attached images/frames. |
| **Step 10** | Resume Cheat-Sheet | Ready-to-copy bullet points and interview Q&A for your resume. |

---

## 🎯 How to Explain This Project in an Interview

### 1. "How did you handle non-text data like Excel and Videos?"
- **Excel:** Standard character splitters fail on tables. We transformed tables into structured Markdown tables and natural language key-value sentences (`Product: X | Revenue: Y`) so dense embeddings match numerical queries cleanly.
- **Video:** Sampled keyframes at regular time intervals (every 2 seconds) with OpenCV, generating visual scene descriptions and indexing them with exact timestamps (`00:02`). When retrieved, the actual video frame is loaded and displayed to the user!

### 2. "Why use HuggingFace embeddings instead of OpenAI or Google Embeddings?"
- Local embeddings (`all-MiniLM-L6-v2`) run directly on Colab's hardware, eliminating API rate limits (HTTP 429), eliminating downtime (HTTP 503), and keeping operational costs at $0.

### 3. "How does cross-modal retrieval work in ChromaDB?"
- ChromaDB indexes vectors along with rich metadata dictionaries (`file_path`, `frame_path`, `timestamp`, `modality`). When a query matches, the retriever returns both the semantic text and the direct path to the visual image or keyframe.

---

## ⚡ Modern React Frontend & FastAPI Engine

The platform has been upgraded to a **React + Tailwind CSS** frontend powered by a **FastAPI** high-performance backend, replacing Streamlit for an enterprise-grade experience.

### ✨ What's in the React UI:
1. **💬 Chat & Retrieval Workspace**:
   - Live pipeline status badge (LLM, Vector DB, Chunker, Top-K)
   - 1-Click suggested questions pills (Excel Revenue, PDF Guidelines, Video Keyframes, Architecture, Dataset Scores)
   - Document Focus dropdown filter
   - Performance telemetry pills (`⚡ Total Latency | Retrieval ms | LLM ms | Model Name`)
   - Interactive Visual Evidence gallery with image lightbox zoom & download
   - Expandable Retrieved Sources inspector with similarity distance & modality chips
2. **📥 Data Ingestion Studio**:
   - 1-Click Quick Demo Data Generator (all 5 modalities: Web, Excel, PDF, JSON Dataset, Image, Video)
   - Custom drag-and-drop upload tabs with progress tracking
   - Real-time active knowledge base status & database reset with confirmation
3. **⚙️ Advanced Configuration Studio (7 Modules)**:
   - Module 1: Chunking Settings with interactive Live Chunk Split Previewer
   - Module 2: Vector DB Hub (ChromaDB, FAISS, Qdrant, Milvus, connection testing)
   - Module 3: Embedding Model Hub with dense latency testing
   - Module 4: Multi-LLM Providers (Gemini, Ollama, Groq, HuggingFace) with API key setup
   - Module 5: Generation Parameters (Temperature, Top-P, token sampling, factual presets)
   - Module 6: Retrieval & Cross-Encoder Reranking
   - Module 7: Pipeline Health & Status
4. **🔬 Comparison Lab**:
   - Benchmark 2 to 4 models and configurations side-by-side on identical queries
5. **📊 Evaluation & Quality Suite**:
   - Ground-Truth vs Reference-Free evaluation
   - Answer Correctness, Context Precision, and Faithfulness metric scorecards

---

## 🏃 How to Run the App

### Option 1: One-Click Run (FastAPI + React)
```bash
python run.py
```
*(Or double-click `start.bat` on Windows)*.
This automatically opens your browser at **`http://127.0.0.1:8000`**.

### Option 2: Live Hot-Reload Frontend Development
```bash
# Terminal 1: Backend
uvicorn server:app --host 127.0.0.1 --port 8000

# Terminal 2: React Vite Dev Server
cd frontend
npm run dev
```
Open **`http://localhost:5173`**.

1. Open [Google Colab](https://colab.research.google.com).
2. Click **Upload** and upload [`colab_multimodal_rag.ipynb`](./colab_multimodal_rag.ipynb).
3. Click:
   > **`Runtime`** ➔ **`Run all`** (<kbd>Ctrl</kbd> + <kbd>F9</kbd>)
4. Run each cell sequentially to inspect:
   - Chunk inspection (`all_chunks[0]`, `all_chunks[3]`)
   - Vector dimensions (`384`)
   - Direct retriever outputs
   - Generated multimodal answers with inline images!
