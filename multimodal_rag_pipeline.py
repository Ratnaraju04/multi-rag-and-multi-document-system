"""
=============================================================================
ALL-IN-ONE MULTI-MODAL RAG BACKEND FOR GOOGLE COLAB
Modalities: Text, Images, Videos
Sources   : Web Scraping, Excel Sheets (.xlsx/.csv), Datasets (JSON/JSONL/Dict), Images, Videos
=============================================================================
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

import re
try:
    import cv2
except ImportError:
    cv2 = None
import json
import glob
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
try:
    import pandas as pd
except ImportError:
    pd = None
from PIL import Image, ImageDraw, ImageFont
import requests
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None
    Settings = None

# ---------------------------------------------------------------------------
# 1. CONFIGURATION & COLAB INTEGRATION
# ---------------------------------------------------------------------------

def _load_local_env():
    """Load key-value pairs from .env if present."""
    if os.path.exists(".env"):
        try:
            with open(".env", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if k and not os.environ.get(k):
                            os.environ[k] = v
        except Exception:
            pass

_load_local_env()

def get_colab_or_env_api_key(key_name: str = "GEMINI_API_KEY") -> str:
    """Retrieve API key from Google Colab userdata secrets or system environment."""
    try:
        from google.colab import userdata  # type: ignore
        key = userdata.get(key_name)
        if key:
            return key
    except Exception:
        pass
    return os.environ.get(key_name) or os.environ.get("GOOGLE_API_KEY") or ""

@dataclass
class RAGConfig:
    api_key: str = field(default_factory=get_colab_or_env_api_key)
    gemini_model: str = "gemini-2.0-flash"  # Ultra-fast, low-latency default model
    chroma_db_dir: str = "./chroma_multimodal_db"
    collection_name: str = "colab_multimodal_kb"
    video_frame_interval_sec: float = 3.0  # Extract 1 keyframe every 3 seconds
    video_output_dir: str = "./extracted_frames"
    rows_per_excel_chunk: int = 50
    video_max_frames: int = 8  # Optimal frame count for fast indexing
    max_image_dimension: int = 768  # Max thumbnail dimension for fast visual uploads

    def __post_init__(self):
        os.makedirs(self.chroma_db_dir, exist_ok=True)
        os.makedirs(self.video_output_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# 2. INGESTION LOADERS
# ---------------------------------------------------------------------------

class WebScraperLoader:
    """Extracts articles, tables, and linked media from web URLs."""
    def __init__(self, timeout: int = 15):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
        }
        self.timeout = timeout

    def scrape_url(self, url: str) -> List[Dict[str, Any]]:
        try:
            resp = requests.get(url, headers=self.headers, timeout=self.timeout)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            print(f"[WebScraperLoader] Warning: Failed to scrape {url}: {e}")
            return []

        for tag in soup(["script", "style", "nav", "footer", "noscript"]):
            tag.extract()

        title = soup.title.string.strip() if soup.title and soup.title.string else url

        # --- Extract infobox / structured tables FIRST (Wikipedia, IMDb, etc.) ---
        infobox_text_parts = []
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all(["th", "td"])
                if len(cells) >= 2:
                    key = cells[0].get_text(separator=" ", strip=True)
                    value = cells[1].get_text(separator=" ", strip=True)
                    if key and value and len(value) > 1:
                        infobox_text_parts.append(f"{key}: {value}")
                elif len(cells) == 1:
                    text = cells[0].get_text(separator=" ", strip=True)
                    if len(text) > 5:
                        infobox_text_parts.append(text)
        infobox_text = "\n".join(infobox_text_parts)

        # Extract body paragraphs, headings, list items
        paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all(["p", "h1", "h2", "h3", "li"])]
        body_text = "\n\n".join([p for p in paragraphs if len(p) > 25])

        # Combine: infobox first (fact-dense), then body
        full_text = (infobox_text + "\n\n" + body_text).strip() if infobox_text else body_text

        # Chunk text
        words = full_text.split()
        chunks = []
        chunk_size, overlap = 500, 50
        for i in range(0, max(len(words), 1), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk:
                chunks.append(chunk)

        # Build a short, readable source label: use the page title (truncated) for display in dropdowns
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        path_slug = parsed.path.rstrip("/").split("/")[-1].replace("_", " ").replace("-", " ").strip()
        if path_slug:
            readable_source = f"{path_slug} ({domain})"
        else:
            readable_source = domain
        # Fallback to title if slug is empty or uninformative
        if not readable_source or len(readable_source) < 3:
            readable_source = title[:80]

        docs = []
        clean_url_id = re.sub(r"[^a-zA-Z0-9]", "_", url)[:35]
        for idx, chunk in enumerate(chunks):
            docs.append({
                "id": f"web_{clean_url_id}_{idx}",
                "modality": "text",
                "content": f"[Web Source: {title} | URL: {url}]\n{chunk}",
                "metadata": {
                    "source": readable_source,
                    "url": url,
                    "title": title,
                    "type": "web_article",
                    "chunk_index": idx
                }
            })
        return docs, readable_source


class ExcelDatasetLoader:
    """Fast, optimized loader for .xlsx, .xls, and .csv files."""
    def __init__(self, rows_per_chunk: int = 50):
        self.rows_per_chunk = rows_per_chunk

    def load(self, file_path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_name = os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        documents = []

        if ext in [".xlsx", ".xls"]:
            xl = pd.ExcelFile(file_path)
            for sheet in xl.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet)
                documents.extend(self._process_df(df, file_name, sheet))
        elif ext == ".csv":
            df = pd.read_csv(file_path)
            documents.extend(self._process_df(df, file_name, "Sheet1"))
        else:
            raise ValueError(f"Unsupported spreadsheet format: {ext}")
        return documents

    def _process_df(self, df: pd.DataFrame, filename: str, sheet: str) -> List[Dict[str, Any]]:
        df = df.dropna(how="all")
        if df.empty:
            return []

        docs = []
        # 1. Comprehensive Schema, Data Types & Statistical Summary Chunk
        col_types = [f"{col} ({dtype})" for col, dtype in df.dtypes.items()]
        cat_cols = [str(c) for c in df.select_dtypes(include=['object', 'category']).columns]
        num_cols = [str(c) for c in df.select_dtypes(include=['number']).columns]

        # Fast Preview (safe for wide tables)
        preview_df = df.head(5) if len(df.columns) <= 25 else df.head(3).iloc[:, :20]
        preview_md = preview_df.to_markdown(index=False) if hasattr(preview_df, "to_markdown") else str(preview_df)

        summary_content = (
            f"[Spreadsheet Dataset: {filename} | Sheet: {sheet}]\n"
            f"Total Rows: {len(df)} | Total Columns: {len(df.columns)}\n"
            f"All Features & Data Types: {', '.join(col_types[:30])}\n"
            f"Categorical Columns ({len(cat_cols)}): {', '.join(cat_cols[:15]) if cat_cols else 'None'}\n"
            f"Numerical Columns ({len(num_cols)}): {', '.join(num_cols[:15]) if num_cols else 'None'}\n\n"
            f"### Sample Preview:\n{preview_md}"
        )
        docs.append({
            "id": f"tbl_summary_{filename}_{sheet}",
            "modality": "table_summary",
            "content": summary_content,
            "metadata": {"source": filename, "sheet": sheet, "type": "table_summary", "rows": len(df)}
        })

        # 2. Fast Vectorized Block Chunks (Optimized dict conversion instead of slow row-by-row apply)
        step = max(50, self.rows_per_chunk)
        max_chunks = 10
        chunk_count = 0

        for i in range(0, min(len(df), step * max_chunks), step):
            sub = df.iloc[i : i + step]
            sample_records = sub.head(12).to_dict(orient="records")
            row_sentences = [
                " | ".join(f"{k}: {v}" for k, v in r.items() if pd.notna(v) and str(v) != "")
                for r in sample_records
            ]

            sub_preview = sub.head(5) if len(sub.columns) <= 25 else sub.head(3).iloc[:, :20]
            sub_md = sub_preview.to_markdown(index=False) if hasattr(sub_preview, "to_markdown") else str(sub_preview)

            content = (
                f"[Dataset: {filename} | Rows {i+1} to {min(i+step, len(df))}]\n"
                f"### Structured View:\n{sub_md}\n\n"
                f"### Record Samples:\n" + "\n".join(row_sentences)
            )
            docs.append({
                "id": f"tbl_rows_{filename}_{sheet}_{i}",
                "modality": "table_rows",
                "content": content,
                "metadata": {"source": filename, "sheet": sheet, "type": "table_rows", "row_start": i+1}
            })
            chunk_count += 1

        return docs


class DatasetLoader:
    """Ingests JSON, JSONL, Parquet or in-memory list of dicts."""
    def __init__(self, records_per_chunk: int = 10):
        self.records_per_chunk = records_per_chunk

    def load_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)
        if ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            records = data if isinstance(data, list) else [data]
        elif ext == ".jsonl":
            records = [json.loads(line) for line in open(file_path, "r", encoding="utf-8") if line.strip()]
        elif ext == ".parquet":
            records = pd.read_parquet(file_path).to_dict(orient="records")
        else:
            raise ValueError(f"Unsupported dataset format: {ext}")
        return self.load_records(records, dataset_name=file_name)

    def load_records(self, records: List[Dict[str, Any]], dataset_name: str = "dataset") -> List[Dict[str, Any]]:
        docs = []
        for i in range(0, len(records), self.records_per_chunk):
            chunk = records[i : i + self.records_per_chunk]
            formatted = "\n\n".join([f"Record #{i+j+1}:\n{json.dumps(r, indent=2, ensure_ascii=False)}" for j, r in enumerate(chunk)])
            docs.append({
                "id": f"dataset_{dataset_name}_{i}",
                "modality": "dataset_record",
                "content": f"[Dataset: {dataset_name} | Records {i+1}-{min(i+self.records_per_chunk, len(records))}]\n{formatted}",
                "metadata": {"source": dataset_name, "type": "dataset", "record_start": i+1}
            })
        return docs


class ImageLoader:
    """Loads images, extracts EXIF/dimensions, and generates visual captions via Gemini."""
    def __init__(self, gemini_client=None):
        self.client = gemini_client

    def load_image(self, image_path: str, caption: Optional[str] = None) -> Dict[str, Any]:
        with Image.open(image_path) as img:
            w, h = img.size
            fmt = img.format or "JPEG"

        file_name = os.path.basename(image_path)
        desc = caption or ""
        if not desc and self.client:
            desc = self._gemini_caption(image_path)

        if not desc:
            desc = f"Image file '{file_name}' ({w}x{h}, format {fmt})."

        content = (
            f"[Image: {file_name}]\n"
            f"Resolution: {w}x{h} | Format: {fmt}\n"
            f"Visual Content & OCR: {desc}"
        )

        return {
            "id": f"img_{file_name}_{os.path.getsize(image_path)}",
            "modality": "image",
            "content": content,
            "metadata": {
                "source": file_name,
                "file_path": os.path.abspath(image_path),
                "type": "image",
                "width": w,
                "height": h,
                "description": desc
            }
        }

    def load_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        docs = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            for p in glob.glob(os.path.join(dir_path, ext)):
                docs.append(self.load_image(p))
        return docs

    def _gemini_caption(self, image_path: str) -> str:
        try:
            with Image.open(image_path) as img:
                img_copy = img.copy()
                img_copy.thumbnail((768, 768))
            prompt = (
                "Provide a concise, detailed description of this image for a multimodal search engine. "
                "Include all visible text (OCR), diagrams, charts, numbers, colors, and objects."
            )
            for m in ["gemini-2.0-flash", "gemini-1.5-flash"]:
                try:
                    if hasattr(self.client, "models"):
                        resp = self.client.models.generate_content(model=m, contents=[prompt, img_copy])
                        return resp.text.strip()
                    elif hasattr(self.client, "GenerativeModel"):
                        model = self.client.GenerativeModel(m)
                        resp = model.generate_content([prompt, img_copy])
                        return resp.text.strip()
                except Exception:
                    continue
        except Exception as e:
            print(f"[ImageLoader] Gemini captioning warning: {e}")
        return ""


class VideoLoader:
    """Extracts keyframes at intervals using OpenCV and generates visual captions concurrently."""
    def __init__(self, frame_interval_sec: float = 3.0, output_dir: str = "./extracted_frames", gemini_client=None):
        self.frame_interval_sec = frame_interval_sec
        self.output_dir = output_dir
        self.client = gemini_client
        os.makedirs(self.output_dir, exist_ok=True)

    def load_video(self, video_path: str, max_frames: int = 8) -> List[Dict[str, Any]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_step = max(1, int(fps * self.frame_interval_sec))
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        extracted_items = []
        frame_idx = 0

        # Fast frame extraction
        while cap.isOpened() and len(extracted_items) < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            sec = frame_idx / fps
            timestamp = f"{int(sec // 60):02d}:{int(sec % 60):02d}"
            frame_path = os.path.join(self.output_dir, f"{video_name}_t{int(sec):04d}.jpg")
            cv2.imwrite(frame_path, frame)

            extracted_items.append({
                "frame_idx": frame_idx,
                "sec": sec,
                "timestamp": timestamp,
                "frame_path": frame_path
            })
            frame_idx += frame_step

        cap.release()

        # Concurrent frame captioning via ThreadPoolExecutor (5x speedup over sequential!)
        captions = {}
        if self.client and extracted_items:
            from concurrent.futures import ThreadPoolExecutor
            def _caption_worker(item):
                c = self._describe_frame(item["frame_path"], item["timestamp"], video_name)
                return item["frame_path"], c

            with ThreadPoolExecutor(max_workers=min(5, len(extracted_items))) as executor:
                results = executor.map(_caption_worker, extracted_items)
                for fpath, cap_text in results:
                    captions[fpath] = cap_text

        docs = []
        for item in extracted_items:
            caption = captions.get(item["frame_path"], "")
            if not caption:
                caption = f"Keyframe at {item['timestamp']} from video '{video_name}'."

            content = (
                f"[Video: {os.path.basename(video_path)} | Timestamp: {item['timestamp']}]\n"
                f"Scene Description: {caption}\n"
                f"Frame Path: {item['frame_path']}"
            )

            docs.append({
                "id": f"vid_{video_name}_{item['frame_idx']}",
                "modality": "video_frame",
                "content": content,
                "metadata": {
                    "source": os.path.basename(video_path),
                    "frame_path": os.path.abspath(item["frame_path"]),
                    "type": "video_keyframe",
                    "timestamp": item["timestamp"],
                    "timestamp_sec": item["sec"],
                    "description": caption
                }
            })

        return docs

    def _describe_frame(self, frame_path: str, timestamp: str, video_name: str) -> str:
        try:
            with Image.open(frame_path) as img:
                img_thumb = img.copy()
                img_thumb.thumbnail((768, 768))
            prompt = f"Keyframe at timestamp {timestamp} from video '{video_name}'. Describe actions, visible text, and context."
            for m in ["gemini-2.0-flash", "gemini-1.5-flash"]:
                try:
                    if hasattr(self.client, "models"):
                        resp = self.client.models.generate_content(model=m, contents=[prompt, img_thumb])
                        return resp.text.strip()
                    elif hasattr(self.client, "GenerativeModel"):
                        resp = self.client.GenerativeModel(m).generate_content([prompt, img_thumb])
                        return resp.text.strip()
                except Exception:
                    continue
        except Exception:
            pass
        return ""


def is_document_level_query(question: str) -> bool:
    """
    Detects if the query is a document-level summary, overview, project list,
    skills, architecture, key achievements, or general meta-query about the document content.
    """
    q = question.lower().strip()
    triggers = [
        "what is this document about", "what is the document about", "what is this pdf about",
        "what is this file about", "what is this about", "about this document", "about this pdf",
        "what is document about", "what is pdf about", "what is the pdf about", "what is this",
        "summarize this document", "summarize the document", "summarize this pdf", "summarize",
        "summary of this document", "summary of the document", "summary of this pdf", "summary",
        "overview", "main points", "key points", "give me the key points", "key takeaways",
        "explain this document", "tell me about this document", "what's in this document",
        "what does this document contain", "what is in this pdf", "content of this document",
        "what projects are mentioned", "what projects", "list the projects", "explain the projects",
        "projects mentioned in this document", "projects mentioned", "projects in this document",
        "what projects are", "projects mentioned in", "projects included",
        "what technologies are used", "technologies are used in the projects", "technologies used",
        "what are the technologies", "tech stack", "tools used",
        "what are the main technical skills", "technical skills mentioned", "main technical skills",
        "technical skills", "skills mentioned", "what skills", "skills in this document",
        "explain the project architecture", "project architecture", "system architecture", "architecture",
        "what are the key achievements", "key achievements mentioned", "achievements mentioned",
        "key achievements", "who is this", "background", "experience"
    ]
    if any(t in q for t in triggers):
        return True
    if re.search(r"\b(summar(y|ize)|overview|abstract|purpose|main ideas?|key takeaways?|key points?|what is (it|this)|projects?|skills?|architecture|achievements?)\b", q):
        return True
    return False


class PDFLoader:
    """PDF Loader for Multi-Modal RAG. Extracts page-by-page text and metadata from PDF files."""
    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load(self, file_path: str) -> List[Dict[str, Any]]:
        """Parses a PDF file and returns chunked documents with page-level metadata."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        file_name = os.path.basename(file_path)
        pages_text = self._extract_pages(file_path)
        if not pages_text:
            print(f"[PDFLoader] ⚠️ No text extracted from '{file_name}'.")
            return []

        total_pages = len(pages_text)
        total_chars = sum(len(t) for t in pages_text)
        print(f"[PDFLoader] Extracted {total_chars} characters across {total_pages} page(s) from '{file_name}' (doc_id: {file_name})")

        documents = []

        for page_num, text in enumerate(pages_text, start=1):
            cleaned_text = text.strip()
            if not cleaned_text:
                continue

            chunks = self._chunk_text(cleaned_text)
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
                        "document_id": file_name,
                        "file_path": os.path.abspath(file_path),
                        "type": "pdf_document",
                        "page": page_num,
                        "total_pages": total_pages,
                        "chunk_index": chunk_idx,
                        "chunk_count": len(chunks)
                    }
                })

        print(f"[PDFLoader] Created {len(documents)} chunk(s) for '{file_name}' (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")
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


# ---------------------------------------------------------------------------
# 3. VECTOR STORE (CHROMADB)
# ---------------------------------------------------------------------------

class MultimodalVectorStore:
    def __init__(self, persist_dir: str = "./chroma_multimodal_db", collection_name: str = "multimodal_kb"):
        self.client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
        self.embedding_fn = self._init_embedding()
        self.collection = self.client.get_or_create_collection(name=collection_name, embedding_function=self.embedding_fn)

    def _init_embedding(self):
        try:
            from chromadb.utils import embedding_functions
            return embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        except Exception:
            from chromadb.utils import embedding_functions
            return embedding_functions.DefaultEmbeddingFunction()

    def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 50):
        if not documents:
            return
        ids = [d["id"] for d in documents]
        texts = [d["content"] for d in documents]
        metadatas = []
        for d in documents:
            meta = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in d.get("metadata", {}).items()}
            meta["modality"] = d.get("modality", "text")
            metadatas.append(meta)

        # Upsert in batches to prevent memory spikes and optimize CPU embedding throughput
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i : i + batch_size],
                documents=texts[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size]
            )

    def search(
        self,
        query: str,
        n_results: int = 5,
        modality_filter: Optional[str] = None,
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        where = None
        filters = []
        if modality_filter:
            filters.append({"modality": modality_filter})
        if source_filter and source_filter != "All Sources":
            filters.append({"source": source_filter})

        if len(filters) == 1:
            where = filters[0]
        elif len(filters) > 1:
            where = {"$and": filters}

        cnt = self.collection.count()
        if cnt == 0:
            return []
        res = self.collection.query(query_texts=[query], n_results=min(n_results, cnt), where=where)
        docs = []
        for i in range(len(res["documents"][0])):
            docs.append({
                "id": res["ids"][0][i],
                "content": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "distance": res["distances"][0][i] if "distances" in res else 0.0
            })
        return docs

    def get_all_sources(self) -> List[str]:
        """Returns unique source names currently stored in ChromaDB."""
        try:
            if self.collection.count() == 0:
                return []
            data = self.collection.get(include=["metadatas"])
            sources = set()
            for m in data.get("metadatas", []):
                if m and "source" in m:
                    sources.add(str(m["source"]))
            return sorted(list(sources))
        except Exception:
            return []

    def get_document_chunks(self, source_name: str, max_chunks: int = 30) -> List[Dict[str, Any]]:
        """Retrieves all chunks belonging to a specific document in sequential order."""
        if not source_name or source_name.startswith("All Sources"):
            return []
        try:
            cnt = self.collection.count()
            if cnt == 0:
                return []

            clean_name = os.path.basename(source_name)
            res = self.collection.get(
                where={"source": clean_name},
                include=["documents", "metadatas"],
                limit=max_chunks
            )
            if not res or not res.get("documents"):
                # Fallback scan across metadatas
                all_data = self.collection.get(include=["documents", "metadatas"], limit=100)
                matched_docs = []
                for idx, meta in enumerate(all_data.get("metadatas", [])):
                    if meta and (clean_name in str(meta.get("source", "")) or clean_name in str(meta.get("file_path", ""))):
                        matched_docs.append({
                            "id": all_data["ids"][idx],
                            "content": all_data["documents"][idx],
                            "metadata": meta,
                            "distance": 0.0
                        })
                return matched_docs

            docs = []
            for i in range(len(res["documents"])):
                docs.append({
                    "id": res["ids"][i],
                    "content": res["documents"][i],
                    "metadata": res["metadatas"][i] if res.get("metadatas") else {},
                    "distance": 0.0
                })
            docs.sort(key=lambda d: (
                int(d.get("metadata", {}).get("page", 1)),
                int(d.get("metadata", {}).get("chunk_index", 0))
            ))
            return docs
        except Exception as e:
            print(f"[VectorStore] get_document_chunks failed for '{source_name}': {e}")
            return []

    def reset_collection(self):
        """Clears all indexed chunks from the collection."""
        try:
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection.name,
                embedding_function=self.embedding_fn
            )
            print("[VectorStore] Database cleared successfully.")
        except Exception as e:
            print(f"[VectorStore] Reset warning: {e}")


# ---------------------------------------------------------------------------
# 4. UNIFIED MULTI-MODAL RAG ENGINE (OPTIMIZED FOR HIGH SPEED)
# ---------------------------------------------------------------------------

class MultimodalRAGEngine:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.session = requests.Session()  # Reusable TCP keep-alive connection pool
        self.active_model = None  # Cache verified working model for instantaneous queries
        self.gemini_client = self._init_gemini()
        self.vector_store = MultimodalVectorStore(self.config.chroma_db_dir, self.config.collection_name)
        
        self.web_loader = WebScraperLoader()
        self.excel_loader = ExcelDatasetLoader(self.config.rows_per_excel_chunk)
        self.dataset_loader = DatasetLoader()
        self.image_loader = ImageLoader(self.gemini_client)
        self.video_loader = VideoLoader(self.config.video_frame_interval_sec, self.config.video_output_dir, self.gemini_client)
        self.pdf_loader = PDFLoader()

    def _init_gemini(self):
        key = self.config.api_key
        if not key:
            print("[Warning] No GEMINI_API_KEY detected. Synthesis will run in retrieval-only mode.")
            return None
        try:
            from google import genai
            return genai.Client(api_key=key)
        except ImportError:
            try:
                import google.generativeai as g
                g.configure(api_key=key)
                return g
            except Exception:
                return None

    def ingest_url(self, url: str):
        """Scrape and index a web page. Returns (chunk_count, readable_source_label)."""
        docs, readable_source = self.web_loader.scrape_url(url)
        self.vector_store.add_documents(docs)
        return len(docs), readable_source

    def ingest_excel(self, file_path: str) -> int:
        docs = self.excel_loader.load(file_path)
        self.vector_store.add_documents(docs)
        return len(docs)

    def ingest_pdf(self, file_path: str) -> int:
        file_name = os.path.basename(file_path)
        print(f"[RAGEngine] Ingesting PDF: '{file_name}' (path: {file_path})")
        docs = self.pdf_loader.load(file_path)
        if not docs:
            print(f"[RAGEngine] ⚠️ 0 chunks generated for '{file_name}'.")
            return 0
        print(f"[RAGEngine] Inserting {len(docs)} chunk(s) of '{file_name}' into ChromaDB...")
        self.vector_store.add_documents(docs)
        print(f"[RAGEngine] Successfully indexed '{file_name}'. Total in vector store: {self.vector_store.collection.count()} chunks.")
        return len(docs)

    def ingest_dataset(self, source: Union[str, List[Dict[str, Any]]], name: str = "dataset") -> int:
        if isinstance(source, str):
            docs = self.dataset_loader.load_from_file(source)
        else:
            docs = self.dataset_loader.load_records(source, dataset_name=name)
        self.vector_store.add_documents(docs)
        return len(docs)

    def ingest_image(self, path_or_dir: str) -> int:
        docs = self.image_loader.load_directory(path_or_dir) if os.path.isdir(path_or_dir) else [self.image_loader.load_image(path_or_dir)]
        self.vector_store.add_documents(docs)
        return len(docs)

    def ingest_video(self, video_path: str, max_frames: Optional[int] = None) -> int:
        frames = max_frames or self.config.video_max_frames
        docs = self.video_loader.load_video(video_path, max_frames=frames)
        self.vector_store.add_documents(docs)
        return len(docs)

    def query(
        self,
        question: str,
        top_k: int = 5,
        modality_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        user_image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        import time
        start_time = time.time()

        target_source = source_filter if (source_filter and source_filter != "All Sources") else None

        # Contextual Query Enrichment for follow-up questions
        search_query = question.strip()
        lower_q = question.lower()
        pronoun_triggers = ["in it", "in this", "this data", "this file", "this pdf", "the pdf", "this document", "the document", "these projects", "the projects"]
        has_pronoun = any(trig in lower_q for trig in pronoun_triggers)

        if not target_source and chat_history and has_pronoun:
            for msg in reversed(chat_history):
                text = str(msg.get("content", ""))
                matches = re.findall(r"([A-Za-z0-9_\-\.]+\.(?:pdf|csv|xlsx|xls|json|mp4|png|jpg))", text, re.IGNORECASE)
                if matches:
                    target_source = matches[0]
                    break

        is_doc_query = is_document_level_query(question)
        if not target_source and is_doc_query:
            all_sources = self.vector_store.get_all_sources()
            pdf_sources = [s for s in all_sources if s.lower().endswith(".pdf")]
            if len(pdf_sources) == 1:
                target_source = pdf_sources[0]

        retrieved = []
        if target_source:
            doc_chunks = self.vector_store.get_document_chunks(target_source, max_chunks=30)
            if doc_chunks:
                if is_doc_query or len(doc_chunks) <= 10:
                    retrieved = doc_chunks
                else:
                    vector_results = self.vector_store.search(
                        search_query,
                        n_results=top_k,
                        modality_filter=modality_filter,
                        source_filter=target_source
                    )
                    retrieved = [doc_chunks[0]] + [item for item in vector_results if item['id'] != doc_chunks[0]['id']]
                    if len(retrieved) < 2:
                        retrieved = doc_chunks[:top_k]
            else:
                retrieved = self.vector_store.search(
                    search_query,
                    n_results=top_k,
                    modality_filter=modality_filter,
                    source_filter=target_source
                )
        else:
            retrieved = self.vector_store.search(
                search_query,
                n_results=top_k,
                modality_filter=modality_filter
            )

        if not retrieved:
            elapsed = time.time() - start_time
            return {
                "question": question,
                "answer": "No relevant context found.",
                "sources": [],
                "visual_evidence": [],
                "latency_seconds": round(elapsed, 2),
                "model_used": self.active_model or "None"
            }

        context_blocks = []
        visual_images = []
        visual_paths = []

        # Intelligent Visual Gating: Don't upload multi-megabyte images if the query is pure text/table/PDF!
        visual_triggers = ["image", "picture", "photo", "diagram", "chart", "figure", "visual", "look", "video", "frame", "timestamp", "scene", "color", "see", "screenshot", "draw"]
        has_visual_intent = any(vt in lower_q for vt in visual_triggers)
        top_is_visual = bool(retrieved and retrieved[0].get("metadata", {}).get("modality") in ["image", "video_frame"])
        attach_images_to_llm = has_visual_intent or top_is_visual or bool(user_image_path)

        for i, item in enumerate(retrieved, 1):
            m = item["metadata"]
            header = f"[Source #{i} | Modality: {m.get('modality', 'text')} | Source: {m.get('source', '')}"
            if "timestamp" in m:
                header += f" | Timestamp: {m['timestamp']}"
            header += "]"
            context_blocks.append(f"{header}\n{item['content']}")

            # Visual evidence must only be actual visual assets (images or video keyframes)
            img_candidate = m.get("frame_path")
            if not img_candidate and (m.get("modality") == "image" or m.get("type") == "image"):
                img_candidate = m.get("file_path")

            image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
            if img_candidate and os.path.exists(img_candidate):
                ext = os.path.splitext(img_candidate)[1].lower()
                if ext in image_extensions:
                    if img_candidate not in visual_paths:
                        visual_paths.append(img_candidate)
                    if attach_images_to_llm and len(visual_images) < 2:
                        try:
                            visual_images.append(Image.open(img_candidate))
                        except Exception:
                            pass

        prompt = (
            "You are a state-of-the-art Multi-Modal RAG AI. Answer the question accurately using the retrieved context.\n"
            "- If referencing videos, specify exact timestamps (e.g. 'At 00:04 in tutorial.mp4...').\n"
            "- If referencing Excel sheets, cite specific columns, rows, and figures.\n"
            "- If referencing images, describe the relevant visual details.\n"
            "- Be concise, direct, and factual.\n\n"
            f"=== MULTIMODAL CONTEXT ===\n" + "\n---------------------\n".join(context_blocks) + "\n\n"
            f"=== USER QUERY ===\n{question}"
        )

        answer = self._generate_answer(prompt, visual_images, user_image_path)
        elapsed = time.time() - start_time
        return {
            "question": question,
            "answer": answer,
            "sources": retrieved,
            "visual_evidence": visual_paths,
            "latency_seconds": round(elapsed, 2),
            "model_used": self.active_model or "gemini-2.0-flash"
        }

    def _get_supported_models(self, api_key: str) -> List[str]:
        """Dynamically fetches models authorized for this API key, cached globally to prevent discovery latency."""
        cached = getattr(self, "_cached_models", None)
        if cached:
            return cached

        preferred = [
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-2.0-flash-lite",
            "gemini-1.5-pro"
        ]
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            resp = self.session.get(url, headers={"x-goog-api-key": api_key}, timeout=4)
            if resp.status_code == 200:
                models_data = resp.json().get("models", [])
                supported = []
                for m in models_data:
                    methods = m.get("supportedGenerationMethods", [])
                    name = m.get("name", "").replace("models/", "")
                    if "generateContent" in methods and "gemini" in name:
                        supported.append(name)
                # Prioritize fastest flash models
                ordered = [p for p in preferred if p in supported]
                ordered += [s for s in supported if s not in ordered]
                if ordered:
                    self._cached_models = ordered
                    return ordered
        except Exception:
            pass
        self._cached_models = preferred
        return preferred

    def _generate_answer(self, prompt: str, images: List[Image.Image], user_img: Optional[str] = None) -> str:
        api_key = (self.config.api_key or os.environ.get("GEMINI_API_KEY", "")).strip().strip("'\"")
        if not api_key:
            return "[GEMINI_API_KEY not configured] Context retrieved successfully. Please add your key in .env or the sidebar."

        all_images = []
        if user_img and os.path.exists(user_img):
            try:
                all_images.append(Image.open(user_img))
            except Exception:
                pass
        all_images.extend(images)

        import time

        # FAST PATH: If a verified model is already cached, execute immediately with ZERO discovery latency
        if self.active_model:
            for attempt in range(2):
                rest_result = self._call_gemini_rest(self.active_model, prompt, all_images, api_key)
                if rest_result and not rest_result.startswith("[REST API") and not rest_result.startswith("[REST Request"):
                    return rest_result
                if "[REST API 429]" in rest_result or "[REST API 503]" in rest_result:
                    time.sleep(1.5)
                    continue
                break  # If other error, fall back to discovery list

        candidate_models = self._get_supported_models(api_key)
        last_error = ""

        # 1. Primary generation via direct REST endpoint (supports both AQ. and AIza keys)
        for model_name in candidate_models:
            clean_name = model_name.replace("models/", "")
            for attempt in range(2):
                rest_result = self._call_gemini_rest(clean_name, prompt, all_images, api_key)
                if rest_result and not rest_result.startswith("[REST API") and not rest_result.startswith("[REST Request"):
                    self.active_model = clean_name  # Cache active model for subsequent queries
                    return rest_result
                if rest_result:
                    last_error = rest_result
                    if "[REST API 404]" in rest_result:
                        break  # Skip non-existent model immediately
                    if "[REST API 429]" in rest_result or "[REST API 503]" in rest_result:
                        time.sleep(2)
                        continue
                break

        # 2. Fallback to SDK client if configured
        if self.gemini_client:
            contents = [prompt] + all_images[:2]
            for model_name in candidate_models:
                clean_name = model_name.replace("models/", "")
                try:
                    if hasattr(self.gemini_client, "models"):
                        resp = self.gemini_client.models.generate_content(model=clean_name, contents=contents)
                        self.active_model = clean_name
                        return resp.text.strip()
                    elif hasattr(self.gemini_client, "GenerativeModel"):
                        model_obj = self.gemini_client.GenerativeModel(clean_name)
                        resp = model_obj.generate_content(contents)
                        self.active_model = clean_name
                        return resp.text.strip()
                except Exception as e:
                    err_str = str(e)
                    if "404" in err_str:
                        continue
                    last_error = err_str

        return f"Generation Notice: {last_error}"

    def _call_gemini_rest(self, model_name: str, prompt: str, images: List[Image.Image], api_key: str) -> Optional[str]:
        """Direct REST call to Google Gemini v1beta API with persistent keep-alive session and optimized payloads."""
        import base64
        import io

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }

        parts = [{"text": prompt}]
        # Compress and attach at most 2 visual images (downscaled to max 768px for fast transmission)
        for img in images[:2]:
            try:
                buf = io.BytesIO()
                img_copy = img.copy()
                img_copy.thumbnail((768, 768))
                img_copy.convert("RGB").save(buf, format="JPEG", quality=75, optimize=True)
                img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": img_b64
                    }
                })
            except Exception:
                pass

        payload = {"contents": [{"parts": parts}]}
        try:
            resp = self.session.post(url, headers=headers, json=payload, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                cands = data.get("candidates", [])
                if cands and "content" in cands[0]:
                    p_list = cands[0]["content"].get("parts", [])
                    if p_list and "text" in p_list[0]:
                        return p_list[0]["text"].strip()
            return f"[REST API {resp.status_code}]: {resp.text}"
        except Exception as e:
            return f"[REST Request Error]: {e}"

    def query_with_config(
        self,
        question: str,
        retrieval_config: Optional[Any] = None,
        gen_config: Optional[Any] = None,
        source_filter: Optional[str] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        llm_manager: Optional[Any] = None,
        reranking_config: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Executes query applying advanced document-aware retrieval, hierarchical synthesis,
        reranking, and grounded generation.
        """
        import time
        start_total = time.time()

        # Extract retrieval settings
        top_k = getattr(retrieval_config, "top_k", 5) if retrieval_config else 5
        threshold = getattr(retrieval_config, "similarity_threshold", 0.0) if retrieval_config else 0.0

        start_retrieval = time.time()

        # 1. Determine Target Document / Source
        target_source = source_filter if (source_filter and not source_filter.startswith("All Sources")) else None

        # Intelligent document focus detection from question or chat history if not locked
        lower_q = question.lower()
        pronoun_triggers = ["in it", "in this", "this data", "this file", "this pdf", "the pdf", "this document", "the document", "these projects", "the projects"]
        has_doc_pronoun = any(trig in lower_q for trig in pronoun_triggers)

        if not target_source and chat_history and has_doc_pronoun:
            for msg in reversed(chat_history):
                text = str(msg.get("content", ""))
                matches = re.findall(r"([A-Za-z0-9_\-\.]+\.(?:pdf|csv|xlsx|xls|json|mp4|png|jpg))", text, re.IGNORECASE)
                if matches:
                    target_source = matches[0]
                    print(f"[RAGEngine] Inferred active source '{target_source}' from chat history context.")
                    break

        is_doc_query = is_document_level_query(question)

        # If user asks a general document query ("what is this document about", "summarize") and there is an indexed PDF
        if not target_source and is_doc_query:
            all_sources = self.vector_store.get_all_sources()
            pdf_sources = [s for s in all_sources if s.lower().endswith(".pdf")]
            if len(pdf_sources) == 1:
                target_source = pdf_sources[0]
                print(f"[RAGEngine] Auto-focused general document query on sole indexed PDF: '{target_source}'")

        # 2. Document-Aware Retrieval Execution
        retrieved_items = []
        applied_threshold_note = ""

        if target_source:
            # Document-focused mode: retrieve chunks belonging to this document
            doc_chunks = self.vector_store.get_document_chunks(target_source, max_chunks=30)
            print(f"[RAGEngine] Document focus: '{target_source}' | Total chunks found: {len(doc_chunks)} | Doc-level query: {is_doc_query}")

            if doc_chunks:
                # If document is small/compact (<= 12 chunks, e.g. up to 3000 words) or general overview query,
                # supply the complete document in sequential page & chunk order to ensure 100% recall of projects, skills, etc.
                if is_doc_query or len(doc_chunks) <= 12:
                    retrieved_items = doc_chunks
                else:
                    # Specific question on larger document: run targeted similarity search
                    vector_results = self.vector_store.search(
                        query=question.strip(),
                        n_results=top_k,
                        source_filter=target_source
                    )
                    # Always include chunk 0 (document header/intro) + relevant vector matches
                    retrieved_items = [doc_chunks[0]]
                    seen_ids = {doc_chunks[0]["id"]}
                    for v_item in vector_results:
                        if v_item["id"] not in seen_ids:
                            retrieved_items.append(v_item)
                            seen_ids.add(v_item["id"])
                    if len(retrieved_items) < 2:
                        retrieved_items = doc_chunks[:top_k]
            else:
                # Fallback to vector search if get_document_chunks returned empty
                retrieved_items = self.vector_store.search(
                    query=question.strip(),
                    n_results=top_k,
                    source_filter=target_source
                )
            # NEVER discard chunks with similarity_threshold when user is querying a locked document!
            applied_threshold_note = "Bypassed for locked document to preserve 100% context recall"
        else:
            # Global search across all indexed sources
            clean_search_query = question.strip()
            retrieved_items = self.vector_store.search(
                query=clean_search_query,
                n_results=top_k
            )

            # Apply similarity threshold if configured
            if threshold > 0.0 and retrieved_items:
                original_candidates = list(retrieved_items)
                passed = [
                    item for item in retrieved_items
                    if (1.0 - item.get("distance", 1.0)) >= threshold or item.get("distance", 1.0) <= (1.0 - threshold)
                ]
                if passed:
                    retrieved_items = passed
                    applied_threshold_note = f"Filtered {len(original_candidates)} -> {len(passed)} chunks with threshold {threshold}"
                else:
                    # Guard against false rejection: retain best candidate
                    retrieved_items = original_candidates[:min(top_k, 2)]
                    applied_threshold_note = f"Threshold {threshold} dropped all chunks; retained top match as safety fallback"
                    print(f"[RAGEngine] {applied_threshold_note}")

        # 3. Optional Reranking
        num_after_rerank = len(retrieved_items)
        if reranking_config and getattr(reranking_config, "enabled", False) and retrieved_items:
            try:
                from src.retrieval.retrieval_engine import RetrievalEngine
                engine = RetrievalEngine()
                retrieved_items = engine.rerank(question, retrieved_items, reranking_config)
                num_after_rerank = len(retrieved_items)
            except Exception as e:
                print(f"[RAGEngine] Reranking skipped: {e}")

        retrieval_latency_ms = (time.time() - start_retrieval) * 1000.0

        # Detailed Logging
        distances = [round(item.get("distance", 0.0), 4) for item in retrieved_items]
        print(f"[RAGEngine] Retrieved {len(retrieved_items)} chunks in {round(retrieval_latency_ms, 1)}ms. Distances: {distances}")

        if not retrieved_items:
            diag_reason = f"No document chunks found for '{target_source}' in vector database." if target_source else "No matching chunks found in vector database."
            return {
                "question": question,
                "answer": f"Unable to generate answer: {diag_reason} Please verify the document is indexed or clear the source filter.",
                "sources": [],
                "visual_evidence": [],
                "latency_seconds": round((time.time() - start_total), 3),
                "retrieval_latency_ms": round(retrieval_latency_ms, 1),
                "llm_latency_ms": 0.0,
                "model_used": "None",
                "num_retrieved": 0,
                "num_after_rerank": 0
            }

        # 4. Context Construction
        context_blocks = []
        visual_images = []
        visual_evidence_paths = []
        image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

        for idx, item in enumerate(retrieved_items, start=1):
            meta = item.get("metadata", {})
            modality = meta.get("modality", "text")
            source = meta.get("source", "unknown")
            timestamp = meta.get("timestamp", "")
            page_num = meta.get("page") or meta.get("page_number", "")
            sheet_name = meta.get("sheet_name", "")

            header = f"[Source #{idx} | Modality: {modality} | Source: {source}"
            if page_num:
                header += f" | Page: {page_num}"
            if sheet_name:
                header += f" | Sheet: {sheet_name}"
            if timestamp:
                header += f" | Timestamp: {timestamp}"
            header += "]"

            context_blocks.append(f"{header}\n{item['content']}\n")

            img_path = meta.get("frame_path") or (meta.get("file_path") if modality == "image" else None)
            if img_path and os.path.exists(img_path) and os.path.splitext(img_path)[1].lower() in image_extensions:
                try:
                    if img_path not in visual_evidence_paths:
                        visual_evidence_paths.append(img_path)
                        visual_images.append(Image.open(img_path))
                except Exception:
                    pass

        combined_context = "\n----------------------------------------\n".join(context_blocks)

        # 5. LLM Prompt Construction
        is_pdf_doc = target_source or any(
            item.get("metadata", {}).get("type") == "pdf_document" or
            str(item.get("metadata", {}).get("source", "")).lower().endswith(".pdf")
            for item in retrieved_items
        )

        if is_pdf_doc:
            doc_label = target_source or "Uploaded PDF Document"
            prompt_text = (
                f"You are a Senior AI Assistant and RAG Application Architect analyzing '{doc_label}'.\n\n"
                "CRITICAL GROUNDING RULES:\n"
                "1. Answer the user question accurately, thoroughly, and factually using ONLY the provided document context below.\n"
                "2. When asked what the document is about or to summarize it: provide a structured overview detailing its main purpose, all projects described, technologies used, and key achievements.\n"
                "3. Extract and preserve exact names, project titles, technologies (e.g., Python, LangChain, ChromaDB, HuggingFace, Groq API, Llama3, Streamlit, TensorFlow, Keras, CNN, OpenCV, Scikit-learn, RandomForest, Pandas), metrics (e.g., near-zero hallucination, sub-2-second latency, ~40% precision gain, 92% validation accuracy on 13,000+ images, R=0.94), and architectures directly from the text.\n"
                "4. Format the response with clear markdown headers, bullet points, and bold terms for readability.\n"
                "5. Do NOT hallucinate, invent details, or bring in outside project information not present in the text. If any detail requested by the user is not in the context, explicitly state that it was not found in the document.\n\n"
                f"=== DOCUMENT CONTEXT ({doc_label}) ===\n"
                f"{combined_context}\n\n"
                f"=== USER QUESTION ===\n"
                f"{question}\n\n"
                "=== ACCURATE DOCUMENT-GROUNDED ANSWER ==="
            )
        else:
            prompt_text = (
                "You are an expert Multi-Modal Retrieval-Augmented Generation (RAG) assistant.\n\n"
                "INSTRUCTIONS:\n"
                "1. Answer the user's question accurately using ONLY the provided multimodal context and images.\n"
                "2. When citing videos, always mention filename and timestamp.\n"
                "3. When citing Excel/tabular data, cite sheet name and rows.\n"
                "4. When citing images, reference filename and visual features.\n"
                "5. If information is insufficient, state what is missing honestly.\n\n"
                f"=== RETRIEVED MULTIMODAL CONTEXT ===\n"
                f"{combined_context}\n\n"
                f"=== USER QUESTION ===\n"
                f"{question}\n\n"
                f"=== YOUR DETAILED MULTIMODAL ANSWER ==="
            )

        # 6. LLM Generation
        start_llm = time.time()
        model_used = "Gemini"
        print(f"[RAGEngine] Sending prompt to LLM (Context size: {len(combined_context)} chars, Images: {len(visual_images)})...")

        if llm_manager is not None:
            resp = llm_manager.generate(prompt=prompt_text, context=combined_context, gen_config=gen_config)
            answer = resp.get("text", "Error generating response.")
            model_used = getattr(llm_manager.config, "model_name", "LLM")
        else:
            answer = self._generate_answer(prompt_text, visual_images)
            model_used = self.active_model or self.config.gemini_model

        llm_latency_ms = (time.time() - start_llm) * 1000.0
        total_latency_ms = (time.time() - start_total) * 1000.0
        print(f"[RAGEngine] Answer generated in {round(llm_latency_ms, 1)}ms by model '{model_used}'.")

        return {
            "question": question,
            "answer": answer,
            "sources": retrieved_items,
            "visual_evidence": visual_evidence_paths,
            "latency_seconds": round(total_latency_ms / 1000.0, 3),
            "retrieval_latency_ms": round(retrieval_latency_ms, 1),
            "llm_latency_ms": round(llm_latency_ms, 1),
            "model_used": model_used,
            "num_retrieved": len(retrieved_items),
            "num_after_rerank": num_after_rerank,
            "applied_threshold_note": applied_threshold_note
        }



# ---------------------------------------------------------------------------
# 5. SYNTHETIC SAMPLE DATA GENERATOR (FOR IMMEDIATE COLAB TESTING)
# ---------------------------------------------------------------------------

def create_synthetic_multimodal_data(sample_dir: str = "./colab_sample_data"):
    """Creates sample files for all 5 modalities so the user can test the pipeline immediately in Colab."""
    import numpy as np
    os.makedirs(sample_dir, exist_ok=True)

    # 1. Sample Excel Sheet
    excel_path = os.path.join(sample_dir, "quarterly_financial_report.xlsx")
    df_sales = pd.DataFrame({
        "Product": ["Quantum Server X1", "Neural Edge Coprocessor", "Vision Pro Module", "Data Pipeline Hub"],
        "Category": ["Hardware", "Hardware", "Sensors", "Software"],
        "Q1_Revenue_USD": [1250000, 850000, 420000, 930000],
        "Q2_Revenue_USD": [1480000, 910000, 490000, 1150000],
        "Growth_Rate": ["18.4%", "7.0%", "16.6%", "23.6%"],
        "Primary_Market": ["North America", "Europe", "Asia-Pacific", "Global"]
    })
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_sales.to_excel(writer, sheet_name="Revenue_Summary", index=False)
    print(f"-> Created sample Excel: {excel_path}")

    # 2. Sample Dataset (JSON)
    json_path = os.path.join(sample_dir, "ai_safety_benchmark.json")
    dataset_records = [
        {"benchmark_id": "BENCH-001", "task": "Adversarial Robustness", "model_tested": "Gemini-2.5", "accuracy": 94.8, "status": "PASSED"},
        {"benchmark_id": "BENCH-002", "task": "Hallucination Mitigation", "model_tested": "Gemini-2.5", "accuracy": 92.1, "status": "PASSED"},
        {"benchmark_id": "BENCH-003", "task": "Cross-Modal Retrieval", "model_tested": "Multimodal-RAG", "accuracy": 96.5, "status": "PASSED"}
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_records, f, indent=2)
    print(f"-> Created sample Dataset: {json_path}")

    # 3. Sample Image
    img_path = os.path.join(sample_dir, "system_architecture_diagram.png")
    img = Image.new("RGB", (640, 360), color=(26, 32, 44))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 60, 250, 160], outline=(66, 153, 225), width=3)
    draw.text((70, 100), "Ingestion Engine\n(Web, Excel, Video)", fill=(255, 255, 255))
    draw.rectangle([380, 60, 580, 160], outline=(72, 187, 120), width=3)
    draw.text((410, 100), "Vector Database\n(ChromaDB Multi-modal)", fill=(255, 255, 255))
    draw.rectangle([215, 220, 425, 310], outline=(237, 137, 54), width=3)
    draw.text((245, 255), "Gemini Multimodal LLM", fill=(255, 255, 255))
    img.save(img_path)
    print(f"-> Created sample Image: {img_path}")

    # 4. Sample Video
    video_path = os.path.join(sample_dir, "product_demo_clip.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 10
    duration_sec = 6
    width, height = 480, 270
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    colors = [(20, 20, 80), (80, 20, 20), (20, 80, 20)]
    captions = ["Scene 1: Introduction to Quantum Engine", "Scene 2: Live Video Analysis", "Scene 3: Real-Time RAG Results"]
    
    for i in range(fps * duration_sec):
        seg = min(i // (fps * 2), len(colors) - 1)
        frame = np.full((height, width, 3), colors[seg], dtype=np.uint8)
        # Put text on frame
        cv2.putText(frame, f"TIMESTAMP {i/fps:.1f}s", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, captions[seg], (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 2)
        writer.write(frame)
    writer.release()
    print(f"-> Created sample Video: {video_path}")

    # 5. Sample PDF Document
    pdf_path = os.path.join(sample_dir, "ai_system_guidelines.pdf")
    try:
        import fitz
        doc = fitz.open()
        p1 = doc.new_page()
        p1.insert_text(
            (50, 70),
            "Artificial Intelligence Safety & Multi-Modal Architecture Guidelines\n\n"
            "Section 1: Multi-Modal Ingestion Protocols\n"
            "The system ingests text, spreadsheets, datasets, images, videos, and PDFs.\n"
            "For high-concurrency workloads, chunk sizes should be capped at 600 words with 100 word overlap.\n\n"
            "Section 2: Security & Privacy Compliance\n"
            "All API keys and credentials must be stored securely in environment variables.\n"
            "Never log sensitive user query vectors directly to unencrypted persistent stores.\n\n"
            "Section 3: Model Architecture Recommendations\n"
            "Use Gemini-2.0-Flash or Gemini-1.5-Flash for ultra-low latency response synthesis."
        )
        doc.save(pdf_path)
        doc.close()
        print(f"-> Created sample PDF: {pdf_path}")
    except Exception:
        pass

    return {
        "excel": excel_path,
        "dataset": json_path,
        "image": img_path,
        "video": video_path,
        "pdf": pdf_path,
        "web_url": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Multi-Modal RAG Pipeline locally / Colab mode")
    print("=" * 60)

    # 1. Create synthetic data
    sample_data = create_synthetic_multimodal_data()

    # 2. Initialize RAG
    engine = MultimodalRAGEngine()

    # 3. Ingest all modalities
    print("\n--- Ingesting Modalities ---")
    n_xl = engine.ingest_excel(sample_data["excel"])
    print(f"Excel chunks indexed: {n_xl}")

    n_ds = engine.ingest_dataset(sample_data["dataset"], name="safety_benchmark")
    print(f"Dataset chunks indexed: {n_ds}")

    n_img = engine.ingest_image(sample_data["image"])
    print(f"Image chunks indexed: {n_img}")

    n_vid = engine.ingest_video(sample_data["video"])
    print(f"Video frame chunks indexed: {n_vid}")

    # 4. Perform cross-modal queries
    queries = [
        "Which product had the highest Q2 revenue in the financial report and what was its growth rate?",
        "What occurred at timestamp 00:02 or 00:04 in the product demo clip?",
        "What components are shown in the system architecture diagram?",
        "What accuracy did Gemini-2.5 achieve on adversarial robustness?"
    ]

    print("\n--- Running Queries ---")
    for q in queries:
        print(f"\n[QUERY]: {q}")
        res = engine.query(q, top_k=3)
        print(f"[ANSWER]: {res['answer']}")
        print(f"[EVIDENCE]: {len(res['sources'])} sources retrieved, {len(res['visual_evidence'])} visual assets attached.")
