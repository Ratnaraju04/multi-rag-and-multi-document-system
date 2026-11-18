"""
Core Multi-Modal RAG Engine.
Coordinates ingestion across web, excel, dataset, image, and video,
retrieves cross-modal evidence, and generates answers using Gemini Multimodal LLM.
"""

import os
from typing import List, Dict, Any, Optional, Union
from PIL import Image

from .config import RAGConfig
from .vector_store import MultimodalVectorStore
from .loaders.web_scraper import WebScraperLoader
from .loaders.excel_loader import ExcelDatasetLoader
from .loaders.dataset_loader import DatasetLoader
from .loaders.image_loader import ImageLoader
from .loaders.video_loader import VideoLoader
from .loaders.pdf_loader import PDFLoader

class MultimodalRAGEngine:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        import requests
        self.session = requests.Session()
        self.active_model = None
        self.gemini_client = self._init_gemini_client()
        
        # Initialize Vector Store
        self.vector_store = MultimodalVectorStore(
            persist_directory=self.config.chroma_db_dir,
            collection_name=self.config.collection_name
        )

        # Initialize Loaders
        self.web_loader = WebScraperLoader()
        self.excel_loader = ExcelDatasetLoader()
        self.dataset_loader = DatasetLoader()
        self.image_loader = ImageLoader(gemini_client=self.gemini_client)
        self.video_loader = VideoLoader(
            frame_interval_sec=self.config.video_frame_interval_sec,
            output_dir=self.config.video_output_dir,
            gemini_client=self.gemini_client
        )
        self.pdf_loader = PDFLoader()

    def _init_gemini_client(self):
        """Initializes Gemini client supporting either new google-genai or legacy SDK."""
        api_key = self.config.api_key
        if not api_key:
            print("[Warning] No GEMINI_API_KEY found. LLM synthesis will be disabled until key is set.")
            return None

        # 1. Try modern google-genai SDK
        try:
            from google import genai
            return genai.Client(api_key=api_key)
        except ImportError:
            pass

        # 2. Fallback to google-generativeai SDK
        try:
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=api_key)
            return genai_legacy
        except ImportError:
            print("[Warning] Neither 'google-genai' nor 'google-generativeai' installed.")
            return None

    # ================= Data Ingestion Pipelines =================

    def ingest_url(self, url: str):
        """Scrapes web page and indexes text and images. Returns (chunk_count, readable_source_label)."""
        docs, readable_source = self.web_loader.scrape_url(url)
        self.vector_store.add_documents(docs)
        return len(docs), readable_source

    def ingest_excel(self, file_path: str) -> int:
        """Parses Excel/CSV sheet into structured markdown tables and row records."""
        docs = self.excel_loader.load(file_path)
        self.vector_store.add_documents(docs)
        return len(docs)

    def ingest_dataset(self, source: Union[str, List[Dict[str, Any]]], dataset_name: str = "dataset") -> int:
        """Ingests JSON, JSONL, Parquet file or in-memory list of dicts."""
        if isinstance(source, str):
            docs = self.dataset_loader.load_from_file(source)
        elif isinstance(source, list):
            docs = self.dataset_loader.load_records(source, dataset_name=dataset_name)
        else:
            raise ValueError("source must be a file path string or a list of dictionaries")
        self.vector_store.add_documents(docs)
        return len(docs)

    def ingest_image(self, path_or_dir: str) -> int:
        """Ingests single image or an entire folder of images with AI visual captions."""
        if os.path.isdir(path_or_dir):
            docs = self.image_loader.load_directory(path_or_dir)
        else:
            docs = [self.image_loader.load_image(path_or_dir)]
        self.vector_store.add_documents(docs)
        return len(docs)

    def ingest_video(self, video_path: str, max_frames: int = 30) -> int:
        """Extracts timestamped keyframes and visual descriptions from a video file."""
        docs = self.video_loader.load_video(video_path, max_frames=max_frames)
        self.vector_store.add_documents(docs)
        return len(docs)

    def ingest_pdf(self, file_path: str) -> int:
        """Parses PDF into page-level chunked documents and indexes into vector store."""
        docs = self.pdf_loader.load(file_path)
        self.vector_store.add_documents(docs)
        return len(docs)

    # ================= Retrieval & Generation =================

    def query(
        self,
        question: str,
        top_k: int = 5,
        modality_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        query_image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes Multi-Modal RAG:
        1. Retrieves relevant text, tables, image captions, and video timestamps.
        2. Loads relevant visual assets (images / keyframes) into memory.
        3. Supplies all multimodal context to Gemini LLM for verified synthesis.
        """
        # Contextual Query Enrichment for follow-up questions ("in it", "this document", etc.)
        search_query = question
        lower_q = question.lower()
        pronoun_triggers = ["in it", "in this", "this data", "this file", "this dataset", "the dataset", "this pdf", "the pdf", "this document", "the document", "the table", "the sheet", "these features", "columns in it"]
        has_pronoun = any(trig in lower_q for trig in pronoun_triggers)

        active_source = None
        if chat_history and has_pronoun:
            import re
            for msg in reversed(chat_history):
                text = str(msg.get("content", ""))
                matches = re.findall(r"([A-Za-z0-9_\-\.]+\.(?:pdf|csv|xlsx|xls|json|mp4|png|jpg))", text, re.IGNORECASE)
                if matches:
                    active_source = matches[0]
                    break

        target_source = source_filter if (source_filter and not source_filter.startswith("All Sources")) else active_source
        if target_source:
            search_query = f"{question} [Source: {target_source}]"

        # 1. Retrieve relevant items
        retrieved_items = self.vector_store.search(
            query=search_query,
            n_results=top_k,
            modality_filter=modality_filter,
            source_filter=target_source
        )

        if not retrieved_items:
            return {
                "question": question,
                "answer": "No relevant documents found in the multimodal index.",
                "sources": [],
                "visual_evidence": []
            }

        # 2. Compile context and gather visual assets
        context_blocks = []
        visual_images = []
        visual_evidence_paths = []

        for idx, item in enumerate(retrieved_items, start=1):
            meta = item.get("metadata", {})
            modality = meta.get("modality", "text")
            source = meta.get("source", "unknown")
            timestamp = meta.get("timestamp", "")
            
            header = f"[Source #{idx} | Modality: {modality} | Source: {source}"
            if timestamp:
                header += f" | Timestamp: {timestamp}"
            header += "]"

            context_blocks.append(f"{header}\n{item['content']}\n")

            # Check for image or video frame attachment
            img_path = meta.get("frame_path")
            if not img_path and (modality == "image" or meta.get("type") == "image"):
                img_path = meta.get("file_path")

            image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
            if img_path and os.path.exists(img_path) and os.path.splitext(img_path)[1].lower() in image_extensions:
                try:
                    img_obj = Image.open(img_path)
                    visual_images.append(img_obj)
                    if img_path not in visual_evidence_paths:
                        visual_evidence_paths.append(img_path)
                except Exception as e:
                    print(f"[RAGEngine] Could not load image {img_path}: {e}")

        combined_context = "\n----------------------------------------\n".join(context_blocks)

        # 3. Formulate Prompt for Multimodal LLM
        system_instructions = (
            "You are an expert Multi-Modal Retrieval-Augmented Generation (RAG) assistant. "
            "You have access to retrieved multimodal context containing scraped web pages, "
            "Excel spreadsheets, datasets, images, and timestamped video keyframes.\n\n"
            "INSTRUCTIONS:\n"
            "1. Answer the user's question accurately using ONLY the provided multimodal context and images.\n"
            "2. When citing videos, always mention the specific filename and TIMESTAMP (e.g. 'At 00:15 in demo.mp4...').\n"
            "3. When citing Excel/tabular data, cite the specific sheet name, column values, and rows.\n"
            "4. When citing images, reference the image file name and visual features.\n"
            "5. If the context does not contain sufficient information, state what is missing honestly."
        )

        prompt_text = (
            f"{system_instructions}\n\n"
            f"=== RETRIEVED MULTIMODAL CONTEXT ===\n"
            f"{combined_context}\n\n"
            f"=== USER QUESTION ===\n"
            f"{question}\n\n"
            f"=== YOUR DETAILED MULTIMODAL ANSWER ==="
        )

        # 4. Generate Answer via Gemini
        answer = self._generate_gemini_response(prompt_text, visual_images, query_image_path)

        return {
            "question": question,
            "answer": answer,
            "sources": retrieved_items,
            "visual_evidence": visual_evidence_paths
        }

    def _get_supported_models(self, api_key: str) -> List[str]:
        """Dynamically fetches models authorized for this specific API key to prevent 404 errors."""
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
                ordered = [p for p in preferred if p in supported]
                ordered += [s for s in supported if s not in ordered]
                if ordered:
                    self._cached_models = ordered
                    return ordered
        except Exception:
            pass
        self._cached_models = preferred
        return preferred

    def _generate_gemini_response(
        self,
        prompt: str,
        attached_images: List[Image.Image],
        query_image_path: Optional[str] = None
    ) -> str:
        api_key = (self.config.api_key or os.environ.get("GEMINI_API_KEY", "")).strip().strip("'\"")
        if not api_key:
            return "[GEMINI_API_KEY not configured] Context retrieved successfully. Please configure your API key."

        all_images = []
        if query_image_path and os.path.exists(query_image_path):
            try:
                all_images.append(Image.open(query_image_path))
            except Exception:
                pass
        all_images.extend(attached_images)

        import time

        # Fast path if working model already established
        if self.active_model:
            for attempt in range(2):
                rest_result = self._call_gemini_rest(self.active_model, prompt, all_images, api_key)
                if rest_result and not rest_result.startswith("[REST API") and not rest_result.startswith("[REST Request"):
                    return rest_result
                if "[REST API 429]" in rest_result or "[REST API 503]" in rest_result:
                    time.sleep(1.5)
                    continue
                break

        candidate_models = self._get_supported_models(api_key)
        last_error = ""

        # 1. Direct REST endpoint with keep-alive
        for model_name in candidate_models:
            clean_name = model_name.replace("models/", "")
            for attempt in range(2):
                rest_result = self._call_gemini_rest(clean_name, prompt, all_images, api_key)
                if rest_result and not rest_result.startswith("[REST API") and not rest_result.startswith("[REST Request"):
                    self.active_model = clean_name
                    return rest_result
                if rest_result:
                    last_error = rest_result
                    if "[REST API 404]" in rest_result:
                        break
                    if "[REST API 429]" in rest_result or "[REST API 503]" in rest_result:
                        time.sleep(2)
                        continue
                break

        # 2. SDK fallback
        if self.gemini_client:
            contents = [prompt] + all_images[:2]
            for model_name in candidate_models:
                clean_name = model_name.replace("models/", "")
                try:
                    if hasattr(self.gemini_client, "models"):
                        response = self.gemini_client.models.generate_content(
                            model=clean_name,
                            contents=contents
                        )
                        self.active_model = clean_name
                        return response.text.strip()
                    elif hasattr(self.gemini_client, "GenerativeModel"):
                        model = self.gemini_client.GenerativeModel(clean_name)
                        response = model.generate_content(contents)
                        self.active_model = clean_name
                        return response.text.strip()
                except Exception as e:
                    err_str = str(e)
                    if "404" in err_str:
                        continue
                    last_error = err_str

        return f"Generation Notice: {last_error}"

    def _call_gemini_rest(self, model_name: str, prompt: str, images: List[Image.Image], api_key: str) -> Optional[str]:
        """Direct REST call to Google Gemini v1beta API with persistent connection and compressed image payloads."""
        import base64
        import io

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }

        parts = [{"text": prompt}]
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
        Executes query applying advanced retrieval, reranking, and generation configurations.
        Maintains backward compatibility while honoring all Module 4-6 parameter overrides.
        """
        import time
        start_total = time.time()

        # Extract retrieval settings
        top_k = getattr(retrieval_config, "top_k", 5) if retrieval_config else 5
        threshold = getattr(retrieval_config, "similarity_threshold", 0.0) if retrieval_config else 0.0

        # Execute base retrieval
        start_retrieval = time.time()
        target_source = source_filter if (source_filter and not source_filter.startswith("All Sources")) else None
        search_query = f"{question} [Source: {target_source}]" if target_source else question

        retrieved_items = self.vector_store.search(
            query=search_query,
            n_results=top_k,
            source_filter=target_source
        )

        # Apply similarity threshold filtering if set
        if threshold > 0.0 and retrieved_items:
            retrieved_items = [
                item for item in retrieved_items
                if (1.0 - item.get("distance", 1.0)) >= threshold or item.get("distance", 1.0) <= (1.0 - threshold)
            ]

        # Apply reranking if enabled
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

        if not retrieved_items:
            return {
                "question": question,
                "answer": "No relevant documents found matching your search criteria and threshold.",
                "sources": [],
                "visual_evidence": [],
                "latency_seconds": round((time.time() - start_total), 3),
                "retrieval_latency_ms": round(retrieval_latency_ms, 1),
                "llm_latency_ms": 0.0,
                "model_used": "None",
                "num_retrieved": 0,
                "num_after_rerank": 0
            }

        # Build context blocks and visual evidence
        context_blocks = []
        visual_images = []
        visual_evidence_paths = []
        image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

        for idx, item in enumerate(retrieved_items, start=1):
            meta = item.get("metadata", {})
            modality = meta.get("modality", "text")
            source = meta.get("source", "unknown")
            timestamp = meta.get("timestamp", "")
            page_num = meta.get("page_number", "")
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

        # Generate via LLM Manager if provided, else via Gemini
        start_llm = time.time()
        model_used = "Gemini"
        if llm_manager is not None:
            resp = llm_manager.generate(prompt=prompt_text, context=combined_context, gen_config=gen_config)
            answer = resp.get("text", "Error generating response.")
            model_used = getattr(llm_manager.config, "model_name", "LLM")
        else:
            answer = self._generate_gemini_response(prompt_text, visual_images)
            model_used = self.active_model or self.config.gemini_model

        llm_latency_ms = (time.time() - start_llm) * 1000.0
        total_latency_ms = (time.time() - start_total) * 1000.0

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
            "num_after_rerank": num_after_rerank
        }

