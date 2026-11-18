"""
Advanced Chunking Engine for Multi-Modal RAG.
Supports 5 configurable chunking strategies and modality-specific chunking pipelines.
"""

import re
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union

@dataclass
class ChunkingConfig:
    strategy: str = "RecursiveCharacterTextSplitter"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    min_chunk_size: int = 100
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])
    preserve_separators: bool = True
    similarity_threshold: float = 0.5  # For SemanticChunker

class ChunkingEngine:
    SUPPORTED_STRATEGIES = [
        "RecursiveCharacterTextSplitter",
        "CharacterTextSplitter",
        "TokenTextSplitter",
        "MarkdownHeaderTextSplitter",
        "SemanticChunker"
    ]

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        if self.config.strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unknown chunking strategy '{self.config.strategy}'. "
                f"Must be one of {self.SUPPORTED_STRATEGIES}"
            )

    # ---------------------------------------------------------
    # Core Splitter Implementations
    # ---------------------------------------------------------
    def _split_recursive(self, text: str, separators: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
        """Recursive character text splitting."""
        final_chunks: List[str] = []
        if not text:
            return final_chunks

        # Find the best separator that actually appears in the text
        separator = separators[-1]
        new_separators = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits: List[str] = []
        for s in splits:
            if not s:
                continue
            if len(s) < chunk_size:
                good_splits.append(s)
            else:
                if new_separators:
                    sub_splits = self._split_recursive(s, new_separators, chunk_size, chunk_overlap)
                    good_splits.extend(sub_splits)
                else:
                    # Hard slice when no separators remain
                    for pos in range(0, len(s), max(1, chunk_size - chunk_overlap)):
                        good_splits.append(s[pos : pos + chunk_size])

        # Merge splits up to chunk_size respecting chunk_overlap
        current_chunk: List[str] = []
        current_len = 0
        step_sep = separator if self.config.preserve_separators else " "

        for piece in good_splits:
            piece_len = len(piece) + (len(step_sep) if current_chunk else 0)
            if current_len + piece_len <= chunk_size:
                current_chunk.append(piece)
                current_len += piece_len
            else:
                if current_chunk:
                    merged = step_sep.join(current_chunk).strip()
                    if merged:
                        final_chunks.append(merged)
                    # Handle overlap by keeping tail pieces
                    overlap_acc: List[str] = []
                    overlap_len = 0
                    for rev_p in reversed(current_chunk):
                        if overlap_len + len(rev_p) <= chunk_overlap:
                            overlap_acc.insert(0, rev_p)
                            overlap_len += len(rev_p) + len(step_sep)
                        else:
                            break
                    current_chunk = overlap_acc
                    current_len = sum(len(p) for p in current_chunk) + max(0, len(current_chunk) - 1) * len(step_sep)
                current_chunk.append(piece)
                current_len += len(piece) + (len(step_sep) if len(current_chunk) > 1 else 0)

        if current_chunk:
            merged = step_sep.join(current_chunk).strip()
            if merged:
                final_chunks.append(merged)

        return final_chunks

    def _split_character(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Simple character splitter by primary separator."""
        sep = self.config.separators[0] if self.config.separators else "\n\n"
        raw_parts = text.split(sep)
        chunks: List[str] = []
        current: List[str] = []
        cur_len = 0

        for part in raw_parts:
            part = part.strip()
            if not part:
                continue
            part_len = len(part) + (len(sep) if current else 0)
            if cur_len + part_len <= chunk_size:
                current.append(part)
                cur_len += part_len
            else:
                if current:
                    chunks.append(sep.join(current))
                    # Overlap
                    current = [part]
                    cur_len = len(part)
                else:
                    chunks.append(part[:chunk_size])
                    cur_len = 0

        if current:
            chunks.append(sep.join(current))
        return chunks

    def _split_token(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Token splitter using words/tokens approximation (~4 chars per token)."""
        words = text.split()
        if not words:
            return []

        # Convert chunk_size and chunk_overlap (in chars or tokens) to word counts
        # Approx 1 token ~ 0.75 words, so words_per_chunk ~ chunk_size / 4
        words_per_chunk = max(1, chunk_size // 4)
        words_overlap = max(0, chunk_overlap // 4)
        step = max(1, words_per_chunk - words_overlap)

        chunks: List[str] = []
        for i in range(0, len(words), step):
            chunk_slice = words[i : i + words_per_chunk]
            if chunk_slice:
                chunks.append(" ".join(chunk_slice))
        return chunks

    def _split_markdown_headers(self, text: str) -> List[str]:
        """Markdown header-based splitter preserving document hierarchy."""
        lines = text.split("\n")
        chunks: List[str] = []
        current_section: List[str] = []
        current_header = ""

        header_pattern = re.compile(r"^(#{1,6})\s+(.*)$")

        for line in lines:
            match = header_pattern.match(line.strip())
            if match:
                if current_section:
                    body = "\n".join(current_section).strip()
                    if body:
                        chunks.append(f"{current_header}\n{body}" if current_header else body)
                    current_section = []
                current_header = line.strip()
            else:
                current_section.append(line)

        if current_section or current_header:
            body = "\n".join(current_section).strip()
            content = f"{current_header}\n{body}" if current_header else body
            if content.strip():
                chunks.append(content.strip())

        # If sections exceed chunk_size, recursively sub-split
        final_chunks: List[str] = []
        for c in chunks:
            if len(c) > self.config.chunk_size:
                sub = self._split_recursive(c, self.config.separators, self.config.chunk_size, self.config.chunk_overlap)
                final_chunks.extend(sub)
            else:
                final_chunks.append(c)

        return final_chunks

    def _split_semantic(self, text: str, chunk_size: int, threshold: float) -> List[str]:
        """
        Semantic chunker that groups sentences by semantic similarity.
        Falls back to sentence boundary splitting if sentence-transformers is unavailable.
        """
        # Split into sentences
        sentence_end = re.compile(r'(?<=[.!?])\s+')
        sentences = [s.strip() for s in sentence_end.split(text) if s.strip()]
        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences

        embeddings = None
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
        except Exception:
            embeddings = None

        chunks: List[str] = []
        current_group: List[str] = [sentences[0]]
        current_length = len(sentences[0])

        for i in range(1, len(sentences)):
            sent = sentences[i]
            sim = 1.0

            if embeddings is not None:
                # Cosine similarity between normalized consecutive vectors
                sim = float((embeddings[i - 1] * embeddings[i]).sum())
            else:
                # Lexical jaccard similarity fallback
                words_a = set(sentences[i - 1].lower().split())
                words_b = set(sent.lower().split())
                union = words_a | words_b
                sim = len(words_a & words_b) / len(union) if union else 0.0

            # If similarity is above threshold and won't exceed max size, keep grouping
            if sim >= threshold and (current_length + len(sent) + 1 <= chunk_size):
                current_group.append(sent)
                current_length += len(sent) + 1
            else:
                # New group
                chunk_str = " ".join(current_group).strip()
                if chunk_str:
                    chunks.append(chunk_str)
                current_group = [sent]
                current_length = len(sent)

        if current_group:
            chunk_str = " ".join(current_group).strip()
            if chunk_str:
                chunks.append(chunk_str)

        return chunks

    # ---------------------------------------------------------
    # Public Methods
    # ---------------------------------------------------------
    def preview_chunks(self, text: str) -> List[str]:
        """Returns preview list of chunk texts based on the configured strategy."""
        strat = self.config.strategy
        size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        if strat == "RecursiveCharacterTextSplitter":
            chunks = self._split_recursive(text, self.config.separators, size, overlap)
        elif strat == "CharacterTextSplitter":
            chunks = self._split_character(text, size, overlap)
        elif strat == "TokenTextSplitter":
            chunks = self._split_token(text, size, overlap)
        elif strat == "MarkdownHeaderTextSplitter":
            chunks = self._split_markdown_headers(text)
        elif strat == "SemanticChunker":
            chunks = self._split_semantic(text, size, self.config.similarity_threshold)
        else:
            chunks = self._split_recursive(text, self.config.separators, size, overlap)

        # Filter out chunks smaller than min_chunk_size unless it's the only chunk
        if len(chunks) > 1 and self.config.min_chunk_size > 0:
            filtered = [c for c in chunks if len(c) >= self.config.min_chunk_size]
            chunks = filtered if filtered else chunks

        return chunks

    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunks text and enriches each chunk with standard metadata.
        """
        metadata = metadata or {}
        raw_chunks = self.preview_chunks(text)
        total_chunks = len(raw_chunks)
        source = metadata.get("source", "raw_text")
        modality = metadata.get("modality", "text")

        documents = []
        for idx, chunk_str in enumerate(raw_chunks):
            doc_meta = dict(metadata)
            doc_meta.update({
                "source": source,
                "modality": modality,
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "chunk_size": len(chunk_str),
                "strategy": self.config.strategy
            })

            documents.append({
                "id": f"{source}_{modality}_{idx}",
                "text": chunk_str,
                "content": chunk_str,
                "metadata": doc_meta
            })

        return documents

    def get_chunk_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Computes statistical metrics over generated chunks."""
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_size": 0,
                "min_size": 0,
                "max_size": 0,
                "total_chars": 0
            }

        sizes = [len(c.get("text", c.get("content", ""))) for c in chunks]
        avg = round(sum(sizes) / len(sizes), 2)
        return {
            "total_chunks": len(chunks),
            "avg_size": avg,
            "avg_chunk_size": avg,
            "min_size": min(sizes),
            "max_size": max(sizes),
            "total_chars": sum(sizes)
        }


    # ---------------------------------------------------------
    # Modality-Specific Chunking Pipelines
    # ---------------------------------------------------------
    def chunk_pdf(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunks PDF pages preserving page_number metadata.
        pages: list of dicts with keys 'page_number' (or 'page'), 'text' (or 'content'), and optional 'source'.
        """
        all_chunks: List[Dict[str, Any]] = []
        global_idx = 0

        for page in pages:
            page_num = page.get("page_number", page.get("page", 1))
            page_text = page.get("text", page.get("content", ""))
            source = page.get("source", "document.pdf")

            if not page_text or not page_text.strip():
                continue

            page_chunks = self.preview_chunks(page_text)
            for chunk_str in page_chunks:
                all_chunks.append({
                    "id": f"pdf_{source}_p{page_num}_{global_idx}",
                    "text": chunk_str,
                    "content": f"[PDF: {source} | Page {page_num}]\n{chunk_str}",
                    "metadata": {
                        "source": source,
                        "modality": "pdf_text",
                        "page_number": page_num,
                        "chunk_index": global_idx,
                        "chunk_size": len(chunk_str),
                        "strategy": self.config.strategy
                    }
                })
                global_idx += 1

        total = len(all_chunks)
        for c in all_chunks:
            c["metadata"]["total_chunks"] = total

        return all_chunks

    def chunk_excel(self, sheets: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Chunks tabular/excel sheets, attaching sheet_name and row index.
        sheets can be {sheet_name: rows_or_text} or a list of sheet dicts.
        """
        all_chunks: List[Dict[str, Any]] = []
        global_idx = 0

        sheet_items = []
        if isinstance(sheets, dict):
            sheet_items = list(sheets.items())
        elif isinstance(sheets, list):
            for item in sheets:
                s_name = item.get("sheet_name", "Sheet1")
                s_content = item.get("content", item.get("text", str(item)))
                sheet_items.append((s_name, s_content))

        for sheet_name, content in sheet_items:
            content_str = content if isinstance(content, str) else str(content)
            chunks = self.preview_chunks(content_str)
            for chunk_str in chunks:
                all_chunks.append({
                    "id": f"excel_{sheet_name}_{global_idx}",
                    "text": chunk_str,
                    "content": f"[Excel Sheet: {sheet_name}]\n{chunk_str}",
                    "metadata": {
                        "source": "spreadsheet",
                        "modality": "table_rows",
                        "sheet_name": sheet_name,
                        "chunk_index": global_idx,
                        "chunk_size": len(chunk_str),
                        "strategy": self.config.strategy
                    }
                })
                global_idx += 1

        total = len(all_chunks)
        for c in all_chunks:
            c["metadata"]["total_chunks"] = total

        return all_chunks

    def chunk_web(self, content: str, url: str) -> List[Dict[str, Any]]:
        """
        Chunks scraped web content, preserving URL and source metadata.
        """
        chunks = self.preview_chunks(content)
        all_chunks: List[Dict[str, Any]] = []
        total = len(chunks)

        for idx, chunk_str in enumerate(chunks):
            all_chunks.append({
                "id": f"web_{idx}",
                "text": chunk_str,
                "content": f"[Web: {url}]\n{chunk_str}",
                "metadata": {
                    "source": url,
                    "url": url,
                    "modality": "web_content",
                    "chunk_index": idx,
                    "total_chunks": total,
                    "chunk_size": len(chunk_str),
                    "strategy": self.config.strategy
                }
            })
        return all_chunks

    def chunk_image_caption(self, caption: str, image_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Chunks or packages image descriptions / captions.
        """
        source = image_path or "image_upload"
        chunk_str = caption.strip()
        return [{
            "id": f"img_{abs(hash(source)) % 100000}",
            "text": chunk_str,
            "content": f"[Image Description: {source}]\n{chunk_str}",
            "metadata": {
                "source": source,
                "image_path": image_path or "",
                "modality": "image_caption",
                "chunk_index": 0,
                "total_chunks": 1,
                "chunk_size": len(chunk_str),
                "strategy": self.config.strategy
            }
        }]

    def chunk_video_transcript(self, transcript: Union[str, List[Dict[str, Any]]], timestamps: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """
        Chunks video transcripts with timestamp tracking.
        transcript can be text or list of segments: [{"timestamp": float, "text": str}]
        """
        all_chunks: List[Dict[str, Any]] = []

        if isinstance(transcript, list):
            # Segment-based transcript
            for idx, segment in enumerate(transcript):
                ts = segment.get("timestamp", timestamps[idx] if (timestamps and idx < len(timestamps)) else 0.0)
                seg_text = segment.get("text", "")
                all_chunks.append({
                    "id": f"video_segment_{idx}",
                    "text": seg_text,
                    "content": f"[Video Timestamp: {ts}s]\n{seg_text}",
                    "metadata": {
                        "source": "video_transcript",
                        "modality": "video_transcript",
                        "timestamp": ts,
                        "chunk_index": idx,
                        "chunk_size": len(seg_text),
                        "strategy": self.config.strategy
                    }
                })
        else:
            # Plain string transcript
            chunks = self.preview_chunks(str(transcript))
            for idx, chunk_str in enumerate(chunks):
                ts = timestamps[idx] if (timestamps and idx < len(timestamps)) else (idx * 5.0)
                all_chunks.append({
                    "id": f"video_chunk_{idx}",
                    "text": chunk_str,
                    "content": f"[Video ~{ts:.1f}s]\n{chunk_str}",
                    "metadata": {
                        "source": "video_transcript",
                        "modality": "video_transcript",
                        "timestamp": ts,
                        "chunk_index": idx,
                        "chunk_size": len(chunk_str),
                        "strategy": self.config.strategy
                    }
                })

        total = len(all_chunks)
        for c in all_chunks:
            c["metadata"]["total_chunks"] = total

        return all_chunks
