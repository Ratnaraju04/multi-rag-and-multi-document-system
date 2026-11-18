"""
Retrieval Engine for Multi-Modal RAG.
Supports semantic similarity retrieval, Maximal Marginal Relevance (MMR) for diversity,
threshold filtering, context token budgeting, and Cross-Encoder reranking.

IMPORTANT ARCHITECTURAL NOTE:
The `top_k` in RetrievalConfig controls the number of relevant document chunks
retrieved from the vector database (e.g. top 5 chunks).
This is completely SEPARATE and INDEPENDENT from the LLM `top_k` generation parameter
in GenerationConfig (which filters the candidate token probability distribution during sampling).
"""

import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# NOTE: This retrieval top_k is SEPARATE from LLM generation top_k.
# Retrieval top_k specifies the number of document chunks fetched from the vector store.
# LLM generation top_k controls token sampling during text decoding.
@dataclass
class RetrievalConfig:
    top_k: int = 5
    similarity_threshold: float = 0.5
    search_type: str = "similarity"  # "similarity" or "mmr"
    mmr_lambda: float = 0.5  # Diversity vs relevance trade-off (1.0 = pure similarity, 0.0 = maximal diversity)
    max_context_tokens: int = 4096

@dataclass
class RerankingConfig:
    enabled: bool = False
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 3

class RetrievalEngine:
    def __init__(self):
        self._cross_encoder = None

    def filter_by_threshold(self, docs: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
        """
        Filters out retrieved document chunks whose similarity score falls below threshold.
        """
        filtered = []
        for doc in docs:
            score = doc.get("score", 0.0)
            if score >= threshold:
                filtered.append(doc)
        return filtered

    def _apply_token_budget(self, docs: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
        """
        Ensures total tokens in retrieved contexts do not exceed max_context_tokens.
        Approximates 1 token ~= 4 characters.
        """
        budgeted = []
        running_tokens = 0

        for doc in docs:
            text = doc.get("content", doc.get("text", ""))
            est_tokens = max(1, len(text) // 4)
            if running_tokens + est_tokens <= max_tokens:
                budgeted.append(doc)
                running_tokens += est_tokens
            else:
                # If room remains, truncate this chunk to fit the budget
                remaining_tokens = max_tokens - running_tokens
                if remaining_tokens > 50:
                    allowed_chars = remaining_tokens * 4
                    truncated_doc = dict(doc)
                    truncated_doc["content"] = text[:allowed_chars] + "... [context truncated]"
                    truncated_doc["text"] = truncated_doc["content"]
                    budgeted.append(truncated_doc)
                break

        return budgeted

    def _compute_mmr(
        self,
        query_embedding: List[float],
        candidate_docs: List[Dict[str, Any]],
        candidate_embeddings: Optional[List[List[float]]],
        top_k: int,
        mmr_lambda: float
    ) -> List[Dict[str, Any]]:
        r"""
        Computes Maximal Marginal Relevance (MMR) to select diverse and relevant chunks.
        MMR = argmax_{d_i in R \ S} [ lambda * Sim(d_i, q) - (1 - lambda) * max_{d_j in S} Sim(d_i, d_j) ]
        """
        if not candidate_docs:
            return []

        if len(candidate_docs) <= top_k:
            return candidate_docs

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        # Prepare document representations
        doc_vectors = []
        if candidate_embeddings and len(candidate_embeddings) == len(candidate_docs):
            for emb in candidate_embeddings:
                v = np.array(emb, dtype=np.float32)
                v_norm = np.linalg.norm(v)
                doc_vectors.append(v / v_norm if v_norm > 0 else v)
        else:
            # Fallback to tf-idf/bag-of-words vectorization if raw embeddings are not available
            words_list = [set(d.get("content", "").lower().split()) for d in candidate_docs]
            all_vocab = list(set().union(*words_list)) if words_list else []
            vocab_idx = {w: i for i, w in enumerate(all_vocab)}
            dim = max(1, len(all_vocab))

            for words in words_list:
                v = np.zeros(dim, dtype=np.float32)
                for w in words:
                    v[vocab_idx[w]] = 1.0
                v_norm = np.linalg.norm(v)
                doc_vectors.append(v / v_norm if v_norm > 0 else v)

        selected_indices: List[int] = []
        remaining_indices = list(range(len(candidate_docs)))

        # Precompute sim with query
        sim_to_query = []
        for i, d in enumerate(candidate_docs):
            if "score" in d and d["score"] is not None:
                sim_to_query.append(float(d["score"]))
            else:
                sim_to_query.append(float(np.dot(doc_vectors[i], q_vec)))

        # Select first doc with highest relevance to query
        first_idx = int(np.argmax(sim_to_query))
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        while len(selected_indices) < top_k and remaining_indices:
            best_score = -float("inf")
            best_cand = None

            for i in remaining_indices:
                relevance = sim_to_query[i]
                # Maximum similarity to already selected documents
                max_sim_to_selected = max(
                    float(np.dot(doc_vectors[i], doc_vectors[j])) for j in selected_indices
                )
                mmr_val = mmr_lambda * relevance - (1.0 - mmr_lambda) * max_sim_to_selected

                if mmr_val > best_score:
                    best_score = mmr_val
                    best_cand = i

            if best_cand is not None:
                selected_indices.append(best_cand)
                remaining_indices.remove(best_cand)
            else:
                break

        return [candidate_docs[idx] for idx in selected_indices]

    def rerank(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        reranking_config: Optional[RerankingConfig] = None
    ) -> List[Dict[str, Any]]:
        """
        Reranks documents using a Cross-Encoder model or robust token relevance scoring.
        """
        config = reranking_config or RerankingConfig()
        if not config.enabled or not docs:
            return docs

        top_n = min(config.top_n, len(docs))

        # Try cross-encoder if installed
        try:
            from sentence_transformers import CrossEncoder
            if self._cross_encoder is None:
                self._cross_encoder = CrossEncoder(config.model)

            pairs = [[query, d.get("content", d.get("text", ""))] for d in docs]
            scores = self._cross_encoder.predict(pairs)

            scored_docs = []
            for doc, score in zip(docs, scores):
                doc_copy = dict(doc)
                doc_copy["original_score"] = doc.get("score")
                doc_copy["rerank_score"] = round(float(score), 4)
                doc_copy["score"] = round(float(score), 4)
                scored_docs.append(doc_copy)

            scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
            return scored_docs[:top_n]

        except Exception as e:
            logger.info(f"CrossEncoder unavailable ({e}), using lexical BM25/term density fallback reranking.")
            q_terms = set(query.lower().split())

            scored_docs = []
            for doc in docs:
                content = doc.get("content", doc.get("text", "")).lower()
                doc_terms = content.split()
                # Term frequency score + phrase bonus
                tf = sum(1 for t in doc_terms if t in q_terms)
                density = tf / max(1, len(doc_terms))
                phrase_bonus = 1.0 if query.lower() in content else 0.0
                lexical_score = (doc.get("score", 0.5) * 0.5) + (density * 0.3) + (phrase_bonus * 0.2)

                doc_copy = dict(doc)
                doc_copy["rerank_score"] = round(lexical_score, 4)
                scored_docs.append(doc_copy)

            scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
            return scored_docs[:top_n]

    def retrieve(
        self,
        query: str,
        embedding: Optional[List[float]] = None,
        vector_adapter: Optional[Any] = None,
        retrieval_config: Optional[RetrievalConfig] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Orchestrates retrieval pipeline:
        1. Fetch candidates from vector_adapter
        2. Filter by similarity threshold
        3. Apply MMR (if search_type == 'mmr') or standard similarity ranking
        4. Budget retrieved chunks against max_context_tokens
        """
        config = retrieval_config or RetrievalConfig()
        if vector_adapter is None:
            raise ValueError("A VectorDBAdapter must be provided for retrieval.")

        # Candidate pool size: pull more if MMR is requested so diversity selection has options
        fetch_k = config.top_k * 3 if config.search_type == "mmr" else config.top_k

        # If query embedding was not provided, check if adapter has embedding logic
        if embedding is None:
            raise ValueError("Query embedding vector must be provided to retrieve().")

        raw_docs = vector_adapter.search(query_embedding=embedding, top_k=fetch_k, filters=filters)

        # 1. Filter by similarity threshold
        threshold_docs = self.filter_by_threshold(raw_docs, config.similarity_threshold)

        # If threshold filtered everything out, take the single best match if available
        if not threshold_docs and raw_docs:
            logger.warning("All documents fell below similarity_threshold. Returning top single result.")
            threshold_docs = [raw_docs[0]]

        # 2. MMR or direct slice
        if config.search_type == "mmr" and len(threshold_docs) > 1:
            mmr_docs = self._compute_mmr(
                query_embedding=embedding,
                candidate_docs=threshold_docs,
                candidate_embeddings=None,
                top_k=config.top_k,
                mmr_lambda=config.mmr_lambda
            )
        else:
            mmr_docs = threshold_docs[:config.top_k]

        # 3. Context token budgeting
        final_docs = self._apply_token_budget(mmr_docs, config.max_context_tokens)

        return final_docs
