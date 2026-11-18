"""
Multi-Modal Vector Database Manager using ChromaDB.
Indexes and retrieves multimodal documents (Text, Images, Video keyframes, Web content, Tabular records).
"""

import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

class MultimodalVectorStore:
    def __init__(
        self,
        persist_directory: str = "./chroma_multimodal_db",
        collection_name: str = "multimodal_kb",
        embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name

        # Initialize Chroma persistent client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )

        # Setup embedding function (using sentence-transformers if available, else Chroma default)
        self.embedding_fn = self._init_embedding_function()
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"description": "Unified multimodal knowledge base for text, images, videos, tables"}
        )

    def _init_embedding_function(self):
        try:
            from chromadb.utils import embedding_functions
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name
            )
        except Exception as e:
            print(f"[VectorStore] Using default embedding function (fallback: {e})")
            from chromadb.utils import embedding_functions
            return embedding_functions.DefaultEmbeddingFunction()

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Batch adds documents into the vector store.
        Each doc has: id, modality, content, metadata
        """
        if not documents:
            return

        ids = []
        documents_text = []
        metadatas = []

        for doc in documents:
            ids.append(str(doc["id"]))
            documents_text.append(str(doc["content"]))
            
            # Clean metadata so Chroma accepts only primitive types
            clean_meta = {}
            for k, v in doc.get("metadata", {}).items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            clean_meta["modality"] = doc.get("modality", "text")
            metadatas.append(clean_meta)

        # Upsert in batches of 50 to maximize embedding throughput on CPU
        batch_size = 50
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i : i + batch_size],
                documents=documents_text[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size]
            )

        print(f"[VectorStore] Successfully indexed {len(documents)} items into '{self.collection_name}'. Total count: {self.collection.count()}")

    def search(
        self,
        query: str,
        n_results: int = 5,
        modality_filter: Optional[str] = None,
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k relevant multimodal chunks for a textual query.
        Optional modality_filter: 'image', 'video_frame', 'text', 'table_rows', 'pdf_text', etc.
        Optional source_filter: locks retrieval to a specific source file (e.g. 'document.pdf').
        """
        where_clause = None
        filters = []
        if modality_filter:
            filters.append({"modality": modality_filter})
        if source_filter and source_filter != "All Sources" and not source_filter.startswith("All Sources"):
            filters.append({"source": source_filter})

        if len(filters) == 1:
            where_clause = filters[0]
        elif len(filters) > 1:
            where_clause = {"$and": filters}

        cnt = self.collection.count()
        if cnt == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, cnt),
            where=where_clause
        )

        retrieved_docs = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)
            ids = results["ids"][0] if "ids" in results else [""] * len(docs)

            for i in range(len(docs)):
                retrieved_docs.append({
                    "id": ids[i],
                    "content": docs[i],
                    "metadata": metas[i],
                    "distance": distances[i]
                })

        return retrieved_docs

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

    def reset_collection(self):
        """Clears the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )
        print(f"[VectorStore] Collection '{self.collection_name}' has been reset.")
