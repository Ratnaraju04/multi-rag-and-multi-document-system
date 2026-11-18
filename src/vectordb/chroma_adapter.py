"""
ChromaDB Adapter implementing VectorDBAdapter.
Safely connects to existing ChromaDB persist directories, preserving all data.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from src.vectordb.base_adapter import VectorDBAdapter

logger = logging.getLogger(__name__)

class ChromaAdapter(VectorDBAdapter):
    def __init__(
        self,
        persist_directory: str = "./chroma_multimodal_db",
        collection_name: str = "multimodal_kb",
        embedding_dimensions: int = 384
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_dimensions = embedding_dimensions
        self.client = None
        self.collection = None
        self._is_connected = False

    def connect(self) -> bool:
        """Connects to the persistent ChromaDB instance without modifying existing data."""
        try:
            import chromadb
            from chromadb.config import Settings

            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )
            # Retrieve or create collection preserving existing documents
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"dimensions": self.embedding_dimensions}
            )
            self._is_connected = True
            return True
        except ImportError:
            logger.error("chromadb is not installed. Run: pip install chromadb")
            self._is_connected = False
            return False
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            self._is_connected = False
            return False

    def disconnect(self) -> None:
        self.client = None
        self.collection = None
        self._is_connected = False

    def test_connection(self) -> Dict[str, Any]:
        """Performs real connection test by checking client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings

            if not os.path.exists(self.persist_directory):
                os.makedirs(self.persist_directory, exist_ok=True)

            test_client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            # Real heartbeat check
            heartbeat = test_client.heartbeat()
            col = test_client.get_or_create_collection(name=self.collection_name)
            doc_count = col.count()

            return {
                "success": True,
                "message": f"ChromaDB connected at '{self.persist_directory}'. Heartbeat: {heartbeat}. Collection '{self.collection_name}' has {doc_count} items.",
                "count": doc_count,
                "path": os.path.abspath(self.persist_directory)
            }
        except ImportError:
            return {
                "success": False,
                "message": "chromadb package is not installed. Please install via: pip install chromadb"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"ChromaDB connection test failed: {str(e)}"
            }

    def create_index(self, name: str, dims: int, metadata: Optional[Dict[str, Any]] = None) -> bool:
        if not self._is_connected and not self.connect():
            return False
        try:
            meta = metadata or {}
            meta["dimensions"] = dims
            self.collection_name = name
            self.embedding_dimensions = dims
            self.collection = self.client.get_or_create_collection(name=name, metadata=meta)
            return True
        except Exception as e:
            logger.error(f"Error creating collection '{name}': {e}")
            return False

    def delete_index(self, name: str) -> bool:
        if not self._is_connected and not self.connect():
            return False
        try:
            self.client.delete_collection(name=name)
            if self.collection_name == name:
                self.collection = None
            return True
        except Exception as e:
            logger.error(f"Error deleting collection '{name}': {e}")
            return False

    def list_indexes(self) -> List[str]:
        if not self._is_connected and not self.connect():
            return []
        try:
            collections = self.client.list_collections()
            return [c.name for c in collections]
        except Exception as e:
            logger.error(f"Error listing Chroma collections: {e}")
            return []

    def get_collection_info(self) -> Dict[str, Any]:
        if not self._is_connected and not self.connect():
            return {
                "name": self.collection_name,
                "count": 0,
                "dims": self.embedding_dimensions,
                "dimensions": self.embedding_dimensions,
                "status": "disconnected"
            }
        try:
            cnt = self.collection.count() if self.collection else 0
            return {
                "name": self.collection_name,
                "count": cnt,
                "dims": self.embedding_dimensions,
                "dimensions": self.embedding_dimensions,
                "status": "connected"
            }
        except Exception as e:
            return {
                "name": self.collection_name,
                "count": 0,
                "dims": self.embedding_dimensions,
                "dimensions": self.embedding_dimensions,
                "status": f"error: {str(e)}"
            }


    def add_documents(self, docs: List[Dict[str, Any]], embeddings: Optional[List[List[float]]] = None) -> int:
        if not docs:
            return 0
        if not self._is_connected and not self.connect():
            raise ConnectionError("ChromaDB is not connected.")

        ids = []
        documents_text = []
        metadatas = []

        for idx, doc in enumerate(docs):
            doc_id = str(doc.get("id", f"doc_{idx}"))
            ids.append(doc_id)
            documents_text.append(str(doc.get("content", doc.get("text", ""))))

            clean_meta = {}
            for k, v in doc.get("metadata", {}).items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            metadatas.append(clean_meta)

        batch_size = 50
        total_added = 0
        for i in range(0, len(ids), batch_size):
            b_ids = ids[i : i + batch_size]
            b_texts = documents_text[i : i + batch_size]
            b_metas = metadatas[i : i + batch_size]
            b_embs = embeddings[i : i + batch_size] if embeddings is not None else None

            if b_embs is not None:
                self.collection.upsert(ids=b_ids, documents=b_texts, metadatas=b_metas, embeddings=b_embs)
            else:
                self.collection.upsert(ids=b_ids, documents=b_texts, metadatas=b_metas)
            total_added += len(b_ids)

        return total_added

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        if not self._is_connected and not self.connect():
            raise ConnectionError("ChromaDB is not connected.")

        where_clause = None
        if filters:
            conditions = []
            for k, v in filters.items():
                if v is not None:
                    conditions.append({k: v})
            if len(conditions) == 1:
                where_clause = conditions[0]
            elif len(conditions) > 1:
                where_clause = {"$and": conditions}

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        if where_clause:
            kwargs["where"] = where_clause

        results = self.collection.query(**kwargs)
        formatted: List[Dict[str, Any]] = []

        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            dists = results["distances"][0] if "distances" in results else [0.0] * len(docs)
            ids = results["ids"][0] if "ids" in results else [""] * len(docs)

            for doc_id, doc_text, meta, dist in zip(ids, docs, metas, dists):
                # Chroma returns L2 squared distance by default (or cosine distance depending on config)
                # Convert distance to similarity score in [0, 1]
                similarity = 1.0 / (1.0 + float(dist)) if dist is not None else 1.0
                formatted.append({
                    "id": doc_id,
                    "content": doc_text,
                    "text": doc_text,
                    "metadata": meta,
                    "distance": float(dist) if dist is not None else 0.0,
                    "score": round(similarity, 4)
                })

        return formatted
