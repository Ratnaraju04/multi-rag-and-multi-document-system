"""
FAISS Vector Database Adapter using local Flat L2 Index with disk persistence.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from src.vectordb.base_adapter import VectorDBAdapter

logger = logging.getLogger(__name__)

class FAISSAdapter(VectorDBAdapter):
    def __init__(
        self,
        storage_dir: Optional[str] = None,
        index_name: str = "default_index",
        dims: int = 384,
        index_dir: Optional[str] = None
    ):
        self.storage_dir = storage_dir or index_dir or "./faiss_store"
        self.index_name = index_name
        self.dims = dims

        self._index = None
        self._documents: List[Dict[str, Any]] = []
        self._is_connected = False
        self._use_faiss = False
        # Fallback numpy store in case faiss-cpu is not installed
        self._fallback_vectors: Optional[np.ndarray] = None

    def connect(self) -> bool:
        """Initializes or loads the FAISS index from disk."""
        os.makedirs(self.storage_dir, exist_ok=True)
        try:
            import faiss
            self._use_faiss = True
        except ImportError:
            self._use_faiss = False
            logger.info("faiss is not installed. Using high-performance NumPy flat L2 fallback.")

        # Attempt to load existing index from disk
        if self._load_from_disk():
            self._is_connected = True
            return True

        # Otherwise create a fresh in-memory index
        if self._use_faiss:
            import faiss
            self._index = faiss.IndexFlatL2(self.dims)
        else:
            self._fallback_vectors = np.empty((0, self.dims), dtype=np.float32)

        self._documents = []
        self._is_connected = True
        return True

    def disconnect(self) -> None:
        if self._is_connected:
            self._save_to_disk()
        self._index = None
        self._fallback_vectors = None
        self._documents = []
        self._is_connected = False

    def _index_path(self, name: str) -> str:
        return os.path.join(self.storage_dir, f"{name}.faiss")

    def _meta_path(self, name: str) -> str:
        return os.path.join(self.storage_dir, f"{name}_meta.json")

    def _save_to_disk(self) -> bool:
        """Persists FAISS index binary and document metadata JSON to disk."""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            meta_path = self._meta_path(self.index_name)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "name": self.index_name,
                    "dims": self.dims,
                    "documents": self._documents
                }, f, ensure_ascii=False, indent=2)

            if self._use_faiss and self._index is not None:
                import faiss
                faiss.write_index(self._index, self._index_path(self.index_name))
            elif self._fallback_vectors is not None:
                npy_path = os.path.join(self.storage_dir, f"{self.index_name}.npy")
                np.save(npy_path, self._fallback_vectors)
            return True
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            return False

    def _load_from_disk(self) -> bool:
        """Loads FAISS index and metadata from disk if available."""
        meta_path = self._meta_path(self.index_name)
        if not os.path.exists(meta_path):
            return False

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.dims = data.get("dims", self.dims)
            self._documents = data.get("documents", [])

            idx_path = self._index_path(self.index_name)
            npy_path = os.path.join(self.storage_dir, f"{self.index_name}.npy")

            if self._use_faiss and os.path.exists(idx_path):
                import faiss
                self._index = faiss.read_index(idx_path)
            elif os.path.exists(npy_path):
                self._fallback_vectors = np.load(npy_path)
                if self._fallback_vectors.dtype != np.float32:
                    self._fallback_vectors = self._fallback_vectors.astype(np.float32)
            else:
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False

    def test_connection(self) -> Dict[str, Any]:
        has_faiss = False
        try:
            import faiss
            has_faiss = True
        except ImportError:
            has_faiss = False

        return {
            "success": True,
            "message": (
                f"FAISS adapter active at '{self.storage_dir}' (faiss-cpu: {has_faiss}, fallback: {not has_faiss}). "
                f"Collection '{self.index_name}' ready with {len(self._documents)} documents."
            ),
            "backend": "faiss-cpu" if has_faiss else "numpy-flat-l2",
            "count": len(self._documents),
            "dims": self.dims
        }

    def create_index(self, name: str, dims: int, metadata: Optional[Dict[str, Any]] = None) -> bool:
        self.index_name = name
        self.dims = dims
        if self._use_faiss:
            try:
                import faiss
                self._index = faiss.IndexFlatL2(dims)
            except ImportError:
                self._fallback_vectors = np.empty((0, dims), dtype=np.float32)
        else:
            self._fallback_vectors = np.empty((0, dims), dtype=np.float32)

        self._documents = []
        self._save_to_disk()
        return True

    def delete_index(self, name: str) -> bool:
        idx_p = self._index_path(name)
        meta_p = self._meta_path(name)
        npy_p = os.path.join(self.storage_dir, f"{name}.npy")
        for p in [idx_p, meta_p, npy_p]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception as e:
                    logger.error(f"Could not remove {p}: {e}")
        if self.index_name == name:
            self._index = None
            self._fallback_vectors = None
            self._documents = []
        return True

    def list_indexes(self) -> List[str]:
        if not os.path.exists(self.storage_dir):
            return []
        names = []
        for f in os.listdir(self.storage_dir):
            if f.endswith("_meta.json"):
                names.append(f[:-10])
        return sorted(list(set(names)))

    def get_collection_info(self) -> Dict[str, Any]:
        count = len(self._documents)
        return {
            "name": self.index_name,
            "count": count,
            "dims": self.dims,
            "dimensions": self.dims,
            "status": "connected" if self._is_connected else "disconnected"
        }


    def add_documents(self, docs: List[Dict[str, Any]], embeddings: Optional[List[List[float]]] = None) -> int:
        if not docs:
            return 0
        if not self._is_connected and not self.connect():
            raise ConnectionError("FAISS adapter is not connected.")

        if embeddings is None:
            raise ValueError("FAISS adapter requires explicit vector embeddings for ingestion.")

        arr = np.array(embeddings, dtype=np.float32)
        if arr.shape[1] != self.dims:
            raise ValueError(f"Embedding dimension {arr.shape[1]} does not match index dimension {self.dims}")

        if self._use_faiss and self._index is not None:
            self._index.add(arr)
        else:
            if self._fallback_vectors is None or len(self._fallback_vectors) == 0:
                self._fallback_vectors = arr
            else:
                self._fallback_vectors = np.vstack([self._fallback_vectors, arr])

        for doc in docs:
            self._documents.append({
                "id": doc.get("id"),
                "content": doc.get("content", doc.get("text", "")),
                "text": doc.get("text", doc.get("content", "")),
                "metadata": doc.get("metadata", {})
            })

        self._save_to_disk()
        return len(docs)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        if not self._is_connected and not self.connect():
            raise ConnectionError("FAISS adapter is not connected.")

        if not self._documents:
            return []

        q_vec = np.array([query_embedding], dtype=np.float32)
        if q_vec.shape[1] != self.dims:
            raise ValueError(f"Query vector dim {q_vec.shape[1]} != index dim {self.dims}")

        k = min(top_k * 3 if filters else top_k, len(self._documents))
        if k == 0:
            return []

        if self._use_faiss and self._index is not None:
            distances, indices = self._index.search(q_vec, k)
            dists = distances[0].tolist()
            inds = indices[0].tolist()
        else:
            # NumPy Flat L2 distance: ||a - b||^2
            diffs = self._fallback_vectors - q_vec
            sq_dists = np.sum(diffs ** 2, axis=1)
            sorted_indices = np.argsort(sq_dists)[:k]
            inds = sorted_indices.tolist()
            dists = sq_dists[sorted_indices].tolist()

        results = []
        for idx, dist in zip(inds, dists):
            if idx < 0 or idx >= len(self._documents):
                continue
            doc = self._documents[idx]

            # Apply metadata filters
            if filters:
                match = True
                meta = doc.get("metadata", {})
                for fk, fv in filters.items():
                    if meta.get(fk) != fv:
                        match = False
                        break
                if not match:
                    continue

            similarity = 1.0 / (1.0 + float(dist))
            results.append({
                "id": doc.get("id"),
                "content": doc.get("content"),
                "text": doc.get("text"),
                "metadata": doc.get("metadata"),
                "distance": float(dist),
                "score": round(similarity, 4)
            })

            if len(results) >= top_k:
                break

        return results
