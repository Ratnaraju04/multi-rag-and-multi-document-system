"""
Qdrant Vector Database Adapter supporting Local (in-memory/disk) and Cloud configurations.
Implements real connection tests before reporting operational status.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from src.vectordb.base_adapter import VectorDBAdapter

logger = logging.getLogger(__name__)

class QdrantAdapter(VectorDBAdapter):
    def __init__(
        self,
        collection_name: str = "multimodal_kb",
        location: str = ":memory:",  # ":memory:", local path e.g. "./qdrant_storage", or cloud url
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        dims: int = 384
    ):
        self.collection_name = collection_name
        self.location = location
        self.url = url
        self.api_key = api_key
        self.dims = dims
        self.client = None
        self._is_connected = False

    def connect(self) -> bool:
        """Connects to Qdrant local or cloud instance."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models

            if self.url:
                self.client = QdrantClient(url=self.url, api_key=self.api_key)
            elif self.location == ":memory:":
                self.client = QdrantClient(location=":memory:")
            else:
                os.makedirs(self.location, exist_ok=True)
                self.client = QdrantClient(path=self.location)

            # Ensure collection exists
            self._ensure_collection()
            self._is_connected = True
            return True
        except ImportError:
            logger.error("qdrant-client is not installed. Run: pip install qdrant-client")
            self._is_connected = False
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            self._is_connected = False
            return False

    def disconnect(self) -> None:
        self.client = None
        self._is_connected = False

    def _ensure_collection(self) -> None:
        if not self.client:
            return
        from qdrant_client.http import models

        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.dims,
                    distance=models.Distance.COSINE
                )
            )

    def test_connection(self) -> Dict[str, Any]:
        """
        Executes a real ping and collections check against Qdrant.
        """
        try:
            from qdrant_client import QdrantClient

            if self.url:
                client = QdrantClient(url=self.url, api_key=self.api_key, timeout=5)
            elif self.location == ":memory:":
                client = QdrantClient(location=":memory:")
            else:
                os.makedirs(self.location, exist_ok=True)
                client = QdrantClient(path=self.location)

            # Real check
            cols = client.get_collections()
            col_names = [c.name for c in cols.collections]

            target_info = None
            if self.collection_name in col_names:
                info = client.get_collection(self.collection_name)
                target_info = {
                    "vectors_count": getattr(info, "vectors_count", getattr(info, "points_count", 0)),
                    "status": str(info.status)
                }

            endpoint = self.url if self.url else f"Local ({self.location})"
            return {
                "success": True,
                "message": f"Successfully connected to Qdrant at {endpoint}.",
                "collections": col_names,
                "collection_details": target_info
            }
        except ImportError:
            return {
                "success": False,
                "message": "qdrant-client is not installed. Please install via: pip install qdrant-client"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Qdrant connection test failed: {str(e)}"
            }

    def create_index(self, name: str, dims: int, metadata: Optional[Dict[str, Any]] = None) -> bool:
        if not self._is_connected and not self.connect():
            return False
        try:
            from qdrant_client.http import models
            self.collection_name = name
            self.dims = dims
            self.client.recreate_collection(
                collection_name=name,
                vectors_config=models.VectorParams(size=dims, distance=models.Distance.COSINE)
            )
            return True
        except Exception as e:
            logger.error(f"Failed to create Qdrant collection {name}: {e}")
            return False

    def delete_index(self, name: str) -> bool:
        if not self._is_connected and not self.connect():
            return False
        try:
            self.client.delete_collection(collection_name=name)
            return True
        except Exception as e:
            logger.error(f"Failed to delete Qdrant collection {name}: {e}")
            return False

    def list_indexes(self) -> List[str]:
        if not self._is_connected and not self.connect():
            return []
        try:
            cols = self.client.get_collections().collections
            return [c.name for c in cols]
        except Exception as e:
            logger.error(f"Failed to list Qdrant collections: {e}")
            return []

    def get_collection_info(self) -> Dict[str, Any]:
        if not self._is_connected and not self.connect():
            return {
                "name": self.collection_name,
                "count": 0,
                "dims": self.dims,
                "status": "disconnected"
            }
        try:
            info = self.client.get_collection(self.collection_name)
            count = getattr(info, "points_count", getattr(info, "vectors_count", 0))
            return {
                "name": self.collection_name,
                "count": count or 0,
                "dims": self.dims,
                "status": "connected"
            }
        except Exception as e:
            return {
                "name": self.collection_name,
                "count": 0,
                "dims": self.dims,
                "status": f"error: {str(e)}"
            }

    def add_documents(self, docs: List[Dict[str, Any]], embeddings: Optional[List[List[float]]] = None) -> int:
        if not docs:
            return 0
        if not self._is_connected and not self.connect():
            raise ConnectionError("Qdrant is not connected.")

        if embeddings is None:
            raise ValueError("Qdrant requires pre-computed vector embeddings.")

        from qdrant_client.http import models

        points = []
        for idx, (doc, emb) in enumerate(zip(docs, embeddings)):
            doc_id = doc.get("id", idx)
            # Qdrant accepts integer or UUID for ID; use hash integer if string
            point_id = abs(hash(str(doc_id))) % (2**63 - 1)

            payload = {
                "raw_id": str(doc_id),
                "content": doc.get("content", doc.get("text", "")),
                "text": doc.get("text", doc.get("content", "")),
                "metadata": doc.get("metadata", {})
            }

            points.append(models.PointStruct(
                id=point_id,
                vector=emb,
                payload=payload
            ))

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        return len(points)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        if not self._is_connected and not self.connect():
            raise ConnectionError("Qdrant is not connected.")

        from qdrant_client.http import models

        query_filter = None
        if filters:
            must_clauses = []
            for k, v in filters.items():
                must_clauses.append(
                    models.FieldCondition(
                        key=f"metadata.{k}",
                        match=models.MatchValue(value=v)
                    )
                )
            if must_clauses:
                query_filter = models.Filter(must=must_clauses)

        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=top_k
        )

        formatted = []
        for hit in search_result:
            payload = hit.payload or {}
            formatted.append({
                "id": payload.get("raw_id", str(hit.id)),
                "content": payload.get("content", payload.get("text", "")),
                "text": payload.get("text", payload.get("content", "")),
                "metadata": payload.get("metadata", {}),
                "score": round(float(hit.score), 4),
                "distance": round(1.0 - float(hit.score), 4)
            })

        return formatted
