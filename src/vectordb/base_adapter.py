"""
Base abstract adapter defining standard interface for all vector database backends.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class VectorDBAdapter(ABC):
    """
    Abstract Vector Database Adapter.
    Standardizes ingestion, similarity retrieval, index lifecycle, and health checking.
    """

    @abstractmethod
    def connect(self) -> bool:
        """Establishes connection to the database. Returns True if connected."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Closes any active connection or releases resources."""
        pass

    @abstractmethod
    def create_index(self, name: str, dims: int, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Creates or switches to an index/collection with specified dimension and metadata."""
        pass

    @abstractmethod
    def delete_index(self, name: str) -> bool:
        """Deletes an index/collection."""
        pass

    @abstractmethod
    def add_documents(self, docs: List[Dict[str, Any]], embeddings: Optional[List[List[float]]] = None) -> int:
        """
        Adds documents and corresponding vector embeddings to the index.
        Returns count of documents successfully added.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs vector similarity search against the active index.
        Returns list of matching document dicts with score/distance and metadata.
        """
        pass

    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Returns status dictionary:
        {
            "name": str,
            "count": int,
            "dims": int,
            "status": "connected" | "disconnected" | "error"
        }
        """
        pass

    @abstractmethod
    def test_connection(self) -> Dict[str, Any]:
        """
        Performs real connection test. Returns:
        {
            "success": bool,
            "message": str
        }
        """
        pass

    @abstractmethod
    def list_indexes(self) -> List[str]:
        """Returns list of available index/collection names."""
        pass
