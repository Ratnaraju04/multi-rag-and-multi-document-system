"""
Vector Database Manager.
Maintains a registry of vector database adapters (Chroma, FAISS, Qdrant),
manages active database selection, and enforces safe switching with re-embedding warnings.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from src.vectordb.base_adapter import VectorDBAdapter
from src.vectordb.chroma_adapter import ChromaAdapter
from src.vectordb.faiss_adapter import FAISSAdapter
from src.vectordb.qdrant_adapter import QdrantAdapter

logger = logging.getLogger(__name__)

class VectorDBManager:
    def __init__(self, default_adapter: str = "chroma"):
        self._adapters: Dict[str, VectorDBAdapter] = {}
        self._active_adapter_name: str = default_adapter
        self._init_default_adapters()

    def _init_default_adapters(self) -> None:
        """Initializes standard adapters with sensible defaults."""
        self._adapters["chroma"] = ChromaAdapter()
        self._adapters["faiss"] = FAISSAdapter()
        self._adapters["qdrant"] = QdrantAdapter()

    def register_adapter(self, name: str, adapter: VectorDBAdapter) -> None:
        """Registers a custom or configured adapter instance."""
        self._adapters[name.lower()] = adapter

    def get_adapter(self, name: Optional[str] = None) -> VectorDBAdapter:
        """Retrieves adapter by name or active adapter."""
        target = (name or self._active_adapter_name).lower()
        if target not in self._adapters:
            raise KeyError(f"Vector DB adapter '{target}' is not registered. Available: {list(self._adapters.keys())}")
        return self._adapters[target]

    def get_active_adapter(self) -> VectorDBAdapter:
        """Returns the currently active VectorDBAdapter."""
        return self.get_adapter(self._active_adapter_name)

    def get_active_adapter_name(self) -> str:
        """Returns name of the currently selected adapter."""
        return self._active_adapter_name

    def set_active_adapter(self, name: str) -> Dict[str, Any]:
        """
        Switches the active vector database.
        Detects if existing data exists in previous DB and produces a re-embedding warning if needed.
        """
        name_lower = name.lower()
        if name_lower not in self._adapters:
            return {
                "success": False,
                "error": f"Unknown database '{name}'. Available: {list(self._adapters.keys())}",
                "warning": None
            }

        prev_name = self._active_adapter_name
        prev_adapter = self._adapters.get(prev_name)
        new_adapter = self._adapters[name_lower]

        warning = None
        # Check if previous DB has indexed items
        prev_count = 0
        if prev_adapter:
            try:
                info = prev_adapter.get_collection_info()
                prev_count = info.get("count", 0)
            except Exception:
                prev_count = 0

        # Check new DB items
        new_count = 0
        try:
            new_info = new_adapter.get_collection_info()
            new_count = new_info.get("count", 0)
        except Exception:
            new_count = 0

        if prev_count > 0 and new_count == 0 and prev_name != name_lower:
            warning = (
                f"RE-INDEXING REQUIRED: You switched active vector database from '{prev_name}' "
                f"({prev_count} documents) to '{name_lower}' (0 documents). "
                f"Your documents are NOT automatically migrated. You must run Ingestion to embed your data into {name_lower}."
            )

        self._active_adapter_name = name_lower
        return {
            "success": True,
            "active_adapter": self._active_adapter_name,
            "warning": warning
        }

    def list_available_adapters(self) -> List[str]:
        """Returns list of registered database adapter names."""
        return list(self._adapters.keys())

    def test_active_connection(self) -> Dict[str, Any]:
        """Runs test_connection on active adapter."""
        adapter = self.get_active_adapter()
        res = adapter.test_connection()
        res["adapter_name"] = self._active_adapter_name
        return res

    def test_all_connections(self) -> Dict[str, Dict[str, Any]]:
        """Tests connectivity across all registered adapters."""
        results = {}
        for name, adapter in self._adapters.items():
            try:
                results[name] = adapter.test_connection()
            except Exception as e:
                results[name] = {"success": False, "message": str(e)}
        return results
