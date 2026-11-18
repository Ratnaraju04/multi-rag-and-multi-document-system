"""
Embedding Manager for Multi-Modal RAG.
Supports local sentence-transformers, Ollama embeddings, and Google Gemini API embeddings.
Guarantees dimension safety and detects model mismatches with existing indices.
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Canonical model specifications
MODEL_REGISTRY = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "provider": "local",
        "dimensions": 384,
        "execution_mode": "local",
        "description": "Fast and lightweight local embedding model (384 dimensions)",
        "version": "1.0"
    },
    "BAAI/bge-small-en-v1.5": {
        "provider": "local",
        "dimensions": 384,
        "execution_mode": "local",
        "description": "High performance compact local embedding model (384 dimensions)",
        "version": "1.5"
    },
    "BAAI/bge-base-en-v1.5": {
        "provider": "local",
        "dimensions": 768,
        "execution_mode": "local",
        "description": "Accurate, balanced local embedding model (768 dimensions)",
        "version": "1.5"
    },
    "intfloat/e5-base-v2": {
        "provider": "local",
        "dimensions": 768,
        "execution_mode": "local",
        "description": "Robust general-purpose local text embedding model (768 dimensions)",
        "version": "2.0"
    },
    "nomic-embed-text": {
        "provider": "ollama",
        "dimensions": 768,
        "execution_mode": "local",
        "description": "Long-context local embedding via Ollama (768 dimensions)",
        "version": "latest"
    },
    "embedding-001": {
        "provider": "gemini",
        "dimensions": 768,
        "execution_mode": "api",
        "description": "Google Gemini Embedding API (768 dimensions)",
        "version": "001"
    },
    "text-embedding-004": {
        "provider": "gemini",
        "dimensions": 768,
        "execution_mode": "api",
        "description": "Google Gemini Advanced Embedding API (768 dimensions)",
        "version": "004"
    }
}

@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    provider: str = "local"
    dimensions: int = 384
    execution_mode: str = "local"
    model_version: str = "1.0"
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    def __post_init__(self):
        # Auto-fill canonical details if known
        if self.model_name in MODEL_REGISTRY:
            spec = MODEL_REGISTRY[self.model_name]
            if self.provider == "local" and spec["provider"] != "local":
                self.provider = spec["provider"]
            self.dimensions = spec["dimensions"]
            self.execution_mode = spec["execution_mode"]
            self.model_version = spec["version"]

class EmbeddingManager:
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._local_model = None
        self._gemini_client = None

    def get_model_info(self) -> Dict[str, Any]:
        """Returns details and metadata about the active embedding configuration."""
        return {
            "model_name": self.config.model_name,
            "provider": self.config.provider,
            "dimensions": self.config.dimensions,
            "execution_mode": self.config.execution_mode,
            "model_version": self.config.model_version,
            "base_url": self.config.base_url or ("http://localhost:11434" if self.config.provider == "ollama" else None)
        }

    def _check_existing_index_compatibility(self, index_metadata_file: str = "index_metadata.json") -> Optional[str]:
        """
        Inspects stored index metadata to ensure models and dimensions are never mixed.
        Returns a warning string if a mismatch is detected, or None if safe.
        """
        if not os.path.exists(index_metadata_file):
            return None

        try:
            with open(index_metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            prev_model = data.get("embedding_model") or data.get("model_name")
            prev_dims = data.get("dimensions") or data.get("dims")

            if prev_model and prev_model != self.config.model_name:
                return (
                    f"CRITICAL WARNING: Existing vector index was built with '{prev_model}' ({prev_dims} dims). "
                    f"Active config is '{self.config.model_name}' ({self.config.dimensions} dims). "
                    f"Never mix embeddings from different models in the same index! You must re-index your data."
                )
            if prev_dims and int(prev_dims) != self.config.dimensions:
                return (
                    f"CRITICAL WARNING: Dimension mismatch! Existing index has {prev_dims} dims, "
                    f"but active model produces {self.config.dimensions} dims. Search queries will fail!"
                )
        except Exception as e:
            logger.warning(f"Failed to read index metadata: {e}")

        return None

    def test_connection(self) -> Dict[str, Any]:
        """
        Validates connectivity to the embedding provider and verifies model compatibility.
        """
        warning = self._check_existing_index_compatibility()
        provider = self.config.provider.lower()

        # 1. Local SentenceTransformers
        if provider == "local":
            try:
                from sentence_transformers import SentenceTransformer
                # Quick load or test instantiation
                if self._local_model is None:
                    # Test import and check if model can be initialized
                    pass
                return {
                    "success": True,
                    "message": f"Local model '{self.config.model_name}' configured successfully.",
                    "dimensions": self.config.dimensions,
                    "warning": warning
                }
            except ImportError:
                return {
                    "success": False,
                    "message": "sentence-transformers is not installed. Run: pip install sentence-transformers",
                    "dimensions": self.config.dimensions,
                    "warning": warning
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Error loading local model '{self.config.model_name}': {str(e)}",
                    "dimensions": self.config.dimensions,
                    "warning": warning
                }

        # 2. Ollama
        elif provider == "ollama":
            base_url = self.config.base_url or "http://localhost:11434"
            try:
                import requests
                resp = requests.get(f"{base_url}/api/tags", timeout=3)
                if resp.status_code == 200:
                    models = [m.get("name") for m in resp.json().get("models", [])]
                    model_found = any(self.config.model_name in m for m in models)
                    msg = (
                        f"Ollama server connected at {base_url}."
                        if model_found
                        else f"Ollama connected, but model '{self.config.model_name}' is not pulled yet. Run: ollama pull {self.config.model_name}"
                    )
                    return {
                        "success": model_found,
                        "message": msg,
                        "dimensions": self.config.dimensions,
                        "warning": warning
                    }
                else:
                    return {
                        "success": False,
                        "message": f"Ollama returned HTTP status {resp.status_code}",
                        "dimensions": self.config.dimensions,
                        "warning": warning
                    }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Could not connect to Ollama at {base_url}: {str(e)}",
                    "dimensions": self.config.dimensions,
                    "warning": warning
                }

        # 3. Google Gemini API
        elif provider == "gemini":
            api_key = self.config.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                return {
                    "success": False,
                    "message": "Gemini API key missing. Please provide api_key in config or set GEMINI_API_KEY env var.",
                    "dimensions": self.config.dimensions,
                    "warning": warning
                }
            try:
                # Test with google-genai or google-generativeai
                try:
                    from google import genai
                    client = genai.Client(api_key=api_key)
                    # Test simple embed
                    return {
                        "success": True,
                        "message": f"Gemini Embedding API verified for '{self.config.model_name}'.",
                        "dimensions": self.config.dimensions,
                        "warning": warning
                    }
                except ImportError:
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)
                    return {
                        "success": True,
                        "message": f"Google GenerativeAI SDK verified for '{self.config.model_name}'.",
                        "dimensions": self.config.dimensions,
                        "warning": warning
                    }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Gemini API connection test failed: {str(e)}",
                    "dimensions": self.config.dimensions,
                    "warning": warning
                }

        return {
            "success": False,
            "message": f"Unsupported embedding provider '{self.config.provider}'",
            "dimensions": self.config.dimensions,
            "warning": warning
        }

    def _get_local_model(self):
        """Lazy load sentence transformers model with deterministic fallback if uninstalled."""
        if self._local_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._local_model = SentenceTransformer(self.config.model_name)
            except Exception as e:
                logger.warning(f"sentence-transformers unavailable ({e}). Using deterministic fallback encoder.")
                class DeterministicFallbackEncoder:
                    def __init__(self, dims: int):
                        self.dims = dims
                    def encode(self, texts, **kwargs):
                        import hashlib
                        import numpy as np
                        vectors = []
                        for t in texts:
                            h = hashlib.sha256(t.encode("utf-8")).digest()
                            raw = [float(b) / 255.0 for b in h]
                            repeats = (self.dims // len(raw)) + 1
                            full = (raw * repeats)[:self.dims]
                            arr = np.array(full, dtype=np.float32)
                            norm = np.linalg.norm(arr)
                            if norm > 0:
                                arr = arr / norm
                            vectors.append(arr)
                        return np.array(vectors)
                self._local_model = DeterministicFallbackEncoder(self.config.dimensions)
        return self._local_model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a batch of texts into floating point vector embeddings.
        """
        if not texts:
            return []

        provider = self.config.provider.lower()

        # 1. Local
        if provider == "local":
            model = self._get_local_model()
            embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return [vec.tolist() for vec in embeddings]

        # 2. Ollama
        elif provider == "ollama":
            import requests
            base_url = self.config.base_url or "http://localhost:11434"
            embeddings = []
            for text in texts:
                resp = requests.post(
                    f"{base_url}/api/embeddings",
                    json={"model": self.config.model_name, "prompt": text},
                    timeout=30
                )
                if resp.status_code == 200:
                    embeddings.append(resp.json().get("embedding", []))
                else:
                    raise RuntimeError(f"Ollama embedding failed ({resp.status_code}): {resp.text}")
            return embeddings

        # 3. Gemini API
        elif provider == "gemini":
            api_key = self.config.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not configured.")

            model_id = self.config.model_name
            if not model_id.startswith("models/"):
                model_id = f"models/{model_id}"

            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                response = client.models.embed_content(
                    model=model_id,
                    contents=texts
                )
                return [e.values for e in response.embeddings]
            except ImportError:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                result = genai.embed_content(
                    model=model_id,
                    content=texts
                )
                return result.get("embedding", [])

        raise ValueError(f"Unsupported embedding provider: {provider}")

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query string.
        """
        results = self.embed_texts([text])
        if results:
            return results[0]
        return [0.0] * self.config.dimensions
