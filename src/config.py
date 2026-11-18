"""
Configuration module for the Multi-Modal RAG system.
Supports both Google Colab (userdata.get) and standard local environments (os.environ).
"""

import os
from dataclasses import dataclass, field
from typing import Optional

def _load_dotenv_fallback():
    """Load key-value pairs from .env if present."""
    for p in [".env", os.path.join(os.path.dirname(__file__), "..", ".env")]:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k, v = k.strip(), v.strip().strip("'\"")
                            if k and not os.environ.get(k):
                                os.environ[k] = v
            except Exception:
                pass

_load_dotenv_fallback()

def get_api_key(key_name: str = "GEMINI_API_KEY") -> str:
    """Retrieve API key from Colab secrets or environment variables."""
    # 1. Try Google Colab userdata
    try:
        from google.colab import userdata  # type: ignore
        key = userdata.get(key_name)
        if key:
            return key
    except Exception:
        pass

    # 2. Try standard environment variables
    key = os.environ.get(key_name) or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key

    return ""

@dataclass
class RAGConfig:
    api_key: str = field(default_factory=get_api_key)
    gemini_model: str = "gemini-1.5-flash"
    chroma_db_dir: str = "./chroma_multimodal_db"
    collection_name: str = "multimodal_knowledge_base"
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    
    # Video processing configs
    video_frame_interval_sec: float = 3.0  # Extract 1 keyframe every 3 seconds
    video_output_dir: str = "./extracted_frames"
    
    # Text chunking configs
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    def __post_init__(self):
        os.makedirs(self.chroma_db_dir, exist_ok=True)
        os.makedirs(self.video_output_dir, exist_ok=True)
