"""
Image Loader for Multi-Modal RAG.
Supports local image files (.jpg, .png, .webp, .jpeg), folders, and URLs.
Extracts visual metadata and generates detailed semantic descriptions via Gemini Vision.
"""

import os
import glob
from typing import List, Dict, Any, Optional
from PIL import Image

class ImageLoader:
    def __init__(self, gemini_client=None):
        self.client = gemini_client

    def load_image(self, image_path: str, caption: Optional[str] = None) -> Dict[str, Any]:
        """Loads a single image file, extracts metadata, and optionally generates an AI visual description."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")

        try:
            with Image.open(image_path) as img:
                width, height = img.size
                img_format = img.format
                mode = img.mode
        except Exception as e:
            raise ValueError(f"Failed to open image {image_path}: {e}")

        file_name = os.path.basename(image_path)
        visual_description = caption or ""

        # If caption not provided and Gemini client is available, generate description
        if not visual_description and self.client:
            visual_description = self._generate_gemini_caption(image_path)

        if not visual_description:
            visual_description = f"Image file '{file_name}' ({width}x{height}, format {img_format})."

        content = (
            f"[Image Source: {file_name}]\n"
            f"Dimensions: {width}x{height} | Format: {img_format}\n"
            f"Visual Description / Content: {visual_description}"
        )

        return {
            "id": f"img_{file_name}_{os.path.getsize(image_path)}",
            "modality": "image",
            "content": content,
            "image_path": os.path.abspath(image_path),
            "metadata": {
                "source": file_name,
                "file_path": os.path.abspath(image_path),
                "type": "image",
                "width": width,
                "height": height,
                "format": img_format,
                "description": visual_description
            }
        }

    def load_directory(self, dir_path: str, extensions: List[str] = None) -> List[Dict[str, Any]]:
        """Loads all images from a directory."""
        if extensions is None:
            extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        
        documents = []
        for ext in extensions:
            pattern = os.path.join(dir_path, ext)
            for path in glob.glob(pattern):
                try:
                    doc = self.load_image(path)
                    documents.append(doc)
                except Exception as e:
                    print(f"[ImageLoader] Warning: Skipping {path} due to error: {e}")
        return documents

    def _generate_gemini_caption(self, image_path: str) -> str:
        """Uses Gemini to describe the image content, OCR any text, and capture diagrams/charts."""
        try:
            with Image.open(image_path) as img:
                img_copy = img.copy()
                img_copy.thumbnail((768, 768))
            prompt = (
                "Provide a comprehensive, detailed description of this image for a search index. "
                "1. Transcribe any visible text (OCR). "
                "2. Describe all main objects, people, diagrams, charts, numbers, and layout. "
                "3. Summarize the overall theme or message."
            )
            for model_name in ["gemini-2.0-flash", "gemini-1.5-flash"]:
                try:
                    if hasattr(self.client, "models"):
                        response = self.client.models.generate_content(
                            model=model_name,
                            contents=[prompt, img_copy]
                        )
                        return response.text.strip()
                    elif hasattr(self.client, "GenerativeModel"):
                        model = self.client.GenerativeModel(model_name)
                        response = model.generate_content([prompt, img_copy])
                        return response.text.strip()
                except Exception:
                    continue
        except Exception as e:
            print(f"[ImageLoader] Captioning fallback error: {e}")
        return ""
