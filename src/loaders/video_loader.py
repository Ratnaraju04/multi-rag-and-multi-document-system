"""
Video Loader for Multi-Modal RAG (Optimized for High Speed & Concurrency).
Extracts keyframes at intervals using OpenCV, stores timestamp metadata,
and generates semantic visual descriptions of video segments concurrently.
"""

import os
import cv2
from typing import List, Dict, Any, Optional
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

class VideoLoader:
    def __init__(self, frame_interval_sec: float = 3.0, output_dir: str = "./extracted_frames", gemini_client=None):
        self.frame_interval_sec = frame_interval_sec
        self.output_dir = output_dir
        self.client = gemini_client
        os.makedirs(self.output_dir, exist_ok=True)

    def load_video(self, video_path: str, max_frames: int = 8) -> List[Dict[str, Any]]:
        """
        Parses video file, extracts keyframes at interval, and returns timestamped RAG documents.
        Uses ThreadPoolExecutor for concurrent captioning across extracted frames.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps
        video_filename = os.path.splitext(os.path.basename(video_path))[0]

        frame_step = int(fps * self.frame_interval_sec)
        if frame_step <= 0:
            frame_step = 1

        extracted_items = []
        frame_idx = 0

        # Fast OpenCV extraction
        while cap.isOpened() and len(extracted_items) < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_sec = frame_idx / fps
            timestamp_formatted = self._format_timestamp(timestamp_sec)

            frame_filename = f"{video_filename}_t{int(timestamp_sec):04d}.jpg"
            frame_path = os.path.join(self.output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)

            extracted_items.append({
                "frame_idx": frame_idx,
                "timestamp_sec": timestamp_sec,
                "timestamp_formatted": timestamp_formatted,
                "frame_path": frame_path
            })

            frame_idx += frame_step

        cap.release()

        # Concurrently generate captions using ThreadPoolExecutor
        captions = {}
        if self.client and extracted_items:
            def _caption_worker(item):
                c = self._describe_frame(item["frame_path"], item["timestamp_formatted"], video_filename)
                return item["frame_path"], c

            with ThreadPoolExecutor(max_workers=min(5, len(extracted_items))) as executor:
                results = executor.map(_caption_worker, extracted_items)
                for fpath, cap_text in results:
                    captions[fpath] = cap_text

        documents = []
        for item in extracted_items:
            caption = captions.get(item["frame_path"], "")
            if not caption:
                caption = f"Video frame at timestamp {item['timestamp_formatted']} from '{video_filename}'."

            content = (
                f"[Video: {os.path.basename(video_path)} | Timestamp: {item['timestamp_formatted']}]\n"
                f"Scene Description: {caption}\n"
                f"Keyframe saved at: {item['frame_path']}"
            )

            documents.append({
                "id": f"vid_{video_filename}_{item['frame_idx']}",
                "modality": "video_frame",
                "content": content,
                "metadata": {
                    "source": os.path.basename(video_path),
                    "video_path": os.path.abspath(video_path),
                    "frame_path": os.path.abspath(item["frame_path"]),
                    "type": "video_keyframe",
                    "timestamp_sec": item["timestamp_sec"],
                    "timestamp": item["timestamp_formatted"],
                    "description": caption
                }
            })

        print(f"[VideoLoader] Fast extracted {len(documents)} keyframes from '{os.path.basename(video_path)}'.")
        return documents

    def _describe_frame(self, frame_path: str, timestamp_str: str, video_name: str) -> str:
        """Calls Gemini Vision with fast downscaled frame thumbnail."""
        try:
            with Image.open(frame_path) as img:
                img_thumb = img.copy()
                img_thumb.thumbnail((768, 768))
            prompt = (
                f"This is a keyframe taken at timestamp {timestamp_str} from video '{video_name}'. "
                "Describe what is happening in this scene, including any visible actions, text, UI elements, or objects."
            )
            for model_name in ["gemini-2.0-flash", "gemini-1.5-flash"]:
                try:
                    if hasattr(self.client, "models"):
                        response = self.client.models.generate_content(
                            model=model_name,
                            contents=[prompt, img_thumb]
                        )
                        return response.text.strip()
                    elif hasattr(self.client, "GenerativeModel"):
                        model = self.client.GenerativeModel(model_name)
                        response = model.generate_content([prompt, img_thumb])
                        return response.text.strip()
                except Exception:
                    continue
        except Exception as e:
            print(f"[VideoLoader] Frame captioning skipped: {e}")
        return ""

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
