"""
Dataset Loader for Multi-Modal RAG.
Supports JSON, JSONL, Parquet files, as well as in-memory Hugging Face Datasets or Python dictionaries.
"""

import os
import json
from typing import List, Dict, Any, Union

class DatasetLoader:
    def __init__(self, records_per_chunk: int = 10):
        self.records_per_chunk = records_per_chunk

    def load_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Loads JSON, JSONL, or Parquet datasets from disk."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        file_ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)

        if file_ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return self.load_records(data, dataset_name=file_name)
            elif isinstance(data, dict):
                return self.load_records([data], dataset_name=file_name)
        elif file_ext == ".jsonl":
            records = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return self.load_records(records, dataset_name=file_name)
        elif file_ext == ".parquet":
            import pandas as pd
            df = pd.read_parquet(file_path)
            records = df.to_dict(orient="records")
            return self.load_records(records, dataset_name=file_name)
        else:
            raise ValueError(f"Unsupported dataset format: {file_ext}. Use .json, .jsonl, or .parquet")

        return []

    def load_records(self, records: List[Dict[str, Any]], dataset_name: str = "custom_dataset") -> List[Dict[str, Any]]:
        """Converts a list of dicts (e.g. from HuggingFace dataset or custom API) into RAG documents."""
        documents = []
        if not records:
            return documents

        # Group records into chunks
        for i in range(0, len(records), self.records_per_chunk):
            chunk_records = records[i : i + self.records_per_chunk]
            formatted_entries = []
            for j, rec in enumerate(chunk_records):
                idx = i + j + 1
                rec_str = json.dumps(rec, indent=2, ensure_ascii=False)
                formatted_entries.append(f"--- Record #{idx} ---\n{rec_str}")

            content = (
                f"[Dataset: {dataset_name} | Records {i+1} to {min(i+self.records_per_chunk, len(records))}]\n"
                + "\n\n".join(formatted_entries)
            )

            documents.append({
                "id": f"dataset_{dataset_name}_{i}",
                "modality": "text",
                "content": content,
                "metadata": {
                    "source": dataset_name,
                    "type": "dataset_records",
                    "record_start": i + 1,
                    "record_end": min(i + self.records_per_chunk, len(records)),
                    "total_records": len(records)
                }
            })

        return documents
