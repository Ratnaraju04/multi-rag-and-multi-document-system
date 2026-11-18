"""
Excel & Tabular Data Loader for Multi-Modal RAG.
Supports .xlsx, .xls, and .csv files. Converts tables into semantic markdown and natural row representations.
"""

import os
from typing import List, Dict, Any, Optional
import pandas as pd

class ExcelDatasetLoader:
    def __init__(self, rows_per_chunk: int = 15):
        self.rows_per_chunk = rows_per_chunk

    def load(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parses an Excel or CSV file and generates structured document chunks.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)
        documents = []

        if file_ext in [".xlsx", ".xls"]:
            excel_file = pd.ExcelFile(file_path)
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                docs = self._process_dataframe(df, file_name, sheet_name=sheet_name)
                documents.extend(docs)
        elif file_ext == ".csv":
            df = pd.read_csv(file_path)
            docs = self._process_dataframe(df, file_name, sheet_name="CSV_Data")
            documents.extend(docs)
        else:
            raise ValueError(f"Unsupported tabular format: {file_ext}. Use .xlsx, .xls, or .csv")

        return documents

    def _process_dataframe(self, df: pd.DataFrame, file_name: str, sheet_name: str) -> List[Dict[str, Any]]:
        documents = []
        df = df.dropna(how="all")  # Drop completely empty rows
        if df.empty:
            return documents

        # 1. Summary Chunk: Schema & Metadata
        columns_str = ", ".join([str(c) for c in df.columns[:30]])
        preview_df = df.head(3) if len(df.columns) <= 25 else df.head(3).iloc[:, :20]
        preview_md = preview_df.to_markdown(index=False) if hasattr(preview_df, "to_markdown") else str(preview_df)

        summary_text = (
            f"[Tabular Dataset: {file_name} | Sheet: {sheet_name}]\n"
            f"Total Rows: {len(df)}, Total Columns: {len(df.columns)}\n"
            f"Columns: {columns_str}\n"
            f"Sample preview:\n{preview_md}"
        )
        documents.append({
            "id": f"table_summary_{file_name}_{sheet_name}",
            "modality": "text",
            "content": summary_text,
            "metadata": {
                "source": file_name,
                "sheet": sheet_name,
                "type": "table_summary",
                "rows": len(df),
                "columns": len(df.columns)
            }
        })

        # 2. Block/Group Chunks (Fast vectorized dict conversion, capped at 12 chunks for high speed)
        step = max(30, self.rows_per_chunk)
        max_chunks = 12
        chunk_count = 0

        for i in range(0, min(len(df), step * max_chunks), step):
            sub_df = df.iloc[i : i + step]
            sub_preview = sub_df.head(5) if len(sub_df.columns) <= 25 else sub_df.head(3).iloc[:, :20]
            markdown_table = sub_preview.to_markdown(index=False) if hasattr(sub_preview, "to_markdown") else str(sub_preview)
            
            sample_records = sub_df.head(10).to_dict(orient="records")
            row_sentences = [
                f"Row {i + r_idx + 1}: " + " | ".join(f"{col}: {val}" for col, val in rec.items() if pd.notna(val) and str(val) != "")
                for r_idx, rec in enumerate(sample_records)
            ]
            natural_text = "\n".join(row_sentences)

            content = (
                f"[Dataset: {file_name} | Sheet: {sheet_name} | Rows {i+1} to {min(i+step, len(df))}]\n"
                f"### Structured Table:\n{markdown_table}\n\n"
                f"### Record Details:\n{natural_text}"
            )

            documents.append({
                "id": f"table_rows_{file_name}_{sheet_name}_{i}",
                "modality": "text",
                "content": content,
                "metadata": {
                    "source": file_name,
                    "sheet": sheet_name,
                    "type": "table_rows",
                    "row_start": i + 1,
                    "row_end": min(i + step, len(df))
                }
            })
            chunk_count += 1

        return documents
