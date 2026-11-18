"""
Evaluation History Manager backed by SQLite.
Persists and compares benchmark runs, pipeline configurations, and metric trajectories over time.
"""

import os
import json
import sqlite3
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class EvalHistoryManager:
    def __init__(self, db_path: str = "eval_history.db"):
        self.db_path = db_path
        self._mem_conn = None
        if self.db_path == ":memory:":
            self._mem_conn = sqlite3.connect(":memory:")
            self._mem_conn.row_factory = sqlite3.Row
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self.db_path == ":memory:":
            return self._mem_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _close_connection(self, conn: sqlite3.Connection) -> None:
        if conn and conn != self._mem_conn:
            conn.close()

    def _init_db(self) -> None:
        """Initializes the database schema."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS eval_sessions (
                    session_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    llm_provider TEXT,
                    llm_model TEXT,
                    embedding_model TEXT,
                    vector_db TEXT,
                    chunking_strategy TEXT,
                    chunk_size INTEGER,
                    num_test_questions INTEGER,
                    aggregate_metrics TEXT,
                    individual_results TEXT
                )
            """)
            conn.commit()
        finally:
            self._close_connection(conn)


    def save_session(self, eval_session: Dict[str, Any]) -> str:
        """
        Saves an evaluation run session into SQLite.
        Returns the unique session_id.
        """
        session_id = eval_session.get("session_id") or f"eval_{uuid.uuid4().hex[:8]}"
        timestamp = eval_session.get("timestamp") or datetime.now(timezone.utc).isoformat()

        agg = eval_session.get("aggregate_metrics", {})
        agg_json = json.dumps(agg) if not isinstance(agg, str) else agg

        ind = eval_session.get("individual_results", [])
        ind_json = json.dumps(ind) if not isinstance(ind, str) else ind

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO eval_sessions (
                    session_id, timestamp, llm_provider, llm_model,
                    embedding_model, vector_db, chunking_strategy,
                    chunk_size, num_test_questions, aggregate_metrics,
                    individual_results
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                timestamp,
                eval_session.get("llm_provider", "unknown"),
                eval_session.get("llm_model", "unknown"),
                eval_session.get("embedding_model", "unknown"),
                eval_session.get("vector_db", "unknown"),
                eval_session.get("chunking_strategy", "unknown"),
                int(eval_session.get("chunk_size", 0)),
                int(eval_session.get("num_test_questions", len(ind) if isinstance(ind, list) else 0)),
                agg_json,
                ind_json
            ))
            conn.commit()
        finally:
            self._close_connection(conn)

        return session_id

    def load_history(self) -> List[Dict[str, Any]]:
        """
        Retrieves all past evaluation sessions sorted newest to oldest.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM eval_sessions ORDER BY timestamp DESC")
            rows = cursor.fetchall()
        finally:
            self._close_connection(conn)

        history = []
        for r in rows:
            row_dict = dict(r)
            try:
                row_dict["aggregate_metrics"] = json.loads(row_dict.get("aggregate_metrics") or "{}")
            except Exception:
                pass
            try:
                row_dict["individual_results"] = json.loads(row_dict.get("individual_results") or "[]")
            except Exception:
                pass
            history.append(row_dict)

        return history

    def load_all(self) -> List[Dict[str, Any]]:
        """Alias for load_history for UI page compatibility."""
        return self.load_history()


    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single session with complete parsed data.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM eval_sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
        finally:
            self._close_connection(conn)


        if not row:
            return None

        data = dict(row)
        try:
            data["aggregate_metrics"] = json.loads(data.get("aggregate_metrics") or "{}")
        except Exception:
            pass
        try:
            data["individual_results"] = json.loads(data.get("individual_results") or "[]")
        except Exception:
            pass

        return data

    def compare_sessions(self, session_ids: List[str]) -> Dict[str, Any]:
        """
        Performs comparative side-by-side analysis across multiple session runs.
        """
        sessions = []
        for sid in session_ids:
            s = self.get_session(sid)
            if s:
                sessions.append(s)

        if not sessions:
            return {"error": "None of the requested sessions were found."}

        comparison = {
            "session_ids": [s["session_id"] for s in sessions],
            "configurations": [],
            "metrics": {
                "avg_correctness": {},
                "avg_faithfulness": {},
                "avg_relevance": {},
                "avg_latency_total_ms": {}
            }
        }

        for s in sessions:
            sid = s["session_id"]
            comparison["configurations"].append({
                "session_id": sid,
                "timestamp": s.get("timestamp"),
                "llm": f"{s.get('llm_provider')}/{s.get('llm_model')}",
                "embedding": s.get("embedding_model"),
                "vector_db": s.get("vector_db"),
                "chunking": f"{s.get('chunking_strategy')} (size={s.get('chunk_size')})"
            })

            agg = s.get("aggregate_metrics", {})
            comparison["metrics"]["avg_correctness"][sid] = agg.get("avg_answer_correctness")
            comparison["metrics"]["avg_faithfulness"][sid] = agg.get("avg_faithfulness")
            comparison["metrics"]["avg_relevance"][sid] = agg.get("avg_answer_relevance")
            lat = agg.get("avg_latencies_ms", {})
            comparison["metrics"]["avg_latency_total_ms"][sid] = lat.get("total_ms")

        return comparison
