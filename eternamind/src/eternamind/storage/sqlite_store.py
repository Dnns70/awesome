from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator


class SQLiteStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_message TEXT NOT NULL,
                    assistant_response TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    context TEXT DEFAULT '',
                    importance REAL DEFAULT 0.5,
                    timestamp TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS reflections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    session_id TEXT DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS social_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT DEFAULT 'default',
                    model_content TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS temporal_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS system_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(content, context, content=memories, content_rowid=rowid);

                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, content, context) VALUES (new.rowid, new.content, new.context);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content, context) VALUES ('delete', old.rowid, old.content, old.context);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content, context) VALUES ('delete', old.rowid, old.content, old.context);
                    INSERT INTO memories_fts(rowid, content, context) VALUES (new.rowid, new.content, new.context);
                END;
            """)

    # --- Interactions ---

    def save_interaction(self, user_message: str, assistant_response: str, metadata: dict[str, Any] | None = None) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO interactions (user_message, assistant_response, timestamp, metadata) VALUES (?, ?, ?, ?)",
                (user_message, assistant_response, datetime.utcnow().isoformat(), json.dumps(metadata or {})),
            )

    def get_recent_interactions(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM interactions ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def count_interactions(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]

    # --- Memories ---

    def save_memory(self, memory_id: str, content: str, context: str = "", importance: float = 0.5) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO memories (id, content, context, importance, timestamp) VALUES (?, ?, ?, ?, ?)",
                (memory_id, content, context, importance, datetime.utcnow().isoformat()),
            )

    def search_memories_fts(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT m.* FROM memories m
                   JOIN memories_fts fts ON m.rowid = fts.rowid
                   WHERE memories_fts MATCH ?
                   ORDER BY rank LIMIT ?""",
                (query, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_memories(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM memories ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def count_memories(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    # --- Reflections ---

    def save_reflection(self, content: str, session_id: str = "") -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO reflections (content, timestamp, session_id) VALUES (?, ?, ?)",
                (content, datetime.utcnow().isoformat(), session_id),
            )

    def get_recent_reflections(self, limit: int = 5) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM reflections ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # --- Social Models ---

    def save_social_model(self, model_content: str, user_id: str = "default") -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO social_models (user_id, model_content, timestamp) VALUES (?, ?, ?)",
                (user_id, model_content, datetime.utcnow().isoformat()),
            )

    def get_latest_social_model(self, user_id: str = "default") -> str:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT model_content FROM social_models WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1",
                (user_id,),
            ).fetchone()
        return row["model_content"] if row else ""

    # --- Temporal Events ---

    def save_temporal_event(self, event_type: str, description: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO temporal_events (event_type, description, timestamp) VALUES (?, ?, ?)",
                (event_type, description, datetime.utcnow().isoformat()),
            )

    def get_session_duration_seconds(self) -> float:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MIN(timestamp) as first FROM temporal_events WHERE event_type = 'session_start'"
            ).fetchone()
        if not row or not row["first"]:
            return 0.0
        first = datetime.fromisoformat(row["first"])
        return (datetime.utcnow() - first).total_seconds()

    def get_last_interaction_seconds_ago(self) -> float:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MAX(timestamp) as last FROM interactions"
            ).fetchone()
        if not row or not row["last"]:
            return 0.0
        last = datetime.fromisoformat(row["last"])
        return (datetime.utcnow() - last).total_seconds()

    # --- System State ---

    def set_state(self, key: str, value: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, datetime.utcnow().isoformat()),
            )

    def get_state(self, key: str, default: str = "") -> str:
        with self._conn() as conn:
            row = conn.execute("SELECT value FROM system_state WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default
