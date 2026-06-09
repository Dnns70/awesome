from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

_MIGRATIONS = [
    "ALTER TABLE memories ADD COLUMN access_count INTEGER DEFAULT 0",
    "ALTER TABLE memories ADD COLUMN last_accessed TEXT DEFAULT NULL",
]


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

                CREATE TABLE IF NOT EXISTS goals (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    priority INTEGER DEFAULT 5,
                    status TEXT DEFAULT 'active',
                    progress_notes TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    source TEXT DEFAULT 'user'
                );

                CREATE TABLE IF NOT EXISTS agent_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_type TEXT NOT NULL,
                    interaction_id INTEGER NOT NULL,
                    score REAL NOT NULL,
                    score_reason TEXT DEFAULT '',
                    timestamp TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_agent_scores_type
                    ON agent_scores(agent_type, timestamp DESC);

                CREATE TABLE IF NOT EXISTS identity_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_content TEXT NOT NULL,
                    drift_score REAL DEFAULT 0.0,
                    timestamp TEXT NOT NULL,
                    interaction_count_at_snapshot INTEGER DEFAULT 0
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
            # Migrate existing databases
            for migration in _MIGRATIONS:
                try:
                    conn.execute(migration)
                except sqlite3.OperationalError:
                    pass  # column already exists

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

    def get_last_interaction_id(self) -> int | None:
        with self._conn() as conn:
            row = conn.execute("SELECT MAX(id) as last_id FROM interactions").fetchone()
        return row["last_id"] if row else None

    # --- Memories ---

    def save_memory(self, memory_id: str, content: str, context: str = "", importance: float = 0.5) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO memories (id, content, context, importance, timestamp, access_count)
                   VALUES (?, ?, ?, ?, ?, 0)
                   ON CONFLICT(id) DO UPDATE SET
                       content=excluded.content,
                       context=excluded.context,
                       importance=excluded.importance,
                       timestamp=excluded.timestamp""",
                (memory_id, content, context, importance, datetime.utcnow().isoformat()),
            )

    def record_memory_access(self, memory_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE memories
                   SET access_count = COALESCE(access_count, 0) + 1,
                       last_accessed = ?
                   WHERE id = ?""",
                (datetime.utcnow().isoformat(), memory_id),
            )

    def update_memory_importance(self, memory_id: str, new_importance: float) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE memories SET importance = ? WHERE id = ?",
                (max(0.0, min(1.0, new_importance)), memory_id),
            )

    def get_high_importance_memories(self, min_importance: float = 0.7, limit: int = 10) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE importance >= ? ORDER BY importance DESC LIMIT ?",
                (min_importance, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_memories_for_rescoring(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, content, importance, access_count, last_accessed, timestamp FROM memories LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

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

    # --- Goals ---

    def save_goal(self, goal_id: str, title: str, description: str = "", priority: int = 5, source: str = "user") -> None:
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO goals (id, title, description, priority, status, created_at, updated_at, source)
                   VALUES (?, ?, ?, ?, 'active', ?, ?, ?)""",
                (goal_id, title, description, priority, now, now, source),
            )

    def get_active_goals(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM goals WHERE status = 'active' ORDER BY priority ASC, created_at ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_goals(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM goals ORDER BY status ASC, priority ASC, created_at ASC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_goal(self, goal_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM goals WHERE id = ?", (goal_id,)).fetchone()
        return dict(row) if row else None

    def update_goal_status(self, goal_id: str, status: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE goals SET status = ?, updated_at = ? WHERE id = ?",
                (status, datetime.utcnow().isoformat(), goal_id),
            )

    def update_goal_priority(self, goal_id: str, priority: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE goals SET priority = ?, updated_at = ? WHERE id = ?",
                (priority, datetime.utcnow().isoformat(), goal_id),
            )

    def append_goal_progress(self, goal_id: str, note: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE goals
                   SET progress_notes = CASE
                       WHEN progress_notes = '' THEN ?
                       ELSE progress_notes || char(10) || ?
                   END,
                   updated_at = ?
                   WHERE id = ?""",
                (note, note, datetime.utcnow().isoformat(), goal_id),
            )

    def delete_goal(self, goal_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM goals WHERE id = ?", (goal_id,))

    def count_active_goals(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM goals WHERE status = 'active'").fetchone()[0]

    # --- Agent Scores ---

    def save_agent_score(self, agent_type: str, interaction_id: int, score: float, reason: str = "") -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO agent_scores (agent_type, interaction_id, score, score_reason, timestamp) VALUES (?, ?, ?, ?, ?)",
                (agent_type, interaction_id, score, reason, datetime.utcnow().isoformat()),
            )

    def get_rolling_agent_scores(self, agent_type: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_scores WHERE agent_type = ? ORDER BY timestamp DESC LIMIT ?",
                (agent_type, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_average_agent_score(self, agent_type: str, last_n: int = 10) -> float:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT AVG(score) as avg_score FROM (
                       SELECT score FROM agent_scores
                       WHERE agent_type = ?
                       ORDER BY timestamp DESC
                       LIMIT ?
                   )""",
                (agent_type, last_n),
            ).fetchone()
        if row and row["avg_score"] is not None:
            return float(row["avg_score"])
        return 0.5

    # --- Identity Snapshots ---

    def save_identity_snapshot(self, snapshot_content: str, drift_score: float, interaction_count: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO identity_snapshots (snapshot_content, drift_score, timestamp, interaction_count_at_snapshot) VALUES (?, ?, ?, ?)",
                (snapshot_content, drift_score, datetime.utcnow().isoformat(), interaction_count),
            )

    def get_recent_identity_snapshots(self, limit: int = 5) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM identity_snapshots ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_latest_identity_snapshot(self) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM identity_snapshots ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

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
