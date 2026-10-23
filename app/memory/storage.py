"""SQLite persistence for local FRIDAY conversations."""
from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


class ConversationStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.executescript("""
                PRAGMA foreign_keys = ON;
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                    content TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages(conversation_id, id);
            """)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Yield one transactional connection and always release its file handle."""
        db = sqlite3.connect(self.path, timeout=10)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys = ON")
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def create_conversation(self, title: str = "New conversation") -> str:
        conversation_id = str(uuid.uuid4())
        now = self._now()
        with self._connection() as db:
            db.execute("INSERT INTO conversations VALUES (?, ?, ?, ?)", (conversation_id, title, now, now))
        return conversation_id

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")
        if not content.strip():
            return
        with self._connection() as db:
            exists = db.execute("SELECT 1 FROM conversations WHERE id=?", (conversation_id,)).fetchone()
            if exists is None:
                raise KeyError("Conversation does not exist")
            db.execute("INSERT INTO messages(conversation_id, role, content, created_at) VALUES(?,?,?,?)",
                       (conversation_id, role, content, self._now()))
            db.execute("UPDATE conversations SET updated_at=? WHERE id=?", (self._now(), conversation_id))
            if role == "user":
                row = db.execute("SELECT title FROM conversations WHERE id=?", (conversation_id,)).fetchone()
                if row and row["title"] == "New conversation":
                    title = content.strip().splitlines()[0][:72]
                    db.execute("UPDATE conversations SET title=? WHERE id=?", (title, conversation_id))

    def list_conversations(self) -> list[dict[str, str]]:
        with self._connection() as db:
            rows = db.execute("SELECT id,title,created_at,updated_at FROM conversations ORDER BY updated_at DESC").fetchall()
        return [dict(row) for row in rows]

    def get_messages(self, conversation_id: str, limit: int = 40) -> list[dict[str, str]]:
        limit = max(1, min(int(limit), 200))
        with self._connection() as db:
            rows = db.execute("SELECT role,content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
                              (conversation_id, limit)).fetchall()
        return [dict(row) for row in reversed(rows)]

    def delete_conversation(self, conversation_id: str) -> None:
        with self._connection() as db:
            db.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
