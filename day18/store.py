"""SQLite persistence for Day 18 task snapshots."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def default_db_path() -> Path:
    override = os.environ.get("DAY18_DB_PATH")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / "data" / "snapshots.db"


def clock_from_stored(value: str) -> str:
    return datetime.fromisoformat(value).strftime("%H:%M:%S")


@dataclass(frozen=True)
class Snapshot:
    id: int
    task_id: str
    status: str
    collected_at: str


class Store:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    collected_at TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def insert_snapshot(
        self,
        task_id: str,
        status: str,
        collected_at: datetime,
    ) -> int:
        stored = collected_at.replace(microsecond=0).isoformat(timespec="seconds")
        conn = self._connect()
        try:
            cursor = conn.execute(
                """
                INSERT INTO snapshots (task_id, status, collected_at)
                VALUES (?, ?, ?)
                """,
                (task_id, status, stored),
            )
            conn.commit()
            return int(cursor.lastrowid)
        finally:
            conn.close()

    def recent(self, limit: int = 20) -> list[Snapshot]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT id, task_id, status, collected_at
                FROM snapshots
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        finally:
            conn.close()
        return [
            Snapshot(
                id=int(row["id"]),
                task_id=str(row["task_id"]),
                status=str(row["status"]),
                collected_at=str(row["collected_at"]),
            )
            for row in rows
        ]

    def count(self) -> int:
        conn = self._connect()
        try:
            row = conn.execute("SELECT COUNT(*) AS n FROM snapshots").fetchone()
        finally:
            conn.close()
        return int(row["n"])

    def latest_collected_at(self) -> str | None:
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT collected_at
                FROM snapshots
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return str(row["collected_at"])

    def summary(self) -> dict:
        """Aggregation used by the MCP server only."""
        conn = self._connect()
        try:
            total = conn.execute("SELECT COUNT(*) AS n FROM snapshots").fetchone()
            grouped = conn.execute(
                """
                SELECT status, COUNT(*) AS n
                FROM snapshots
                GROUP BY status
                """
            ).fetchall()
            last = conn.execute(
                """
                SELECT collected_at
                FROM snapshots
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
        finally:
            conn.close()

        counts: dict[str, int] = {"IN_PROGRESS": 0, "DONE": 0}
        for row in grouped:
            key = str(row["status"]).strip().upper()
            counts[key] = counts.get(key, 0) + int(row["n"])

        last_collection = None
        if last is not None:
            last_collection = clock_from_stored(str(last["collected_at"]))

        return {
            "total_snapshots": int(total["n"]),
            "counts": counts,
            "last_collection": last_collection,
        }
