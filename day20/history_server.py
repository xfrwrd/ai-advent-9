"""MCP server history_server: snapshots already stored by Day 18."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path

from mcp.server.mcpserver import MCPServer

mcp = MCPServer(
    name="history_server",
    instructions="Historical task snapshots from the Day 18 SQLite database.",
)


def _db_path() -> Path:
    override = os.environ.get("DAY18_DB_PATH")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "day18" / "data" / "snapshots.db"


def _clock(value: str | None) -> str | None:
    if not value:
        return None
    return datetime.fromisoformat(value).strftime("%H:%M:%S")


def _rows(task_id: str) -> list[sqlite3.Row]:
    path = _db_path()
    if not path.exists():
        return []
    conn = sqlite3.connect(path, timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        return list(
            conn.execute(
                """
                SELECT task_id, status, collected_at
                FROM snapshots
                WHERE task_id = ?
                ORDER BY id DESC
                """,
                (task_id,),
            )
        )
    finally:
        conn.close()


@mcp.tool(
    description=(
        "Summarize stored snapshots for one task from SQLite history. "
        "Use this for history, not for the live task status."
    )
)
def get_task_summary(task_id: str) -> dict:
    """Aggregate snapshots of task_id. Does not call the Task API."""
    task_id = (task_id or "").strip()
    if not task_id:
        return {"error": "invalid_input", "message": "task_id is required"}
    rows = _rows(task_id)
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"]).strip().lower()
        counts[status] = counts.get(status, 0) + 1
    last = _clock(str(rows[0]["collected_at"])) if rows else None
    return {
        "task_id": task_id,
        "total_snapshots": len(rows),
        "by_status": counts,
        "last_collection": last,
    }


@mcp.tool(
    description="Return the latest stored snapshots for one task from SQLite history."
)
def get_recent_snapshots(task_id: str, limit: int = 10) -> dict:
    """List recent snapshots. Does not call the Task API."""
    task_id = (task_id or "").strip()
    if not task_id:
        return {"error": "invalid_input", "message": "task_id is required"}
    size = limit if isinstance(limit, int) and limit > 0 else 10
    snapshots = [
        {
            "task_id": str(row["task_id"]),
            "status": str(row["status"]),
            "collected_at": _clock(str(row["collected_at"])),
        }
        for row in _rows(task_id)[:size]
    ]
    return {"task_id": task_id, "snapshots": snapshots}


if __name__ == "__main__":
    mcp.run(transport="stdio")
