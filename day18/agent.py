"""Day 18 agent: asks MCP for the SQLite summary and returns that result."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp_client import call_tool, tool_text_payload


@dataclass
class AgentResult:
    payload: dict[str, Any] | None
    output: str
    error: str | None = None


def format_summary(payload: dict[str, Any]) -> str:
    counts = payload.get("counts") or {}
    lines = [f"Total snapshots: {payload.get('total_snapshots', 0)}"]
    seen: set[str] = set()
    for key in ("IN_PROGRESS", "DONE"):
        lines.append(f"{key}: {counts.get(key, 0)}")
        seen.add(key)
    for key in sorted(counts):
        if key not in seen:
            lines.append(f"{key}: {counts[key]}")
    last = payload.get("last_collection") or "—"
    lines.append(f"Last collection: {last}")
    return "\n".join(lines)


class Agent:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    async def fetch_summary(self) -> AgentResult:
        mcp_raw = await call_tool(self.db_path, "get_task_summary", {})
        payload = _parse_payload(mcp_raw)
        if not isinstance(payload, dict):
            return AgentResult(
                payload=None,
                output=f"Agent received unexpected MCP payload: {payload!r}",
                error="unexpected_payload",
            )
        if payload.get("error"):
            message = str(payload.get("message") or payload["error"])
            return AgentResult(payload=payload, output=message, error=message)
        if "total_snapshots" not in payload:
            return AgentResult(
                payload=payload,
                output=f"Agent received unexpected MCP payload: {payload!r}",
                error="unexpected_payload",
            )
        return AgentResult(
            payload=payload,
            output=format_summary(payload),
            error=None,
        )


def _parse_payload(mcp_raw: Any) -> Any:
    text = tool_text_payload(mcp_raw).strip()
    if not text:
        structured = getattr(mcp_raw, "structuredContent", None)
        if structured is not None:
            return structured
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        structured = getattr(mcp_raw, "structuredContent", None)
        if structured is not None:
            return structured
        return {"error": "parse_error", "message": text}
