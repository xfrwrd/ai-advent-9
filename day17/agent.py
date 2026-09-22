"""Day 17: Agent that calls MCP get_task and uses the result (no LLM)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from mcp_client import call_tool, list_tools, tool_text_payload


@dataclass
class TaskView:
    """Application-facing task summary built from MCP tool result."""

    id: str
    title: str
    status: str
    assignee: str
    raw: dict[str, Any]

    def format_report(self) -> str:
        return (
            f"Task {self.id}: {self.title}\n"
            f"Status: {self.status}\n"
            f"Assignee: {self.assignee}"
        )


@dataclass
class AgentResult:
    tools: list
    mcp_raw: Any
    task: TaskView | None
    output: str
    error: str | None = None


class Agent:
    """
    Day 17 agent: uses MCP client to call get_task, then formats the result.

    No LLM — the point is Agent → MCP → API → result used by the app.
    """

    def __init__(self, api_base: str) -> None:
        self.api_base = api_base.rstrip("/")
        self.last_mcp_result: Any = None
        self.last_task: TaskView | None = None

    async def list_mcp_tools(self) -> list:
        return await list_tools(self.api_base)

    async def fetch_task(self, task_id: str) -> AgentResult:
        tools = await self.list_mcp_tools()
        mcp_raw = await call_tool(
            self.api_base,
            "get_task",
            {"task_id": task_id},
        )
        self.last_mcp_result = mcp_raw

        payload = self._parse_payload(mcp_raw)
        if isinstance(payload, dict) and payload.get("error"):
            message = payload.get("message") or payload["error"]
            output = f"Agent could not load task {task_id}: {message}"
            self.last_task = None
            return AgentResult(
                tools=tools,
                mcp_raw=mcp_raw,
                task=None,
                output=output,
                error=str(message),
            )

        if not isinstance(payload, dict) or "id" not in payload:
            output = f"Agent received unexpected MCP payload: {payload!r}"
            return AgentResult(
                tools=tools,
                mcp_raw=mcp_raw,
                task=None,
                output=output,
                error="unexpected_payload",
            )

        task = TaskView(
            id=str(payload.get("id", "")),
            title=str(payload.get("title", "")),
            status=str(payload.get("status", "")),
            assignee=str(payload.get("assignee", "")),
            raw=payload,
        )
        self.last_task = task
        return AgentResult(
            tools=tools,
            mcp_raw=mcp_raw,
            task=task,
            output=task.format_report(),
            error=None,
        )

    def _parse_payload(self, mcp_raw: Any) -> Any:
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
