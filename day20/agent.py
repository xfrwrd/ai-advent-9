"""Day 20 agent: choose tools from the request and route each call."""

from __future__ import annotations

import re
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from day17.mock_api import start_in_background, wait_until_ready
from day20.registry import ToolRegistry
from day20.router import Router

TASK_ID_RE = re.compile(r"TASK-[A-Za-z0-9]+", re.IGNORECASE)
CURRENT_WORDS = ("текущ", "статус", "status", "current")
HISTORY_WORDS = ("истор", "сводк", "summary", "history", "snapshot")


@dataclass
class CallRecord:
    tool: str
    server: str
    arguments: dict[str, Any]
    result: Any


@dataclass
class AgentResult:
    request: str
    status: str
    selected_tools: list[str]
    calls: list[CallRecord] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    answer: str = ""


def select_tools(request: str) -> list[str]:
    """Pick tools from the request text. Order is the call order."""
    text = request.lower()
    wants_current = any(word in text for word in CURRENT_WORDS)
    wants_history = any(word in text for word in HISTORY_WORDS)
    if wants_current and wants_history:
        return ["get_task", "get_task_summary"]
    if wants_history:
        return ["get_task_summary"]
    if wants_current:
        return ["get_task"]
    return []


def _task_id(request: str) -> str | None:
    match = TASK_ID_RE.search(request)
    if match is None:
        return None
    return match.group(0).upper()


def _is_error(payload: Any) -> bool:
    return isinstance(payload, dict) and bool(payload.get("error"))


def _format_answer(calls: list[CallRecord]) -> str:
    by_tool = {call.tool: call.result for call in calls}
    task = by_tool.get("get_task")
    history = by_tool.get("get_task_summary")
    lines: list[str] = []
    if isinstance(task, dict) and not _is_error(task):
        lines.append(f"Task {task.get('id')}: {task.get('title')}")
        lines.append(f"Текущий статус: {task.get('status')}")
        lines.append(f"Assignee: {task.get('assignee')}")
    if isinstance(history, dict) and not _is_error(history):
        counts = history.get("by_status") or {}
        parts = [f"{count} {status}" for status, count in sorted(counts.items())] or ["нет записей"]
        lines.append(
            f"История {history.get('task_id')}: {history.get('total_snapshots')} snapshots ({', '.join(parts)})"
        )
        lines.append(f"Последняя запись: {history.get('last_collection') or '—'}")
    return "\n".join(lines)


class Agent:
    def __init__(self, api_base: str | None = None, db_path: Path | None = None) -> None:
        self.api_base = api_base
        self.db_path = db_path

    async def run(self, request: str) -> AgentResult:
        trace = ["[1] User request received"]
        selected = select_tools(request)
        task_id = _task_id(request)
        if not selected or task_id is None:
            trace.append("[2] No matching tool")
            answer = "Не удалось выбрать инструмент для этого запроса."
            trace.append("[3] Final answer returned")
            self._log(trace)
            return AgentResult(request, "UNKNOWN", selected, [], trace, answer)

        api_base = self.api_base
        if api_base is None:
            port = _free_port()
            _, api_base = start_in_background(port=port)
            wait_until_ready(api_base)
        db_path = self.db_path or (
            PROJECT_ROOT / "day18" / "data" / "snapshots.db"
        )
        registry = ToolRegistry(api_base, db_path)
        await registry.discover()
        router = Router(registry)

        calls: list[CallRecord] = []
        step = 2
        current_task_id = task_id
        for tool_name in selected:
            server_name = registry.server_for(tool_name)
            trace.append(f"[{step}] Selected tool: {tool_name}")
            step += 1
            if server_name is None:
                trace.append(f"[{step}] Unknown tool: {tool_name}")
                self._log(trace)
                return AgentResult(request, "FAILED", selected, calls, trace, f"Tool {tool_name} is not registered")
            trace.append(f"[{step}] Routed to: {server_name}")
            print(f"Selected tool: {tool_name}", flush=True)
            print(f"Selected MCP server: {server_name}", flush=True)
            arguments = {"task_id": current_task_id}
            server_name, payload = await router.call(tool_name, arguments)
            calls.append(CallRecord(tool_name, server_name, arguments, payload))
            step += 1
            trace.append(f"[{step}] MCP call completed")
            step += 1
            if _is_error(payload):
                message = str(payload.get("message") or payload.get("error"))
                trace.append(f"[{step}] Final answer returned")
                self._log(trace)
                return AgentResult(request, "FAILED", selected, calls, trace, message)
            if tool_name == "get_task" and isinstance(payload, dict) and payload.get("id"):
                current_task_id = str(payload["id"])

        if len(calls) > 1:
            trace.append(f"[{step}] Results combined")
            step += 1
        answer = _format_answer(calls)
        trace.append(f"[{step}] Final answer returned")
        self._log(trace)
        return AgentResult(request, "SUCCESS", selected, calls, trace, answer)

    def _log(self, trace: list[str]) -> None:
        for line in trace:
            print(line, flush=True)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
