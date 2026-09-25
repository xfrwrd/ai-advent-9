"""MCP client that opens one of the Day 20 servers by name."""

from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SERVER_MODULES = {
    "task_server": "day20.task_server",
    "history_server": "day20.history_server",
}


def server_params(server_name: str, api_base: str, db_path: Path) -> StdioServerParameters:
    module = SERVER_MODULES[server_name]
    env = {
        **os.environ,
        "DAY20_API_BASE": api_base.rstrip("/"),
        "DAY18_DB_PATH": str(db_path),
    }
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", module],
        env=env,
        cwd=str(PROJECT_ROOT),
    )


@asynccontextmanager
async def mcp_session(
    server_name: str,
    api_base: str,
    db_path: Path,
) -> AsyncIterator[ClientSession]:
    params = server_params(server_name, api_base, db_path)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def list_tools(server_name: str, api_base: str, db_path: Path) -> list:
    async with mcp_session(server_name, api_base, db_path) as session:
        result = await session.list_tools()
        return list(result.tools)


async def call_tool(
    server_name: str,
    api_base: str,
    db_path: Path,
    name: str,
    arguments: dict[str, Any] | None = None,
) -> Any:
    async with mcp_session(server_name, api_base, db_path) as session:
        return await session.call_tool(name, arguments or {})


def parse_tool_payload(call_result: Any) -> Any:
    content = getattr(call_result, "content", None) or []
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        parts.append(str(text) if text is not None else str(block))
    text = "\n".join(parts).strip()
    if not text:
        structured = getattr(call_result, "structuredContent", None)
        return structured if structured is not None else {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        structured = getattr(call_result, "structuredContent", None)
        if structured is not None:
            return structured
        return {"error": "parse_error", "message": text}
