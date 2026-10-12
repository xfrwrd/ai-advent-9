"""MCP client for the Day 31 Git server. Same stdio session shape as Day 19 and Day 20."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def server_params(repo: Path | None = None) -> StdioServerParameters:
    env = dict(os.environ)
    env["DAY31_REPO"] = str((repo or PROJECT_ROOT).resolve())
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "day31.mcp_server"],
        env=env,
        cwd=str(PROJECT_ROOT),
    )


@asynccontextmanager
async def mcp_session(repo: Path | None = None) -> AsyncIterator[ClientSession]:
    params = server_params(repo)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def call_tool(
    name: str,
    arguments: dict[str, Any] | None = None,
    repo: Path | None = None,
) -> Any:
    async with mcp_session(repo) as session:
        return await session.call_tool(name, arguments or {})


def call_tool_sync(
    name: str,
    arguments: dict[str, Any] | None = None,
    repo: Path | None = None,
) -> dict:
    raw = asyncio.run(call_tool(name, arguments, repo))
    payload = parse_tool_payload(raw)
    if not isinstance(payload, dict):
        return {"result": payload}
    return payload


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
