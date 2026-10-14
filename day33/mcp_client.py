"""MCP client for the Day 33 support server."""

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


def server_params(data_path: Path | None = None) -> StdioServerParameters:
    env = dict(os.environ)
    if data_path is not None:
        env["DAY33_DATA"] = str(data_path.resolve())
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "day33.mcp_server"],
        env=env,
        cwd=str(PROJECT_ROOT),
    )


@asynccontextmanager
async def mcp_session(data_path: Path | None = None) -> AsyncIterator[ClientSession]:
    params = server_params(data_path)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def call_tool(
    name: str,
    arguments: dict[str, Any] | None = None,
    data_path: Path | None = None,
) -> Any:
    async with mcp_session(data_path) as session:
        return await session.call_tool(name, arguments or {})


def call_tool_sync(
    name: str,
    arguments: dict[str, Any] | None = None,
    data_path: Path | None = None,
) -> dict:
    raw = asyncio.run(call_tool(name, arguments, data_path))
    payload = parse_tool_payload(raw)
    if not isinstance(payload, dict):
        return {"result": payload}
    return payload


def call_many_sync(
    calls: list[tuple[str, dict[str, Any]]],
    data_path: Path | None = None,
) -> list[dict]:
    async def _run() -> list[dict]:
        async with mcp_session(data_path) as session:
            found = []
            for name, arguments in calls:
                raw = await session.call_tool(name, arguments)
                payload = parse_tool_payload(raw)
                found.append(payload if isinstance(payload, dict) else {"result": payload})
            return found

    return asyncio.run(_run())


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
