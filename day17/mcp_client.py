"""Day 17: MCP client helpers (stdio) — list_tools / call_tool."""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = Path(__file__).resolve().parent / "mcp_server.py"


def server_params(api_base: str) -> StdioServerParameters:
    env = {**os.environ, "DAY17_API_BASE": api_base.rstrip("/")}
    return StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_SCRIPT)],
        env=env,
    )


@asynccontextmanager
async def mcp_session(api_base: str) -> AsyncIterator[ClientSession]:
    """Connect to Day 17 MCP server, initialize session, yield ClientSession."""
    async with stdio_client(server_params(api_base)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def list_tools(api_base: str) -> list:
    async with mcp_session(api_base) as session:
        result = await session.list_tools()
        return list(result.tools)


async def call_tool(
    api_base: str,
    name: str,
    arguments: dict[str, Any] | None = None,
) -> Any:
    async with mcp_session(api_base) as session:
        return await session.call_tool(name, arguments or {})


def tool_text_payload(call_result: Any) -> str:
    """Extract text content from CallToolResult (MCP SDK)."""
    content = getattr(call_result, "content", None) or []
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(str(text))
        else:
            parts.append(str(block))
    if parts:
        return "\n".join(parts)
    structured = getattr(call_result, "structuredContent", None)
    if structured is not None:
        return str(structured)
    return str(call_result)
