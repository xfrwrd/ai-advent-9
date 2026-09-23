"""Day 18 MCP client — call_tool over stdio."""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = Path(__file__).resolve().parent / "mcp_server.py"


def server_params(db_path: Path) -> StdioServerParameters:
    env = {**os.environ, "DAY18_DB_PATH": str(db_path)}
    return StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_SCRIPT)],
        env=env,
    )


@asynccontextmanager
async def mcp_session(db_path: Path) -> AsyncIterator[ClientSession]:
    async with stdio_client(server_params(db_path)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def call_tool(
    db_path: Path,
    name: str,
    arguments: dict[str, Any] | None = None,
) -> Any:
    async with mcp_session(db_path) as session:
        return await session.call_tool(name, arguments or {})


def tool_text_payload(call_result: Any) -> str:
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
