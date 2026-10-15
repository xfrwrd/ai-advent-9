"""stdio MCP client for the Day 34 servers. Same session shape as Day 31 and Day 33."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from day31.mcp_client import parse_tool_payload

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVERS = {"files": "day34.mcp_files", "changes": "day34.mcp_changes"}


def server_params(server: str, repo: Path | None = None, artifacts: Path | None = None) -> StdioServerParameters:
    env = dict(os.environ)
    if repo is not None:
        env["DAY34_REPO"] = str(repo.resolve())
    if artifacts is not None:
        env["DAY34_ARTIFACTS"] = str(artifacts.resolve())
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", SERVERS[server]],
        env=env,
        cwd=str(PROJECT_ROOT),
    )


async def call_tool(
    server: str,
    name: str,
    arguments: dict[str, Any] | None = None,
    repo: Path | None = None,
    artifacts: Path | None = None,
) -> Any:
    async with stdio_client(server_params(server, repo, artifacts)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(name, arguments or {})


async def list_tool_names(server: str, repo: Path | None = None, artifacts: Path | None = None) -> list[str]:
    async with stdio_client(server_params(server, repo, artifacts)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            return [tool.name for tool in listed.tools]


def call_tool_sync(
    server: str,
    name: str,
    arguments: dict[str, Any] | None = None,
    repo: Path | None = None,
    artifacts: Path | None = None,
) -> dict:
    payload = parse_tool_payload(asyncio.run(call_tool(server, name, arguments, repo, artifacts)))
    return payload if isinstance(payload, dict) else {"result": payload}
