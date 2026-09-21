"""Day 16: Minimal MCP client — connect, initialize, list_tools, print."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = Path(__file__).resolve().parent / "server.py"


async def fetch_tools() -> list:
    """
    Connect to the Day 16 MCP server over stdio, initialize the session,
    and return the tools list from the server (not hardcoded).
    """
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_SCRIPT)],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return list(result.tools)


def format_tools_report(tools: list) -> str:
    lines = [
        "Connected to MCP server",
        "MCP session initialized",
        f"Available tools: {len(tools)}",
        "",
    ]
    for tool in tools:
        name = getattr(tool, "name", "?")
        description = getattr(tool, "description", None) or "(no description)"
        lines.append(f"* {name}")
        lines.append(f"  description: {description}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


async def main() -> None:
    tools = await fetch_tools()
    print(format_tools_report(tools))
    if not tools:
        raise SystemExit("ERROR: MCP server returned an empty tools list.")


if __name__ == "__main__":
    asyncio.run(main())
