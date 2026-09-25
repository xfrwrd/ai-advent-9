"""Send a selected tool to the MCP server recorded in the registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from day20.mcp_client import call_tool, parse_tool_payload
from day20.registry import ToolRegistry


class Router:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> tuple[str, Any]:
        server_name = self.registry.server_for(tool_name)
        if server_name is None:
            return "", {
                "error": "unknown_tool",
                "message": f"Tool {tool_name} is not registered",
            }
        raw = await call_tool(
            server_name,
            self.registry.api_base,
            Path(self.registry.db_path),
            tool_name,
            arguments,
        )
        return server_name, parse_tool_payload(raw)
