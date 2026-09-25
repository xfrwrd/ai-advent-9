"""Discover tools from each MCP server and remember which server owns each tool."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from day20.mcp_client import SERVER_MODULES, list_tools


@dataclass(frozen=True)
class ToolInfo:
    name: str
    server: str
    description: str
    input_schema: dict


class ToolRegistry:
    def __init__(self, api_base: str, db_path: Path) -> None:
        self.api_base = api_base
        self.db_path = Path(db_path)
        self.tools: dict[str, ToolInfo] = {}

    async def discover(self) -> dict[str, ToolInfo]:
        found: dict[str, ToolInfo] = {}
        for server_name in SERVER_MODULES:
            listed = await list_tools(server_name, self.api_base, self.db_path)
            for tool in listed:
                schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}
                found[tool.name] = ToolInfo(
                    name=tool.name,
                    server=server_name,
                    description=tool.description or "",
                    input_schema=schema if isinstance(schema, dict) else {},
                )
        self.tools = found
        return found

    def server_for(self, tool_name: str) -> str | None:
        info = self.tools.get(tool_name)
        if info is None:
            return None
        return info.server
