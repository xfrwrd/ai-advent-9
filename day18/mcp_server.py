"""Day 18 MCP server — get_task_summary aggregates SQLite snapshots."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from store import Store, default_db_path

mcp = MCPServer(
    name="day18-task-summary",
    instructions="MCP tools that summarize task snapshots stored by the Day 18 scheduler.",
)


@mcp.tool(
    description=(
        "Aggregate stored task snapshots from SQLite. "
        "Returns total count, counts by status, and the time of the last collection."
    )
)
def get_task_summary() -> dict:
    """Summarize snapshots persisted by the background scheduler."""
    return Store(default_db_path()).summary()


if __name__ == "__main__":
    mcp.run(transport="stdio")
