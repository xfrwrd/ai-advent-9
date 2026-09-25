"""MCP server task_server: current task data from the existing Task API."""

from __future__ import annotations

import os

import requests
from mcp.server.mcpserver import MCPServer

API_BASE = os.environ.get("DAY20_API_BASE", "http://127.0.0.1:8765").rstrip("/")

mcp = MCPServer(
    name="task_server",
    instructions="Current task data. Tool get_task calls the Task API.",
)


@mcp.tool(
    description=(
        "Get the current task from the Task API by task ID. "
        "Use this for the live status, title, and assignee. "
        "Does not read history."
    )
)
def get_task(task_id: str) -> dict:
    """Fetch one task with GET /tasks/{task_id}."""
    task_id = (task_id or "").strip()
    if not task_id:
        return {"error": "invalid_input", "message": "task_id is required"}
    url = f"{API_BASE}/tasks/{task_id}"
    try:
        response = requests.get(url, timeout=5)
    except requests.RequestException as exc:
        return {
            "error": "api_unreachable",
            "message": f"Failed to reach Task API: {exc.__class__.__name__}",
            "url": url,
        }
    try:
        payload = response.json()
    except ValueError:
        payload = {"error": "invalid_response", "message": response.text[:300]}
    if response.status_code == 404:
        return payload if isinstance(payload, dict) else {"error": "not_found"}
    if not response.ok:
        return {
            "error": "api_error",
            "status_code": response.status_code,
            "message": response.text[:300],
        }
    return payload if isinstance(payload, dict) else {"error": "invalid_response"}


if __name__ == "__main__":
    mcp.run(transport="stdio")
