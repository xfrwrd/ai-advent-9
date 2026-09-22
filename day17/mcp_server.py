"""Day 17: MCP server — get_task tool calls the mock Task API over HTTP."""

from __future__ import annotations

import os

import requests
from mcp.server.mcpserver import MCPServer

API_BASE = os.environ.get("DAY17_API_BASE", "http://127.0.0.1:8765").rstrip("/")

mcp = MCPServer(
    name="day17-task-api",
    instructions="MCP tools that fetch tasks from the Day 17 mock Task API.",
)


@mcp.tool(
    description=(
        "Get task information from the Task API by task ID. "
        "Calls GET /tasks/{task_id} on the mock API."
    )
)
def get_task(task_id: str) -> dict:
    """
    Fetch a task by id from the mock Task API.

    Args:
        task_id: Task identifier, e.g. TASK-123.
    """
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

    if response.status_code == 404:
        try:
            payload = response.json()
        except ValueError:
            payload = {"error": "not_found", "message": f"Task {task_id} not found"}
        return payload

    if not response.ok:
        return {
            "error": "api_error",
            "status_code": response.status_code,
            "message": response.text[:300],
        }

    try:
        return response.json()
    except ValueError:
        return {
            "error": "invalid_response",
            "message": "API returned non-JSON body",
            "raw": response.text[:300],
        }


if __name__ == "__main__":
    mcp.run(transport="stdio")
