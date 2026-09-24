"""Day 19 MCP server: get_tasks → summarize_tasks → save_summary."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import requests
from mcp.server.mcpserver import MCPServer

API_BASE = os.environ.get("DAY19_API_BASE", "http://127.0.0.1:8765").rstrip("/")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def summary_path() -> Path:
    override = os.environ.get("DAY19_SUMMARY_PATH")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / "data" / "day19_summary.json"


mcp = MCPServer(
    name="day19-pipeline",
    instructions="MCP tools that fetch tasks, summarize them, and save the summary.",
)


def _forced_error(tool_name: str) -> dict | None:
    if os.environ.get("DAY19_FAIL_STEP") == tool_name:
        return {"error": "forced_failure", "message": f"{tool_name} failed"}
    return None


@mcp.tool(
    description=(
        "Fetch the current task list from the existing Task API. "
        "Reads task ids from GET /health, then GET /tasks/{id} for each."
    )
)
def get_tasks() -> dict:
    """Load tasks from the existing Task API. Does not summarize or save them."""
    forced = _forced_error("get_tasks")
    if forced is not None:
        return forced

    health_url = f"{API_BASE}/health"
    try:
        health = requests.get(health_url, timeout=5)
    except requests.RequestException as exc:
        return {
            "error": "api_unreachable",
            "message": f"Failed to reach Task API: {exc.__class__.__name__}",
            "url": health_url,
        }
    if not health.ok:
        return {
            "error": "api_error",
            "status_code": health.status_code,
            "message": health.text[:300],
        }
    try:
        task_ids = health.json().get("tasks")
    except ValueError:
        return {"error": "invalid_response", "message": "API returned non-JSON body"}
    if not isinstance(task_ids, list):
        return {"error": "invalid_response", "message": "API did not return task ids"}

    tasks: list[dict] = []
    for task_id in task_ids:
        url = f"{API_BASE}/tasks/{task_id}"
        try:
            response = requests.get(url, timeout=5)
        except requests.RequestException as exc:
            return {
                "error": "api_unreachable",
                "message": f"Failed to reach Task API: {exc.__class__.__name__}",
                "url": url,
            }
        if not response.ok:
            return {
                "error": "api_error",
                "status_code": response.status_code,
                "message": response.text[:300],
                "url": url,
            }
        try:
            task = response.json()
        except ValueError:
            return {"error": "invalid_response", "message": f"Non-JSON task {task_id}"}
        tasks.append(task)
    return {"tasks": tasks}


@mcp.tool(
    description=(
        "Aggregate a task list that was already fetched. "
        "Uses only the tasks argument and does not call the Task API."
    )
)
def summarize_tasks(tasks: list[dict]) -> dict:
    """Build a deterministic summary from the tasks passed in."""
    forced = _forced_error("summarize_tasks")
    if forced is not None:
        return forced
    if not isinstance(tasks, list):
        return {"error": "invalid_input", "message": "tasks must be a list"}

    counts: dict[str, int] = {}
    for item in tasks:
        if not isinstance(item, dict) or "id" not in item or "status" not in item:
            return {
                "error": "invalid_input",
                "message": "each task needs id and status",
            }
        status = str(item["status"]).strip().lower()
        counts[status] = counts.get(status, 0) + 1

    total = len(tasks)
    done = counts.get("done", 0)
    in_progress = counts.get("in_progress", 0)
    parts = [f"{done} done", f"{in_progress} in progress"]
    for status in sorted(counts):
        if status not in {"done", "in_progress"}:
            parts.append(f"{counts[status]} {status}")
    return {
        "total": total,
        "done": done,
        "in_progress": in_progress,
        "by_status": counts,
        "summary": f"{total} tasks: " + ", ".join(parts),
    }


@mcp.tool(
    description=(
        "Save a task summary produced by summarize_tasks to a JSON file. "
        "Writes the summary object it receives."
    )
)
def save_summary(summary: dict) -> dict:
    """Persist the given summary and return where it was written."""
    forced = _forced_error("save_summary")
    if forced is not None:
        return forced
    if not isinstance(summary, dict) or "summary" not in summary:
        return {"error": "invalid_input", "message": "summary object is required"}

    path = summary_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "saved": True,
        "path": str(path),
        "timestamp": datetime.now().replace(microsecond=0).isoformat(timespec="seconds"),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
