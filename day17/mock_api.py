"""Day 17: Local mock Task API (no OAuth / external services)."""

from __future__ import annotations

import argparse
import threading
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

TASKS: dict[str, dict[str, Any]] = {
    "TASK-123": {
        "id": "TASK-123",
        "title": "Implement MCP integration",
        "status": "in_progress",
        "assignee": "Xenia",
    },
    "TASK-456": {
        "id": "TASK-456",
        "title": "Write Day 17 tests",
        "status": "todo",
        "assignee": "Alex",
    },
}


async def get_task(request: Request) -> JSONResponse:
    task_id = request.path_params["task_id"]
    task = TASKS.get(task_id)
    if task is None:
        return JSONResponse(
            {"error": "not_found", "message": f"Task {task_id} not found"},
            status_code=404,
        )
    return JSONResponse(task)


async def health(_: Request) -> JSONResponse:
    return JSONResponse({"ok": True, "tasks": list(TASKS.keys())})


app = Starlette(
    routes=[
        Route("/health", health),
        Route("/tasks/{task_id}", get_task),
    ]
)


def create_app() -> Starlette:
    return app


def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    uvicorn.run(app, host=host, port=port, log_level="warning")


def start_in_background(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> tuple[threading.Thread, str]:
    """Start mock API in a daemon thread. Returns (thread, base_url)."""
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    def _run() -> None:
        asyncio_run_server(server)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread, f"http://{host}:{port}"


def asyncio_run_server(server: uvicorn.Server) -> None:
    import asyncio

    asyncio.run(server.serve())


def wait_until_ready(base_url: str, timeout: float = 5.0) -> None:
    import time

    import requests

    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/health", timeout=0.5)
            if response.ok:
                return
        except Exception as exc:  # noqa: BLE001 — readiness probe
            last_error = exc
        time.sleep(0.05)
    raise RuntimeError(f"Mock API did not become ready: {last_error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 17 mock Task API")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    print(f"Mock API listening on http://{args.host}:{args.port}")
    run_server(args.host, args.port)
