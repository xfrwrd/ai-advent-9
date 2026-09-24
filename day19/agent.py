"""Day 19 agent. One call runs the whole MCP pipeline."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

from day19.pipeline import Pipeline, PipelineResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from day17.mock_api import start_in_background, wait_until_ready


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Agent:
    def __init__(self, summary_path: Path | None = None, fail_step: str | None = None) -> None:
        self.summary_path = Path(summary_path) if summary_path is not None else (
            Path(__file__).resolve().parent / "data" / "day19_summary.json"
        )
        self.fail_step = fail_step

    async def run_tasks_pipeline(self, api_base: str | None = None) -> PipelineResult:
        if api_base is None:
            port = _free_port()
            _, api_base = start_in_background(port=port)
            wait_until_ready(api_base)
        pipeline = Pipeline(api_base, self.summary_path, fail_step=self.fail_step)
        return await pipeline.run()
