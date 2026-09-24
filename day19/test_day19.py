"""Day 19: MCP pipeline get_tasks → summarize_tasks → save_summary."""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from day17.mock_api import start_in_background, wait_until_ready
from day19.agent import Agent
from day19.mcp_client import call_tool, list_tools, parse_tool_payload
from day19.pipeline import STEP_GET, STEP_SAVE, STEP_SUMMARIZE


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Day19PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        port = free_port()
        cls._thread, cls.api_base = start_in_background(port=port)
        wait_until_ready(cls.api_base)

    def _summary_path(self, tmp: str) -> Path:
        return Path(tmp) / "day19_summary.json"

    def test_three_tools_are_registered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = asyncio.run(list_tools(self.api_base, self._summary_path(tmp)))
        names = {tool.name for tool in tools}
        self.assertEqual(names, {"get_tasks", "summarize_tasks", "save_summary"})
        for tool in tools:
            self.assertTrue(tool.description)
            schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)
            self.assertIsInstance(schema, dict)

        summarize = next(tool for tool in tools if tool.name == "summarize_tasks")
        save = next(tool for tool in tools if tool.name == "save_summary")
        summarize_schema = getattr(summarize, "input_schema", None) or {}
        save_schema = getattr(save, "input_schema", None) or {}
        self.assertIn("tasks", summarize_schema.get("properties") or {})
        self.assertIn("summary", save_schema.get("properties") or {})

    def test_get_tasks_returns_api_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = asyncio.run(call_tool(self.api_base, self._summary_path(tmp), "get_tasks", {}))
        payload = parse_tool_payload(raw)
        self.assertIsInstance(payload.get("tasks"), list)
        tasks = payload["tasks"]
        ids = {item["id"] for item in tasks}
        self.assertIn("TASK-123", ids)
        self.assertIn("TASK-456", ids)
        task = next(item for item in tasks if item["id"] == "TASK-123")
        self.assertEqual(task["title"], "Implement MCP integration")
        self.assertEqual(task["status"], "in_progress")

    def test_summarize_uses_get_tasks_output_not_a_refetch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._summary_path(tmp)
            tasks = parse_tool_payload(
                asyncio.run(call_tool(self.api_base, path, "get_tasks", {}))
            )["tasks"]
            edited = list(tasks) + [
                {"id": "TASK-EXTRA", "title": "Extra", "status": "done"}
            ]
            summary = parse_tool_payload(
                asyncio.run(
                    call_tool(
                        self.api_base,
                        path,
                        "summarize_tasks",
                        {"tasks": edited},
                    )
                )
            )
        self.assertEqual(summary["total"], len(edited))
        self.assertEqual(summary["done"], 1)
        self.assertEqual(summary["in_progress"], 1)
        self.assertIn(f"{len(edited)} tasks:", summary["summary"])

    def test_save_summary_writes_the_summarize_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._summary_path(tmp)
            tasks = parse_tool_payload(
                asyncio.run(call_tool(self.api_base, path, "get_tasks", {}))
            )["tasks"]
            summary = parse_tool_payload(
                asyncio.run(
                    call_tool(self.api_base, path, "summarize_tasks", {"tasks": tasks})
                )
            )
            saved = parse_tool_payload(
                asyncio.run(
                    call_tool(self.api_base, path, "save_summary", {"summary": summary})
                )
            )
            on_disk = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(saved["saved"])
        self.assertEqual(saved["path"], str(path))
        self.assertTrue(saved["timestamp"])
        self.assertEqual(on_disk, summary)

    def test_run_tasks_pipeline_runs_all_three_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._summary_path(tmp)
            result = asyncio.run(
                Agent(summary_path=path).run_tasks_pipeline(api_base=self.api_base)
            )
            on_disk = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(result.status, "SUCCESS")
        self.assertEqual([step.name for step in result.steps], [STEP_GET, STEP_SUMMARIZE, STEP_SAVE])
        self.assertEqual(result.steps[1].tool_input["tasks"], result.tasks)
        self.assertEqual(result.steps[2].tool_input["summary"], result.summary)
        self.assertEqual(on_disk, result.summary)
        self.assertIn("PIPELINE SUCCESS", result.report)
        self.assertIn("STEP 1: get_tasks", result.report)
        self.assertIn("STEP 2: summarize_tasks", result.report)
        self.assertIn("STEP 3: save_summary", result.report)

    def test_first_step_failure_stops_the_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._summary_path(tmp)
            result = asyncio.run(
                Agent(summary_path=path).run_tasks_pipeline(api_base="http://127.0.0.1:9")
            )
        self.assertEqual(result.status, "FAILED")
        self.assertEqual(result.failed_step, STEP_GET)
        self.assertEqual([step.name for step in result.steps], [STEP_GET])
        self.assertFalse(path.exists())

    def test_second_step_failure_does_not_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._summary_path(tmp)
            result = asyncio.run(
                Agent(summary_path=path, fail_step=STEP_SUMMARIZE).run_tasks_pipeline(
                    api_base=self.api_base
                )
            )
        self.assertEqual(result.status, "FAILED")
        self.assertEqual(result.failed_step, STEP_SUMMARIZE)
        self.assertEqual([step.name for step in result.steps], [STEP_GET, STEP_SUMMARIZE])
        self.assertEqual(result.steps[0].status, "SUCCESS")
        self.assertEqual(result.steps[1].status, "FAILED")
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
