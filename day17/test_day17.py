"""Day 17 tests: MCP get_task + Agent uses result (no LLM)."""

from __future__ import annotations

import asyncio
import json
import socket
import unittest

from agent import Agent
from mock_api import start_in_background, wait_until_ready
from mcp_client import call_tool, list_tools, tool_text_payload


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Day17McpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        port = free_port()
        cls._thread, cls.api_base = start_in_background(port=port)
        wait_until_ready(cls.api_base)

    def test_list_tools_includes_get_task_with_task_id_schema(self) -> None:
        tools = asyncio.run(list_tools(self.api_base))
        names = {t.name for t in tools}
        self.assertIn("get_task", names)
        get_task = next(t for t in tools if t.name == "get_task")
        self.assertTrue(get_task.description)
        schema = getattr(get_task, "input_schema", None) or {}
        # MCP JSON Schema: properties.task_id required
        props = schema.get("properties") or {}
        self.assertIn("task_id", props)
        required = schema.get("required") or []
        self.assertIn("task_id", required)

    def test_call_tool_returns_task_123(self) -> None:
        raw = asyncio.run(
            call_tool(self.api_base, "get_task", {"task_id": "TASK-123"})
        )
        payload = json.loads(tool_text_payload(raw))
        self.assertEqual(payload["id"], "TASK-123")
        self.assertEqual(payload["title"], "Implement MCP integration")
        self.assertEqual(payload["status"], "in_progress")
        self.assertEqual(payload["assignee"], "Xenia")

    def test_unknown_task_id(self) -> None:
        raw = asyncio.run(
            call_tool(self.api_base, "get_task", {"task_id": "TASK-MISSING"})
        )
        payload = json.loads(tool_text_payload(raw))
        self.assertEqual(payload.get("error"), "not_found")

    def test_agent_uses_mcp_result(self) -> None:
        agent = Agent(api_base=self.api_base)
        result = asyncio.run(agent.fetch_task("TASK-123"))
        self.assertIsNone(result.error)
        self.assertIsNotNone(result.task)
        self.assertEqual(result.task.id, "TASK-123")
        self.assertIn("Task TASK-123: Implement MCP integration", result.output)
        self.assertIn("Status: in_progress", result.output)
        self.assertIn("Assignee: Xenia", result.output)
        self.assertIsNotNone(agent.last_mcp_result)
        self.assertIsNotNone(agent.last_task)

    def test_agent_handles_missing_task(self) -> None:
        agent = Agent(api_base=self.api_base)
        result = asyncio.run(agent.fetch_task("NOPE"))
        self.assertIsNotNone(result.error)
        self.assertIn("could not load task", result.output.lower())


if __name__ == "__main__":
    unittest.main()
