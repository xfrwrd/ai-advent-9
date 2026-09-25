"""Day 20: two MCP servers, registry, routing, and a long flow."""

from __future__ import annotations

import asyncio
import socket
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from day17.mock_api import start_in_background, wait_until_ready
from day20.agent import Agent, select_tools
from day20.registry import ToolRegistry


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def make_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            status TEXT NOT NULL,
            collected_at TEXT NOT NULL
        )
        """
    )
    conn.executemany(
        "INSERT INTO snapshots (task_id, status, collected_at) VALUES (?, ?, ?)",
        [
            ("TASK-123", "in_progress", "2026-09-25T19:00:00"),
            ("TASK-123", "in_progress", "2026-09-25T19:00:10"),
            ("TASK-456", "todo", "2026-09-25T19:00:00"),
        ],
    )
    conn.commit()
    conn.close()


class Day20Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        port = free_port()
        cls._thread, cls.api_base = start_in_background(port=port)
        wait_until_ready(cls.api_base)
        cls._tmp = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls._tmp.name) / "snapshots.db"
        make_db(cls.db_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _agent(self) -> Agent:
        return Agent(api_base=self.api_base, db_path=self.db_path)

    def test_discovery_maps_tools_to_servers(self) -> None:
        registry = ToolRegistry(self.api_base, self.db_path)
        tools = asyncio.run(registry.discover())
        self.assertEqual(set(tools), {"get_task", "get_task_summary", "get_recent_snapshots"})
        self.assertEqual(tools["get_task"].server, "task_server")
        self.assertEqual(tools["get_task_summary"].server, "history_server")
        self.assertEqual(tools["get_recent_snapshots"].server, "history_server")
        self.assertTrue(tools["get_task"].description)
        self.assertIn("task_id", tools["get_task"].input_schema.get("properties") or {})
        self.assertIn("task_id", tools["get_task_summary"].input_schema.get("properties") or {})

    def test_current_status_uses_only_task_server(self) -> None:
        request = "Покажи текущий статус TASK-123"
        self.assertEqual(select_tools(request), ["get_task"])
        result = asyncio.run(self._agent().run(request))
        self.assertEqual(result.status, "SUCCESS")
        self.assertEqual([(call.server, call.tool) for call in result.calls], [("task_server", "get_task")])
        self.assertEqual(result.calls[0].result["id"], "TASK-123")
        self.assertEqual(result.calls[0].result["status"], "in_progress")
        self.assertIn("Текущий статус: in_progress", result.answer)
        self.assertNotIn("История", result.answer)

    def test_history_uses_only_history_server(self) -> None:
        request = "Покажи историю TASK-123"
        self.assertEqual(select_tools(request), ["get_task_summary"])
        result = asyncio.run(self._agent().run(request))
        self.assertEqual(
            [(call.server, call.tool) for call in result.calls],
            [("history_server", "get_task_summary")],
        )
        self.assertEqual(result.calls[0].result["total_snapshots"], 2)
        self.assertIn("2 snapshots", result.answer)
        self.assertNotIn("Текущий статус", result.answer)

    def test_long_flow_uses_both_servers_in_order(self) -> None:
        request = "Получи текущий статус TASK-123 и подготовь сводку по истории"
        result = asyncio.run(self._agent().run(request))
        self.assertEqual(result.status, "SUCCESS")
        self.assertEqual(
            [(call.server, call.tool) for call in result.calls],
            [
                ("task_server", "get_task"),
                ("history_server", "get_task_summary"),
            ],
        )
        self.assertEqual(result.calls[1].arguments["task_id"], result.calls[0].result["id"])
        self.assertIn("Implement MCP integration", result.answer)
        self.assertIn("Текущий статус: in_progress", result.answer)
        self.assertIn("2 snapshots", result.answer)
        self.assertIn("19:00:10", result.answer)
        joined = "\n".join(result.trace)
        self.assertLess(joined.index("Selected tool: get_task"), joined.index("Routed to: task_server"))
        self.assertLess(joined.index("Routed to: task_server"), joined.index("Selected tool: get_task_summary"))
        self.assertLess(joined.index("Selected tool: get_task_summary"), joined.index("Routed to: history_server"))
        self.assertIn("Results combined", joined)

    def test_unknown_request_makes_no_calls(self) -> None:
        result = asyncio.run(self._agent().run("Расскажи анекдот"))
        self.assertEqual(result.status, "UNKNOWN")
        self.assertEqual(result.calls, [])
        self.assertIn("Не удалось выбрать инструмент", result.answer)


if __name__ == "__main__":
    unittest.main()
