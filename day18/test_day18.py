"""Day 18: scheduler, SQLite persistence, MCP summary, Streamlit does not aggregate."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import time
import unittest
from datetime import datetime
from pathlib import Path

from agent import Agent, format_summary
from mcp_client import call_tool, tool_text_payload
from scheduler import Scheduler, get_scheduler, reset_scheduler_for_tests
from store import Store


class Day18StoreTests(unittest.TestCase):
    def test_rows_survive_reopen_and_summary_counts_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snapshots.db"
            store = Store(path)
            store.insert_snapshot(
                "TASK-123", "done", datetime(2026, 9, 23, 16, 41, 50)
            )
            store.insert_snapshot(
                "TASK-123", "in_progress", datetime(2026, 9, 23, 16, 42, 0)
            )
            store.insert_snapshot(
                "TASK-123", "in_progress", datetime(2026, 9, 23, 16, 42, 10)
            )

            reopened = Store(path)
            rows = reopened.recent(20)
            self.assertEqual(
                [(row.task_id, row.status, row.collected_at[-8:]) for row in rows],
                [
                    ("TASK-123", "in_progress", "16:42:10"),
                    ("TASK-123", "in_progress", "16:42:00"),
                    ("TASK-123", "done", "16:41:50"),
                ],
            )
            summary = reopened.summary()
            self.assertEqual(summary["total_snapshots"], 3)
            self.assertEqual(summary["counts"]["IN_PROGRESS"], 2)
            self.assertEqual(summary["counts"]["DONE"], 1)
            self.assertEqual(summary["last_collection"], "16:42:10")
            self.assertEqual(
                format_summary(summary),
                "\n".join(
                    [
                        "Total snapshots: 3",
                        "IN_PROGRESS: 2",
                        "DONE: 1",
                        "Last collection: 16:42:10",
                    ]
                ),
            )


class Day18SchedulerTests(unittest.TestCase):
    def test_collects_on_interval_without_a_second_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "snapshots.db")
            scheduler = Scheduler(store, interval_seconds=0.25)
            scheduler.start()
            try:
                first_thread = scheduler._thread
                count_after_start = store.count()
                scheduler.start()
                self.assertIs(scheduler._thread, first_thread)
                self.assertEqual(store.count(), count_after_start)
                self.assertTrue(scheduler.view().running)

                deadline = time.time() + 3
                while time.time() < deadline and store.count() < 3:
                    time.sleep(0.05)
                self.assertGreaterEqual(store.count(), 3)
                rows = store.recent(10)
                self.assertEqual(rows[0].task_id, "TASK-123")
                self.assertEqual(rows[0].status, "in_progress")
                stamps = [row.collected_at for row in rows]
                self.assertEqual(stamps, sorted(stamps, reverse=True))
                self.assertIsNotNone(scheduler.view().next_run_at)
            finally:
                scheduler.stop()

    def test_get_scheduler_returns_the_same_instance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["DAY18_DB_PATH"] = str(Path(tmp) / "snapshots.db")
            os.environ["DAY18_INTERVAL_SECONDS"] = "30"
            reset_scheduler_for_tests()
            try:
                first = get_scheduler()
                second = get_scheduler()
                self.assertIs(first, second)
                self.assertEqual(first.store.count(), 1)

                errors: list[str] = []
                barrier = threading.Barrier(4)

                def _call() -> None:
                    barrier.wait()
                    got = get_scheduler()
                    if got is not first:
                        errors.append("different instance")

                threads = [threading.Thread(target=_call) for _ in range(4)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=2)
                self.assertEqual(errors, [])
                self.assertEqual(first.store.count(), 1)
            finally:
                reset_scheduler_for_tests()
                os.environ.pop("DAY18_DB_PATH", None)
                os.environ.pop("DAY18_INTERVAL_SECONDS", None)


class Day18McpTests(unittest.TestCase):
    def test_agent_summary_comes_from_mcp_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snapshots.db"
            store = Store(path)
            store.insert_snapshot(
                "TASK-123", "in_progress", datetime(2026, 9, 23, 16, 42, 0)
            )
            store.insert_snapshot(
                "TASK-123", "in_progress", datetime(2026, 9, 23, 16, 42, 10)
            )
            store.insert_snapshot(
                "TASK-123", "done", datetime(2026, 9, 23, 16, 41, 50)
            )

            raw = asyncio.run(call_tool(path, "get_task_summary", {}))
            payload = json.loads(tool_text_payload(raw))
            self.assertEqual(payload["total_snapshots"], 3)
            self.assertEqual(payload["counts"]["IN_PROGRESS"], 2)
            self.assertEqual(payload["counts"]["DONE"], 1)

            result = asyncio.run(Agent(path).fetch_summary())
            self.assertIsNone(result.error)
            self.assertEqual(result.payload, payload)
            self.assertEqual(result.output, format_summary(payload))
            self.assertIn("Total snapshots: 3", result.output)
            self.assertIn("IN_PROGRESS: 2", result.output)
            self.assertIn("DONE: 1", result.output)

    def test_streamlit_page_does_not_aggregate_sqlite(self) -> None:
        source = (Path(__file__).parent / "app.py").read_text()
        self.assertIn("Получить сводку через MCP", source)
        self.assertIn("fetch_summary", source)
        self.assertNotIn(".summary(", source)
        self.assertNotIn("st.button", source.split("MCP Summary")[0])


if __name__ == "__main__":
    unittest.main()
