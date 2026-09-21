"""Day 16: MCP client connects and lists tools (no LLM)."""

from __future__ import annotations

import asyncio
import unittest

from client import fetch_tools, format_tools_report


class McpListToolsTests(unittest.TestCase):
    def test_connect_and_list_tools_nonempty(self) -> None:
        tools = asyncio.run(fetch_tools())
        self.assertGreater(len(tools), 0)
        names = {t.name for t in tools}
        self.assertIn("add", names)
        self.assertIn("echo", names)
        self.assertIn("greet", names)

    def test_report_contains_names(self) -> None:
        tools = asyncio.run(fetch_tools())
        report = format_tools_report(tools)
        self.assertIn("Connected to MCP server", report)
        self.assertIn("Available tools:", report)
        self.assertIn("* add", report)
        self.assertIn("* echo", report)
        self.assertIn("* greet", report)


if __name__ == "__main__":
    unittest.main()
