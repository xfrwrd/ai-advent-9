"""Run the Day 19 MCP pipeline once and print the real result."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from day19.agent import Agent


async def main() -> None:
    result = await Agent().run_tasks_pipeline()
    print(result.report, end="")


if __name__ == "__main__":
    asyncio.run(main())
