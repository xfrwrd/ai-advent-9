"""Run the Day 20 long flow once and print the real trace."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from day20.agent import Agent

REQUEST = "Получи текущий статус TASK-123 и подготовь сводку по истории"


async def main() -> None:
    result = await Agent().run(REQUEST)
    print()
    print(result.answer)


if __name__ == "__main__":
    asyncio.run(main())
