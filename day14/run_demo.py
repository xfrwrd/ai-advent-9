"""Live Day 14 demo: compatible + conflicts + clear STM."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from agent import Agent
from invariants import (
    BUSINESS_CONFLICT_PROMPT,
    COMPATIBLE_PROMPT,
    STACK_CONFLICT_PROMPT,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise SystemExit("DEEPSEEK_API_KEY missing")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        agent = Agent(api_key=api_key, base_dir=Path(tmp))
        agent.setup_demo_task()

        print("=== Active invariants ===")
        for inv in agent.get_invariants(active_only=True):
            print(f"- [{inv.category.value}] {inv.id}: {inv.rule}")

        print("\n=== Compatible ===")
        r1 = agent.ask(COMPATIBLE_PROMPT)
        print(f"rejected={r1.rejected}")
        print(r1.content[:500])
        print("---")

        print("\n=== Stack conflict (MongoDB) ===")
        r2 = agent.ask(STACK_CONFLICT_PROMPT)
        print(f"rejected={r2.rejected} inv={r2.conflict.invariant.id if r2.conflict else None}")
        print(r2.content)
        print("---")

        print("\n=== Business conflict ===")
        r3 = agent.ask(BUSINESS_CONFLICT_PROMPT)
        print(f"rejected={r3.rejected} inv={r3.conflict.invariant.id if r3.conflict else None}")
        print(r3.content[:400])
        print("---")

        print("\n=== Clear Short-Term, repeat MongoDB ===")
        agent.clear_short_term()
        r4 = agent.ask(STACK_CONFLICT_PROMPT)
        print(f"short_term_len={len(agent.memory.short_term)}")
        print(f"rejected={r4.rejected} inv={r4.conflict.invariant.id if r4.conflict else None}")
        assert r4.rejected
        assert any(i.id == "database" for i in agent.get_invariants(active_only=True))
        print("DEMO OK")


if __name__ == "__main__":
    main()
