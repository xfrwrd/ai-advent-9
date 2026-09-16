"""Live Day 13 demo: pause → clear STM → Продолжай + full stage path."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from agent import DEMO_GOAL, Agent
from task_state import TaskStage

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise SystemExit("DEEPSEEK_API_KEY missing")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        agent = Agent(api_key=api_key, base_dir=base)

        print("=== 1. Create task ===")
        task = agent.create_task(DEMO_GOAL)
        print(task.to_dict())

        print("\n=== 2. planning → execution + step ===")
        agent.transition_task(TaskStage.EXECUTION)
        agent.update_task_step(
            "определить API endpoints",
            "описать request/response models",
        )
        print(agent.get_task_state().to_dict())

        print("\n=== 3. Seed dialogue then Pause ===")
        agent.append_message("user", "Давай спланируем REST API")
        agent.append_message("assistant", "Ок, начнём с endpoints...")
        agent.pause_task()
        print(agent.get_task_state().to_dict())

        print("\n=== 4. Clear Short-Term ===")
        agent.clear_short_term()
        print(f"short_term={agent.memory.short_term}")
        print(f"task still: {agent.get_task_state().to_dict()}")

        print("\n=== 5. New session + Resume ===")
        agent2 = Agent(api_key=api_key, base_dir=base)
        restored = agent2.get_task_state()
        print(f"restored: {restored.to_dict()}")
        agent2.resume_task()

        print("\n=== 6. Ask «Продолжай» ===")
        result = agent2.ask("Продолжай")
        print("--- agent reply ---")
        print(result.content)
        print("--- end reply ---")
        print(f"tokens={result.token_stats.total_tokens} cost=${result.token_stats.estimated_cost:.6f}")

        # Check system had CURRENT TASK
        preview = agent2.preview_prompt()
        assert "CURRENT TASK" in preview
        assert "execution" in preview

        print("\n=== 7. Full path → validation → done ===")
        agent2.transition_task(TaskStage.VALIDATION)
        agent2.update_task_step("проверить контракты API", "отметить done если ок")
        print(agent2.get_task_state().to_dict())
        agent2.transition_task(TaskStage.DONE)
        print(agent2.get_task_state().to_dict())
        print("\nDEMO OK")


if __name__ == "__main__":
    main()
