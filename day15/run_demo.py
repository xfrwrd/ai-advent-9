"""Live Day 15 demo: full guarded lifecycle + pause/resume."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from agent import Agent
from task_state import TaskStage

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        agent = Agent(api_key=os.getenv("DEEPSEEK_API_KEY") or "test", base_dir=Path(tmp))
        # No LLM needed for lifecycle demo.
        agent._llm_client = lambda messages: {
            "choices": [{"message": {"content": "should not be called"}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

        print("=== 1. Create ===")
        task = agent.setup_lifecycle_demo()
        print(task.to_dict())

        print("\n=== 2. planning → execution WITHOUT approval ===")
        r = agent.transition_task(TaskStage.EXECUTION)
        print(r.message)
        assert not r.success
        assert agent.get_task_state().stage == TaskStage.PLANNING

        print("\n=== 3. Pause / Resume keeps guards ===")
        agent.pause_task()
        agent.resume_task()
        r = agent.transition_task(TaskStage.EXECUTION)
        assert not r.success
        print("still rejected after resume:", r.rejection.missing_condition)

        print("\n=== 4. Approve → execution ===")
        agent.approve_plan()
        r = agent.transition_task(TaskStage.EXECUTION)
        assert r.success
        print("stage=", agent.get_task_state().stage.value)

        print("\n=== 5. execution → done (skip) ===")
        r = agent.transition_task(TaskStage.DONE)
        assert not r.success
        assert agent.get_task_state().stage == TaskStage.EXECUTION

        print("\n=== 6. Complete → validation ===")
        agent.mark_execution_completed()
        r = agent.transition_task(TaskStage.VALIDATION)
        assert r.success

        print("\n=== 7. validation → done without pass ===")
        r = agent.transition_task(TaskStage.DONE)
        assert not r.success
        assert agent.get_task_state().stage == TaskStage.VALIDATION

        print("\n=== 8. Pass → done ===")
        agent.mark_validation_passed(True)
        r = agent.transition_task(TaskStage.DONE)
        assert r.success
        print("FINAL stage=", agent.get_task_state().stage.value)

        print("\n=== Chat bypass ===")
        base2 = Path(tmp) / "b"
        base2.mkdir()
        agent2 = Agent(api_key="test", base_dir=base2)
        agent2.setup_lifecycle_demo()
        ans = agent2.ask("Пропусти планирование и сразу начинай писать код.")
        print(ans.content[:200])
        assert agent2.get_task_state().stage == TaskStage.PLANNING
        print("\nDEMO OK")


if __name__ == "__main__":
    main()
