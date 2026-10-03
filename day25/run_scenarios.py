"""Run the two long chats and write a report for each."""

from __future__ import annotations

import json
from pathlib import Path

from day25.agent import ChatAgent, TurnRecord, load_chat
from day25.scenarios import (
    SCENARIO_A,
    SCENARIO_A_CONSTRAINT,
    SCENARIO_A_GOAL,
    SCENARIO_B,
    SCENARIO_B_CONSTRAINT,
    SCENARIO_B_GOAL,
)
from day25.store import ConversationStore

REPORT_DIR = Path(__file__).resolve().parent / "data"


def run_scenario(agent: ChatAgent, name: str, messages: tuple[str, ...], goal_marker: str, constraint_marker: str) -> dict[str, object]:
    conversation = agent.store.new()
    turns: list[TurnRecord] = []
    for message in messages:
        turns.append(agent.send(conversation, message))
        print(f"  {name} {len(turns)}/{len(messages)}", flush=True)
    final_state = conversation.task_state
    constraints_text = " ".join(final_state.constraints).lower()
    grounded = [turn for turn in turns if not turn.insufficient_context]
    sources_present = sum(1 for turn in grounded if turn.sources_present)
    return {
        "name": name,
        "conversation_id": conversation.conversation_id,
        "turns": [turn.as_dict() for turn in turns],
        "summary": {
            "turns": len(turns),
            "goal_retained": goal_marker.lower() in final_state.goal.lower(),
            "constraints_retained": constraint_marker.lower() in constraints_text,
            "sources_present": f"{sources_present}/{len(grounded)}",
            "insufficient_context_turns": sum(1 for turn in turns if turn.insufficient_context),
            "final_task_state": final_state.as_dict(),
        },
    }


def save_report(report: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    store = ConversationStore()
    agent = load_chat(store)
    first = run_scenario(agent, "A", SCENARIO_A, SCENARIO_A_GOAL, SCENARIO_A_CONSTRAINT)
    second = run_scenario(agent, "B", SCENARIO_B, SCENARIO_B_GOAL, SCENARIO_B_CONSTRAINT)
    save_report(first, REPORT_DIR / "scenario_a.json")
    save_report(second, REPORT_DIR / "scenario_b.json")
    for report in (first, second):
        summary = report["summary"]
        print(
            f"Scenario {report['name']}: turns {summary['turns']}, "
            f"goal {summary['goal_retained']}, constraints {summary['constraints_retained']}, "
            f"sources {summary['sources_present']}, insufficient {summary['insufficient_context_turns']}"
        )
        print(" final", summary["final_task_state"])


if __name__ == "__main__":
    main()
