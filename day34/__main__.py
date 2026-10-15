"""CLI. Writing needs the word "да" typed for a specific diff.

    python -m day34 "Найди все места использования LocalLLMClient"
"""

from __future__ import annotations

import argparse
import json
import sys

from day34.agent import run_agent
from day34.llm import AgentLLM
from day34.mcp_client import call_tool_sync
from day34.report import record_apply, save_run


def _show(step: dict) -> None:
    if step.get("kind") == "tool":
        result = step.get("result") or {}
        status = result.get("error") or "ok"
        print(f"[{step['step']}] {step['tool']} {json.dumps(step.get('args'), ensure_ascii=False)} → {status}")
    elif step.get("kind") == "final":
        print(f"[{step['step']}] финальный ответ")
    else:
        print(f"[{step['step']}] {step.get('kind')}: {step.get('error')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Файловый ассистент Day 34")
    parser.add_argument("goal")
    parser.add_argument("--no-apply", action="store_true", help="только показать diff")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)

    run = run_agent(args.goal, llm=AgentLLM(), on_step=_show)
    folder = save_run(run)
    print()
    print(run.answer)
    print()
    print(f"Прочитано: {', '.join(run.files_read) or 'нет'}")
    for item in run.proposals:
        print()
        print(f"Изменение {item['proposal_id']} · {item['path']}")
        print(item["diff"])
        if args.no_apply:
            continue
        answer = input("Применить этот diff? Введите «да»: ").strip().lower()
        if answer != "да":
            result = call_tool_sync("changes", "cancel_file_change", {"proposal_id": item["proposal_id"]})
            print(f"Отменено: {result}")
            record_apply(run, {"proposal_id": item["proposal_id"], **result})
            continue
        result = call_tool_sync(
            "changes",
            "apply_file_change",
            {"proposal_id": item["proposal_id"], "confirm_token": item["diff_sha256"]},
        )
        print(f"Результат: {result}")
        record_apply(run, {"proposal_id": item["proposal_id"], **result})
    print(f"\nОтчёт: {folder / 'report.md'}")
    return 0 if run.stop_reason == "final" else 1


if __name__ == "__main__":
    raise SystemExit(main())
