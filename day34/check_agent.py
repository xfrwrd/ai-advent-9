"""Live run with qwen3:8b on this repository. Nothing is applied.

    .venv/bin/python -m day34.check_agent usage
    .venv/bin/python -m day34.check_agent docs
"""

from __future__ import annotations

import json
import sys

from day34.agent import run_agent
from day34.llm import AgentLLM
from day34.report import save_run

GOALS = {
    "usage": "Найди все места использования LocalLLMClient и объясни, какие модули от него зависят.",
    "docs": "Проверь документацию Day 31 (day31/README.md) и обнови её по фактическому коду day31.",
    "demo": "Проверь документацию day34/demo/README.md и обнови её по фактическому коду day34/demo.",
    "skip": "Проверь, какие файлы и каталоги по day31/README.md не индексируются, сверь с кодом day31 и обнови README.",
}


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(line_buffering=True)
    names = (argv if argv is not None else sys.argv[1:]) or list(GOALS)
    for name in names:
        print(f"=== {name} ===")
        run = run_agent(GOALS[name], llm=AgentLLM(), on_step=_show)
        folder = save_run(run)
        print(run.answer)
        print(f"stop {run.stop_reason}, {run.seconds} s, files {run.files_read}")
        for item in run.proposals:
            print(item["diff"])
        print(f"report {folder / 'report.md'}")
        print()
    return 0


def _show(step: dict) -> None:
    if step.get("kind") == "tool":
        status = (step.get("result") or {}).get("error") or "ok"
        print(f"[{step['step']}] {step['tool']} {json.dumps(step.get('args'), ensure_ascii=False)[:160]} → {status}")
    else:
        print(f"[{step['step']}] {step.get('kind')}: {step.get('error', '')}")


if __name__ == "__main__":
    raise SystemExit(main())
