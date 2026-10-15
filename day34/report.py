"""Run reports: report.json and report.md under artifacts/runs/<run_id>/."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from day34.agent import Run
from day34.policy import artifacts_dir, redact


def save_run(run: Run, artifacts: Path | None = None) -> Path:
    folder = (artifacts or artifacts_dir()) / "runs" / run.run_id
    folder.mkdir(parents=True, exist_ok=True)
    payload = asdict(run)
    payload["tools_used"] = run.tools_used()
    for item in payload["proposals"]:
        item.pop("diff_sha256", None)
    (folder / "report.json").write_text(redact(json.dumps(payload, ensure_ascii=False, indent=2)), encoding="utf-8")
    (folder / "report.md").write_text(redact(render_markdown(run)), encoding="utf-8")
    return folder


def record_apply(run: Run, result: dict, artifacts: Path | None = None) -> Path:
    run.applied.append(result)
    return save_run(run, artifacts)


def render_markdown(run: Run) -> str:
    lines = [
        f"# Day 34 run {run.run_id}",
        "",
        f"Задача: {run.goal}",
        f"Остановка: {run.stop_reason}, {run.seconds} с, шагов {len(run.steps)}",
        "",
        "## Шаги",
    ]
    for step in run.steps:
        if step.get("kind") == "tool":
            result = step.get("result") or {}
            status = result.get("error") or "ok"
            args = json.dumps(step.get("args") or {}, ensure_ascii=False)
            lines.append(f"- {step['step']}. `{step['tool']}` {args} → {status}")
        elif step.get("kind") == "final":
            lines.append(f"- {step['step']}. финальный ответ")
        else:
            lines.append(f"- {step['step']}. {step.get('kind')}: {step.get('error')}")
    lines.extend(["", "## Прочитанные файлы"])
    lines.extend([f"- `{path}`" for path in run.files_read] or ["- нет"])
    lines.extend(["", "## Ответ", "", run.answer, "", "## Изменения"])
    if run.proposals:
        for item in run.proposals:
            lines.extend([f"### `{item['path']}` · {item['proposal_id']}", "", "```diff", item["diff"].rstrip(), "```", ""])
    else:
        lines.append("- не подготовлены")
    lines.extend(["", "## Применение"])
    if run.applied:
        lines.extend(f"- {json.dumps(item, ensure_ascii=False)}" for item in run.applied)
    else:
        lines.append("- не применялось")
    if run.errors:
        lines.extend(["", "## Ошибки"])
        lines.extend(f"- {error}" for error in run.errors)
    return "\n".join(lines) + "\n"
