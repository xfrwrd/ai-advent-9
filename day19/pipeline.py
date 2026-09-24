"""Automatic MCP pipeline: get_tasks → summarize_tasks → save_summary."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from day19.mcp_client import call_tool, parse_tool_payload

STEP_GET = "get_tasks"
STEP_SUMMARIZE = "summarize_tasks"
STEP_SAVE = "save_summary"


@dataclass
class StepResult:
    name: str
    status: str
    tool_input: Any
    output: Any


@dataclass
class PipelineResult:
    status: str
    failed_step: str | None
    steps: list[StepResult] = field(default_factory=list)
    report: str = ""

    @property
    def tasks(self) -> list:
        step = self._step(STEP_GET)
        if step is None or not isinstance(step.output, list):
            return []
        return step.output

    @property
    def summary(self) -> dict | None:
        step = self._step(STEP_SUMMARIZE)
        if step is None or not isinstance(step.output, dict) or step.output.get("error"):
            return None
        return step.output

    @property
    def saved(self) -> dict | None:
        step = self._step(STEP_SAVE)
        if step is None or not isinstance(step.output, dict) or step.output.get("error"):
            return None
        return step.output

    def _step(self, name: str) -> StepResult | None:
        for step in self.steps:
            if step.name == name:
                return step
        return None


def _is_error(payload: Any) -> bool:
    return isinstance(payload, dict) and bool(payload.get("error"))


def format_report(result: PipelineResult) -> str:
    lines = ["Starting MCP pipeline...", ""]
    labels = {
        STEP_GET: "STEP 1: get_tasks",
        STEP_SUMMARIZE: "STEP 2: summarize_tasks",
        STEP_SAVE: "STEP 3: save_summary",
    }
    for step in result.steps:
        lines.append(labels.get(step.name, step.name))
        lines.append(step.status)
        if step.name == STEP_GET and isinstance(step.output, list):
            lines.append(f"Received {len(step.output)} tasks")
        elif step.name == STEP_SUMMARIZE and isinstance(step.output, dict) and not _is_error(step.output):
            lines.append(f"Input tasks: {len(step.tool_input.get('tasks', []))}")
            lines.append(f"Total: {step.output.get('total')}")
            lines.append(f"Done: {step.output.get('done')}")
            lines.append(f"In progress: {step.output.get('in_progress')}")
            lines.append(str(step.output.get("summary", "")))
        elif step.name == STEP_SAVE and isinstance(step.output, dict) and not _is_error(step.output):
            lines.append(f"Saved to: {step.output.get('path')}")
        elif _is_error(step.output):
            message = step.output.get("message") or step.output.get("error")
            lines.append(f"Error: {message}")
        lines.append("")
    if result.status == "SUCCESS":
        lines.append("PIPELINE SUCCESS")
    else:
        lines.append(f"PIPELINE FAILED at {result.failed_step}")
    return "\n".join(lines).rstrip() + "\n"


class Pipeline:
    def __init__(
        self,
        api_base: str,
        summary_path: Path,
        fail_step: str | None = None,
    ) -> None:
        self.api_base = api_base
        self.summary_path = Path(summary_path)
        self.fail_step = fail_step

    async def run(self) -> PipelineResult:
        steps: list[StepResult] = []

        tasks_payload = await self._call(STEP_GET, {})
        tasks = _task_list(tasks_payload)
        get_output: Any = tasks if tasks is not None else tasks_payload
        steps.append(StepResult(STEP_GET, _status(tasks_payload), {}, get_output))
        if _is_error(tasks_payload) or tasks is None:
            return self._finish("FAILED", STEP_GET, steps)

        summary_input = {"tasks": tasks}
        summary_payload = await self._call(STEP_SUMMARIZE, summary_input)
        steps.append(
            StepResult(STEP_SUMMARIZE, _status(summary_payload), summary_input, summary_payload)
        )
        if _is_error(summary_payload):
            return self._finish("FAILED", STEP_SUMMARIZE, steps)

        save_input = {"summary": summary_payload}
        save_payload = await self._call(STEP_SAVE, save_input)
        steps.append(StepResult(STEP_SAVE, _status(save_payload), save_input, save_payload))
        if _is_error(save_payload):
            return self._finish("FAILED", STEP_SAVE, steps)
        return self._finish("SUCCESS", None, steps)

    async def _call(self, name: str, arguments: dict[str, Any]) -> Any:
        raw = await call_tool(
            self.api_base,
            self.summary_path,
            name,
            arguments,
            fail_step=self.fail_step,
        )
        return parse_tool_payload(raw)

    def _finish(
        self,
        status: str,
        failed_step: str | None,
        steps: list[StepResult],
    ) -> PipelineResult:
        result = PipelineResult(status=status, failed_step=failed_step, steps=steps)
        result.report = format_report(result)
        return result


def _status(payload: Any) -> str:
    return "FAILED" if _is_error(payload) else "SUCCESS"


def _task_list(payload: Any) -> list | None:
    if isinstance(payload, dict) and isinstance(payload.get("tasks"), list) and not payload.get("error"):
        return payload["tasks"]
    if isinstance(payload, list):
        return payload
    return None
