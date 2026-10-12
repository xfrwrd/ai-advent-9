"""Choose documentation, a Git tool, or both. The model does not make this choice."""

from __future__ import annotations

from dataclasses import dataclass

_DOC_HINTS = (
    "как ",
    "как устр",
    "что наход",
    "что в ",
    "где ",
    "папк",
    "запуст",
    "readme",
    "документ",
    "компонент",
    "ollama",
    "rag",
    "day",
    "архитектур",
    "индекс",
    "сервис",
    "fastapi",
    "qwen",
)


@dataclass(frozen=True)
class Plan:
    use_docs: bool
    tools: tuple[str, ...]


def plan_for(question: str) -> Plan:
    text = question.strip()
    if text.lower().startswith("/help"):
        text = text[5:].strip()
    folded = text.lower().replace("ё", "е")
    tools: list[str] = []
    if any(token in folded for token in ("ветк", "branch", "detached")):
        tools.append("git_current_branch")
    if any(token in folded for token in ("git status", "статус репозитор", "незакоммич")):
        tools.append("git_status")
    if "diff" in folded or "отличия в git" in folded:
        tools.append("git_diff_summary")
    if any(token in folded for token in ("список файлов", "какие файлы", "файлы репозитор")):
        tools.append("list_project_files")
    unique = tuple(dict.fromkeys(tools))
    wants_docs = not unique or any(hint in folded for hint in _DOC_HINTS)
    return Plan(use_docs=wants_docs, tools=unique)
