"""Ten control questions grounded in the project files."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ControlQuestion:
    id: int
    question: str
    expected: str
    sources: tuple[str, ...]


QUESTIONS: tuple[ControlQuestion, ...] = (
    ControlQuestion(
        1,
        "Какой MCP tool в day20 читает текущую задачу из Task API: get_task?",
        "get_task",
        ("day20/task_server.py",),
    ),
    ControlQuestion(
        2,
        "Какой MCP server в day20 отвечает за snapshots и get_task_summary?",
        "history_server",
        ("day20/history_server.py",),
    ),
    ControlQuestion(
        3,
        "В каком порядке day19 pipeline вызывает get_tasks, summarize_tasks и save_summary?",
        "get_tasks, затем summarize_tasks, затем save_summary",
        ("day19/pipeline.py",),
    ),
    ControlQuestion(
        4,
        "Как называется embedding model в day21 и чему равен EMBEDDING_DIM?",
        "local-token-hash-v1, 256",
        ("day21/README.md", "day21/embeddings.py"),
    ),
    ControlQuestion(
        5,
        "Чему равен CHUNK_SIZE_CHARS в day21 config?",
        "2000",
        ("day21/config.py",),
    ),
    ControlQuestion(
        6,
        "Какой title и status у TASK-123 в day17 mock API?",
        "Implement MCP integration, in_progress",
        ("day17/mock_api.py",),
    ),
    ControlQuestion(
        7,
        "Какое значение по умолчанию у DAY18_INTERVAL_SECONDS в scheduler?",
        "10",
        ("day18/scheduler.py",),
    ),
    ControlQuestion(
        8,
        "В какой файл day19 save_summary записывает сводку: day19_summary.json?",
        "day19/data/day19_summary.json",
        ("day19/mcp_server.py",),
    ),
    ControlQuestion(
        9,
        "Какой transport использует day16 MCP server: stdio?",
        "stdio",
        ("day16/server.py",),
    ),
    ControlQuestion(
        10,
        "Что делает кнопка Очистить в day19/app.py с файлом day19_summary.json?",
        "удаляет day19_summary.json",
        ("day19/app.py",),
    ),
)
