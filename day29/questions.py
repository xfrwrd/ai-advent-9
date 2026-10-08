"""Five document questions. Both modes see the same retrieved chunks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExamQuestion:
    id: int
    kind: str
    text: str


QUESTIONS: tuple[ExamQuestion, ...] = (
    ExamQuestion(1, "fact", "Чему равен CHUNK_SIZE_CHARS в day21 config?"),
    ExamQuestion(
        2,
        "multi_chunk",
        "В каком порядке day19 pipeline вызывает get_tasks, summarize_tasks и save_summary, "
        "и в какой файл save_summary пишет сводку?",
    ),
    ExamQuestion(3, "absent", "Какой номер телефона дежурного администратора указан в oncall-ротации?"),
    ExamQuestion(
        4,
        "verbatim",
        "Процитируй дословно строку из day16, где MCP server задаёт transport.",
    ),
    ExamQuestion(
        5,
        "conditions",
        "По day21 назови embedding model, EMBEDDING_DIM и CHUNK_SIZE_CHARS. "
        "Если какого-то из этих трёх значений нет в контексте, прямо напиши, какого не хватает.",
    ),
)

DEMO_PHRASES = tuple(f"{item.id}. {item.text}" for item in QUESTIONS)

SCORE_FIELDS = (
    "factual_correctness",
    "source_faithfulness",
    "completeness",
    "instruction_following",
    "correct_refusal",
)
