"""Three questions for the Local vs Cloud demo. Same retrieval for both models."""

from __future__ import annotations

FACT = "Чему равен CHUNK_SIZE_CHARS в day21 config?"
MULTI = (
    "В каком порядке day19 pipeline вызывает get_tasks, summarize_tasks и save_summary, "
    "и в какой файл save_summary пишет сводку?"
)
ABSENT = "Какой номер телефона дежурного администратора указан в oncall-ротации?"

DEMO_PHRASES = (
    f"1. {FACT}",
    f"2. {MULTI}",
    f"3. {ABSENT}",
)
