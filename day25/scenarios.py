"""Two dialogues of follow-up questions about files that are in the Day 21 index."""

from __future__ import annotations

SCENARIO_A = (
    "Моя цель — разобраться, как day19 pipeline вызывает MCP tools.",
    "Какие инструменты он вызывает?",
    "В каком они идут порядке?",
    "Ограничение: смотри только day19 pipeline, не day20.",
    "А куда сохраняется сводка?",
    "Как называется этот файл?",
    "Какая функция собирает путь к нему?",
    "summarize_tasks сам запрашивает задачи или берёт уже переданный список?",
    "Напомни последовательность трёх шагов.",
    "А что делает первый шаг с Task API?",
    "Мы всё ещё про day19 pipeline. Повтори, куда пишется json.",
)

SCENARIO_B = (
    "Хочу разобраться в индексации day21. Это цель диалога.",
    "Как называется embedding model?",
    "Уточняю термин: модель local-token-hash-v1.",
    "Чему равна её размерность?",
    "Ограничение: отвечай только по коду day21, не по UI.",
    "Чему равен CHUNK_SIZE_CHARS?",
    "Это размер в символах?",
    "Какой overlap у фиксированного чанкера?",
    "Меняю цель: теперь хочу узнать transport MCP server в day16.",
    "Ограничение про ответы только по коду остаётся. Какой transport?",
    "Это stdio?",
    "В каком файле day16 это задано?",
)

SCENARIO_A_GOAL = "day19"
SCENARIO_A_CONSTRAINT = "day19"
SCENARIO_B_GOAL = "day16"
SCENARIO_B_CONSTRAINT = "day21"
