"""Sequential baseline then optimized runs. Does not download models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from day29.answer import answer_with, config_for
from day29.config import BASELINE, MODES, MODEL_FACTS, OPTIMIZED
from day29.questions import QUESTIONS, SCORE_FIELDS
from day29.retrieve import load_frozen_index, retrieve

RESULTS_PATH = Path(__file__).resolve().parent / "data" / "results.json"


def run(repeats: int = 1) -> dict[str, object]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    index = load_frozen_index()
    runs: list[dict[str, object]] = []
    for question in QUESTIONS:
        context = retrieve(question.text, index)
        for mode in MODES:
            config = config_for(mode)
            for repeat in range(1, repeats + 1):
                result = answer_with(context, config)
                runs.append(
                    {
                        "question_id": question.id,
                        "kind": question.kind,
                        "question": question.text,
                        "mode": mode,
                        "repeat": repeat,
                        "result": result.as_dict(),
                    }
                )
                print(
                    f"{question.id} {mode} #{repeat} "
                    f"retrieval {result.retrieval_s:.2f}s generation {result.generation_s:.2f}s",
                    flush=True,
                )
    report = {
        "experiment_ran": True,
        "repeats": repeats,
        "model_inventory": MODEL_FACTS,
        "baseline": config_for(BASELINE).as_dict(),
        "optimized": config_for(OPTIMIZED).as_dict(),
        "runs": runs,
        "manual_scores": _blank_scores(),
        "conclusion": None,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _blank_scores() -> list[dict[str, object]]:
    empty = {field: None for field in SCORE_FIELDS}
    return [
        {"question_id": question.id, "baseline": dict(empty), "optimized": dict(empty)}
        for question in QUESTIONS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and optimized local RAG answers.")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    run(args.repeats)


if __name__ == "__main__":
    main()
