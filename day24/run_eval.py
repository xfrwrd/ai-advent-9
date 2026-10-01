"""Save the Day 24 report for the 10 control questions and the negative ones."""

from __future__ import annotations

from day24.agent import load_day24
from day24.evaluate import REPORT_PATH, evaluate, save_report


def main() -> None:
    report = evaluate(load_day24())
    save_report(report)
    summary = report["summary"]
    print(f"threshold={report['similarity_threshold']}")
    print(
        f"sources {summary['answers_with_sources']}/{summary['grounded_answers']} "
        f"quotes {summary['answers_with_quotes']}/{summary['grounded_answers']} "
        f"valid {summary['valid_quotes']}/{summary['grounded_answers']} "
        f"expected {summary['expected_sources_found']}/{summary['questions']} "
        f"supported {summary['answers_supported_by_quotes']}/{summary['grounded_answers']}"
    )
    print(
        f"negative {summary['correct_i_dont_know']}/{summary['negative_questions']} "
        f"control insufficient {summary['insufficient_on_control']}"
    )
    print(f"saved {REPORT_PATH}")


if __name__ == "__main__":
    main()
