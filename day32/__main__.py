"""Local review: python -m day32 --base main --head HEAD"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from day32.review import run_review, write_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only code review between two Git refs.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    repo = Path(args.repo).resolve()
    result = run_review(repo, args.base, args.head)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    short = (result.diff.head_commit or "norev")[:8]
    root = Path(args.output).resolve() if args.output else repo / "day32" / "reports"
    markdown_path, json_path = write_report(result, root / f"{stamp}-{short}")
    counts = result.stats()
    omitted = sum(1 for change in result.diff.files if change.omitted_reason)
    print(f"base {result.diff.base_ref} {result.diff.base_commit}")
    print(f"head {result.diff.head_ref} {result.diff.head_commit}")
    print(f"files {len(result.diff.files)}, omitted {omitted}")
    print(f"docs {len(result.context.docs)}, code {len(result.context.code)}")
    print(
        "findings "
        + " ".join(f"{name}={counts[name]}" for name in ("critical", "high", "medium", "low"))
    )
    print(result.summary)
    print(f"report {markdown_path}")
    print(f"json {json_path}")
    if result.diff.error and not result.diff.base_commit:
        return 2
    return 1 if result.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
