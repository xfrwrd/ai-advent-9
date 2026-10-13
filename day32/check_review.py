"""Manual review with real Qwen. Not part of unittest.

    .venv/bin/python -m day32.index_code
    .venv/bin/python -m day32.check_review
"""

from __future__ import annotations

from day32.__main__ import main
from day32.gitdiff import run_git
from day32.index_code import PROJECT_ROOT


def run() -> int:
    code, _out, _err = run_git(PROJECT_ROOT, ["rev-parse", "--verify", "--end-of-options", "HEAD~1^{commit}"])
    if code != 0:
        print("Нужен хотя бы один родительский коммит. Ревью не запускалось.")
        return 0
    print("Это живой вызов Qwen. Автотесты его не выполняют.")
    return main(["--base", "HEAD~1", "--head", "HEAD", "--repo", str(PROJECT_ROOT)])


if __name__ == "__main__":
    raise SystemExit(run())
