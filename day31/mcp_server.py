"""Read-only Git tools. The process never pushes, checks out, resets, or deletes."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from mcp.server.mcpserver import MCPServer

mcp = MCPServer(
    name="day31-git",
    instructions="Read-only Git facts for one allowed repository. No push, reset, checkout, or delete.",
)


def _repo() -> Path:
    raw = os.environ.get("DAY31_REPO", "").strip()
    if raw:
        return Path(raw).resolve()
    return Path(__file__).resolve().parent.parent


def _git(args: list[str]) -> tuple[int, str, str]:
    repo = _repo()
    if not repo.is_dir():
        return 128, "", "directory does not exist"
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except FileNotFoundError:
        return 127, "", "git is not installed"
    except subprocess.TimeoutExpired:
        return 124, "", "git timed out"
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def _ensure_repo() -> dict | None:
    code, out, _err = _git(["rev-parse", "--is-inside-work-tree"])
    if code != 0 or out != "true":
        return {
            "error": "not_a_git_repo",
            "message": "Директория не является Git-репозиторием.",
            "path": str(_repo()),
        }
    return None


@mcp.tool(description="Return the current Git branch of the allowed repository. Read-only.")
def git_current_branch() -> dict:
    """Current branch, or detached HEAD with the commit id."""
    failed = _ensure_repo()
    if failed is not None:
        return failed
    code, name, err = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    commit_code, commit, _commit_err = _git(["rev-parse", "HEAD"])
    if code != 0 or commit_code != 0:
        return {"error": "git_failed", "message": err or "git rev-parse failed"}
    if name == "HEAD":
        return {
            "detached": True,
            "branch": None,
            "commit": commit,
            "message": "HEAD detached, именованной ветки нет.",
        }
    return {"detached": False, "branch": name, "commit": commit}


@mcp.tool(description="Return a short git status. Read-only, does not stage or commit.")
def git_status() -> dict:
    """Porcelain status, capped so the payload stays small."""
    failed = _ensure_repo()
    if failed is not None:
        return failed
    code, out, err = _git(["status", "--porcelain=v1", "-b"])
    if code != 0:
        return {"error": "git_failed", "message": err or "git status failed"}
    lines = out.splitlines()
    cap = 40
    return {"status": "\n".join(lines[:cap]), "truncated": len(lines) > cap}


@mcp.tool(description="Return git diff --stat for unstaged and staged changes. Read-only.")
def git_diff_summary() -> dict:
    """Stat summary only. The patch itself is not returned."""
    failed = _ensure_repo()
    if failed is not None:
        return failed
    code, out, err = _git(["diff", "--stat"])
    staged_code, staged, staged_err = _git(["diff", "--cached", "--stat"])
    if code != 0 or staged_code != 0:
        return {"error": "git_failed", "message": err or staged_err or "git diff failed"}
    return {"unstaged": out, "staged": staged}


@mcp.tool(description="List tracked files in the allowed repository. Read-only.")
def list_project_files() -> dict:
    """git ls-files, at most 80 paths."""
    failed = _ensure_repo()
    if failed is not None:
        return failed
    code, out, err = _git(["ls-files"])
    if code != 0:
        return {"error": "git_failed", "message": err or "git ls-files failed"}
    files = [line for line in out.splitlines() if line]
    cap = 80
    return {"files": files[:cap], "count": len(files), "truncated": len(files) > cap}


if __name__ == "__main__":
    mcp.run(transport="stdio")
