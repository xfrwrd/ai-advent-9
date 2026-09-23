"""Task currently watched by the Day 18 background collector."""

from __future__ import annotations


def current_task() -> tuple[str, str]:
    """Return (task_id, status) to snapshot. Status is the live source value."""
    return "TASK-123", "in_progress"
