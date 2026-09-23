"""Background snapshot scheduler.

One process-wide instance. Streamlit reruns call get_scheduler() and get the
same object; they do not start another thread.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

from store import Store, default_db_path
from task_source import current_task

TaskSource = Callable[[], tuple[str, str]]

_guard = threading.Lock()
_scheduler: Scheduler | None = None


def configured_interval_seconds() -> float:
    raw = os.environ.get("DAY18_INTERVAL_SECONDS", "10")
    try:
        value = float(raw)
    except ValueError:
        return 10.0
    if value <= 0:
        return 10.0
    return value


@dataclass(frozen=True)
class SchedulerView:
    running: bool
    interval_seconds: float
    last_success_at: str | None
    next_run_at: datetime | None
    snapshot_count: int


class Scheduler:
    def __init__(
        self,
        store: Store,
        interval_seconds: float,
        source: TaskSource = current_task,
    ) -> None:
        self.store = store
        self.interval_seconds = interval_seconds
        self._source = source
        self._stop = threading.Event()
        self._start_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._next_run_at: datetime | None = None

    def start(self) -> None:
        with self._start_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._collect()
            self._thread = threading.Thread(
                target=self._loop,
                name="day18-scheduler",
                daemon=True,
            )
            self._thread.start()
            print(
                f"Day 18 scheduler started interval={self.interval_seconds:g}s "
                f"db={self.store.path}",
                flush=True,
            )

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def view(self) -> SchedulerView:
        with self._state_lock:
            next_run_at = self._next_run_at
        return SchedulerView(
            running=self.is_running(),
            interval_seconds=self.interval_seconds,
            last_success_at=self.store.latest_collected_at(),
            next_run_at=next_run_at,
            snapshot_count=self.store.count(),
        )

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._collect()

    def _collect(self) -> None:
        task_id, status = self._source()
        collected_at = datetime.now().replace(microsecond=0)
        self.store.insert_snapshot(task_id, status, collected_at)
        with self._state_lock:
            self._next_run_at = collected_at + timedelta(seconds=self.interval_seconds)


def get_scheduler() -> Scheduler:
    """Return the running scheduler, starting it once per process."""
    global _scheduler
    with _guard:
        if _scheduler is not None and _scheduler.is_running():
            return _scheduler
        _scheduler = Scheduler(Store(default_db_path()), configured_interval_seconds())
        _scheduler.start()
        return _scheduler


def reset_scheduler_for_tests() -> None:
    global _scheduler
    with _guard:
        if _scheduler is not None:
            _scheduler.stop()
            _scheduler = None
