"""In-memory request cap and a hard gate in front of Ollama."""

from __future__ import annotations

import threading
import time

from day30.config import MAX_PARALLEL, RATE_LIMIT, RATE_WINDOW_S


class RateLimiter:
    def __init__(self, limit: int = RATE_LIMIT, window_s: float = RATE_WINDOW_S) -> None:
        self.limit = limit
        self.window_s = window_s
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            recent = [stamp for stamp in self._hits.get(key, []) if now - stamp < self.window_s]
            if len(recent) >= self.limit:
                retry = int(self.window_s - (now - recent[0])) + 1
                self._hits[key] = recent
                return False, max(retry, 1)
            recent.append(now)
            self._hits[key] = recent
            return True, 0


class GenerationGate:
    """Rejects immediately when both slots are busy. No waiting queue."""

    def __init__(self, limit: int = MAX_PARALLEL) -> None:
        self._slots = threading.BoundedSemaphore(limit)

    def try_enter(self) -> bool:
        return self._slots.acquire(blocking=False)

    def leave(self) -> None:
        self._slots.release()
