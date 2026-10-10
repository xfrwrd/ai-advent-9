"""Service settings. Secrets come from the environment, not from this file."""

from __future__ import annotations

import os

from day29.config import BASE_URL, OPTIMIZED_CONFIG

# Same starting point as Day 29 optimized mode. Override with env vars.
DEFAULT_TEMPERATURE = OPTIMIZED_CONFIG.temperature
DEFAULT_NUM_PREDICT = OPTIMIZED_CONFIG.num_predict
DEFAULT_NUM_CTX = OPTIMIZED_CONFIG.num_ctx
DEFAULT_MODEL = OPTIMIZED_CONFIG.model

# Day 29 estimates mixed Russian and code at about 3 characters per token.
# This is not the model tokenizer.
CHARS_PER_TOKEN = 3
SYSTEM_RESERVE_TOKENS = 200
MAX_SESSIONS = 100
MAX_MESSAGES = 20
SESSION_TTL_S = 30 * 60
RATE_LIMIT = 10
RATE_WINDOW_S = 60
MAX_PARALLEL = 2


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


class Settings:
    def __init__(self) -> None:
        self.api_key = os.environ.get("DAY30_API_KEY", "").strip()
        self.model = os.environ.get("DAY30_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self.base_url = os.environ.get("DAY30_OLLAMA_URL", BASE_URL).rstrip("/")
        self.temperature = _float("DAY30_TEMPERATURE", float(DEFAULT_TEMPERATURE or 0.1))
        self.num_predict = _int("DAY30_NUM_PREDICT", int(DEFAULT_NUM_PREDICT or 512))
        self.num_ctx = _int("DAY30_NUM_CTX", int(DEFAULT_NUM_CTX or 4096))
        self.timeout_s = _float("DAY30_TIMEOUT", 180.0)
        self.host = "0.0.0.0" if os.environ.get("DAY30_LAN") == "1" else "127.0.0.1"
        self.port = _int("DAY30_PORT", 8030)

    def max_message_tokens(self) -> int:
        room = self.num_ctx - self.num_predict - SYSTEM_RESERVE_TOKENS
        return max(room, 1)


def load_settings() -> Settings:
    return Settings()
