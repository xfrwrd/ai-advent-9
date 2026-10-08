"""One Ollama generate call, with the timing fields the API actually returns."""

from __future__ import annotations

from dataclasses import dataclass, field

import requests

from day29.config import BASE_URL, GenerateConfig, request_body


@dataclass
class Generation:
    text: str
    prompt_eval_count: int | None
    eval_count: int | None
    load_duration_ns: int | None
    eval_duration_ns: int | None
    prompt_eval_duration_ns: int | None
    total_duration_ns: int | None
    tokens_per_sec: float | None
    cold_start: bool | None
    memory: dict[str, object] | None = field(default=None)

    def as_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "prompt_eval_count": self.prompt_eval_count,
            "eval_count": self.eval_count,
            "load_duration_ns": self.load_duration_ns,
            "eval_duration_ns": self.eval_duration_ns,
            "prompt_eval_duration_ns": self.prompt_eval_duration_ns,
            "total_duration_ns": self.total_duration_ns,
            "tokens_per_sec": self.tokens_per_sec,
            "cold_start": self.cold_start,
            "memory": self.memory,
        }


def generate(prompt: str, config: GenerateConfig, base_url: str = BASE_URL) -> Generation:
    url = f"{base_url.rstrip('/')}/api/generate"
    try:
        response = requests.post(url, json=request_body(config, prompt), timeout=(10, 180))
    except requests.ConnectionError as exc:
        raise RuntimeError(
            f"Нет соединения с Ollama по адресу {base_url}. Запустите сервер командой: ollama serve"
        ) from exc
    except requests.Timeout as exc:
        raise RuntimeError(f"Ollama не ответила вовремя ({base_url}, модель {config.model}).") from exc
    response.raise_for_status()
    payload = response.json()
    text = payload.get("response")
    if not isinstance(text, str):
        raise RuntimeError("Ответ Ollama не содержит текстовое поле response.")
    return Generation(
        text=text,
        prompt_eval_count=_int(payload.get("prompt_eval_count")),
        eval_count=_int(payload.get("eval_count")),
        load_duration_ns=_int(payload.get("load_duration")),
        eval_duration_ns=_int(payload.get("eval_duration")),
        prompt_eval_duration_ns=_int(payload.get("prompt_eval_duration")),
        total_duration_ns=_int(payload.get("total_duration")),
        tokens_per_sec=_tokens_per_sec(payload.get("eval_count"), payload.get("eval_duration")),
        cold_start=_cold_start(payload.get("load_duration")),
        memory=running_model(base_url, config.model),
    )


def running_model(base_url: str, model: str) -> dict[str, object] | None:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/ps", timeout=(5, 10))
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return None
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return None
    for item in models:
        if not isinstance(item, dict):
            continue
        if item.get("name") == model or item.get("model") == model:
            return {
                "size": item.get("size"),
                "size_vram": item.get("size_vram"),
                "context_length": (item.get("details") or {}).get("context_length")
                if isinstance(item.get("details"), dict)
                else None,
            }
    return None


def _tokens_per_sec(eval_count: object, eval_duration_ns: object) -> float | None:
    count = _int(eval_count)
    duration = _int(eval_duration_ns)
    if count is None or duration is None or duration <= 0:
        return None
    return count / (duration / 1_000_000_000)


def _cold_start(load_duration_ns: object) -> bool | None:
    duration = _int(load_duration_ns)
    if duration is None:
        return None
    return duration >= 1_000_000_000


def _int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
