"""Turn a short history into one Ollama prompt. Uses the Day 29 request shape."""

from __future__ import annotations

import math

import requests

from day29.config import GenerateConfig, request_body
from day30.config import CHARS_PER_TOKEN, SYSTEM_RESERVE_TOKENS, Settings
from day30.sessions import Session, Turn

SYSTEM = (
    "Ты локальный помощник. Отвечай на языке пользователя. "
    "Не выдумывай факты. Если данных не хватает, так и скажи. "
    "Учитывай предыдущие реплики этой сессии."
)


class UpstreamDown(Exception):
    pass


class UpstreamTimeout(Exception):
    pass


class ModelMissing(Exception):
    pass


def estimated_tokens(text: str) -> int:
    if not text:
        return 0
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def message_fits(message: str, settings: Settings) -> bool:
    return estimated_tokens(message) <= settings.max_message_tokens()


def fit_history(history: list[Turn], message: str, settings: Settings) -> list[Turn]:
    budget = settings.num_ctx - settings.num_predict - SYSTEM_RESERVE_TOKENS - estimated_tokens(message)
    if budget < 0:
        return []
    kept: list[Turn] = []
    used = 0
    for turn in reversed(history):
        cost = estimated_tokens(turn.content)
        if used + cost > budget:
            break
        kept.append(turn)
        used += cost
    kept.reverse()
    return kept


def render(history: list[Turn], message: str) -> str:
    lines = [f"SYSTEM:\n{SYSTEM}"]
    for turn in history:
        lines.append(f"{turn.role.upper()}:\n{turn.content}")
    lines.append(f"USER:\n{message}")
    return "\n\n".join(lines)


def complete(session: Session, message: str, settings: Settings) -> str:
    history = fit_history(session.messages, message, settings)
    config = GenerateConfig(
        name="day30",
        model=settings.model,
        think=False,
        temperature=settings.temperature,
        num_predict=settings.num_predict,
        num_ctx=settings.num_ctx,
        prompt_name="day30_chat",
    )
    prompt = render(history, message)
    try:
        response = requests.post(
            f"{settings.base_url}/api/generate",
            json=request_body(config, prompt),
            timeout=(10, settings.timeout_s),
        )
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout as exc:
        raise UpstreamTimeout from exc
    except requests.ConnectionError as exc:
        raise UpstreamDown from exc
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 0
        if status == 404:
            raise ModelMissing from exc
        raise UpstreamDown from exc
    except ValueError as exc:
        raise UpstreamDown from exc
    text = payload.get("response")
    if not isinstance(text, str):
        raise UpstreamDown
    return text
