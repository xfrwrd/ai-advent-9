"""Private HTTP chat. Ollama stays on loopback; this process does not reconfigure it."""

from __future__ import annotations

import hmac
import logging
import os
import threading
import time

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from day30.chat import (
    ModelMissing,
    UpstreamDown,
    UpstreamTimeout,
    complete,
    message_fits,
)
from day30.config import Settings, load_settings
from day30.limiters import GenerationGate, RateLimiter
from day30.sessions import SessionStore

logger = logging.getLogger("day30")

_settings = load_settings()
_sessions = SessionStore()
_rates = RateLimiter()
_gate = GenerationGate()
_lock = threading.Lock()
_metrics = {
    "requests": 0,
    "chat_ok": 0,
    "chat_error": 0,
    "rate_limited": 0,
    "rejected_busy": 0,
    "generation_seconds": 0.0,
}


class ChatIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1)


def create_app(
    settings: Settings | None = None,
    sessions: SessionStore | None = None,
    rates: RateLimiter | None = None,
    gate: GenerationGate | None = None,
) -> FastAPI:
    app = FastAPI(title="Day 30 private LLM", docs_url=None, redoc_url=None)
    app.state.settings = settings or _settings
    app.state.sessions = sessions or _sessions
    app.state.rates = rates or _rates
    app.state.gate = gate or _gate

    @app.exception_handler(RequestValidationError)
    async def invalid_body(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": "Некорректный JSON. Нужны поля session_id и message."},
        )

    @app.exception_handler(Exception)
    async def hidden(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, HTTPException):
            raise exc
        logger.exception("unhandled")
        return JSONResponse(status_code=500, content={"detail": "Внутренняя ошибка сервиса."})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request, authorization: str | None = Header(default=None)) -> JSONResponse:
        _authorize(request, authorization)
        allowed, retry = request.app.state.rates.allow(_client(request))
        if not allowed:
            return _too_many(request, retry)
        return _readiness(request.app.state.settings)

    @app.get("/metrics")
    def metrics(request: Request, authorization: str | None = Header(default=None)) -> JSONResponse:
        _authorize(request, authorization)
        allowed, retry = request.app.state.rates.allow(_client(request))
        if not allowed:
            return _too_many(request, retry)
        with _lock:
            snapshot = dict(_metrics)
        snapshot["host"] = request.app.state.settings.host
        return JSONResponse(snapshot)

    @app.post("/chat")
    def chat(
        body: ChatIn,
        request: Request,
        authorization: str | None = Header(default=None),
        x_session_token: str | None = Header(default=None),
    ) -> JSONResponse:
        _authorize(request, authorization)
        allowed, retry = request.app.state.rates.allow(_client(request))
        if not allowed:
            return _too_many(request, retry)
        settings: Settings = request.app.state.settings
        if not message_fits(body.message, settings):
            return JSONResponse(
                status_code=413,
                content={
                    "detail": (
                        "Сообщение длиннее окна, которое сервис готов отправить в модель. "
                        "Оценка размера — около 3 символов на токен, это не точный токенизатор. "
                        "Сократите текст."
                    )
                },
            )
        session, error = request.app.state.sessions.open(body.session_id, x_session_token)
        if error == "bad_token":
            logger.info("chat rejected session=%s reason=token", body.session_id)
            return JSONResponse(
                status_code=403,
                content={"detail": "Сессия есть, но токен не совпал. Знать session_id недостаточно."},
            )
        if not request.app.state.gate.try_enter():
            with _lock:
                _metrics["rejected_busy"] += 1
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "Модель уже отвечает на два запроса. Повторите чуть позже.",
                    "session_id": session.session_id,
                    "session_token": session.token,
                },
            )
        started = time.perf_counter()
        try:
            answer = complete(session, body.message, settings)
        except UpstreamTimeout:
            with _lock:
                _metrics["chat_error"] += 1
            return JSONResponse(
                status_code=504,
                content={"detail": "Модель не ответила вовремя.", "session_token": session.token},
            )
        except ModelMissing:
            with _lock:
                _metrics["chat_error"] += 1
            return JSONResponse(
                status_code=503,
                content={"detail": "Модель не найдена в Ollama.", "session_token": session.token},
            )
        except UpstreamDown:
            with _lock:
                _metrics["chat_error"] += 1
            return JSONResponse(
                status_code=503,
                content={"detail": "Ollama недоступна.", "session_token": session.token},
            )
        else:
            request.app.state.sessions.remember(session, body.message, answer)
            elapsed = time.perf_counter() - started
            with _lock:
                _metrics["chat_ok"] += 1
                _metrics["generation_seconds"] += elapsed
            logger.info("chat ok session=%s seconds=%.2f", body.session_id, elapsed)
            return JSONResponse(
                {"session_id": session.session_id, "answer": answer, "session_token": session.token}
            )
        finally:
            request.app.state.gate.leave()

    return app


def _authorize(request: Request, authorization: str | None) -> None:
    expected = request.app.state.settings.api_key
    if not expected:
        raise _http(503, "DAY30_API_KEY не задан.")
    prefix = "Bearer "
    if not authorization or not authorization.startswith(prefix):
        raise _http(401, "Нужен заголовок Authorization: Bearer <token>.")
    sent = authorization[len(prefix) :].strip()
    if not hmac.compare_digest(sent.encode(), expected.encode()):
        raise _http(401, "Неверный API key.")


def _http(status: int, detail: str) -> None:
    raise HTTPException(status_code=status, detail=detail)


def _client(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _too_many(request: Request, retry: int) -> JSONResponse:
    with _lock:
        _metrics["rate_limited"] += 1
    return JSONResponse(
        status_code=429,
        content={"detail": "Слишком много запросов. Лимит — 10 в минуту."},
        headers={"Retry-After": str(retry)},
    )


def _readiness(settings: Settings) -> JSONResponse:
    try:
        response = requests.get(f"{settings.base_url}/api/tags", timeout=(3, 5))
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        return JSONResponse(status_code=503, content={"ready": False, "detail": "Ollama недоступна."})
    names = []
    for item in payload.get("models") or []:
        if isinstance(item, dict):
            names.append(str(item.get("name") or item.get("model") or ""))
    if settings.model not in names and f"{settings.model}:latest" not in names:
        return JSONResponse(
            status_code=503,
            content={"ready": False, "detail": "Модель не установлена в Ollama."},
        )
    return JSONResponse({"ready": True, "model": settings.model})


app = create_app()


def main() -> None:
    load_dotenv()
    settings = load_settings()
    global app
    app = create_app(settings)
    if not settings.api_key:
        raise SystemExit("Задайте DAY30_API_KEY в окружении. В репозиторий токен не кладите.")
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
