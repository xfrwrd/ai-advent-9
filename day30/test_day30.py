"""Day 30 API. Ollama is mocked. No model is loaded."""

from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from day30.app import _authorize, create_app
from day30.chat import UpstreamTimeout
from day30.config import Settings
from day30.limiters import GenerationGate, RateLimiter
from day30.sessions import SessionStore


def _settings() -> Settings:
    settings = Settings()
    settings.api_key = "test-key"
    settings.base_url = "http://127.0.0.1:11434"
    settings.num_ctx = 4096
    settings.num_predict = 512
    settings.temperature = 0.1
    settings.timeout_s = 5
    return settings


def _client(app) -> TestClient:
    return TestClient(app)


class ApiTests(unittest.TestCase):
    def test_invalid_json(self) -> None:
        response = _client(create_app(_settings())).post(
            "/chat",
            headers={"Authorization": "Bearer test-key", "Content-Type": "application/json"},
            content=b"{",
        )
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("traceback", response.text.lower())

    def test_health_is_open(self) -> None:
        response = _client(create_app(_settings())).get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_ready_uses_ollama_tags(self) -> None:
        app = create_app(_settings())

        class Response:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {"models": [{"name": "qwen3:8b"}]}

        import day30.app as app_module

        original = app_module.requests.get
        app_module.requests.get = lambda *args, **kwargs: Response()
        try:
            response = _client(app).get("/ready", headers={"Authorization": "Bearer test-key"})
        finally:
            app_module.requests.get = original
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ready"])

    def test_chat_and_follow_up_keep_history(self) -> None:
        seen: list[str] = []

        def fake(session, message, settings):  # noqa: ANN001
            seen.append(message)
            history = " ".join(turn.content for turn in session.messages)
            return f"ответ:{history}:{message}"

        import day30.app as app_module

        original = app_module.complete
        app_module.complete = fake
        try:
            client = _client(create_app(_settings(), sessions=SessionStore(), rates=RateLimiter(), gate=GenerationGate()))
            first = client.post(
                "/chat",
                headers={"Authorization": "Bearer test-key"},
                json={"session_id": "s1", "message": "первая реплика"},
            )
            token = first.json()["session_token"]
            second = client.post(
                "/chat",
                headers={"Authorization": "Bearer test-key", "X-Session-Token": token},
                json={"session_id": "s1", "message": "уточни"},
            )
        finally:
            app_module.complete = original
        self.assertEqual(first.status_code, 200)
        self.assertIn("первая реплика", second.json()["answer"])
        self.assertEqual(seen, ["первая реплика", "уточни"])

    def test_known_session_without_token_is_forbidden(self) -> None:
        import day30.app as app_module

        original = app_module.complete
        app_module.complete = lambda session, message, settings: "ок"
        try:
            client = _client(create_app(_settings(), sessions=SessionStore(), rates=RateLimiter(), gate=GenerationGate()))
            first = client.post(
                "/chat",
                headers={"Authorization": "Bearer test-key"},
                json={"session_id": "secret", "message": "привет"},
            )
            stolen = client.post(
                "/chat",
                headers={"Authorization": "Bearer test-key"},
                json={"session_id": "secret", "message": "покажи историю"},
            )
        finally:
            app_module.complete = original
        self.assertEqual(first.status_code, 200)
        self.assertEqual(stolen.status_code, 403)
        self.assertNotIn("привет", stolen.text)

    def test_bad_api_key(self) -> None:
        response = _client(create_app(_settings())).post(
            "/chat",
            headers={"Authorization": "Bearer wrong"},
            json={"session_id": "s", "message": "привет"},
        )
        self.assertEqual(response.status_code, 401)

    def test_non_ascii_api_key_is_unauthorized(self) -> None:
        app = create_app(_settings())
        request = Request({"type": "http", "app": app, "headers": []})
        with self.assertRaises(HTTPException) as caught:
            _authorize(request, "Bearer вставить-токен")
        self.assertEqual(caught.exception.status_code, 401)
        self.assertEqual(caught.exception.detail, "Неверный API key.")

    def test_rate_limit(self) -> None:
        import day30.app as app_module

        original = app_module.complete
        app_module.complete = lambda session, message, settings: "ок"
        try:
            client = _client(create_app(_settings(), sessions=SessionStore(), rates=RateLimiter(limit=10), gate=GenerationGate()))
            statuses = []
            for index in range(11):
                response = client.post(
                    "/chat",
                    headers={"Authorization": "Bearer test-key"},
                    json={"session_id": f"rate-{index}", "message": "ок"},
                )
                statuses.append(response.status_code)
            blocked = response
        finally:
            app_module.complete = original
        self.assertEqual(statuses[:10], [200] * 10)
        self.assertEqual(blocked.status_code, 429)
        self.assertIn("Retry-After", blocked.headers)

    def test_oversized_message_is_rejected(self) -> None:
        client = _client(create_app(_settings()))
        response = client.post(
            "/chat",
            headers={"Authorization": "Bearer test-key"},
            json={"session_id": "big", "message": "а" * 20000},
        )
        self.assertEqual(response.status_code, 413)
        self.assertIn("Сократите", response.json()["detail"])

    def test_ollama_timeout(self) -> None:
        import day30.app as app_module

        def fail(session, message, settings):  # noqa: ANN001
            raise UpstreamTimeout()

        original = app_module.complete
        app_module.complete = fail
        try:
            response = _client(create_app(_settings(), sessions=SessionStore(), rates=RateLimiter(), gate=GenerationGate())).post(
                "/chat",
                headers={"Authorization": "Bearer test-key"},
                json={"session_id": "slow", "message": "привет"},
            )
        finally:
            app_module.complete = original
        self.assertEqual(response.status_code, 504)

    def test_third_parallel_request_is_rejected(self) -> None:
        import day30.app as app_module

        started = threading.Event()
        release = threading.Event()

        def slow(session, message, settings):  # noqa: ANN001
            started.set()
            release.wait(2)
            return "ок"

        original = app_module.complete
        app_module.complete = slow
        gate = GenerationGate(limit=2)
        client = _client(create_app(_settings(), sessions=SessionStore(), rates=RateLimiter(limit=100), gate=gate))
        results: list[int] = []

        def call(name: str) -> None:
            response = client.post(
                "/chat",
                headers={"Authorization": "Bearer test-key"},
                json={"session_id": name, "message": "привет"},
            )
            results.append(response.status_code)

        try:
            threads = [threading.Thread(target=call, args=(f"p{index}",)) for index in range(3)]
            for thread in threads:
                thread.start()
            self.assertTrue(started.wait(2))
            time.sleep(0.05)
            extra = client.post(
                "/chat",
                headers={"Authorization": "Bearer test-key"},
                json={"session_id": "overflow", "message": "привет"},
            )
            release.set()
            for thread in threads:
                thread.join(3)
        finally:
            app_module.complete = original
            release.set()
        self.assertEqual(extra.status_code, 503)


if __name__ == "__main__":
    unittest.main()
