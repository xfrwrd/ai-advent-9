"""Day 09: Agent with summary compression for context management."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

API_URL = "https://api.deepseek.com/chat/completions"
THINKING_DISABLED = {"type": "disabled"}

# Official DeepSeek pricing for deepseek-v4-flash (off-peak, cache miss).
MAX_CONTEXT_TOKENS = 1_000_000
INPUT_PRICE_PER_MILLION = 0.22
OUTPUT_PRICE_PER_MILLION = 0.66

RECENT_MESSAGES_LIMIT = 8
COMPRESSION_THRESHOLD = 10

HISTORY_PATH = Path(__file__).resolve().parent / "history.json"
SUMMARY_PATH = Path(__file__).resolve().parent / "summary.json"
STATS_PATH = Path(__file__).resolve().parent / "stats.json"

SUMMARIZE_SYSTEM_PROMPT = """Сожми историю диалога.
Сохрани:
- важные факты;
- решения;
- требования;
- предпочтения пользователя;
- незавершённые вопросы.

Не добавляй новых фактов.
Удаляй повторы и несущественные детали.
Ответ должен быть кратким."""


@dataclass
class TokenStats:
    current_message_tokens: int
    history_tokens: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float
    summary_tokens: int = 0
    context_tokens_estimate: int = 0


@dataclass
class AgentResponse:
    content: str
    token_stats: TokenStats
    compression_enabled: bool
    did_compress: bool = False


@dataclass
class CumulativeStats:
    cumulative_tokens: int = 0
    cumulative_cost: float = 0.0


class AgentError(Exception):
    """Raised when the agent cannot get a valid answer from the API."""


def estimate_tokens(text: str) -> int:
    """Approximate token count. NOT an exact DeepSeek tokenizer result."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def estimate_messages_tokens(messages: list[dict]) -> int:
    total = 0
    for message in messages:
        total += estimate_tokens(str(message.get("role", "")))
        total += estimate_tokens(str(message.get("content", "")))
        total += 4
    return total


def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    input_cost = prompt_tokens / 1_000_000 * INPUT_PRICE_PER_MILLION
    output_cost = completion_tokens / 1_000_000 * OUTPUT_PRICE_PER_MILLION
    return input_cost + output_cost


class Agent:
    """Agent with optional summary compression of older dialog history."""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-v4-flash",
        max_context_tokens: int = MAX_CONTEXT_TOKENS,
        compression_enabled: bool = True,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.compression_enabled = compression_enabled
        self.history_path = HISTORY_PATH
        self.summary_path = SUMMARY_PATH
        self.stats_path = STATS_PATH
        self.messages = self._load_history()
        self.summary = self._load_summary()
        self.cumulative = self._load_stats()

    def _load_history(self) -> list[dict]:
        if not self.history_path.exists():
            return []
        try:
            data = json.loads(self.history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return data if isinstance(data, list) else []

    def _save_history(self) -> None:
        self.history_path.write_text(
            json.dumps(self.messages, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_summary(self) -> str:
        if not self.summary_path.exists():
            return ""
        try:
            data = json.loads(self.summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ""
        if isinstance(data, dict):
            return str(data.get("summary") or "").strip()
        if isinstance(data, str):
            return data.strip()
        return ""

    def _save_summary(self) -> None:
        self.summary_path.write_text(
            json.dumps({"summary": self.summary}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_stats(self) -> CumulativeStats:
        if not self.stats_path.exists():
            return CumulativeStats()
        try:
            data = json.loads(self.stats_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return CumulativeStats()
        return CumulativeStats(
            cumulative_tokens=int(data.get("cumulative_tokens", 0)),
            cumulative_cost=float(data.get("cumulative_cost", 0.0)),
        )

    def _save_stats(self) -> None:
        self.stats_path.write_text(
            json.dumps(asdict(self.cumulative), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def clear_history(self) -> None:
        self.messages = []
        self.summary = ""
        self.cumulative = CumulativeStats()
        self._save_history()
        self._save_summary()
        self._save_stats()

    def replace_messages(self, messages: list[dict], clear_summary: bool = True) -> None:
        """Replace history from a demo/test helper. Does not call the API."""
        self.messages = list(messages)
        if clear_summary:
            self.summary = ""
            self._save_summary()
        self._save_history()

    def _post_chat(self, messages: list[dict]) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "thinking": THINKING_DISABLED,
        }
        try:
            response = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=90,
            )
        except requests.Timeout as exc:
            raise AgentError("Таймаут при обращении к DeepSeek API.") from exc
        except requests.RequestException as exc:
            raise AgentError(
                f"Не удалось связаться с DeepSeek API: {exc.__class__.__name__}."
            ) from exc

        if not response.ok:
            raise AgentError(
                f"DeepSeek API вернул ошибку HTTP {response.status_code}."
            )

        try:
            return response.json()
        except ValueError as exc:
            raise AgentError("Не удалось разобрать ответ DeepSeek API.") from exc

    def _summarize(self, previous_summary: str, messages: list[dict]) -> str:
        parts: list[str] = []
        if previous_summary.strip():
            parts.append(f"Предыдущий summary:\n{previous_summary.strip()}")
        parts.append("Сообщения для сжатия:")
        for message in messages:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            parts.append(f"{role}: {content}")

        data = self._post_chat(
            [
                {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
                {"role": "user", "content": "\n\n".join(parts)},
            ]
        )
        try:
            content = data["choices"][0]["message"].get("content") or ""
            usage = data.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            completion_tokens = int(usage.get("completion_tokens", 0))
            total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise AgentError("Не удалось разобрать summary-ответ DeepSeek API.") from exc

        summary = content.strip()
        if not summary:
            raise AgentError("DeepSeek API вернул пустой summary.")

        self.cumulative.cumulative_tokens += total_tokens
        self.cumulative.cumulative_cost += calculate_cost(prompt_tokens, completion_tokens)
        self._save_stats()
        return summary

    def maybe_compress(self) -> bool:
        """Compress old messages into summary when over threshold. Returns True if compressed."""
        if not self.compression_enabled:
            return False
        if len(self.messages) <= COMPRESSION_THRESHOLD:
            return False
        if len(self.messages) <= RECENT_MESSAGES_LIMIT:
            return False

        old_messages = self.messages[:-RECENT_MESSAGES_LIMIT]
        recent_messages = self.messages[-RECENT_MESSAGES_LIMIT:]
        if not old_messages:
            return False

        new_summary = self._summarize(
            previous_summary=self.summary,
            messages=old_messages,
        )
        self.summary = new_summary
        self.messages = recent_messages
        self._save_summary()
        self._save_history()
        return True

    def build_api_messages(self) -> list[dict]:
        """Build the message list that will be sent to the LLM for a normal ask."""
        api_messages: list[dict] = []
        if self.compression_enabled and self.summary.strip():
            api_messages.append(
                {
                    "role": "system",
                    "content": f"Summary предыдущего диалога:\n{self.summary.strip()}",
                }
            )
            api_messages.extend(self.messages)
        else:
            api_messages.extend(self.messages)
        return api_messages

    def estimate_context_tokens(self) -> int:
        return estimate_messages_tokens(self.build_api_messages())

    def ask(self, user_message: str) -> AgentResponse:
        history_before = list(self.messages)
        history_tokens = estimate_messages_tokens(history_before)
        current_message_tokens = estimate_tokens(user_message)
        summary_tokens = estimate_tokens(self.summary)
        did_compress = False

        self.messages.append({"role": "user", "content": user_message})

        if self.compression_enabled:
            # Compress older turns before the API call if history grew past threshold.
            # Keep the newest RECENT_MESSAGES_LIMIT messages (including this user turn).
            try:
                did_compress = self.maybe_compress()
            except AgentError:
                self.messages.pop()
                raise
            summary_tokens = estimate_tokens(self.summary)

        api_messages = self.build_api_messages()
        context_estimate = estimate_messages_tokens(api_messages)

        if context_estimate > self.max_context_tokens:
            self.messages.pop()
            self._save_history()
            raise AgentError(
                "История диалога превышает контекстное окно модели "
                f"(~{context_estimate} estimated tokens > "
                f"{self.max_context_tokens} MAX_CONTEXT_TOKENS)."
            )

        try:
            data = self._post_chat(api_messages)
        except AgentError:
            self.messages.pop()
            self._save_history()
            raise

        try:
            content = data["choices"][0]["message"].get("content") or ""
            usage = data.get("usage") or {}
            prompt_tokens = int(usage["prompt_tokens"])
            completion_tokens = int(usage["completion_tokens"])
            total_tokens = int(usage["total_tokens"])
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            self.messages.pop()
            self._save_history()
            raise AgentError("Не удалось разобрать ответ DeepSeek API.") from exc

        answer = content.strip()
        if not answer:
            self.messages.pop()
            self._save_history()
            raise AgentError("DeepSeek API вернул пустой ответ.")

        self.messages.append({"role": "assistant", "content": answer})

        # Compress again after the assistant reply if we crossed the threshold.
        if self.compression_enabled:
            try:
                did_compress = self.maybe_compress() or did_compress
            except AgentError as exc:
                # Keep the successful turn; report compression failure separately.
                self._save_history()
                raise AgentError(f"Ответ получен, но compression не удался: {exc}") from exc

        self._save_history()

        cost = calculate_cost(prompt_tokens, completion_tokens)
        self.cumulative.cumulative_tokens += total_tokens
        self.cumulative.cumulative_cost += cost
        self._save_stats()

        return AgentResponse(
            content=answer,
            token_stats=TokenStats(
                current_message_tokens=current_message_tokens,
                history_tokens=history_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_cost=cost,
                summary_tokens=summary_tokens,
                context_tokens_estimate=context_estimate,
            ),
            compression_enabled=self.compression_enabled,
            did_compress=did_compress,
        )
