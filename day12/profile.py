"""Day 12: UserProfile — how the agent should respond to this user."""

from __future__ import annotations

import json
from dataclasses import dataclass, field


STYLE_OPTIONS = (
    "concise",
    "detailed",
    "friendly",
    "formal",
    "technical",
    "simple",
)

FORMAT_OPTIONS = (
    "plain_text",
    "bullets",
    "step_by_step",
    "structured",
)

EXPERTISE_OPTIONS = (
    "beginner",
    "intermediate",
    "advanced",
    "technical",
)

CONSTRAINT_OPTIONS = (
    "no_basic_explanations",
    "use_technical_terminology",
    "explain_terminology",
    "use_simple_examples",
    "avoid_unexplained_jargon",
    "no_emojis",
    "avoid_unnecessary_introductions",
)


@dataclass
class UserProfile:
    """Stable personalization preferences (stored in Long-Term Memory)."""

    name: str | None = None
    language: str = "ru"
    style: str = "concise"
    format: str = "structured"
    expertise_level: str = "intermediate"
    constraints: list[str] = field(default_factory=list)

    def update(self, **kwargs: object) -> UserProfile:
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown profile field: {key}")
            if key == "constraints":
                if value is None:
                    self.constraints = []
                elif isinstance(value, list):
                    self.constraints = [str(item) for item in value]
                else:
                    raise TypeError("constraints must be a list of strings")
            elif key == "name":
                self.name = None if value is None else str(value)
            else:
                setattr(self, key, str(value))
        return self

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "language": self.language,
            "style": self.style,
            "format": self.format,
            "expertise_level": self.expertise_level,
            "constraints": list(self.constraints),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict | None) -> UserProfile:
        if not data or not isinstance(data, dict):
            return cls()
        constraints = data.get("constraints") or []
        if isinstance(constraints, str):
            try:
                parsed = json.loads(constraints)
                constraints = parsed if isinstance(parsed, list) else [constraints]
            except json.JSONDecodeError:
                constraints = [constraints]
        name = data.get("name")
        return cls(
            name=None if name in (None, "") else str(name),
            language=str(data.get("language") or "ru"),
            style=str(data.get("style") or "concise"),
            format=str(data.get("format") or "structured"),
            expertise_level=str(data.get("expertise_level") or "intermediate"),
            constraints=[str(item) for item in constraints],
        )

    @classmethod
    def from_json(cls, raw: str | None) -> UserProfile:
        if not raw or not str(raw).strip():
            return cls()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return cls()
        return cls.from_dict(data if isinstance(data, dict) else None)

    def format_prompt_block(self) -> str:
        """Dedicated USER PROFILE block for the system prompt."""
        lines = [
            "USER PROFILE:",
            f"Name: {self.name or '(unknown)'}",
            f"Language: {self.language}",
            f"Style: {self.style}",
            f"Format: {self.format}",
            f"Expertise: {self.expertise_level}",
            "Constraints:",
        ]
        if self.constraints:
            for item in self.constraints:
                lines.append(f"- {item}")
        else:
            lines.append("- (none)")
        lines.append("")
        lines.append(
            "Follow USER PROFILE for every answer. "
            "The user should not need to repeat these preferences."
        )
        return "\n".join(lines)

    def summary_lines(self) -> list[str]:
        return [
            f"Name: {self.name or '(unknown)'}",
            f"Language: {self.language}",
            f"Style: {self.style}",
            f"Format: {self.format}",
            f"Expertise: {self.expertise_level}",
            f"Constraints: {', '.join(self.constraints) if self.constraints else '(none)'}",
        ]


def profile_a_technical() -> UserProfile:
    """Profile A — Technical / Concise."""
    return UserProfile(
        name="Ксения",
        language="ru",
        style="concise",
        format="structured",
        expertise_level="advanced",
        constraints=[
            "no_basic_explanations",
            "use_technical_terminology",
            "avoid_unnecessary_introductions",
            "no_emojis",
        ],
    )


def profile_b_beginner() -> UserProfile:
    """Profile B — Beginner / Detailed."""
    return UserProfile(
        name="Алекс",
        language="ru",
        style="detailed",
        format="step_by_step",
        expertise_level="beginner",
        constraints=[
            "explain_terminology",
            "use_simple_examples",
            "avoid_unexplained_jargon",
            "no_emojis",
        ],
    )


DEMO_PROMPT = (
    "Объясни, что такое circuit breaker в микросервисах."
)

IDEMPOTENCY_PROMPT = "Что такое idempotency?"
