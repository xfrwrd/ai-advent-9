"""Day 14: Task invariants — hard boundaries for the current task."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum


class InvariantCategory(str, Enum):
    ARCHITECTURE = "architecture"
    TECHNICAL_DECISION = "technical_decision"
    TECH_STACK = "tech_stack"
    BUSINESS_RULE = "business_rule"


CATEGORY_LABELS = {
    InvariantCategory.ARCHITECTURE: "ARCHITECTURE",
    InvariantCategory.TECHNICAL_DECISION: "TECHNICAL DECISION",
    InvariantCategory.TECH_STACK: "TECH STACK",
    InvariantCategory.BUSINESS_RULE: "BUSINESS RULE",
}


class InvariantError(Exception):
    """Invalid invariant operation."""


@dataclass
class Invariant:
    """Structured constraint that must not be violated for the current task."""

    id: str
    category: InvariantCategory
    rule: str
    reason: str
    active: bool = True
    # Deterministic conflict signals (lowercase substrings / phrases).
    conflict_signals: list[str] = field(default_factory=list)
    # Short hint for a safe alternative when this invariant is hit.
    alternative_hint: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category.value,
            "rule": self.rule,
            "reason": self.reason,
            "active": self.active,
            "conflict_signals": list(self.conflict_signals),
            "alternative_hint": self.alternative_hint,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict | None) -> Invariant | None:
        if not data or not isinstance(data, dict):
            return None
        inv_id = str(data.get("id") or "").strip()
        rule = str(data.get("rule") or "").strip()
        if not inv_id or not rule:
            return None
        try:
            category = InvariantCategory(
                str(data.get("category") or InvariantCategory.TECHNICAL_DECISION.value)
            )
        except ValueError as exc:
            raise InvariantError(f"Некорректная категория: {exc}") from exc
        signals = data.get("conflict_signals") or []
        if not isinstance(signals, list):
            signals = []
        return cls(
            id=inv_id,
            category=category,
            rule=rule,
            reason=str(data.get("reason") or "").strip(),
            active=bool(data.get("active", True)),
            conflict_signals=[str(s).strip().lower() for s in signals if str(s).strip()],
            alternative_hint=str(data.get("alternative_hint") or "").strip(),
        )


@dataclass
class InvariantConflict:
    invariant: Invariant
    user_request: str
    matched_signal: str

    def format_rejection(self) -> str:
        alt = self.invariant.alternative_hint or (
            "Могу предложить решение в рамках действующих ограничений задачи."
        )
        return "\n".join(
            [
                "CONFLICT",
                "",
                "1. Факт: запрос конфликтует с ограничением текущей задачи.",
                "",
                "2. Нарушенный invariant:",
                f'   id={self.invariant.id}',
                f'   rule="{self.invariant.rule}"',
                f"   category={self.invariant.category.value}",
                "",
                f"3. Причина (reason): {self.invariant.reason or '(not set)'}",
                f"   Почему запрос противоречит: «{self.user_request.strip()}» "
                f"(детерминированный сигнал: «{self.matched_signal}»).",
                "",
                f"4. Допустимая альтернатива: {alt}",
                "",
                "Инвариант не удалён. Изменить ограничение можно только "
                "явно через управление invariants (add/deactivate/remove), "
                "не через обычный чат.",
            ]
        )


@dataclass
class InvariantStore:
    """Collection of task invariants (persisted as one Working Memory blob)."""

    items: list[Invariant] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {"items": [item.to_dict() for item in self.items]},
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, raw: str | None) -> InvariantStore:
        if not raw or not str(raw).strip():
            return cls()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return cls()
        if not isinstance(data, dict):
            return cls()
        items_raw = data.get("items") or []
        if not isinstance(items_raw, list):
            return cls()
        items: list[Invariant] = []
        for entry in items_raw:
            inv = Invariant.from_dict(entry if isinstance(entry, dict) else None)
            if inv is not None:
                items.append(inv)
        return cls(items=items)

    def get(self, inv_id: str) -> Invariant | None:
        for item in self.items:
            if item.id == inv_id:
                return item
        return None

    def active_items(self) -> list[Invariant]:
        return [item for item in self.items if item.active]

    def add(self, invariant: Invariant) -> None:
        if not invariant.id.strip():
            raise InvariantError("id инварианта не может быть пустым.")
        if not invariant.rule.strip():
            raise InvariantError("rule инварианта не может быть пустым.")
        existing = self.get(invariant.id)
        if existing is not None:
            idx = self.items.index(existing)
            self.items[idx] = invariant
        else:
            self.items.append(invariant)

    def deactivate(self, inv_id: str) -> Invariant:
        inv = self.get(inv_id)
        if inv is None:
            raise InvariantError(f"Инвариант «{inv_id}» не найден.")
        inv.active = False
        return inv

    def activate(self, inv_id: str) -> Invariant:
        inv = self.get(inv_id)
        if inv is None:
            raise InvariantError(f"Инвариант «{inv_id}» не найден.")
        inv.active = True
        return inv

    def remove(self, inv_id: str) -> None:
        before = len(self.items)
        self.items = [item for item in self.items if item.id != inv_id]
        if len(self.items) == before:
            raise InvariantError(f"Инвариант «{inv_id}» не найден.")

    def format_prompt_block(self) -> str:
        active = self.active_items()
        if not active:
            return "ACTIVE INVARIANTS\n(none)"
        lines = [
            "ACTIVE INVARIANTS",
            "These are hard boundaries for the current task.",
            "Do not propose solutions that violate them — including semantic "
            "rephrasings without the exact forbidden keywords "
            "(e.g. «документоориентированная БД» ≈ leave PostgreSQL; "
            "«завершить заказ до оплаты» ≈ violate payment-before-COMPLETED).",
            "If the user request violates an invariant by meaning, refuse with: "
            "CONFLICT + which rule + reason + allowed alternative.",
            "Do not remove or rewrite them unless the user uses the "
            "explicit invariant management API (not ordinary chat).",
            "",
        ]
        by_cat: dict[InvariantCategory, list[Invariant]] = {}
        for inv in active:
            by_cat.setdefault(inv.category, []).append(inv)
        for category in InvariantCategory:
            group = by_cat.get(category)
            if not group:
                continue
            lines.append(f"[{CATEGORY_LABELS[category]}]")
            for inv in group:
                lines.append(f"- ({inv.id}) {inv.rule}")
                if inv.reason:
                    lines.append(f"  Reason: {inv.reason}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def ui_groups(self) -> list[tuple[str, list[Invariant]]]:
        active = self.active_items()
        result: list[tuple[str, list[Invariant]]] = []
        for category in InvariantCategory:
            group = [i for i in active if i.category == category]
            if group:
                result.append((CATEGORY_LABELS[category], group))
        return result


_BYPASS_PATTERNS = (
    r"\bзабудь\b",
    r"\bforget\b",
    r"удали\s+инвариант",
    r"remove\s+invariant",
    r"deactivate\s+invariant",
    r"отмени\s+ограничен",
)


def normalize_request(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def check_invariant_conflict(
    user_request: str,
    store: InvariantStore,
) -> InvariantConflict | None:
    """
    Deterministic conflict check against active invariants.

    Source of truth is code + structured signals — not LLM judgment.
    Ordinary chat cannot deactivate/remove invariants.
    """
    text = normalize_request(user_request)
    if not text:
        return None

    active = store.active_items()
    if not active:
        return None

    for pattern in _BYPASS_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            for inv in active:
                needles = [inv.id.lower(), *inv.conflict_signals]
                rule_tokens = [
                    t
                    for t in re.findall(r"[a-zа-я0-9_+-]+", inv.rule.lower())
                    if len(t) >= 4
                ]
                needles.extend(rule_tokens)
                for needle in needles:
                    if needle and needle in text:
                        return InvariantConflict(
                            invariant=inv,
                            user_request=user_request,
                            matched_signal=f"bypass:{needle}",
                        )
            return InvariantConflict(
                invariant=active[0],
                user_request=user_request,
                matched_signal="bypass:forget/remove",
            )

    for inv in active:
        for signal in inv.conflict_signals:
            if signal and signal in text:
                return InvariantConflict(
                    invariant=inv,
                    user_request=user_request,
                    matched_signal=signal,
                )
    return None


def demo_invariants() -> list[Invariant]:
    """Canonical Day 14 demo constraints for the orders backend task."""
    return [
        Invariant(
            id="api_style",
            category=InvariantCategory.ARCHITECTURE,
            rule="Use REST API",
            reason="Existing system architecture",
            conflict_signals=[
                "graphql",
                "grpc",
                "soap",
                "заменим rest",
                "вместо rest",
                "не rest",
            ],
            alternative_hint="Могу продолжить проектирование на REST API.",
        ),
        Invariant(
            id="backend_language",
            category=InvariantCategory.TECH_STACK,
            rule="Backend language must be Python",
            reason="Approved project stack",
            conflict_signals=[
                "на go",
                "на golang",
                "use go",
                "перейти на go",
                "заменим python",
                "вместо python",
                "backend на java",
                "на rust",
                "на node",
                "на typescript",
            ],
            alternative_hint=(
                "Могу продолжить реализацию требуемой функциональности на Python."
            ),
        ),
        Invariant(
            id="database",
            category=InvariantCategory.TECH_STACK,
            rule="Database must be PostgreSQL",
            reason="Approved project stack",
            conflict_signals=[
                "mongodb",
                "mongo",
                "mysql",
                "sqlite",
                "dynamodb",
                "cassandra",
                "заменим postgresql",
                "вместо postgresql",
                "не postgresql",
            ],
            alternative_hint=(
                "Могу предложить решение проблемы в рамках PostgreSQL "
                "(схема, индексы, JSONB и т.п.)."
            ),
        ),
        Invariant(
            id="payment_before_completed",
            category=InvariantCategory.BUSINESS_RULE,
            rule="Order cannot become COMPLETED before payment is confirmed",
            reason="Business requirement",
            conflict_signals=[
                "completed сразу",
                "даже если оплата",
                "без оплаты",
                "без подтверждения оплаты",
                "payment is not",
                "completed before payment",
                "completed без",
                "в completed сразу",
            ],
            alternative_hint=(
                "Могу предложить статусы вроде CREATED → AWAITING_PAYMENT → "
                "PAID → COMPLETED, где COMPLETED только после подтверждения оплаты."
            ),
        ),
    ]


DEMO_GOAL = "Разработать backend для сервиса заказов"
COMPATIBLE_PROMPT = "Добавь endpoint для получения заказа."
STACK_CONFLICT_PROMPT = "Давай заменим PostgreSQL на MongoDB."
BUSINESS_CONFLICT_PROMPT = (
    "Давай переводить заказ в COMPLETED сразу после создания, "
    "даже если оплата ещё не подтверждена."
)

# Semantic paraphrases — no explicit conflict_signals match.
# These must reach the LLM with ACTIVE INVARIANTS in context (branch 2).
SEMANTIC_DB_PROMPTS = (
    "Давай вместо текущей реляционной базы перейдём на документоориентированную БД.",
    "Предлагаю отказаться от текущей базы и хранить документы без реляционной схемы.",
)
SEMANTIC_BUSINESS_PROMPT = (
    "Давай считать заказ завершённым сразу после создания, а оплату проверим потом."
)
