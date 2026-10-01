"""Plausible questions the project files do not answer.

Scores on the Day 21 index stay under the Day 23 threshold of 9.5.
"""

from __future__ import annotations

from day22.questions import ControlQuestion

NEGATIVE_QUESTIONS: tuple[ControlQuestion, ...] = (
    ControlQuestion(
        1,
        "Какой номер телефона дежурного администратора указан в oncall-ротации?",
        "не знаю",
        (),
    ),
    ControlQuestion(
        2,
        "Какая сумма ежемесячного инвойса Stripe записана в договоре?",
        "не знаю",
        (),
    ),
    ControlQuestion(
        3,
        "Сколько дней отпуска осталось у сотрудника по табелю HR?",
        "не знаю",
        (),
    ),
)
