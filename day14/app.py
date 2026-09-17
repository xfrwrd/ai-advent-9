"""Day 14: Streamlit UI — Task Invariants on Day 11–13 architecture."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent import (
    INPUT_PRICE_PER_MILLION,
    OUTPUT_PRICE_PER_MILLION,
    Agent,
    AgentError,
)
from invariants import (
    BUSINESS_CONFLICT_PROMPT,
    COMPATIBLE_PROMPT,
    DEMO_GOAL,
    STACK_CONFLICT_PROMPT,
    Invariant,
    InvariantCategory,
    InvariantError,
)
from memory import INVARIANTS_KEY, PROFILE_KEY, TASK_STATE_KEY
from task_state import STAGE_ORDER, TaskStateError

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

st.set_page_config(page_title="Day 14 — Invariants", layout="wide")
st.title("Day 14 — Инварианты и ограничения состояния")
st.caption(
    "UserProfile → HOW · TaskState → WHAT WE'RE DOING · "
    "Invariants → WHAT MUST NOT BE VIOLATED"
)

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    st.error("DEEPSEEK_API_KEY не задан. Добавьте ключ в .env в корне проекта.")
    st.stop()

agent = Agent(api_key=api_key)
task = agent.get_task_state()
store = agent.get_invariant_store()

left, right = st.columns([2, 1])

with right:
    st.subheader("Current Task")
    if task is None:
        st.info("Нет активной задачи.")
        goal = st.text_input("Goal", value=DEMO_GOAL)
        if st.button("Create task", type="primary"):
            try:
                agent.create_task(goal)
                st.rerun()
            except TaskStateError as exc:
                st.error(str(exc))
    else:
        st.markdown(f"**Goal:** {task.goal}")
        st.markdown(f"**Stage:** `{task.stage.value.upper()}`")
        st.markdown(f"**Status:** `{task.status.value}`")
        parts = []
        for stage in STAGE_ORDER:
            label = stage.value.capitalize()
            parts.append(f"**[{label}]**" if stage == task.stage else label)
        st.markdown(" → ".join(parts))

    st.divider()
    st.subheader("ACTIVE INVARIANTS")
    groups = store.ui_groups()
    if not groups:
        st.caption("(none)")
    else:
        for label, items in groups:
            st.markdown(f"**[{label}]**")
            for inv in items:
                st.markdown(f"✓ `{inv.id}`: {inv.rule}")

    with st.expander("Manage invariants"):
        new_id = st.text_input("id", value="custom_rule")
        new_cat = st.selectbox(
            "category",
            options=[c.value for c in InvariantCategory],
        )
        new_rule = st.text_input("rule", value="")
        new_reason = st.text_input("reason", value="")
        new_signals = st.text_input(
            "conflict_signals (comma-separated)",
            value="",
            help="Детерминированные сигналы конфликта, напр. mongodb, graphql",
        )
        if st.button("Add invariant"):
            try:
                signals = [s.strip() for s in new_signals.split(",") if s.strip()]
                agent.add_invariant(
                    Invariant(
                        id=new_id.strip(),
                        category=InvariantCategory(new_cat),
                        rule=new_rule.strip(),
                        reason=new_reason.strip(),
                        conflict_signals=signals,
                    )
                )
                st.rerun()
            except (InvariantError, ValueError) as exc:
                st.error(str(exc))

        ids = [inv.id for inv in store.items]
        if ids:
            pick = st.selectbox("Select invariant", options=ids)
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("Deactivate"):
                    try:
                        agent.deactivate_invariant(pick)
                        st.rerun()
                    except InvariantError as exc:
                        st.error(str(exc))
            with c2:
                if st.button("Activate"):
                    try:
                        agent.activate_invariant(pick)
                        st.rerun()
                    except InvariantError as exc:
                        st.error(str(exc))
            with c3:
                if st.button("Remove"):
                    try:
                        agent.remove_invariant(pick)
                        st.rerun()
                    except InvariantError as exc:
                        st.error(str(exc))

    st.divider()
    st.subheader("Demo")
    if st.button("Setup demo task + invariants", type="primary"):
        agent.setup_demo_task()
        st.success("Задача заказов + 4 invariants загружены.")
        st.rerun()

    st.caption("Быстрые проверки:")
    if st.button("Compatible: endpoint"):
        try:
            result = agent.ask(COMPATIBLE_PROMPT)
            st.session_state.last_result = result
        except AgentError as exc:
            st.error(str(exc))
        st.rerun()
    if st.button("Conflict: MongoDB"):
        try:
            result = agent.ask(STACK_CONFLICT_PROMPT)
            st.session_state.last_result = result
        except AgentError as exc:
            st.error(str(exc))
        st.rerun()
    if st.button("Conflict: COMPLETED"):
        try:
            result = agent.ask(BUSINESS_CONFLICT_PROMPT)
            st.session_state.last_result = result
        except AgentError as exc:
            st.error(str(exc))
        st.rerun()

    st.divider()
    st.subheader("Memory")
    if st.button("Clear Short-Term"):
        agent.clear_short_term()
        st.info("Short-Term очищен — TaskState и Invariants остаются.")
        st.rerun()
    if st.button("Clear Working Memory"):
        agent.clear_working_memory()
        st.warning("Working очищен — TaskState и Invariants удалены.")
        st.rerun()
    if st.button("Очистить всё"):
        agent.clear_all()
        st.session_state.pop("last_result", None)
        st.rerun()

with left:
    st.subheader("Диалог (Short-Term)")
    for message in agent.memory.short_term:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_message = st.chat_input("Запрос… (конфликт проверяется до LLM)")
    if user_message:
        with st.chat_message("user"):
            st.markdown(user_message)
        with st.chat_message("assistant"):
            try:
                result = agent.ask(user_message)
                if result.rejected:
                    st.warning(result.content)
                else:
                    st.markdown(result.content)
                st.session_state.last_result = result
            except AgentError as exc:
                st.error(str(exc))

st.divider()
st.subheader("Что сейчас попадёт в prompt")
st.code(agent.preview_prompt(), language="text")

with st.expander("Working Memory"):
    st.json(agent.memory.working or {"(empty)": True})

if "last_result" in st.session_state and st.session_state.last_result is not None:
    result = st.session_state.last_result
    stats = result.token_stats
    st.divider()
    if result.rejected and result.conflict is not None:
        st.subheader("Last conflict")
        st.error(
            f"Violated: `{result.conflict.invariant.id}` — "
            f"{result.conflict.invariant.rule}"
        )
    st.subheader("Token statistics")
    st.markdown(
        f"""
- rejected (no LLM): **{result.rejected}**
- prompt_tokens: **{stats.prompt_tokens}**
- completion_tokens: **{stats.completion_tokens}**
- total_tokens: **{stats.total_tokens}**
- Estimated cost: **${stats.estimated_cost:.6f}**
  (input ${INPUT_PRICE_PER_MILLION}/1M · output ${OUTPUT_PRICE_PER_MILLION}/1M)
"""
    )

st.caption(
    f"Keys: `{TASK_STATE_KEY}` · `{INVARIANTS_KEY}` · `{PROFILE_KEY}`"
)
