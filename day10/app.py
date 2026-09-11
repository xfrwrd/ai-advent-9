"""Day 10: Streamlit UI for Sliding Window / Sticky Facts / Branching."""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent import (
    FACTS_PATH,
    HISTORY_PATH,
    INPUT_PRICE_PER_MILLION,
    OUTPUT_PRICE_PER_MILLION,
    STRATEGY_LABELS,
    Agent,
    AgentError,
    Strategy,
    estimate_messages_tokens,
)
from demo_helpers import (
    BRANCH_A_PROMPT,
    BRANCH_B_PROMPT,
    FINAL_QUESTION,
    build_checkpoint_messages,
    load_shared_scenario,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

st.set_page_config(page_title="Day 10 — Стратегии контекста", layout="wide")
st.title("Day 10 — Стратегии контекста")
st.caption("Sliding Window · Sticky Facts · Branching. Без summary/compression.")

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    st.error("DEEPSEEK_API_KEY не задан. Добавьте ключ в .env в корне проекта.")
    st.stop()

if "strategy" not in st.session_state:
    st.session_state.strategy = Strategy.SLIDING_WINDOW.value
if "comparison" not in st.session_state:
    st.session_state.comparison = {
        Strategy.SLIDING_WINDOW.value: None,
        Strategy.STICKY_FACTS.value: None,
        Strategy.BRANCHING.value: None,
    }
if "branch_answers" not in st.session_state:
    st.session_state.branch_answers = {"branch_a": None, "branch_b": None}

strategy_options = list(Strategy)
selected = st.selectbox(
    "Стратегия управления контекстом",
    options=strategy_options,
    format_func=lambda s: STRATEGY_LABELS[s],
    index=strategy_options.index(Strategy(st.session_state.strategy)),
)
st.session_state.strategy = selected.value

agent = Agent(api_key=api_key, strategy=selected)

left, right = st.columns([2, 1])

with right:
    st.subheader("Управление")
    if st.button("Очистить всё"):
        agent.clear_all()
        st.session_state.comparison = {key: None for key in st.session_state.comparison}
        st.session_state.branch_answers = {"branch_a": None, "branch_b": None}
        st.session_state.pop("last_result", None)
        st.rerun()

    if selected == Strategy.STICKY_FACTS and st.button("Очистить facts"):
        agent.clear_facts()
        st.rerun()

    st.subheader("Контекст")
    if selected == Strategy.SLIDING_WINDOW:
        st.markdown(
            f"""
- Window size: **{agent.window_size}**
- Messages stored: **{len(agent.messages)}**
- History tokens (estimate): **~{estimate_messages_tokens(agent.messages)}**
"""
        )
    elif selected == Strategy.STICKY_FACTS:
        filled = sum(1 for value in agent.facts.values() if value)
        st.markdown(
            f"""
- Window size: **{agent.window_size}**
- Messages stored: **{len(agent.messages)}**
- Facts filled: **{filled}/{len(agent.facts)}**
"""
        )
        with st.expander("Показать facts", expanded=True):
            st.json(agent.facts)
    else:
        st.markdown(
            f"""
- Current branch: **{agent.get_current_branch()}**
- Branches: **{', '.join(agent.list_branches())}**
- Messages in branch: **{len(agent.messages)}**
- Checkpoint size: **{len(agent.branches_data.get('checkpoint') or [])}**
"""
        )
        if st.button("Create checkpoint"):
            agent.create_checkpoint()
            st.success("Checkpoint сохранён.")
            st.rerun()

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Create branch_a"):
                try:
                    agent.create_branch("branch_a")
                    st.success("branch_a от checkpoint.")
                except AgentError as exc:
                    st.error(str(exc))
                st.rerun()
        with col_b:
            if st.button("Create branch_b"):
                try:
                    agent.create_branch("branch_b")
                    st.success("branch_b от checkpoint.")
                except AgentError as exc:
                    st.error(str(exc))
                st.rerun()

        branch_name = st.selectbox(
            "Переключить ветку",
            options=agent.list_branches(),
            index=max(agent.list_branches().index(agent.get_current_branch()), 0)
            if agent.get_current_branch() in agent.list_branches()
            else 0,
        )
        if st.button("Switch branch"):
            try:
                agent.switch_branch(branch_name)
                st.success(f"Активна: {branch_name}")
            except AgentError as exc:
                st.error(str(exc))
            st.rerun()

    st.subheader("Demo")
    if selected in (Strategy.SLIDING_WINDOW, Strategy.STICKY_FACTS):
        if st.button("Загрузить единый сценарий ContextFlow"):
            if selected == Strategy.STICKY_FACTS:
                estimated = load_shared_scenario(HISTORY_PATH, FACTS_PATH)
                agent.facts = agent._load_facts()
            else:
                estimated = load_shared_scenario(HISTORY_PATH)
            agent.replace_messages(agent._load_json_list(HISTORY_PATH))
            st.success(
                f"Сценарий загружен (~{estimated} tokens estimate). "
                f"Сейчас в памяти: {len(agent.messages)} messages "
                f"(window={agent.window_size})."
            )
            st.rerun()

        if st.button("Задать финальный вопрос ТЗ"):
            try:
                result = agent.ask(FINAL_QUESTION)
                st.session_state.last_result = result
                st.session_state.comparison[selected.value] = {
                    "answer": result.content,
                    "prompt_tokens": result.token_stats.prompt_tokens,
                    "completion_tokens": result.token_stats.completion_tokens,
                    "total_tokens": result.token_stats.total_tokens,
                    "estimated_cost": result.token_stats.estimated_cost,
                    "messages": len(agent.messages),
                    "facts": dict(agent.facts)
                    if selected == Strategy.STICKY_FACTS
                    else None,
                }
            except AgentError as exc:
                st.error(str(exc))
            st.rerun()
    else:
        if st.button("Подготовить checkpoint (базовые требования)"):
            msgs = build_checkpoint_messages()
            agent.messages = list(msgs)
            agent.branches_data = {
                "checkpoint": list(msgs),
                "current": "main",
                "branches": {"main": list(msgs)},
            }
            agent._save_history()
            agent._save_branches()
            st.success("Checkpoint = базовые требования ContextFlow.")
            st.rerun()

        st.markdown(
            f"""
После checkpoint + create branch_a/branch_b:

1. Switch → **branch_a** → кнопка решения A  
2. Switch → **branch_b** → кнопка решения B  
3. Сравни ответы внизу
"""
        )
        if st.button("В этой ветке: решение A (mobile да)"):
            try:
                agent.ask(BRANCH_A_PROMPT)
                result = agent.ask(
                    "Нужно ли мобильное приложение в первой версии? Ответь кратко почему."
                )
                st.session_state.last_result = result
                st.session_state.branch_answers["branch_a"] = result.content
            except AgentError as exc:
                st.error(str(exc))
            st.rerun()

        if st.button("В этой ветке: решение B (только web)"):
            try:
                agent.ask(BRANCH_B_PROMPT)
                result = agent.ask(
                    "Нужно ли мобильное приложение в первой версии? Ответь кратко почему."
                )
                st.session_state.last_result = result
                st.session_state.branch_answers["branch_b"] = result.content
            except AgentError as exc:
                st.error(str(exc))
            st.rerun()

with left:
    st.subheader(f"Диалог — {STRATEGY_LABELS[selected]}")
    for message in agent.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_message = st.chat_input("Введите запрос...")
    if user_message:
        with st.chat_message("user"):
            st.markdown(user_message)
        with st.chat_message("assistant"):
            try:
                result = agent.ask(user_message)
                st.markdown(result.content)
                st.session_state.last_result = result
            except AgentError as exc:
                st.error(str(exc))

if "last_result" in st.session_state and st.session_state.last_result is not None:
    result = st.session_state.last_result
    stats = result.token_stats
    st.divider()
    st.subheader(f"Токены — {STRATEGY_LABELS[result.strategy]}")
    st.markdown(
        f"""
- prompt_tokens: **{stats.prompt_tokens}**
- completion_tokens: **{stats.completion_tokens}**
- total_tokens: **{stats.total_tokens}**
- Estimated cost (off-peak cache-miss): **${stats.estimated_cost:.6f}**
  (input ${INPUT_PRICE_PER_MILLION}/1M · output ${OUTPUT_PRICE_PER_MILLION}/1M)
- context estimate: **~{stats.context_tokens_estimate}**
"""
    )

st.divider()
st.subheader("Cumulative по стратегиям")
cum_rows = []
for strategy in Strategy:
    bucket = agent.cumulative.by_strategy.get(strategy.value, {})
    cum_rows.append(
        {
            "strategy": STRATEGY_LABELS[strategy],
            "cumulative_tokens": bucket.get("cumulative_tokens", 0),
            "cumulative_cost": f"${float(bucket.get('cumulative_cost', 0.0)):.6f}",
        }
    )
st.dataframe(cum_rows, hide_index=True, use_container_width=True)

st.divider()
st.subheader("Сравнение (ручное)")
st.caption("Quality / Facts retained / Ease of use — заполни вручную после теста.")
comp_rows = []
for strategy in Strategy:
    data = st.session_state.comparison.get(strategy.value)
    comp_rows.append(
        {
            "Strategy": STRATEGY_LABELS[strategy],
            "Quality": "(ручная оценка)",
            "Important facts retained": "(ручная оценка)",
            "Prompt tokens": data["prompt_tokens"] if data else "—",
            "Ease of use": "(ручная оценка)",
        }
    )
st.dataframe(comp_rows, hide_index=True, use_container_width=True)

for strategy in Strategy:
    data = st.session_state.comparison.get(strategy.value)
    if data and data.get("answer"):
        with st.expander(f"Ответ — {STRATEGY_LABELS[strategy]}"):
            st.markdown(data["answer"])
            if data.get("facts"):
                st.json(data["facts"])

if any(st.session_state.branch_answers.values()):
    st.subheader("Branching: ответы веток")
    col_ba, col_bb = st.columns(2)
    with col_ba:
        st.markdown("**branch_a**")
        st.write(st.session_state.branch_answers.get("branch_a") or "—")
    with col_bb:
        st.markdown("**branch_b**")
        st.write(st.session_state.branch_answers.get("branch_b") or "—")
