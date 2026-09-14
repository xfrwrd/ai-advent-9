"""Day 11: Streamlit UI — explicit Short / Working / Long-Term memory."""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent import (
    INPUT_PRICE_PER_MILLION,
    OUTPUT_PRICE_PER_MILLION,
    TEST_QUESTION,
    Agent,
    AgentError,
)
from memory import LAYER_LABELS, MemoryLayer

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

st.set_page_config(page_title="Day 11 — Explicit Memory Layers", layout="wide")
st.title("Day 11 — Explicit Memory Layers")
st.caption(
    "Short-Term · Working · Long-Term. Без summary как основной памяти, "
    "без авто-extraction."
)

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    st.error("DEEPSEEK_API_KEY не задан. Добавьте ключ в .env в корне проекта.")
    st.stop()

agent = Agent(api_key=api_key)
memory = agent.memory

left, right = st.columns([2, 1])

with right:
    st.subheader("Явно запомнить")
    st.caption("Данные пишутся только в выбранный слой — без fan-out.")
    layer = st.selectbox(
        "Слой памяти",
        options=list(MemoryLayer),
        format_func=lambda layer_item: LAYER_LABELS[layer_item],
        key="remember_layer",
    )
    remember_key = st.text_input("Key", key="remember_key")
    remember_value = st.text_area("Value", key="remember_value", height=80)
    if st.button("Save to selected layer", type="primary"):
        try:
            if layer == MemoryLayer.SHORT_TERM:
                if not remember_value.strip():
                    st.error("Для Short-Term нужен Value (текст сообщения).")
                else:
                    agent.remember(layer, remember_key, remember_value.strip())
                    st.success("Добавлено в Short-Term как user-сообщение.")
                    st.rerun()
            else:
                if not remember_key.strip():
                    st.error("Для Working / Long-Term нужен Key.")
                else:
                    agent.remember(layer, remember_key.strip(), remember_value)
                    st.success(f"Сохранено в {LAYER_LABELS[layer]}.")
                    st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    st.divider()
    st.subheader("Test scenario")
    st.markdown(
        """
1. Загрузить тест (LT / WM / ST)  
2. Задать контрольный вопрос  
3. Clear Short-Term → повторить вопрос  
4. Clear Working → останется только Long-Term
"""
    )
    if st.button("1. Загрузить test scenario"):
        agent.load_test_scenario()
        st.session_state.pop("last_result", None)
        st.success(
            "LONG TERM: name=Alex · WORKING: backend=Go, current_task=API design · "
            "SHORT TERM: «Сейчас обсуждаем retry.»"
        )
        st.rerun()

    if st.button("2. Задать контрольный вопрос"):
        try:
            result = agent.ask(TEST_QUESTION)
            st.session_state.last_result = result
        except AgentError as exc:
            st.error(str(exc))
        st.rerun()

    if st.button("3. Clear Short-Term и повторить вопрос"):
        agent.clear_short_term()
        try:
            result = agent.ask(TEST_QUESTION)
            st.session_state.last_result = result
        except AgentError as exc:
            st.error(str(exc))
        st.rerun()

    if st.button("4. Clear Working Memory"):
        agent.clear_working_memory()
        st.warning("Working очищен — остаётся Long-Term (+ текущий short-term диалог).")
        st.rerun()

    if st.button("Очистить всё"):
        agent.clear_all()
        st.session_state.pop("last_result", None)
        st.rerun()

with left:
    st.subheader("Диалог (Short-Term)")
    for message in memory.short_term:
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

st.divider()
st.subheader("Слои памяти")

col_st, col_wm, col_lt = st.columns(3)

with col_st:
    with st.expander("Short-Term Memory", expanded=True):
        sizes = memory.layer_size_estimates()
        st.caption(f"Messages: {len(memory.short_term)} · ~{sizes['short_term']} tokens")
        if memory.short_term:
            st.json(memory.short_term)
        else:
            st.write("(empty)")
        if st.button("Clear conversation", key="clear_st"):
            agent.clear_short_term()
            st.rerun()

with col_wm:
    with st.expander("Working Memory", expanded=True):
        sizes = memory.layer_size_estimates()
        st.caption(f"Keys: {len(memory.working)} · ~{sizes['working']} tokens")
        wm_key = st.text_input("Key", key="wm_key")
        wm_value = st.text_input("Value", key="wm_value")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save", key="wm_save"):
                try:
                    agent.set_working_memory(wm_key, wm_value)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        with c2:
            if st.button("Clear", key="wm_clear"):
                agent.clear_working_memory()
                st.rerun()
        if memory.working:
            st.json(memory.working)
            remove_key = st.selectbox(
                "Remove key",
                options=[""] + list(memory.working.keys()),
                key="wm_remove",
            )
            if remove_key and st.button("Remove selected", key="wm_remove_btn"):
                agent.remove_working_memory(remove_key)
                st.rerun()
        else:
            st.write("(empty)")

with col_lt:
    with st.expander("Long-Term Memory", expanded=True):
        sizes = memory.layer_size_estimates()
        st.caption(f"Keys: {len(memory.long_term)} · ~{sizes['long_term']} tokens")
        lt_key = st.text_input("Key", key="lt_key")
        lt_value = st.text_input("Value", key="lt_value")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save", key="lt_save"):
                try:
                    agent.set_long_term_memory(lt_key, lt_value)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        with c2:
            if st.button("Clear", key="lt_clear"):
                agent.clear_long_term_memory()
                st.rerun()
        if memory.long_term:
            st.json(memory.long_term)
            remove_key = st.selectbox(
                "Remove key",
                options=[""] + list(memory.long_term.keys()),
                key="lt_remove",
            )
            if remove_key and st.button("Remove selected", key="lt_remove_btn"):
                agent.remove_long_term_memory(remove_key)
                st.rerun()
        else:
            st.write("(empty)")

st.divider()
st.subheader("Что сейчас попадёт в prompt")
st.caption(
    "System: Long-term + Working (отдельные блоки). "
    "Recent dialogue = Short-Term как chat messages."
)
st.code(agent.preview_prompt(), language="text")

with st.expander("API messages (как уйдёт в LLM)"):
    st.json(agent.build_api_messages())

sizes = memory.layer_size_estimates()
st.markdown(
    f"""
**Размер слоёв (оценка tokens):**  
Short-Term ~{sizes['short_term']} · Working ~{sizes['working']} · Long-Term ~{sizes['long_term']}
"""
)

if "last_result" in st.session_state and st.session_state.last_result is not None:
    result = st.session_state.last_result
    stats = result.token_stats
    st.divider()
    st.subheader("Token statistics")
    st.markdown(
        f"""
- prompt_tokens: **{stats.prompt_tokens}**
- completion_tokens: **{stats.completion_tokens}**
- total_tokens: **{stats.total_tokens}**
- Estimated cost (off-peak cache-miss): **${stats.estimated_cost:.6f}**
  (input ${INPUT_PRICE_PER_MILLION}/1M · output ${OUTPUT_PRICE_PER_MILLION}/1M)
- context estimate: **~{stats.context_tokens_estimate}**
- layer sizes: Short-Term ~{stats.short_term_tokens} ·
  Working ~{stats.working_tokens} · Long-Term ~{stats.long_term_tokens}
"""
    )

st.caption(
    f"Cumulative: {agent.cumulative.cumulative_tokens} tokens · "
    f"${agent.cumulative.cumulative_cost:.6f}"
)
