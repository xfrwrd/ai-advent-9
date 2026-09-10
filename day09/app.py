"""Day 09: Streamlit UI — summary compression vs full history."""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent import (
    COMPRESSION_THRESHOLD,
    HISTORY_PATH,
    INPUT_PRICE_PER_MILLION,
    OUTPUT_PRICE_PER_MILLION,
    RECENT_MESSAGES_LIMIT,
    SUMMARY_PATH,
    Agent,
    AgentError,
    estimate_messages_tokens,
    estimate_tokens,
)
from demo_helpers import RECALL_QUESTION, load_contextflow_demo

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

st.set_page_config(page_title="Day 9 — Compression / Summary", layout="centered")
st.title("Day 9 — Compression / Summary")
st.caption(
    "UI → Agent → DeepSeek API. "
    "Старые сообщения сжимаются в summary, recent history остаётся как есть."
)

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    st.error("DEEPSEEK_API_KEY не задан. Добавьте ключ в .env в корне проекта.")
    st.stop()

if "compression_enabled" not in st.session_state:
    st.session_state.compression_enabled = True
if "comparison" not in st.session_state:
    st.session_state.comparison = {"off": None, "on": None}

compression_enabled = st.toggle(
    "Compression ON / OFF",
    value=st.session_state.compression_enabled,
    help="OFF = полная history. ON = summary + recent messages.",
)
st.session_state.compression_enabled = compression_enabled

agent = Agent(
    api_key=api_key,
    model="deepseek-v4-flash",
    compression_enabled=compression_enabled,
)

col_clear, col_compress = st.columns(2)
with col_clear:
    if st.button("Очистить историю"):
        agent.clear_history()
        st.session_state.comparison = {"off": None, "on": None}
        st.session_state.pop("last_token_stats", None)
        st.session_state.pop("last_answer", None)
        st.rerun()

with col_compress:
    if st.button("Сжать сейчас", disabled=not compression_enabled):
        try:
            did = agent.maybe_compress()
            if did:
                st.success("Compression выполнен: старые сообщения → summary.")
            else:
                st.info(
                    f"Сжимать пока нечего "
                    f"(messages ≤ {COMPRESSION_THRESHOLD} или ≤ {RECENT_MESSAGES_LIMIT})."
                )
        except AgentError as exc:
            st.error(str(exc))
        st.rerun()

st.subheader("Контекст")
summary_tokens = estimate_tokens(agent.summary)
context_tokens = agent.estimate_context_tokens()
st.markdown(
    f"""
- Compression: **{"ON" if agent.compression_enabled else "OFF"}**
- Messages in recent history: **{len(agent.messages)}**
- Summary exists: **{"yes" if bool(agent.summary.strip()) else "no"}**
- Summary tokens: **~{summary_tokens}** *(estimate)*
- Current context tokens: **~{context_tokens}** *(estimate)*
- Threshold / recent limit: **{COMPRESSION_THRESHOLD} / {RECENT_MESSAGES_LIMIT}**
"""
)

with st.expander("Показать summary"):
    if agent.summary.strip():
        st.markdown(agent.summary)
    else:
        st.caption("Summary пока пустой.")

with st.expander("Тест ContextFlow (сравнение ON/OFF)"):
    st.markdown(
        f"""
1. Нажми **Загрузить тестовый диалог** — локально пишется длинная history
   с фактами (ContextFlow / Go / Postgres / AI-планировщик), без API.
2. Выставь **Compression OFF** → **Задать контрольный вопрос** → сохранится результат OFF.
3. Снова загрузи тестовый диалог.
4. Выставь **Compression ON** → **Сжать сейчас** → **Задать контрольный вопрос**.
5. Сравни ответы и `prompt_tokens` внизу.

Контрольный вопрос:
`{RECALL_QUESTION}`
"""
    )

    if st.button("Загрузить тестовый диалог"):
        estimated = load_contextflow_demo(HISTORY_PATH, SUMMARY_PATH)
        st.session_state.pop("last_token_stats", None)
        st.session_state.pop("last_answer", None)
        st.success(f"Диалог загружен. Оценка полной history: ~{estimated} tokens.")
        st.rerun()

    if st.button("Задать контрольный вопрос"):
        try:
            result = agent.ask(RECALL_QUESTION)
            st.session_state.last_answer = result.content
            st.session_state.last_token_stats = result.token_stats
            key = "on" if agent.compression_enabled else "off"
            st.session_state.comparison[key] = {
                "answer": result.content,
                "prompt_tokens": result.token_stats.prompt_tokens,
                "completion_tokens": result.token_stats.completion_tokens,
                "total_tokens": result.token_stats.total_tokens,
                "estimated_cost": result.token_stats.estimated_cost,
                "context_estimate": result.token_stats.context_tokens_estimate,
                "summary_tokens": result.token_stats.summary_tokens,
                "history_len": len(agent.messages),
                "did_compress": result.did_compress,
            }
        except AgentError as exc:
            st.error(str(exc))
        st.rerun()

st.subheader("Диалог (recent history)")
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
            st.session_state.last_answer = result.content
            st.session_state.last_token_stats = result.token_stats
            if result.did_compress:
                st.info("Во время этого хода сработала compression.")
        except AgentError as exc:
            st.error(str(exc))

if "last_token_stats" in st.session_state:
    stats = st.session_state.last_token_stats
    st.divider()
    st.subheader("Токены последнего запроса")
    st.caption(
        "history / summary / context — estimate. "
        "prompt / completion / total — фактический usage DeepSeek."
    )
    st.markdown(
        f"""
- Текущий запрос: **{stats.current_message_tokens}** tokens *(estimate)*
- Размер history (до запроса): **{stats.history_tokens}** tokens *(estimate)*
- Summary tokens: **~{stats.summary_tokens}** *(estimate)*
- Current context tokens: **~{stats.context_tokens_estimate}** *(estimate)*
- Input API (prompt_tokens): **{stats.prompt_tokens}** *(usage)*
- Ответ: **{stats.completion_tokens}** *(usage)*
- Всего: **{stats.total_tokens}** *(usage)*
- Estimated cost (cache miss, off-peak): **${stats.estimated_cost:.6f}**
  (input ${INPUT_PRICE_PER_MILLION}/1M · output ${OUTPUT_PRICE_PER_MILLION}/1M)
"""
    )
    st.caption("Оценка стоимости по off-peak cache-miss тарифу")

st.divider()
st.subheader("Накопительная статистика")
st.markdown(
    f"""
- Cumulative tokens: **{agent.cumulative.cumulative_tokens}**
- Cumulative cost: **${agent.cumulative.cumulative_cost:.6f}**
- Messages on disk: **{len(agent.messages)}**
- History estimate: **~{estimate_messages_tokens(agent.messages)}** tokens
"""
)

comparison = st.session_state.comparison
if comparison["off"] or comparison["on"]:
    st.divider()
    st.subheader("Сравнение Compression OFF vs ON")
    col_off, col_on = st.columns(2)
    with col_off:
        st.markdown("**OFF (полная history)**")
        off = comparison["off"]
        if off:
            st.markdown(
                f"""
- prompt_tokens: **{off['prompt_tokens']}**
- context estimate: **~{off['context_estimate']}**
- summary tokens: **~{off['summary_tokens']}**
- history len after ask: **{off['history_len']}**
- cost: **${off['estimated_cost']:.6f}**
"""
            )
            with st.expander("Ответ OFF"):
                st.markdown(off["answer"])
        else:
            st.caption("Ещё нет результата OFF.")

    with col_on:
        st.markdown("**ON (summary + recent)**")
        on = comparison["on"]
        if on:
            st.markdown(
                f"""
- prompt_tokens: **{on['prompt_tokens']}**
- context estimate: **~{on['context_estimate']}**
- summary tokens: **~{on['summary_tokens']}**
- history len after ask: **{on['history_len']}**
- cost: **${on['estimated_cost']:.6f}**
- did_compress: **{on['did_compress']}**
"""
            )
            with st.expander("Ответ ON"):
                st.markdown(on["answer"])
        else:
            st.caption("Ещё нет результата ON.")

    if comparison["off"] and comparison["on"]:
        delta = comparison["off"]["prompt_tokens"] - comparison["on"]["prompt_tokens"]
        st.success(
            f"Разница prompt_tokens (OFF − ON): **{delta}**. "
            "ON обычно меньше, если summary короче полной старой history."
        )
