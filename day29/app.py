"""Day 28 page, plus a Baseline / Optimized switch for the local model."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from day28.agent import CLOUD, TimedRAG, load_day28
from day29.answer import ModeAnswer, answer_with, config_for
from day29.config import BASELINE, OPTIMIZED
from day29.questions import DEMO_PHRASES, SCORE_FIELDS
from day29.retrieve import load_frozen_index, retrieve

st.set_page_config(page_title="Day 29 — оптимизация локальной LLM", layout="wide")
st.title("Day 29 — оптимизация локальной LLM")
st.caption(
    "Local сравнивает Baseline Day 28 и Optimized Day 29 на одних и тех же фрагментах. "
    "Cloud по-прежнему отвечает через DeepSeek."
)

MODE_LABELS = {BASELINE: "Baseline", OPTIMIZED: "Optimized"}
SCORE_LABELS = {
    "factual_correctness": "Фактическая правильность",
    "source_faithfulness": "Соответствие источникам",
    "completeness": "Полнота",
    "instruction_following": "Следование инструкциям",
    "correct_refusal": "Корректный отказ",
}


@st.cache_resource
def get_cloud():
    return load_day28()


@st.cache_resource
def get_index():
    return load_frozen_index()


def _show_local(result: ModeAnswer) -> None:
    response = result.response
    st.write(response.answer)
    if response.insufficient_context:
        st.warning("Релевантных фрагментов нет. Генерация не вызывалась.")
    speed = "—"
    if result.generation is not None and result.generation.tokens_per_sec is not None:
        speed = f"{result.generation.tokens_per_sec:.1f}"
    st.markdown(
        f"**Retrieval** {result.retrieval_s:.2f} s · "
        f"**Generation** {result.generation_s:.2f} s · "
        f"**Total** {result.total_s:.2f} s · "
        f"**tokens/s** {speed}"
    )
    if response.sources:
        st.markdown("**Sources**")
        for source in response.sources:
            st.markdown(f"- `{source.source}` / {source.section}")
    for field in SCORE_FIELDS:
        st.slider(
            SCORE_LABELS[field],
            min_value=1,
            max_value=5,
            value=3,
            key=f"day29_{result.config.name}_{field}",
        )


def _show_cloud(result: TimedRAG) -> None:
    response = result.response
    st.subheader("Cloud")
    st.write(response.answer)
    st.markdown(
        f"**Retrieval** {result.retrieval_s:.2f} s · "
        f"**Generation** {result.generation_s:.2f} s · "
        f"**Total** {result.total_s:.2f} s"
    )
    if response.sources:
        st.markdown("**Sources**")
        for source in response.sources:
            st.markdown(f"- `{source.source}` / {source.section}")


if "day29_local" not in st.session_state:
    st.session_state["day29_local"] = {}
if "day29_context" not in st.session_state:
    st.session_state["day29_context"] = {}
if "day29_cloud" not in st.session_state:
    st.session_state["day29_cloud"] = None

provider = st.radio("Модель", ["Local", "Cloud"], horizontal=True)
mode = BASELINE
if provider == "Local":
    mode_label = st.radio("Конфигурация", [MODE_LABELS[BASELINE], MODE_LABELS[OPTIMIZED]], horizontal=True)
    mode = BASELINE if mode_label == MODE_LABELS[BASELINE] else OPTIMIZED
    config = config_for(mode)
    st.markdown(
        f"**Параметры {MODE_LABELS[mode]}:** model `{config.model}`, "
        f"think `{config.think}`, temperature `{config.temperature if config.temperature is not None else 'не передаётся, в Modelfile 0.6'}`, "
        f"num_predict `{config.num_predict if config.num_predict is not None else 'не передаётся'}`, "
        f"num_ctx `{config.num_ctx if config.num_ctx is not None else 'не передаётся'}`, "
        f"prompt `{config.prompt_name}`."
    )
else:
    st.markdown("**Cloud:** DeepSeek через Day 28. Параметры Ollama к этому ответу не применяются.")

phrase = st.selectbox("Фраза", DEMO_PHRASES)
queued = phrase.split(". ", 1)[1] if st.button("Спросить") else None
typed = st.chat_input("Вопрос")
prompt = queued or typed
if prompt:
    with st.spinner("Ищу фрагменты и отвечаю…"):
        try:
            if provider == "Cloud":
                st.session_state["day29_cloud"] = get_cloud().ask(prompt, CLOUD)
            else:
                context = st.session_state["day29_context"].get(prompt)
                if context is None:
                    context = retrieve(prompt, get_index())
                    st.session_state["day29_context"][prompt] = context
                st.session_state["day29_local"][mode] = answer_with(context, config_for(mode))
        except Exception as exc:  # noqa: BLE001 — show Ollama or DeepSeek errors on the page
            st.error(str(exc))

if provider == "Local":
    columns = st.columns(2)
    for column, name in zip(columns, (BASELINE, OPTIMIZED)):
        with column:
            st.subheader(MODE_LABELS[name])
            result = st.session_state["day29_local"].get(name)
            if result is None:
                st.write("Этот вариант ещё не запускали.")
                continue
            _show_local(result)
else:
    cloud = st.session_state["day29_cloud"]
    if cloud is None:
        st.write("Cloud ещё не запускали.")
    else:
        _show_cloud(cloud)
