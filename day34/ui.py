"""Day 34 page. The agent prepares diffs. Only the «Применить» button writes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from day34.agent import run_agent
from day34.llm import AgentLLM
from day34.mcp_client import call_tool_sync
from day34.report import record_apply, save_run

GOALS = (
    "Найди все места использования LocalLLMClient и объясни, какие модули от него зависят.",
    "Проверь документацию day34/demo/README.md и обнови её по фактическому коду day34/demo.",
    "Проверь, какие файлы и каталоги по day31/README.md не индексируются, сверь с кодом day31 и обнови README.",
    "Проверь документацию Day 31 (day31/README.md) и обнови её по фактическому коду day31.",
    "Подготовь короткий changelog Day 34 в новом файле day34/CHANGELOG.md по коду папки day34.",
)

st.set_page_config(page_title="Day 34 — файловый ассистент", layout="wide")
st.title("Day 34 — ассистент для файлов проекта")
st.caption(
    "Модель сама выбирает инструменты: список файлов, поиск, чтение, подготовка правки. "
    "Записывает файл только кнопка «Применить» после просмотра diff."
)

choice = st.selectbox("Пример задачи", GOALS)
goal = st.text_area("Задача", value=choice, height=80)

if st.button("Запустить агента", type="primary"):
    status = st.status("Агент работает…", expanded=True)

    def _live(step: dict) -> None:
        if step.get("kind") == "tool":
            result = step.get("result") or {}
            mark = result.get("error") or "ok"
            status.write(f"Шаг {step['step']}: `{step['tool']}` {json.dumps(step.get('args'), ensure_ascii=False)[:160]} → {mark}")
        elif step.get("kind") == "final":
            status.write(f"Шаг {step['step']}: финальный ответ")
        else:
            status.write(f"Шаг {step['step']}: {step.get('kind')} — {step.get('error')}")

    run = run_agent(goal, llm=AgentLLM(), on_step=_live)
    folder = save_run(run)
    status.update(label=f"Готово: {run.stop_reason}, {run.seconds} с", state="complete", expanded=False)
    st.session_state["day34_run"] = run
    st.session_state["day34_report"] = str(folder / "report.md")
    st.session_state["day34_applied"] = {}

run = st.session_state.get("day34_run")
if run is not None:
    st.subheader("Ответ ассистента")
    st.markdown(run.answer)
    if run.stop_reason != "final":
        st.warning(f"Остановка: {run.stop_reason}")
    for error in run.errors:
        st.error(error)

    left, right = st.columns(2)
    with left:
        st.markdown("**Прочитанные файлы**")
        if run.files_read:
            for path in run.files_read:
                st.write(f"`{path}`")
        else:
            st.write("нет")
    with right:
        st.markdown("**Инструменты по порядку**")
        st.write(" → ".join(run.tools_used()) or "нет")

    st.markdown("**История шагов**")
    for step in run.steps:
        title = f"Шаг {step['step']}: {step.get('tool') or step.get('kind')}"
        if step.get("kind") == "tool" and (step.get("result") or {}).get("error"):
            title += f" — {step['result']['error']}"
        with st.expander(title):
            if step.get("args") is not None:
                st.json(step.get("args"))
            if step.get("result") is not None:
                shown = dict(step["result"])
                shown.pop("diff_sha256", None)
                st.json(shown)
            if step.get("error"):
                st.warning(step["error"])
            if step.get("raw"):
                st.code(step["raw"])

    st.subheader("Подготовленные изменения")
    applied: dict = st.session_state.setdefault("day34_applied", {})
    if not run.proposals:
        st.write("Изменений нет. Кнопка «Применить» появляется только при подготовленном diff.")
    for item in run.proposals:
        pid = item["proposal_id"]
        st.markdown(f"`{item['path']}` · {item['mode']} · +{item['lines_added']} −{item['lines_removed']}")
        if item.get("reason"):
            st.caption(item["reason"])
        st.code(item["diff"], language="diff")
        outcome = applied.get(pid)
        if outcome is None:
            apply_col, cancel_col = st.columns(2)
            if apply_col.button("Применить", key=f"apply-{pid}", type="primary"):
                outcome = call_tool_sync(
                    "changes",
                    "apply_file_change",
                    {"proposal_id": pid, "confirm_token": item["diff_sha256"]},
                )
            elif cancel_col.button("Отменить", key=f"cancel-{pid}"):
                outcome = call_tool_sync("changes", "cancel_file_change", {"proposal_id": pid})
            if outcome is not None:
                applied[pid] = outcome
                record_apply(run, {"proposal_id": pid, **outcome})
                st.rerun()
        else:
            if outcome.get("error"):
                st.error(outcome.get("message") or outcome["error"])
            elif outcome.get("status") == "cancelled":
                st.info("Изменение отменено, файл не тронут.")
            else:
                st.success(
                    f"Сохранено: {outcome.get('path')}, проверка повторным чтением: "
                    f"{'совпадает' if outcome.get('verified') else outcome.get('status')}"
                )

    st.caption(f"Отчёт: {st.session_state.get('day34_report')}")
