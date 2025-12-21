from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

import base64
import uuid

import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.graph import create_graph
from src.agent.models import ExecResult, ReasonDecision, ReportOutput


st.set_page_config(page_title="Data Analysis AI Agent", layout="wide")


def _init_session():
    if "app" not in st.session_state:
        st.session_state.app = create_graph()
    if "state" not in st.session_state:
        st.session_state.state = None
    # ChatGPT風に表示するチャット履歴（表示用。LLM用の state["messages"] とは分離）
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def _reset():
    st.session_state.state = None
    st.session_state.chat_history = []


def _render_report(report: dict):
    ro = ReportOutput.model_validate(report)
    st.markdown("**📊 Report Summary**")
    st.markdown(ro.summary)

    if ro.table_markdown:
        st.markdown("**📋 Tables**")
        for t in ro.table_markdown:
            st.markdown(t)

    if ro.plot_png_base64:
        st.markdown("**📈 Plots**")
        for b64 in ro.plot_png_base64:
            st.image(base64.b64decode(b64))

    if ro.json:
        st.markdown("**🔧 JSON**")
        for j in ro.json:
            st.json(j)


_init_session()

with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if st.button("Reset session"):
        _reset()
    st.caption("env: OPENAI_API_KEY / OPENAI_MODEL")

if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

df = pd.read_csv(uploaded)
st.subheader("Data Preview")
with st.expander(f"Data Preview ({df.shape[0]} rows × {df.shape[1]} columns)", expanded=False):
    st.dataframe(df, use_container_width=True)

if "processing" not in st.session_state:
    st.session_state.processing = False

if st.session_state.state is None:
    st.session_state.state = {
        "messages": [],
        "df": df,
        "decision": None,
        "last_code": None,
        "last_exec": None,
        "report": None,
    }
else:
    # dfは常に最新アップロードを優先（単一CSV前提）
    st.session_state.state["df"] = df

state = st.session_state.state

st.subheader("Chat")
# ChatGPT風: chat_history を時系列で表示（LLM用messagesとは分離）
for e in st.session_state.get("chat_history", []):
    etype = e.get("type")
    if etype == "user":
        with st.chat_message("user"):
            st.write(e.get("text", ""))
    elif etype == "assistant":
        with st.chat_message("assistant"):
            st.write(e.get("text", ""))
    elif etype == "code":
        with st.chat_message("assistant"):
            with st.expander("📝 実行コード", expanded=False):
                st.code(e.get("code", ""), language="python")
    elif etype == "report":
        with st.chat_message("assistant"):
            _render_report(e["report"])

 # レポート表示は chat_history 側に一本化（時系列の中に残す）

# 処理中：スピナーを表示しながら実行
if st.session_state.processing:
    with st.chat_message("assistant"):
        with st.spinner("分析中..."):
            prev_report = st.session_state.state.get("report")
            prev_last_code = st.session_state.state.get("last_code")
            prev_messages = list(st.session_state.state.get("messages", []))
            agent_result = st.session_state.app.invoke(st.session_state.state)
            st.session_state.state = agent_result

            # run_code で生成されたコードを履歴に積む（同一内容の重複は避ける）
            new_last_code = agent_result.get("last_code")
            if new_last_code and new_last_code != prev_last_code:
                last = st.session_state.chat_history[-1] if st.session_state.chat_history else None
                if not (last and last.get("type") == "code" and last.get("code") == new_last_code):
                    st.session_state.chat_history.append({"type": "code", "code": new_last_code})

            # ask_clarification 等で増えたAIMessageを chat_history に積む（report_summaryタグは除外）
            new_messages = list(agent_result.get("messages", []))
            if len(new_messages) > len(prev_messages):
                for m in new_messages[len(prev_messages) :]:
                    if isinstance(m, AIMessage) and m.additional_kwargs.get("source") == "report_summary":
                        continue
                    if isinstance(m, AIMessage):
                        st.session_state.chat_history.append({"type": "assistant", "text": m.content})

            new_report = agent_result.get("report")
            if new_report and new_report != prev_report:
                st.session_state.chat_history.append({"type": "report", "report": new_report})
    st.session_state.processing = False
    st.rerun()

# チャット入力
user_text = st.chat_input("分析内容を入力...")
if user_text:
    # 表示用の履歴に積む（ChatGPT風）
    st.session_state.chat_history.append({"type": "user", "text": user_text})
    st.session_state.state["messages"] = list(st.session_state.state["messages"]) + [
        HumanMessage(content=user_text)
    ]
    st.session_state.processing = True
    st.rerun()



