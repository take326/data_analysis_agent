from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

import base64
import uuid

import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage

from src.agent.graph import create_graph
from src.agent.models import ExecResult, ReasonDecision, ReportOutput


st.set_page_config(page_title="Data Analysis AI Agent", layout="wide")


def _init_session():
    if "app" not in st.session_state:
        st.session_state.app = create_graph()
    if "state" not in st.session_state:
        st.session_state.state = None


def _reset():
    st.session_state.state = None


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
# メッセージ履歴を表示
for msg in state.get("messages", []):
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    else:
        with st.chat_message("assistant"):
            st.write(msg.content)

# 実行コードを表示
if state.get("last_code"):
    with st.chat_message("assistant"):
        with st.expander("📝 実行コード", expanded=False):
            st.code(state["last_code"], language="python")

# レポートを表示
if state.get("report"):
    with st.chat_message("assistant"):
        _render_report(state["report"])

# 処理中：スピナーを表示しながら実行
if st.session_state.processing:
    with st.chat_message("assistant"):
        with st.spinner("分析中..."):
            out = st.session_state.app.invoke(st.session_state.state)
            st.session_state.state = out
    st.session_state.processing = False
    st.rerun()

# チャット入力
user_text = st.chat_input("分析内容を入力...")
if user_text:
    st.session_state.state["messages"] = list(st.session_state.state["messages"]) + [
        HumanMessage(content=user_text)
    ]
    st.session_state.processing = True
    st.rerun()



