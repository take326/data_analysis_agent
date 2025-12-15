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
    st.subheader("Report Summary")
    st.markdown(ro.summary)

    if ro.table_markdown:
        st.subheader("Tables")
        for t in ro.table_markdown:
            st.markdown(t)

    if ro.plot_png_base64:
        st.subheader("Plots")
        for b64 in ro.plot_png_base64:
            st.image(base64.b64decode(b64))

    if ro.json:
        st.subheader("JSON")
        for j in ro.json:
            st.json(j)


_init_session()

st.title("Data Analysis AI Agent (LangGraph prototype)")

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
st.write("Preview")
st.dataframe(df.head(20), use_container_width=True)

st.divider()
st.subheader("Chat")

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

user_text = st.chat_input("分析したいことを入力してください（例：クラスタリングして特徴を要約して）")
if user_text:
    st.session_state.state["messages"] = list(st.session_state.state["messages"]) + [
        HumanMessage(content=user_text)
    ]

    # Invoke the graph until END
    out = st.session_state.app.invoke(st.session_state.state)
    st.session_state.state = out

state = st.session_state.state

# Render latest decision / outputs
if state.get("decision"):
    decision = ReasonDecision.model_validate(state["decision"])
    st.caption(f"Decision: {decision.action}")
    if decision.action == "ask_clarification":
        st.info(decision.clarification_question or "確認したい点があります。")

if state.get("last_exec"):
    last_exec = ExecResult.model_validate(state["last_exec"])
    with st.expander("Last exec (stdout/stderr)"):
        st.text_area("stdout", last_exec.stdout, height=150)
        st.text_area("stderr", last_exec.stderr, height=150)

if state.get("report"):
    _render_report(state["report"])


