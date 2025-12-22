from __future__ import annotations

import os
from typing import Any, Sequence

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..models import ExecResult, MemoryCategory, ReasonDecision
from ..state import AgentState


def _df_schema_hint(df: pd.DataFrame, max_cols: int = 80) -> str:
    cols = list(df.columns)[:max_cols]
    dtypes = df.dtypes.astype(str).to_dict()
    parts = [f"rows={len(df)}, cols={len(df.columns)}"]
    parts.append("columns:")
    for c in cols:
        parts.append(f"- {c}: {dtypes.get(c)}")
    if len(df.columns) > max_cols:
        parts.append(f"... and {len(df.columns) - max_cols} more columns")
    return "\n".join(parts)


def reason_node(state: AgentState) -> dict:
    """
    Reasonノード（LLM）。
    - messages / df / last_exec / last_code を元に action を選ぶ
    - ask_clarification の場合は質問をmessagesに追加して終了させる
    """
    load_dotenv()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0).with_structured_output(ReasonDecision)

    df = state["df"]
    last_exec = ExecResult.model_validate(state["last_exec"]) if state.get("last_exec") else None
    last_code = state.get("last_code")

    sys = SystemMessage(
        content=(
            "You are the Reasoning node of a data analysis agent.\n"
            "You must output a JSON object matching the ReasonDecision schema.\n"
            "Allowed actions: ask_clarification, run_code, report.\n"
            "- If user request is unclear OR last_exec failed because more user info is needed, choose ask_clarification "
            "and provide clarification_question.\n"
            "- If the next step is to run/repair analysis code, choose run_code and provide analysis_instruction.\n"
            "- If results are sufficient, choose report and (preferably) provide analysis_instruction describing what was done.\n"
            "Do NOT include markdown code fences.\n"
        )
    )

    hint = _df_schema_hint(df)
    context_msgs: list[BaseMessage] = [sys]

    # ユーザーメモリをコンテキストに追加（state経由）
    memories = state.get("memories") or []
    if memories:
        memory_lines = [f"- [{m['category']}] {m['content']}" for m in memories]
        memory_text = "\n".join(memory_lines)
        context_msgs.append(
            SystemMessage(content=f"User Memory (learned preferences from past sessions):\n{memory_text}")
        )

    context_msgs.extend(list(state["messages"]))
    context_msgs.append(SystemMessage(content=f"Data schema hint:\n{hint}"))

    if last_exec is not None:
        context_msgs.append(SystemMessage(content=f"Last exec result:\n{last_exec.model_dump()}"))
    if last_code:
        context_msgs.append(SystemMessage(content=f"Last generated code (for debugging/fix):\n{last_code}"))

    decision = llm.invoke(context_msgs)
    patch: dict[str, Any] = {"decision": decision.model_dump()}

    if decision.action == "ask_clarification":
        q = decision.clarification_question or "追加で確認したい点があります。目的や対象列を教えてください。"
        patch["messages"] = [AIMessage(content=q)]

    return patch


