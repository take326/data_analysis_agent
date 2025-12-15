from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from ..models import ExecPythonInput, ExecResult, ReasonDecision
from ..state import AgentState
from ..tools.exec_python import exec_python


_CODE_SYSTEM_PROMPT = """You are a Python data analyst.
Write Python code to accomplish the task.

Rules:
- The input DataFrame is available as variable `df`.
- You MAY use pandas/numpy/matplotlib/seaborn/scikit-learn.
- Keep outputs concise. Prefer printing a few key numbers.
- If you produce tables, append markdown strings to a list variable TABLE_MARKDOWN.
- If you produce structured results, append dict/list objects to a list variable JSON_OUT.
- If you plot figures with matplotlib/seaborn, just create the plots; the runner will automatically capture figures.
- Do NOT read/write files. Do NOT access network. Do NOT use open()/eval()/exec().
- Output ONLY the python code (no markdown fences).
"""


def run_code_node(state: AgentState) -> dict:
    """
    Run Codeノード（LLMでコード生成→exec_pythonツールで実行）。
    """
    load_dotenv()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0)

    decision = ReasonDecision.model_validate(state["decision"])
    if not decision.analysis_instruction:
        # ここに来るのは設計上の想定外だが、最低限のフォールバック
        raise ValueError("analysis_instruction is required for run_code")

    prompt = (
        _CODE_SYSTEM_PROMPT
        + "\nTask (natural language):\n"
        + decision.analysis_instruction
        + "\n"
    )

    code = llm.invoke([HumanMessage(content=prompt)]).content

    df: pd.DataFrame = state["df"]
    exec_in = ExecPythonInput(
        code=code,
        timeout_sec=180,
        max_output_chars=20000,
        context={"df": df, "pd": pd, "np": np},
    )
    exec_out = exec_python(exec_in)
    last_exec = ExecResult.model_validate(exec_out.result.model_dump()).model_dump()

    return {
        "last_code": code,
        "last_exec": last_exec,
    }


