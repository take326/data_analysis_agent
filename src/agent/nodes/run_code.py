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
- If you produce tables, append markdown strings to a list variable TABLE_MARKDOWN.
- If you produce structured results, append dict/list objects to a list variable JSON_OUT.
- If you plot figures with matplotlib/seaborn, just create the plots; the runner will automatically capture figures.
- Do NOT read/write files. Do NOT access network. Do NOT use open()/eval()/exec().
- Output ONLY the python code (no markdown fences).

IMPORTANT - Analysis Output Requirements:
You MUST print detailed analysis information to stdout using print(). This output is critical for the reasoning agent to understand what was done and decide next steps. Include:

1. **Data Structure Info**: Print shape, dtypes, column names, and sample values when first exploring data.
   Example: print(f"DataFrame shape: {df.shape}"), print(df.dtypes), print(df.head())

2. **Statistical Summaries**: Print key statistics (mean, std, min, max, quartiles) for analyzed columns.
   Example: print(df['column'].describe())

3. **Graph Source Data**: When creating plots, ALWAYS print the underlying data used.
   - For histograms: print value counts or binned data
   - For scatter plots: print correlation coefficients and sample points
   - For bar charts: print the aggregated values being plotted
   - For time series: print key data points (first, last, min, max, trends)
   Example: print(f"Correlation: {df['x'].corr(df['y']):.4f}")

4. **Intermediate Calculations**: Print intermediate results, filtered row counts, groupby results, etc.
   Example: print(f"Filtered rows: {len(filtered_df)} / {len(df)}")

5. **Analysis Conclusions**: Print a brief summary of what the analysis reveals.
   Example: print(f"Key finding: Column X has {missing_pct:.1f}% missing values")

The stdout output helps the reasoning agent understand results and determine if further analysis is needed.
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


