from __future__ import annotations

import json
import os
from typing import Any, Sequence

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..models import ExecResult, ReasonDecision
from ..state import AgentState


def _build_df_schema(df) -> dict:
    """
    Build a compact dataframe schema summary for LLM use (feature selection / grounding).
    Avoids sending raw rows; only aggregates and small samples.
    """
    rows = len(df)
    cols = len(df.columns)
    columns: list[dict] = []

    for c in df.columns:
        s = df[c]
        nunique = int(s.nunique(dropna=True))
        missing = int(s.isna().sum())
        missing_pct = (missing / rows * 100.0) if rows else 0.0
        unique_rate = (nunique / rows) if rows else 0.0

        col_info: dict[str, Any] = {
            "name": str(c),
            "dtype": str(s.dtype),
            "missing_pct": round(missing_pct, 3),
            "nunique": nunique,
            "unique_rate": round(unique_rate, 6),
        }

        # Include top values only for low-cardinality columns (helps categorical vs id/text).
        if nunique <= 10:
            try:
                vc = s.value_counts(dropna=True, normalize=True).head(3)
                col_info["top_values"] = [[str(k), round(float(v), 4)] for k, v in vc.items()]
            except Exception:
                pass

        columns.append(col_info)

    return {"rows": rows, "cols": cols, "columns": columns}


def reason_node(state: AgentState) -> dict:
    """
    Reasonノード（LLM）。
    - messages（直近の会話履歴）/ last_exec / last_code を元に action を選ぶ
    - ask_clarification の場合は質問をmessagesに追加して終了させる
    """
    load_dotenv()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0).with_structured_output(ReasonDecision)

    last_exec = ExecResult.model_validate(state["last_exec"]) if state.get("last_exec") else None
    last_code = state.get("last_code")

    sys = SystemMessage(
        content="""You are the Reasoning node of a data analysis agent.
    You must output a JSON object matching the ReasonDecision schema.
    Allowed actions: ask_clarification, run_code, report.
    
    ## When to Ask for Clarification
    Choose `ask_clarification` when you cannot proceed confidently. Focus on these aspects:

    - Target is unclear: "平均を計算して" (which column?)
    - Task is vague: "分析して" (what kind of analysis?)
    Otherwise, make a reasonable assumption and proceed.
    
    ## When to Run Code
    Choose run_code when user request is clear and the next step is to run or repair analysis code.
    Provide analysis_instruction - a clear, actionable task for code generation.
    
    ### What to Base analysis_instruction On
    Use these information sources from the state:
    
    1. **User's Request**
       - A normalized question with explicit column names
       - What does the user want to know?
       - What is their goal?

    2. **Previous Execution (last_exec)**
       - ONLY use if relevant to CURRENT request
       - ✅ Use when: Fixing errors, building upon previous analysis
       - ❌ Ignore when: User asks new unrelated question
       - Check: Did it succeed? What error occurred?
    
    3. **Previous Code (last_code)**
       - ONLY use if relevant to CURRENT request
       - ✅ Use when: User says "also", "too", or modifying previous work
       - ❌ Ignore when: User changed topic
       - Check: What was attempted? What needs fixing?
    
    ## When to Report
    Choose report when ALL of these are true:
    - ✅ code_run_count >= 1 (at least one code execution for current request)
    - ✅ Execution was successful (last_exec.ok = True)
    - ✅ Results answer the CURRENT question
   
   ## Machine Learning Model Creation
    When the user asks to create a prediction/classification/regression model:
    
    ### Step 1: Identify Target Variable
    - Extract from user's request (e.g., 'predict survival' → target is 'survived')
    - If unclear, use ask_clarification
    
    ### Step 2: Analyze Dataset Schema
    You will receive df_schema (JSON) with per-column dtype/missing/nunique/top_values.
    
    ### Step 3: Select Features
    **EXCLUDE these columns:**
    - ID columns: 'id', 'ID', 'PassengerId', 'customer_id' (pattern: ends with 'id')
    - Names/Text: 'Name', 'Description', 'Comment' (free text)
    - High cardinality: categorical with >50 unique values (e.g., 'Ticket', 'Cabin')
    - Too many missing: >70% missing values
    - Constant columns: only 1 unique value (e.g., all rows have same value)
    - Target variable itself
    **INCLUDE these columns:**
    - Numeric features: int/float columns that seem relevant
    - Low cardinality categorical: <20 unique values (e.g., 'Sex', 'Embarked')
    - Domain-relevant features (use common sense)
    **Examples:**
    - Titanic survival: INCLUDE [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked], EXCLUDE [PassengerId, Name, Ticket, Cabin]
    - Housing price: INCLUDE [LotArea, YearBuilt, OverallQual, GrLivArea, Neighborhood], EXCLUDE [Id, Street]
    
    ### Step 4: Categorize Features
    - categorical_features: object/string dtype
    - numeric_features: int64/float64 dtype
    
    ### Step 5: Determine Task Type
    - 'classification': target is categorical (0/1, categories)
    - 'regression': target is continuous numeric
    
    ### Step 6: Output ml_config
    Set ml_config with all the above information.
    """
    )

    context_msgs: list[BaseMessage] = [sys]

    # df_schema をプロンプトに追加（df本体は渡さない）
    df_schema = _build_df_schema(state["df"])
    context_msgs.append(SystemMessage(content="df_schema (JSON):\n" + json.dumps(df_schema, ensure_ascii=False)))


    # 直近の会話履歴（最大12件）をそのまま渡す（文脈を踏まえた判断のため）
    recent_msgs = list(state.get("messages", []))[-12:]
    context_msgs.extend(recent_msgs)
    
    # Data observation is done in run_code node

    # last_exec と last_code は、現在のリクエストで実行済みの場合のみ表示
    code_run_count = state.get("code_run_count", 0)
    if code_run_count >= 1:
        if last_exec:
            # 画像データを除外してLLMに渡す（トークン削減のため）
            exec_info = {
                "ok": last_exec.ok,
                "stdout": last_exec.stdout,
                "stderr": last_exec.stderr,
            }
            context_msgs.append(SystemMessage(content=f"Last exec result:\n{exec_info}"))
        if last_code:
            context_msgs.append(SystemMessage(content=f"Last generated code (for debugging/fix):\n{last_code}"))
    
    # code_run_count を最後に伝える（最も重要なので最後）
    context_msgs.append(
        SystemMessage(content=f"""
=== CRITICAL CONSTRAINT ===
Code execution count for current request: {code_run_count}

ABSOLUTE RULE: If code_run_count is 0, you must not choose 'report'.
This is a hard constraint. No exceptions.
==========================
""")
    )

    decision = llm.invoke(context_msgs)
    patch: dict[str, Any] = {"decision": decision.model_dump(), "df_schema": df_schema}

    # Persist the latest analysis instruction so downstream nodes (e.g., report) can reference it
    # even after decision switches from run_code -> report.
    if decision.action == "run_code" and decision.analysis_instruction:
        patch["last_analysis_instruction"] = decision.analysis_instruction

    # 上限到達時の強制ask_clarification
    MAX_CODE_RUNS = 3
    if state.get("code_run_count", 0) >= MAX_CODE_RUNS:
        if last_exec and not last_exec.ok:
            # 3回失敗したので、ユーザーに助けを求める
            error_msg = last_exec.stderr 
            q = (
                f"コード実行が{MAX_CODE_RUNS}回失敗しました。以下のエラーを確認して、"
                "データの形式や要件について追加情報を教えてください:\n\n"
                f"```\n{error_msg}\n```"
            )
            patch["messages"] = [AIMessage(content=q)]
            return patch

    if decision.action == "ask_clarification":
        q = decision.clarification_question
        patch["messages"] = [AIMessage(content=q)]

    return patch


