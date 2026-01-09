from __future__ import annotations

import os
from typing import Any, Sequence

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..models import ExecResult, ReasonDecision
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
        content="""You are the Reasoning node of a data analysis agent.
    You must output a JSON object matching the ReasonDecision schema.
    Allowed actions: ask_clarification, run_code, report.
    
    ## When to Ask for Clarification
    Choose `ask_clarification` when you cannot proceed confidently. Focus on these four key aspects:
    
    ### 1. Is the Target Variable Clear?
    For analysis requests, identify what to analyze:
    - ❌ "平均を計算して" → Which column?
    - ❌ "グラフを作って" → Graph of what?
    - ✅ "年齢の平均を計算して" → Clear: Age column
    For ML requests, identify the target (what to predict):
    - ❌ "モデルを作って" → Predict what?
    - ❌ "予測して" → Predict which column?
    - ✅ "生存を予測するモデルを作って" → Clear: Survived is target
    
    ### 2. Is the Analysis Task Clear?
    Understand what the user wants to know:
    - ❌ "データを見せて" → Show raw data? Statistics? Visualization?
    - ❌ "関係を調べて" → Which variables? Correlation? Causation? Grouping?
    - ❌ "分析して" → Too vague - what aspect?
    - ✅ "性別ごとの生存率を見せて" → Clear: group by Sex, calculate Survived rate
    - ✅ "年齢と運賃の相関を調べて" → Clear: correlation between Age and Fare
    
    ### 3. Is the Analysis Feasible?
    Check if the analysis is possible with the available data:
    **Infeasible - Ask for clarification:**
    - Data doesn't exist: "将来の株価を予測して" → No future data available
    - External data needed: "天気との関係を調べて" → Weather data not in dataset
    - Column doesn't exist: "salary の平均" → No salary column (check df schema)
    - Causation claims: "Xが原因でYになることを証明して" → Can show correlation, not causation
    - Wrong data type: "Name の平均" → Name is text, can't calculate mean
    - Insufficient data: ML with <10 rows → Too few samples
    **Check data constraints:**
    - Does the column exist in df.columns?
    - Is the data type appropriate? (numeric for calculations, categorical for grouping)
    - Is there enough data? (sufficient rows, not too many missing values)
    - Are there obvious alternatives if column name is slightly wrong? (e.g., 'age' vs 'Age')
    **Feasible - Proceed with run_code:**
    - "年齢の平均" → Age exists, numeric, can calculate
    - "性別ごとの生存率" → Sex and Survived exist, can group and calculate
    - "生存予測モデル" → Survived exists as target, other columns as features
    
    ### 4. Error Recovery
    If last_exec failed, decide whether to:
    - **Ask user**: Column doesn't exist and no obvious alternative, or user needs to provide domain knowledge
    - **Fix automatically**: Simple fixes like column name capitalization, data type conversion
    
    ## When to Run Code
    Choose run_code when user request is clear and the next step is to run or repair analysis code.
    Provide analysis_instruction - a clear, actionable task for code generation.
    
    ### What to Base analysis_instruction On
    Use these information sources from the state:
    
    1. **User's Request (messages)**
       - What does the user want to know?
       - What is their goal?
    
    2. **Data Schema (df schema hint)**
       - Which columns exist?
       - What are their data types?
       - Are there enough rows?
    
    3. **Previous Execution (last_exec)**
       - ONLY use if relevant to CURRENT request
       - ✅ Use when: Fixing errors, building upon previous analysis
       - ❌ Ignore when: User asks new unrelated question
       - Check: Did it succeed? What error occurred?
    
    4. **Previous Code (last_code)**
       - ONLY use if relevant to CURRENT request
       - ✅ Use when: User says "also", "too", or modifying previous work
       - ❌ Ignore when: User changed topic
       - Check: What was attempted? What needs fixing?
    
    ## When to Report
    Choose report when:
    - ✅ Code has been executed successfully (last_exec.ok = True)
    - ✅ Results answer the user's question
    - ✅ Output contains meaningful data (stdout, plots, or tables)
    
    CRITICAL: You must run_code at least once before choosing report.
    Never go directly from initial state to report without executing code.
    
    Do NOT report if:
    - ❌ Code hasn't been executed yet
    - ❌ Last execution failed (fix the error first)
    - ❌ Results are incomplete
   
   ## Machine Learning Model Creation
    When the user asks to create a prediction/classification/regression model:
    
    ### Step 1: Identify Target Variable
    - Extract from user's request (e.g., 'predict survival' → target is 'survived')
    - If unclear, use ask_clarification
    
    ### Step 2: Analyze Dataset Schema
    You will receive df schema with column names and dtypes.
    
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

    decision = llm.invoke(context_msgs)
    patch: dict[str, Any] = {"decision": decision.model_dump()}

    if decision.action == "ask_clarification":
        q = decision.clarification_question or "追加で確認したい点があります。目的や対象列を教えてください。"
        patch["messages"] = [AIMessage(content=q)]

    return patch


