from __future__ import annotations

import json
import os

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
- If you plot figures with matplotlib/seaborn, just create the plots; the runner will automatically capture figures.
- Do NOT read/write files. Do NOT access network. Do NOT use open()/eval()/exec().
- Output ONLY the python code (no markdown fences).

IMPORTANT - Analysis Output Requirements:
You will be provided a `df_schema` summary (columns/dtypes/missing/top_values).
Print key intermediate results and conclusions so the agent can verify correctness.
The stdout output helps the reasoning agent understand results and determine if further analysis is needed.

## Machine Learning Code Generation

When ml_config is provided in the decision, you MUST follow these rules:

### Required Steps:
1. **Feature Selection**: Use ONLY the features in ml_config.feature_names
2. **Categorical Encoding**: Encode ml_config.categorical_features with LabelEncoder
   - Fill categorical missing values with 'missing' BEFORE encoding
3. **Missing Value Handling**: Use KNNImputer for numeric missing values
   - Categorical missing values are already handled in step 2
4. **Train/Test Split**: Split data 80/20 with train_test_split
5. **Model Training**: Choose the best model for accuracy

Model Selection Guidelines (prioritize ACCURACY):
- For regression:
  * RandomForestRegressor (default): High accuracy, robust, works well in most cases
  * LinearRegression: Only if data shows clear linear relationship
  
- For classification:
  * RandomForestClassifier (default): High accuracy, robust, works well in most cases
  * LogisticRegression: Only if data shows clear linear relationship

Default: Use RandomForest unless you have a specific reason to choose otherwise

6. **Evaluation**: Print train and test scores with metric names
   - For regression: Print "R² score" (e.g., "Train R² score: 0.95")
   - For classification: Print "Accuracy" (e.g., "Train Accuracy: 0.95")

7. **Model Assignment** (REQUIRED):
   You MUST assign the trained model for automatic saving.
   ALWAYS include these lines at the end of your code:
   
   # Build categorical mappings for prediction UI
   cat_mappings = {}
   for cat_feat in ml_config.categorical_features:
       encoder = locals()[f'le_{cat_feat}']
       cat_mappings[cat_feat] = {int(i): str(label) for i, label in enumerate(encoder.classes_)}
   
   MODEL = model
   MODEL_METADATA = {
       'model_name': 'Descriptive Model Name',
       'feature_names': list(X.columns),
       'target_name': target,
       'model_type': type(model).__name__,
       'task_type': ml_config.task_type,
       'train_score': train_score,
       'test_score': test_score,
       'categorical_features': ml_config.categorical_features,
       'categorical_mappings': cat_mappings
   }

### Important Notes:
- Prioritize accuracy over simplicity when choosing models
- RandomForest is the recommended default for most cases
- Check ml_config.task_type to determine classification vs regression
- Do NOT use df.drop() to select features - use ml_config.feature_names directly
- Encode categorical features BEFORE using KNNImputer
- ALWAYS split into train/test sets
- ALWAYS print evaluation metrics (both train and test scores)
"""


def run_code_node(state: AgentState) -> dict:
    """
    Run Codeノード（LLMでコード生成→exec_pythonツールで実行）。
    """
    load_dotenv()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0, max_tokens=2048)  # 出力制限を追加

    decision = ReasonDecision.model_validate(state["decision"])
    if not decision.analysis_instruction:
        # ここに来るのは設計上の想定外だが、最低限のフォールバック
        raise ValueError("analysis_instruction is required for run_code")

    df: pd.DataFrame = state["df"]

    # df_schema is produced in reason_node and stored in state; use it to ground code generation.
    df_schema = state.get("df_schema") or {}

    # Check if this is a retry (error from previous attempt)
    last_exec = state.get("last_exec")
    last_code = state.get("last_code")
    
    # Build prompt in readable sections (titles only; no separator bars)
    parts: list[str] = []
    parts.append(_CODE_SYSTEM_PROMPT)
    parts.append(f"Task (natural language):\n{decision.analysis_instruction}")

    # ML config (optional, but required for the ML rules in _CODE_SYSTEM_PROMPT)
    if decision.ml_config is not None:
        parts.append(
            "ML_CONFIG (JSON):\n" + json.dumps(decision.ml_config.model_dump(), ensure_ascii=False)
        )

    # DF schema (ground truth for column names/types/missing/cardinality)
    parts.append("DF_SCHEMA (JSON):\n" + json.dumps(df_schema, ensure_ascii=False))

    parts.append(
        "\n".join(
            [
                "CRITICAL INSTRUCTIONS:",
                "Based on the df_schema above, write appropriate analysis code.",
                "- Use the EXACT column names and types from df_schema",
                "- If ML_CONFIG is provided, follow it EXACTLY for feature selection and preprocessing.",
            ]
        )
    )

    # Add error context if retrying
    if last_exec and not last_exec.get("ok"):
        retry_lines: list[str] = ["PREVIOUS ATTEMPT FAILED - FIX THE ERROR:"]

        if last_code:
            retry_lines.append("Previous code that failed:\n```python\n" + last_code + "\n```")

        retry_lines.append("Error details:")
        if last_exec.get("error_type"):
            retry_lines.append(f"- Error type: {last_exec['error_type']}")
        if last_exec.get("error_message"):
            retry_lines.append(f"- Error message: {last_exec['error_message']}")
        if last_exec.get("stderr"):
            retry_lines.append(f"- Stderr output:\n{last_exec['stderr']}")

        retry_lines.append("IMPORTANT: Analyze the error above and rewrite the code to fix it.")
        retry_lines.append("Do NOT repeat the same mistake!")

        parts.append("\n".join(retry_lines))

    analysis_prompt = "\n\n".join(parts)

    # Generate analysis code
    analysis_code = llm.invoke([HumanMessage(content=analysis_prompt)]).content

    # Execute analysis code
    exec_in = ExecPythonInput(
        code=analysis_code,
        max_output_chars=20000,
        context={"df": df, "pd": pd, "np": np},
    )
    exec_out = exec_python(exec_in)
    last_exec = ExecResult.model_validate(exec_out.result.model_dump()).model_dump()

    return {
        "last_code": analysis_code,
        "last_exec": last_exec,
        "code_run_count": state.get("code_run_count", 0) + 1,
    }
