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
  * GradientBoostingRegressor: Highest accuracy, use if you need maximum performance
  * LinearRegression: Only if data shows clear linear relationship
  
- For classification:
  * RandomForestClassifier (default): High accuracy, robust, works well in most cases
  * GradientBoostingClassifier: Highest accuracy, use if you need maximum performance
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

    prompt = (
        _CODE_SYSTEM_PROMPT
        + "\nTask (natural language):\n"
        + decision.analysis_instruction
        + "\n"
    )
    
    # ml_configがある場合は追加
    if decision.ml_config:
        ml_config = decision.ml_config
        prompt += (
            f"\nML Configuration:\n"
            f"- Target: {ml_config.target_name}\n"
            f"- Features: {ml_config.feature_names}\n"
            f"- Categorical features: {ml_config.categorical_features}\n"
            f"- Numeric features: {ml_config.numeric_features}\n"
            f"- Task type: {ml_config.task_type}\n"
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


