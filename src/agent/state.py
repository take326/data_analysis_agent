from __future__ import annotations

from typing import Annotated, Optional, Sequence
from typing_extensions import TypedDict

import pandas as pd
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    プロトタイプState（InMemory実行前提）。
    - df（DataFrame本体）をStateに保持する。
    - messages は reducer(add_messages) で追記する。
    - memories はinvoke時に読み込んだユーザーメモリ。
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    df: pd.DataFrame
    uploaded_filename: Optional[str]  # アップロードされたCSVファイル名

    # ユーザーメモリ（invoke時に読み込み）
    memories: Optional[list[dict]]

    # Reasonの決定（Pydanticモデルをmodel_dumpしたdict）
    decision: Optional[dict]

    # Run Codeの実行結果
    last_code: Optional[str]
    last_exec: Optional[dict]
    last_analysis_instruction: Optional[str]  # Run Codeで保存されたanalysis_instruction
    code_run_count: Optional[int]  # 現在のリクエストに対するコード実行回数（無限ループ防止用）

    # Phase 0/1: Data schema snapshot (for ML feature selection, and to ground code gen)
    df_schema: Optional[dict]

    # Reportの最終結果
    report: Optional[dict]


