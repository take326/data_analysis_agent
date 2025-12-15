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
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    df: pd.DataFrame

    # Reasonの決定（Pydanticモデルをmodel_dumpしたdict）
    decision: Optional[dict]

    # Run Codeの実行結果
    last_code: Optional[str]
    last_exec: Optional[dict]

    # Reportの最終結果
    report: Optional[dict]


