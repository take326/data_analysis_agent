from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ExecResult(BaseModel):
    """exec実行の観測結果（プロトタイプは成果物の実体をそのまま保持）。"""

    ok: bool
    stdout: str = ""
    stderr: str = ""

    # 生成物
    plot_png_base64: list[str] = Field(default_factory=list)
    table_markdown: list[str] = Field(default_factory=list)
    json: list[Any] = Field(default_factory=list)

    error_type: Optional[str] = None
    error_message: Optional[str] = None


class ExecPythonInput(BaseModel):
    code: str = Field(..., description="実行するPythonコード")
    timeout_sec: int = Field(default=180, ge=1, le=300, description="実行タイムアウト秒（簡易）")
    max_output_chars: int = Field(default=20000, ge=1000, le=200000, description="stdout/stderr合計の上限")
    context: dict[str, Any] = Field(default_factory=dict, description="execのグローバルに渡す変数")


class ExecPythonOutput(BaseModel):
    result: ExecResult


ReasonAction = Literal["ask_clarification", "run_code", "report"]


class ReasonDecision(BaseModel):
    action: ReasonAction

    # action == ask_clarification のとき必須
    clarification_question: Optional[str] = None

    # action == run_code のとき必須（reportでも参照されるので、report時もセット推奨）
    analysis_instruction: Optional[str] = Field(
        default=None,
        description="Run Code に渡す自然言語タスク。目的/出力（表/グラフ）/使用列/手順を含む。",
    )


class ReportOutput(BaseModel):
    summary: str
    plot_png_base64: list[str] = Field(default_factory=list)
    table_markdown: list[str] = Field(default_factory=list)
    json: list[Any] = Field(default_factory=list)


