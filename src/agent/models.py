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
    
    # Phase 2: モデル保存
    saved_model_id: Optional[str] = None


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
    ml_config: Optional["MLConfig"] = Field(
        default=None,
        description="機械学習モデル作成時の設定（特徴量選択と前処理）",
    )



class ReportOutput(BaseModel):
    summary: str
    plot_png_base64: list[str] = Field(default_factory=list)
    table_markdown: list[str] = Field(default_factory=list)
    json: list[Any] = Field(default_factory=list)


# ========== Memory Models ==========

from enum import Enum


class MemoryAction(str, Enum):
    """メモリ更新操作の種類"""
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    NONE = "none"


class MemoryCategory(str, Enum):
    """メモリのカテゴリ"""
    ANALYSIS_STYLE = "analysis_style"          # 分析スタイルの好み
    DOMAIN_KNOWLEDGE = "domain_knowledge"      # ドメイン知識レベル
    DATA_PREFERENCE = "data_preference"        # よく扱うデータタイプ
    REPORT_FORMAT = "report_format"            # レポート形式の好み
    COMMUNICATION_STYLE = "communication_style"  # 文体・言語の好み
    WORKFLOW = "workflow"                      # 作業フローの癖
    OTHER = "other"                            # その他


class MemoryItem(BaseModel):
    """個別のメモリエントリ（CSVの1行に対応）"""
    id: int = Field(..., ge=1, description="メモリID（1から始まる整数）")
    category: MemoryCategory
    content: str = Field(..., description="メモリの内容")


class MemoryUpdate(BaseModel):
    """単一のメモリ更新操作"""
    action: MemoryAction
    target_id: Optional[int] = Field(
        default=None,
        ge=1,
        description="UPDATE/DELETE時の対象ID"
    )
    new_item: Optional[MemoryItem] = Field(
        default=None,
        description="ADD/UPDATE時の新しい内容"
    )
    reason: str = Field(..., description="この更新を行う理由")


class MemoryUpdateDecision(BaseModel):
    """LLMのメモリ更新判断（structured output用）"""
    updates: list[MemoryUpdate] = Field(
        default_factory=list,
        description="実行する更新操作のリスト（変更なしは空リスト）"
    )


# ========== ML Model Configuration ==========

class MLConfig(BaseModel):
    """機械学習モデルの設定（特徴量選択と前処理）"""
    target_name: str = Field(..., description="目的変数名")
    feature_names: list[str] = Field(..., description="使用する特徴量のリスト")
    categorical_features: list[str] = Field(..., description="カテゴリ変数のリスト")
    numeric_features: list[str] = Field(..., description="数値変数のリスト")
    preprocessing_steps: list[str] = Field(
        default_factory=lambda: ["encode_categorical", "handle_missing"],
        description="前処理ステップ名のリスト（例: ['encode_categorical', 'handle_missing']）"
    )
    task_type: str = Field(..., description="'classification' or 'regression'")


# ========== ML Model Metadata ==========

class ModelMetadata(BaseModel):
    """保存されたMLモデルのメタデータ"""
    model_id: str = Field(..., description="モデルID（タイムスタンプベース）")
    model_name: str = Field(..., description="ユーザーフレンドリーな名前")
    feature_names: list[str] = Field(..., description="特徴量名のリスト")
    target_name: str = Field(..., description="目的変数名")
    model_type: str = Field(..., description="モデルタイプ（例: RandomForestClassifier）")
    task_type: str = Field(..., description="'classification' or 'regression'")
    created_at: str = Field(..., description="作成日時（ISO形式）")
    train_score: Optional[float] = Field(default=None, description="訓練スコア")
    test_score: Optional[float] = Field(default=None, description="テストスコア")


