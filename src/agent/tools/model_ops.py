"""モデル保存・読み込みツール"""
from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

MODELS_DIR = Path("user_data/models")


def save_model_automatically(model: Any, metadata: dict) -> str:
    """
    モデルを自動保存
    
    Args:
        model: scikit-learnのモデルオブジェクト
        metadata: MODEL_METADATAの辞書
        
    Returns:
        model_id: 保存されたモデルのID
    """
    # ディレクトリ作成
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # model_id生成（タイムスタンプ）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_id = f"model_{timestamp}"
    
    # ファイルパス
    model_path = MODELS_DIR / f"{model_id}.pkl"
    metadata_path = MODELS_DIR / f"{model_id}.json"
    
    # モデル保存（pickle）
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # メタデータ保存（JSON）
    full_metadata = {
        "model_id": model_id,
        "created_at": datetime.now().isoformat(),
        **metadata  # MODEL_METADATAの内容を追加
    }
    with open(metadata_path, 'w') as f:
        json.dump(full_metadata, f, indent=2)
    
    return model_id


def load_model(model_id: str) -> tuple[Any, dict]:
    """
    モデルとメタデータを読み込み
    
    Args:
        model_id: モデルID
        
    Returns:
        (model, metadata): モデルオブジェクトとメタデータ辞書
    """
    model_path = MODELS_DIR / f"{model_id}.pkl"
    metadata_path = MODELS_DIR / f"{model_id}.json"
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return model, metadata


def list_saved_models() -> list[dict]:
    """
    保存済みモデルの一覧を取得
    
    Returns:
        メタデータのリスト（新しい順）
    """
    if not MODELS_DIR.exists():
        return []
    
    models = []
    for json_file in MODELS_DIR.glob("*.json"):
        with open(json_file, 'r') as f:
            metadata = json.load(f)
            models.append(metadata)
    
    # 新しい順にソート
    models.sort(key=lambda x: x['created_at'], reverse=True)
    return models
