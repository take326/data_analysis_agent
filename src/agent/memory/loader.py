"""
Memory loader module for CSV-based user memory persistence.

ファイル名形式: user_memory(updated at: YYYYMMDD).csv
シングルユーザー、単一ファイルを想定。
"""

from __future__ import annotations

import glob
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..models import MemoryCategory, MemoryItem, MemoryUpdateDecision

MEMORY_DIR = Path(__file__).parents[3] / "user_data"


def get_memory_file() -> Path | None:
    """
    メモリファイルを取得する。
    ファイル形式: user_memory(updated at: YYYYMMDD).csv
    単一ファイルを想定。
    """
    pattern = str(MEMORY_DIR / "user_memory(updated at: *).csv")
    files = glob.glob(pattern)
    if not files:
        return None
    return Path(files[0])


def load_memory() -> list[MemoryItem]:
    """
    メモリをCSVから読み込む。
    ファイルが存在しない場合は空リストを返す。
    """
    filepath = get_memory_file()
    if filepath is None or not filepath.exists():
        return []

    df = pd.read_csv(filepath)
    items = []
    for _, row in df.iterrows():
        items.append(
            MemoryItem(
                id=int(row["id"]),
                category=MemoryCategory(row["category"]),
                content=str(row["content"]),
            )
        )
    return items


def save_memory(items: list[MemoryItem]) -> Path:
    """
    メモリをCSVに保存する。
    既存ファイルがあれば削除して新規作成。
    """
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    # 古いファイルを削除
    for old in MEMORY_DIR.glob("user_memory(updated at: *).csv"):
        old.unlink()

    # 新しいファイルを作成
    today = datetime.now().strftime("%Y%m%d")
    filepath = MEMORY_DIR / f"user_memory(updated at: {today}).csv"

    if items:
        df = pd.DataFrame([item.model_dump() for item in items])
        df.to_csv(filepath, index=False)
    else:
        # 空の場合もヘッダーだけ作成
        df = pd.DataFrame(columns=["id", "category", "content"])
        df.to_csv(filepath, index=False)

    return filepath


def apply_updates(
    current_items: list[MemoryItem], decision: MemoryUpdateDecision
) -> list[MemoryItem]:
    """
    MemoryUpdateDecisionを適用して新しいメモリリストを返す。
    """
    items = list(current_items)  # コピー
    existing_ids = {item.id for item in items}

    for update in decision.updates:
        if update.action == "none":
            continue

        elif update.action == "add":
            if update.new_item is None:
                continue
            # 新規IDを自動採番（新規ID: 既存の最大ID + 1）
            new_id = max(existing_ids, default=0) + 1
            new_item = MemoryItem(
                id=new_id,
                category=update.new_item.category,
                content=update.new_item.content,
            )
            items.append(new_item)
            existing_ids.add(new_id)

        elif update.action == "update":
            if update.target_id is None or update.new_item is None:
                continue
            for i, item in enumerate(items):
                if item.id == update.target_id:
                    items[i] = MemoryItem(
                        id=update.target_id,
                        category=update.new_item.category,
                        content=update.new_item.content,
                    )
                    break

        elif update.action == "delete":
            if update.target_id is None:
                continue
            items = [item for item in items if item.id != update.target_id]
            existing_ids.discard(update.target_id)

    return items
