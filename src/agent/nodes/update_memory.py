"""
update_memory node - reasonノードの前にユーザーメモリを更新するノード

セッション内容を分析してユーザーの特徴・好みを学習し、
更新後のメモリをstateに反映して後続ノードで使用可能にする。
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..memory.loader import apply_updates, load_memory, save_memory
from ..models import MemoryCategory, MemoryItem, MemoryUpdateDecision
from ..state import AgentState


def _format_memory_for_prompt(items: list[MemoryItem]) -> str:
    """メモリをプロンプト用のテキストに変換"""
    if not items:
        return "(メモリは空です)"

    lines = []
    for item in items:
        lines.append(f"[ID:{item.id}] [{item.category.value}] {item.content}")
    return "\n".join(lines)


def _format_messages_for_prompt(messages: list) -> str:
    """メッセージ履歴を要約用にフォーマット"""
    lines = []
    for msg in messages[-10:]:  # 直近10件に制限
        role = "User" if msg.type == "human" else "Assistant"
        content = msg.content[:500] if len(msg.content) > 500 else msg.content
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def update_memory_node(state: AgentState) -> dict:
    """
    update_memoryノード。

    1. 既存メモリをロード（CSVまたはstate経由）
    2. セッション内容（messages）をLLMに渡す
    3. LLMがMemoryUpdateDecisionを返す
    4. 更新を適用してCSVに保存
    5. 更新後のメモリをstateに反映（後続ノードで使用）
    """
    load_dotenv()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0).with_structured_output(
        MemoryUpdateDecision
    )

    # 既存メモリをロード（CSVから最新を取得）
    current_memory = load_memory()
    memory_text = _format_memory_for_prompt(current_memory)

    # セッション内容を取得
    messages = list(state.get("messages", []))
    messages_text = _format_messages_for_prompt(messages)

    # カテゴリの説明を生成
    category_descriptions = "\n".join(
        [f"- {cat.value}: {cat.name}" for cat in MemoryCategory]
    )

    system_prompt = f"""あなたはユーザーメモリを管理するAIです。
セッションの会話内容を分析し、ユーザーの特徴・好みを既存メモリに反映してください。

## 利用可能なカテゴリ
{category_descriptions}

## 現在のメモリ
{memory_text}

## 更新ルール
1. ADD: 新しい特徴を発見した場合。new_itemにcategoryとcontentを指定（idは自動採番されるので1を指定）
2. UPDATE: 既存の特徴が変化/より詳細になった場合。target_idとnew_itemを指定
3. DELETE: 古くなった/間違っていた特徴を削除。target_idを指定
4. 変更がなければ空のupdatesリストを返す

## 注意
- ユーザーが明示的に好みを変更した場合（例:「英語で出力して」）は必ず更新する
- 具体的で有用な情報のみ保存する
- 短すぎる/曖昧なメモリは避ける
"""

    user_prompt = f"""## 今回のセッション内容

### 会話履歴
{messages_text}

このセッションからユーザーの特徴・好みを分析し、メモリを更新してください。
"""

    decision = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )

    # 更新を適用して保存
    updated_memory = current_memory
    if decision.updates:
        updated_memory = apply_updates(current_memory, decision)
        save_memory(updated_memory)
    elif not current_memory:
        # 初回で変更なしの場合も空ファイルを作成
        save_memory([])

    # 更新後のメモリをstateに反映（後続のreason/reportで使用）
    return {
        "memories": [m.model_dump() for m in updated_memory]
    }
