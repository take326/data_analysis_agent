from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from ..models import ExecResult, ReportOutput
from ..state import AgentState


def report_node(state: AgentState) -> dict:
    """
    Reportノード（LLM）。
    last_exec を根拠に summary を生成し、
    実行結果に表/グラフがあればそれも含めて返す。
    ユーザーメモリも参照して、好みに合わせたレポートを生成する。
    """
    load_dotenv()

    last_exec = ExecResult.model_validate(state["last_exec"])
    last_code = state.get("last_code") or ""
    last_analysis_instruction = state.get("last_analysis_instruction") or ""

    has_plots = bool(last_exec.plot_png_base64)
    if has_plots:
        model = os.getenv("OPENAI_MODEL_VISION", "gpt-4o")
    else:
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0)
    
    # 3. レポート関連のメモリだけをフィルタリング
    memories = state.get("memories") or []
    REPORT_RELEVANT_CATEGORIES = ["report_format", "communication_style", "data_preference"]
    relevant_memories = [
        m for m in memories 
        if m['category'] in REPORT_RELEVANT_CATEGORIES
    ]
    
    memory_instruction = ""
    if relevant_memories:
        memory_lines = [f"- [{m['category']}] {m['content']}" for m in relevant_memories]
        memory_instruction = (
            "\n\nUser Preferences (apply only if relevant):\n"
            + "\n".join(memory_lines)
        )

    # 4. 改善されたプロンプト
    tables = last_exec.table_markdown or []
    tables_text = "\n\n".join(tables)
    prompt = f"""
You are generating a concise analysis report in Markdown.

ANALYSIS TASK (last_analysis_instruction):
{last_analysis_instruction}

LAST GENERATED CODE (for context only; do not paste code into the report):
{last_code}

EVIDENCE (use these as the only ground truth):

STDOUT:
{last_exec.stdout}

TABLES (markdown):
{tables_text}

CRITICAL RULES:
- Start with the direct answer to the analysis task (key numbers first).
- Then cite the minimal supporting evidence from stdout/tables (use exact numbers).
- Then write brief interpretation/insights (only what is supported by the evidence).
- If plots are provided, you MUST include at least one observation per plot (what you see in the figure).

{memory_instruction}
"""
    
    if has_plots:
        # Vision: send text + images in one message content
        content: list[dict] = [{"type": "text", "text": prompt}]
        for b64 in last_exec.plot_png_base64:
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            )
        response = llm.invoke([HumanMessage(content=content)]).content
    else:
        response = llm.invoke([HumanMessage(content=prompt)]).content

    # 返答はそのままMarkdownサマリーとして扱う（title分離/保存は行わない）
    summary = response.strip()

    report = ReportOutput(
        summary=summary,
        plot_png_base64=last_exec.plot_png_base64,
        table_markdown=last_exec.table_markdown,
    )

    
    # report.summary を messages にも残す（LLMの過去コンテキスト用）。
    # UIでは二重表示を避けるため、app.py 側でこのメッセージを非表示にする。
    return {
        "report": report.model_dump(),
        "messages": [
            AIMessage(
                content=summary,
                additional_kwargs={"source": "report_summary"},
            )
        ],
    }


