from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from ..models import ExecResult, ReasonDecision, ReportOutput
from ..state import AgentState


def report_node(state: AgentState) -> dict:
    """
    Reportノード（LLM）。
    last_exec を根拠に summary を生成し、
    実行結果に表/グラフがあればそれも含めて返す。
    ユーザーメモリも参照して、好みに合わせたレポートを生成する。
    """
    load_dotenv()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0)

    last_exec = ExecResult.model_validate(state["last_exec"])

    # ユーザーメモリを参照（state経由）
    memories = state.get("memories") or []
    memory_instruction = ""
    if memories:
        memory_lines = [f"- [{m['category']}] {m['content']}" for m in memories]
        memory_instruction = (
            "\n\n## User Preferences (from memory)\n"
            "Follow these user preferences when generating the report:\n"
            + "\n".join(memory_lines)
        )

    prompt = (
        "Write a concise analysis report summary based on the execution results below.\n"
        "Describe what was analyzed and what was found.\n"
        "Include key numbers, visualizations created, and important insights.\n"
        f"{memory_instruction}\n\n"
        f"stdout:\n{last_exec.stdout}\n\n"
    )
    summary = llm.invoke([HumanMessage(content=prompt)]).content

    report = ReportOutput(
        summary=summary,
        plot_png_base64=last_exec.plot_png_base64,
        table_markdown=last_exec.table_markdown,
        json=last_exec.json,
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


