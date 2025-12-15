from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from ..models import ExecResult, ReasonDecision, ReportOutput
from ..state import AgentState


def report_node(state: AgentState) -> dict:
    """
    Reportノード（LLM）。
    decision.analysis_instruction と last_exec を根拠に summary を生成し、
    実行結果に表/グラフがあればそれも含めて返す。
    """
    load_dotenv()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0)

    decision = ReasonDecision.model_validate(state["decision"])
    last_exec = ExecResult.model_validate(state["last_exec"])
    analysis_instruction = decision.analysis_instruction or "(analysis_instruction not provided)"

    prompt = (
        "Write a concise analysis report summary in Japanese.\n"
        "Use the task and the execution outputs as evidence.\n"
        "Include key numbers if present. Mention important caveats if obvious.\n\n"
        f"Task:\n{analysis_instruction}\n\n"
        f"stdout:\n{last_exec.stdout}\n\n"
        f"stderr:\n{last_exec.stderr}\n"
    )
    summary = llm.invoke([HumanMessage(content=prompt)]).content

    report = ReportOutput(
        summary=summary,
        plot_png_base64=last_exec.plot_png_base64,
        table_markdown=last_exec.table_markdown,
        json=last_exec.json,
    )
    return {"report": report.model_dump()}


