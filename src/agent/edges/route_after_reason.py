from __future__ import annotations

from langgraph.graph import END

from ..models import ReasonDecision
from ..state import AgentState


def route_after_reason(state: AgentState) -> str:
    decision = ReasonDecision.model_validate(state["decision"])
    if decision.action == "ask_clarification":
        return END
    if decision.action == "report":
        return "report"
    return "run_code"


