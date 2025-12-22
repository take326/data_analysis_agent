from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .edges.route_after_reason import route_after_reason
from .nodes.reason import reason_node
from .nodes.report import report_node
from .nodes.run_code import run_code_node
from .nodes.update_memory import update_memory_node
from .state import AgentState


def create_graph():
    """
    グラフを作成する。
    update_memory → reason → [run_code → reason]* → report → END
    """
    g = StateGraph(AgentState)
    g.add_node("update_memory", update_memory_node)
    g.add_node("reason", reason_node)
    g.add_node("run_code", run_code_node)
    g.add_node("report", report_node)

    g.add_edge(START, "update_memory")
    g.add_edge("update_memory", "reason")
    g.add_conditional_edges("reason", route_after_reason, {"run_code": "run_code", "report": "report", END: END})
    g.add_edge("run_code", "reason")
    g.add_edge("report", END)

    return g.compile()


