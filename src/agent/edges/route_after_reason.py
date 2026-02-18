from __future__ import annotations

from langgraph.graph import END

from ..models import ReasonDecision
from ..state import AgentState


def route_after_reason(state: AgentState) -> str:
    MAX_CODE_RUNS = 3
    
    decision = ReasonDecision.model_validate(state["decision"])
    
    if decision.action == "ask_clarification":
        return END
    
    if decision.action == "report":
        return "report"
    
    # run_code を選択した場合、上限チェック
    if decision.action == "run_code":
        code_run_count = state.get("code_run_count", 0)
        
        # 上限に達している場合
        if code_run_count >= MAX_CODE_RUNS:
            last_exec = state.get("last_exec")
            
            # 成功している場合はレポート生成
            if last_exec and last_exec.get("ok"):
                return "report"
            
            # 失敗している場合はask_clarificationで終了
            # reasonノードで強制的にask_clarificationを選ばせる必要がある
            # ここではENDを返して、reasonノードで処理させる
            return END
    
    return "run_code"


