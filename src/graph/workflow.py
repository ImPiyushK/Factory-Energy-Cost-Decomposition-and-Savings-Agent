"""LangGraph workflow: the minimal 5-node pipeline with conditional edges."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.graph import nodes
from src.graph.state import AgentState


def _needs_brief(state: AgentState) -> str:
    """Conditional edge: only LLM-supported answer types go through the brief node."""
    if state.get("answer_type") in ("decomposition", "recommendation"):
        return "brief"
    return "persist"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("classify", nodes.classify)
    g.add_node("context", nodes.context)
    g.add_node("compute", nodes.compute)
    g.add_node("brief", nodes.brief)
    g.add_node("persist", nodes.persist)
    g.add_edge(START, "classify")
    g.add_edge("classify", "context")
    g.add_edge("context", "compute")
    g.add_conditional_edges("compute", _needs_brief, {"brief": "brief", "persist": "persist"})
    g.add_edge("brief", "persist")
    g.add_edge("persist", END)
    return g.compile()


GRAPH = build_graph()


def run(question: str, period: str | None = None, compare_to: str | None = None,
        run_id: str | None = None) -> dict:
    """Invoke the workflow; returns the final payload (already trace-saved)."""
    state: AgentState = {"question": question, "period": period, "compare_to": compare_to,
                         "run_id": run_id}
    result = GRAPH.invoke(state, {"recursion_limit": 25})
    return result.get("final", {})
