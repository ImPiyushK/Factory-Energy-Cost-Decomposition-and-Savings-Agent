"""LangGraph state definition."""
from __future__ import annotations

from typing import Annotated, Any, TypedDict


def _append(left: list | None, right: list | None) -> list:
    return (left or []) + (right or [])


class AgentState(TypedDict, total=False):
    question: str
    period: str | None
    compare_to: str | None
    run_id: str
    intent: str
    answer_type: str
    resolved: dict[str, Any]     # {'period':..., 'compare_to':...}
    ctx: dict[str, Any]          # data/context (frames omitted in traces, only fingerprints)
    verified: dict[str, Any]     # VRES (verified results)
    summary: str
    faithfulness: dict[str, Any]
    llm: dict[str, Any]
    errors: Annotated[list[dict[str, Any]], _append]
    steps: Annotated[list[dict[str, Any]], _append]
    tool_calls: Annotated[list[dict[str, Any]], _append]
    final: dict[str, Any]        # AnalyzeResponse payload
