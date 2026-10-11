"""Pydantic v2 contract models — the frozen API contract (spec Section 6).

These models are the single source of truth for:
  - FastAPI request/response validation
  - the LangGraph state's verified-results payload ("VRES")
  - Streamlit rendering

Field notes (resolving spec ambiguities, see docs/architecture.md):
  * `estimated_impact` (prose requirement) is provided as a short human-readable string
    summarising the same computed ranges as saving_*_per_year. No invented numbers.
  * `bridge` is populated only for answer_type == "decomposition"; null otherwise.
  * Ranges are always [low, high] with low <= high.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

AnswerType = Literal[
    "decomposition", "recommendation", "calculated", "not_in_data", "clarification_needed", "refused"
]
Confidence = Literal["low", "medium", "high"]
Ease = Literal["easy", "medium", "hard"]
Cause = Literal["tariff", "volume", "product_mix", "weather", "waste"]


class AnalyzeRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, examples=["Why did the bill go up 15%?"])
    period: Optional[str] = Field(None, pattern=r"^\d{4}-Q[1-4]$", examples=["2026-Q3"])
    compare_to: Optional[str] = Field(None, pattern=r"^\d{4}-Q[1-4]$", examples=["2026-Q2"])


class BridgeItem(BaseModel):
    cause: Cause
    pct_points: float
    range: list[float] = Field(..., min_length=2, max_length=2)

    @model_validator(mode="after")
    def _ordered(self):
        lo, hi = self.range
        if lo > hi:
            raise ValueError("bridge range must be [low, high]")
        return self


class Recommendation(BaseModel):
    action: str
    rationale: str
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: Confidence
    requires_approval: bool
    ease: Ease
    payback_months: list[float] = Field(..., min_length=2, max_length=2)
    saving_kwh_per_year: list[float] = Field(..., min_length=2, max_length=2)
    saving_money_per_year: list[float] = Field(..., min_length=2, max_length=2)
    saving_co2e_t_per_year: list[float] = Field(..., min_length=2, max_length=2)
    estimated_impact: str  # human-readable summary of the computed ranges above

    @model_validator(mode="after")
    def _ordered(self):
        for k in ("payback_months", "saving_kwh_per_year", "saving_money_per_year", "saving_co2e_t_per_year"):
            lo, hi = getattr(self, k)
            if lo > hi:
                raise ValueError(f"{k} must be [low, high]")
        return self


class AnalyzeResponse(BaseModel):
    run_id: str = Field(..., pattern=r"^en-[0-9a-f]{6}$")
    answer_type: AnswerType
    summary: str
    bridge: Optional[list[BridgeItem]] = None
    recommendations: list[Recommendation] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    # Recommended additions (contract extension, documented):
    data_quality: Optional[dict[str, Any]] = None
    computed_change_pct: Optional[float] = None  # actual computed bill change (may differ from "15%")
    trace_hint: Optional[str] = None

    @model_validator(mode="after")
    def _bridge_only_for_decomposition(self):
        if self.answer_type == "decomposition" and (self.bridge is None or len(self.bridge) != 5):
            raise ValueError("decomposition answers must carry a 5-cause bridge")
        if self.answer_type != "decomposition" and self.bridge not in (None, []):
            raise ValueError("bridge is only allowed for decomposition answers")
        return self


class IngestResponse(BaseModel):
    table: str
    rows_loaded: int
    rows_rejected: int
    status: Literal["loaded", "loaded_with_warnings", "rejected"]
    version_id: Optional[int] = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    validation: dict[str, Any] = Field(default_factory=dict)  # condensed validation report


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    api: dict[str, Any]
    data_store: dict[str, Any]
    llm: dict[str, Any]
    active_data_versions: dict[str, Any]
    last_validation: Optional[dict[str, Any]] = None
    version: str


class TraceResponse(BaseModel):
    """Full audit record for one run (superset design, remappable to Annex C when provided)."""
    model_config = {"extra": "allow"}

    run_id: str
    created_at: str
    question: str
    params: dict[str, Any]
    data_fingerprints: dict[str, Any]
    validation: dict[str, Any]
    steps: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    model: dict[str, Any]
    llm_calls: int
    tokens: dict[str, Any]
    errors: list[dict[str, Any]]
    faithfulness: dict[str, Any]
    final_answer: dict[str, Any]
    latency_ms: int
