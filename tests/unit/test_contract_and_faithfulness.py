"""Unit tests: contract schemas, validation checks, tariff math, faithfulness validator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.llm.faithfulness import build_allowlist, extract_numbers, validate_numbers
from src.schemas import AnalyzeRequest, AnalyzeResponse, BridgeItem, Recommendation


# ------------------------------- contract schemas ------------------------------- #

def test_analyze_request_valid():
    r = AnalyzeRequest(question="Why did the bill go up?")
    assert r.period is None
    r = AnalyzeRequest(question="Why did the bill go up?", period="2026-Q3", compare_to="2026-Q2")
    assert r.compare_to == "2026-Q2"


@pytest.mark.parametrize("period", ["2026-Q5", "26-Q3", "2026-q3", "abc"])
def test_analyze_request_rejects_bad_period(period):
    with pytest.raises(Exception):
        AnalyzeRequest(question="x", period=period)


def _bridge():
    return [BridgeItem(cause=c, pct_points=p, range=[p - 1, p + 1])
            for c, p in zip(("tariff", "volume", "product_mix", "weather", "waste"),
                            (1, 2, 3, 4, 5))]


def test_decomposition_requires_5_cause_bridge():
    with pytest.raises(Exception):
        AnalyzeResponse(run_id="en-abc123", answer_type="decomposition", summary="s",
                        bridge=_bridge()[:4])


def test_bridge_forbidden_on_other_types():
    with pytest.raises(Exception):
        AnalyzeResponse(run_id="en-abc123", answer_type="calculated", summary="s", bridge=_bridge())


def test_recommendation_range_ordering_enforced():
    base = dict(action="a", rationale="r", evidence_refs=["x"], confidence="medium",
                requires_approval=True, ease="easy",
                saving_kwh_per_year=[100, 200], saving_money_per_year=[10, 20],
                saving_co2e_t_per_year=[1, 2], estimated_impact="x")
    Recommendation(payback_months=[1, 6], **base)
    with pytest.raises(Exception):
        Recommendation(payback_months=[6, 1], **base)
    with pytest.raises(Exception):
        Recommendation(payback_months=[1, 6], saving_co2e_t_per_year=[3, 2], **base)


# ------------------------------- faithfulness module ------------------------------- #

def test_extract_numbers_handles_commas_and_signs():
    vals = [v for _, v in extract_numbers("Bill rose 15.0% to 1,234.5 units; -7 lower; 4 causes.")]
    assert 15.0 in vals and 1234.5 in vals and -7.0 in vals and 4.0 in vals


def test_number_faithfulness_allows_tool_numbers_and_roundings():
    allow = [15.0, 3.3, 120000.0]
    v = validate_numbers("The bill rose 15% while waste added 3.3 pp; savings 120,000 kWh.",
                         allow)
    assert v["ok"], v["violations"]


def test_number_faithfulness_rejects_invented_numbers():
    allow = [15.0]
    v = validate_numbers("The furnace wasted exactly 47,531 kWh last quarter.", allow)
    assert not v["ok"]
    assert any(v_["value"] == 47531.0 for v_ in v["violations"])


def test_number_faithfulness_derived_percent_allowed():
    allow = [20.0, 100.0]
    # 20% of 100 = 20 (derived a/b*100) and sum 120 (a+b)
    v = validate_numbers("Spend went 20% up, now 120 total.", allow)
    assert v["ok"], v["violations"]


def test_allowlist_builder_collects_all_numbers():
    vres = {"bill": {"base": 10.0, "current": 11.5, "change_pct": 15.0},
            "bridge": [{"cause": "tariff", "pct_points": 8.0, "range": [7.0, 9.0]}]}
    allow = build_allowlist(vres)
    for v in (10.0, 11.5, 15.0, 8.0, 7.0, 9.0):
        assert v in allow
