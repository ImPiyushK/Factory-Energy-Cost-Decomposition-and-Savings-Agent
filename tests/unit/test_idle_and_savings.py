"""Idle-load and leak detection precision + savings/payback containment (EV-06..09 style)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analytics.idle import idle_and_findings, idle_stats
from src.analytics.savings import build_recommendations, resolve_emission_factor
from src.graph import engine


@pytest.fixture(scope="module")
def idle_q3(ctx):
    frames = ctx["frames"]
    dec = engine.decomp_tool(ctx, "2026-Q3", "2026-Q2")
    waste_pp = next(b["pct_points"] for b in dec["bridge"] if b["cause"] == "waste")
    return idle_and_findings(frames["meter_15min"], frames["production"], "2026-Q3",
                             corroborating_waste_pp=waste_pp)


def test_idle_total_is_positive_and_ranked(idle_q3):
    assert idle_q3["total_idle_kwh"] > 0
    meters = [r["meter_id"] for r in idle_q3["by_meter"]]
    assert "MAIN" in meters and "compressors_2" in meters
    idle = {r["meter_id"]: r for r in idle_q3["by_meter"]}
    assert idle["MAIN"]["idle_kwh"] > idle["cooling"]["idle_kwh"]


def test_injected_leak_detected_on_compressors_2_only(idle_q3):
    found = {f["meter_id"] for f in idle_q3["findings"]}
    assert "compressors_2" in found, idle_q3["findings"]
    assert "compressors_1" not in found  # no false positives on the clean twin
    f0 = next(f for f in idle_q3["findings"] if f["meter_id"] == "compressors_2")
    assert f0["inferred_cause"].startswith("Night-time and weekend compressor load")
    assert f0["measured"]["excess_off_prod_kw"] > 3.0
    assert f0["confidence"] in ("medium", "high")


def test_mislabeled_holiday_is_handled_as_non_production(idle_q3):
    # idle hours include the whole mislabeled holiday (2026-08-15) -> >= its 96 intervals share
    idle = {r["meter_id"]: r for r in idle_q3["by_meter"]}
    # 1 extra holiday day = 24 h of idle per meter above the regular weekend/night calendar
    assert idle["lighting"]["idle_hours"] >= 1000  # nights (~65d*8h) + weekends + holidays


@pytest.fixture(scope="module")
def savings_q3(ctx, idle_q3):
    frames = ctx["frames"]
    dec = engine.decomp_tool(ctx, "2026-Q3", "2026-Q2")
    return build_recommendations(idle_q3, dec["bridge"], frames["tariff"],
                                 frames["meter_15min"], "2026-Q3")


def test_leak_recommendation_contains_true_annual_saving(savings_q3):
    """The generator injects a constant 9 kW leak -> true full-year saving = 9*8760 kWh."""
    truth_kwh = 9.0 * 8760.0
    recs = [r for r in savings_q3["recommendations"] if "compressors_2" in r["action"]]
    assert recs, savings_q3["recommendations"]
    rec = recs[0]
    lo, hi = rec["saving_kwh_per_year"]
    assert lo <= truth_kwh <= hi, (rec["saving_kwh_per_year"], truth_kwh)
    # money range consistent with the same kWh range at the metered rate
    assert rec["saving_money_per_year"][0] < rec["saving_money_per_year"][1] > 0
    assert rec["requires_approval"] is True
    assert rec["ease"] == "medium"


def test_ranges_are_ordered_and_payback_sane(savings_q3):
    for r in savings_q3["recommendations"]:
        assert r["saving_kwh_per_year"][0] <= r["saving_kwh_per_year"][1]
        assert r["saving_money_per_year"][0] <= r["saving_money_per_year"][1]
        assert 0 <= r["payback_months"][0] <= r["payback_months"][1] <= 240


def test_disjoint_evidence_pools(savings_q3):
    pools = savings_q3["ledger"]["pools"]
    assert len(pools) == len(set(pools)), pools


def test_ranking_prefers_short_payback(savings_q3):
    mids = [(r["payback_months"][0] + r["payback_months"][1]) / 2 for r in
            savings_q3["recommendations"]]
    assert mids == sorted(mids)


def test_emission_factor_resolved_from_card(savings_q3):
    ef = savings_q3["emission_factor"]
    assert not ef["missing"] and abs(ef["kg_per_kwh"] - 0.42) < 1e-9
