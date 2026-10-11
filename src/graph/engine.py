"""Analysis engine: shared context loading + deterministic tool dispatch.

This is the single place where verified results are produced; the LangGraph nodes
only orchestrate. Every tool writes its outputs into the VerifiedResults (VRES)
dict, from which the number-allowlist for the faithfulness check is built.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.analytics import baseline as bl
from src.analytics import decomposition as dc
from src.analytics import whatif
from src.analytics.assets import asset_breakdown
from src.analytics.idle import idle_and_findings
from src.analytics.periods import available_quarters, quarter_bounds
from src.analytics.savings import build_recommendations, resolve_emission_factor
from src.config import settings
from src.ingest import loader
from src.store import db
from src.validate import checks as vchecks


class AmbiguousPeriod(Exception):
    pass


class OutOfData(Exception):
    pass


def load_context() -> dict[str, Any]:
    """Load active data + strict validation; raise OutOfData when blockers exist."""
    conn = db.get_conn()
    try:
        versions = db.get_active_versions(conn)
    finally:
        conn.close()
    frames = loader.load_active_frames()
    missing = [t for t in loader.TABLES if frames.get(t) is None or frames[t].empty]
    if missing:
        raise OutOfData(f"data not loaded for: {', '.join(missing)} "
                        f"(POST /ingest or the CLI loader first)")
    card_path = settings.resolve(settings.data_dir) + "/data_card.json"
    import json as _json
    from pathlib import Path as _Path
    card = _json.loads(_Path(card_path).read_text()) if _Path(card_path).exists() else None
    report = vchecks.run_all_checks(frames, tolerance=settings.recon_tolerance, card=card, strict=True)
    if not report["ok"]:
        raise OutOfData("data validation failed: " + "; ".join(report["blockers"]))
    quarters = available_quarters(frames["meter_15min"]["ts"])
    ef, ef_source = resolve_emission_factor()
    return {"frames": frames, "versions": versions, "validation": report, "card": card,
            "quarters": quarters, "emission_factor": {"kg_per_kwh": ef, "source": ef_source,
                                                      "missing": ef is None}}


def resolve_periods(period: str | None, compare_to: str | None, quarters: list[str]) -> tuple[str, str]:
    """Default: latest complete quarter vs the one before it; ambiguous -> clarify."""
    if len(quarters) < 2:
        raise AmbiguousPeriod("need at least two quarters of data to compare")
    if period and period not in quarters:
        raise OutOfData(f"quarter {period} is not (fully) covered by the loaded data "
                        f"(available: {', '.join(quarters)})")
    p = period or quarters[-1]
    if compare_to and compare_to not in quarters:
        raise OutOfData(f"quarter {compare_to} is not covered by the loaded data")
    if compare_to is None:
        idx = quarters.index(p)
        if idx == 0:
            raise AmbiguousPeriod("no earlier quarter is available to compare with "
                                  f"{p}; asked-for baseline missing")
        compare_to = quarters[idx - 1]
    if compare_to >= p:
        raise AmbiguousPeriod("compare_to quarter must be earlier than period")
    return p, compare_to


def decomp_tool(ctx: dict, period: str, compare_to: str, method: str = "shapley",
                n_boot: int = 60) -> dict:
    f = ctx["frames"]
    return dc.decompose(f["meter_15min"], f["tariff"], f["production"], f["weather"],
                        period, compare_to, method=method, n_boot=n_boot)


def assets_tool(ctx: dict, period: str) -> dict:
    return asset_breakdown(ctx["frames"]["meter_15min"], period)


def idle_tool(ctx: dict, period: str, corroborating_waste_pp: float | None = None) -> dict:
    f = ctx["frames"]
    return idle_and_findings(f["meter_15min"], f["production"], period,
                             corroborating_waste_pp=corroborating_waste_pp)


def savings_tool(ctx: dict, period: str, compare_to: str, dec: dict | None = None) -> dict:
    f = ctx["frames"]
    if dec is None:
        dec = decomp_tool(ctx, period, compare_to)
    idle = idle_tool(ctx, period, corroborating_waste_pp=(
        next((b["pct_points"] for b in dec["bridge"] if b["cause"] == "waste"), 0.0)))
    savings = build_recommendations(idle, dec["bridge"], f["tariff"], f["meter_15min"], period)
    return {"bill": dec["bill"], "bridge": dec["bridge"], "model": dec["model"],
            "n_bootstrap_draws": dec["n_bootstrap_draws"], "n_failed_draws": dec["n_failed_draws"],
            "assumptions": dec["assumptions"], "limitations": dec["limitations"],
            "idle": idle, "savings": savings}


def whatif_tool(ctx: dict, question: str, period: str, compare_to: str) -> dict:
    f = ctx["frames"]
    q = question.lower()
    # future tariffs / forecasts are out of data scope -> declared, not computed
    if any(w in q for w in ("next year", "next quarter", "next month", "forecast",
                            "predict", "will the tariff")):
        raise OutOfData("future tariffs/forecasts are not in the data")
    if "new product" in q or "new line" in q or "new asset" in q:
        raise OutOfData("no data exists for hypothetical new products/assets")
    import re as _re
    pct = _re.search(r"(-?\d+(?:\.\d+)?)\s*%", q)
    direction = -1.0 if _re.search(r"\b(lower|cheaper|reduce[d]?|down|less)\b", q) else 1.0
    if _re.search(r"last (quarter|quarter's) tariff|q2 tariff|old tariff|previous.*tariff", q):
        return whatif.apply_base_tariff(f["tariff"], f["meter_15min"], period, compare_to)
    if pct and _re.search(r"(price|tariff|rate)", q):
        bucket = None
        for b in ("peak", "shoulder", "off-peak", "offpeak", "off peak"):
            if b in q:
                bucket = b.replace("off peak", "offpeak").replace("off-peak", "offpeak")
                break
        return whatif.apply_tariff_pct(f["tariff"], f["meter_15min"], period, bucket,
                                       direction * abs(float(pct.group(1))) / 100.0)
    if pct and _re.search(r"volume|production|output|more|less", q):
        return whatif.apply_volume_pct(f["tariff"], f["meter_15min"], f["production"], f["weather"],
                                       period, compare_to, direction * abs(float(pct.group(1))) / 100.0)
    if _re.search(r"what if", q):
        raise AmbiguousPeriod("what-if scenario not recognized - supported: 'what if we had last "
                              "quarter's tariff', 'what if peak price were X% lower', "
                              "'what if volume were X% higher'")
    raise OutOfData("not answerable from the available data")


def result_envelope(
    *, answer_type: str, vres: dict, summary: str, period: str | None, compare_to: str | None,
) -> dict:
    """Assemble the final AnalyzeResponse payload from verified results."""
    payload = {
        "answer_type": answer_type,
        "summary": summary,
        "bridge": None,
        "recommendations": [],
        "assumptions": list(dict.fromkeys(vres.get("assumptions", [])))[:6],
        "limitations": list(dict.fromkeys(vres.get("limitations", [])))[:8],
        "data_quality": vres.get("data_quality"),
        "computed_change_pct": (vres.get("bill") or {}).get("change_pct"),
        "vres": vres,  # kept for the trace; stripped from the API response
    }
    if answer_type == "decomposition":
        payload["bridge"] = [{"cause": b["cause"], "pct_points": b["pct_points"], "range": b["range"]}
                             for b in vres.get("bridge", [])]
    elif answer_type == "recommendation":
        payload["recommendations"] = vres.get("savings", {}).get("recommendations", [])
    return payload


def strip_vres(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k != "vres"}
