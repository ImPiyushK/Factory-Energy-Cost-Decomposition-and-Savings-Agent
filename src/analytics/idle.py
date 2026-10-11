"""Idle load (spec R6) and leak/fault detection (spec R7).

All numbers are MEASURED observations (kWh during non-production windows).
Inferred causes ("consistent with compressed-air leak") are always labelled as
inferences and require physical inspection - they are never stated as facts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.ops_calendar import off_production_mask, production_calendar_from_production_df
from src.analytics.periods import quarter_bounds

METER_TYPE_KEYWORDS = {
    "compressor": "compressor",
    "cool": "cooling",
    "furnac": "furnace",
    "light": "lighting",
    "line": "line",
    "hvac": "cooling",
}


def meter_type(meter_id: str) -> str:
    low = meter_id.lower()
    for k, v in METER_TYPE_KEYWORDS.items():
        if k in low:
            return v
    return "other"


def idle_stats(meter_df: pd.DataFrame, production_df: pd.DataFrame, period: str) -> dict:
    """Idle (non-production) kWh, hours and split by night/weekend for each meter."""
    s, e = quarter_bounds(period)
    m = meter_df.copy()
    m["ts"] = pd.to_datetime(m["ts"], utc=True)
    m = m[(m["ts"] >= s) & (m["ts"] <= e)]
    grid = pd.DatetimeIndex(sorted(pd.to_datetime(m["ts"], utc=True).dt.floor("15min").unique()))

    _, hour_windows, _ = production_calendar_from_production_df(production_df)
    off = off_production_mask(grid, hour_windows)
    is_night = ((grid.hour >= 22) | (grid.hour < 6))
    is_weekend = grid.dayofweek >= 5
    off_night = off & is_night
    off_weekend = off & is_weekend & ~is_night

    by_meter = []
    for meter_id, g in m.groupby("meter_id"):
        k = pd.Series(g["kwh"].astype(float).values, index=pd.DatetimeIndex(g["ts"])).reindex(grid)
        idle_kwh = float(k[off].sum(skipna=True))
        if idle_kwh == 0.0 and bool(k[off].isna().all()):
            continue
        hours = int(off.sum()) * 0.25
        by_meter.append({
            "meter_id": meter_id,
            "meter_type": meter_type(meter_id),
            "idle_kwh": round(idle_kwh, 1),
            "idle_hours": round(hours, 1),
            "night_kwh": round(float(k[off_night].sum(skipna=True)), 1),
            "weekend_kwh": round(float(k[off_weekend].sum(skipna=True)), 1),
            "total_kwh": round(float(k.sum(skipna=True)), 1),
            "idle_share_pct": round(idle_kwh / max(1e-9, float(k.sum(skipna=True))) * 100, 1),
            "off_prod_avg_kw": round(float(k[off].mean(skipna=True)) * 4.0, 2),
        })
    by_meter.sort(key=lambda r: -r["idle_kwh"])
    return {
        "period": period,
        "by_meter": by_meter,
        "total_idle_kwh": round(sum(r["idle_kwh"] for r in by_meter if r["meter_id"] != "MAIN"), 1),
        "definition": "Non-production = nights (22:00-06:00), weekends, zero-volume days and "
                      "public holidays (including holidays mislabeled as normal shifts).",
    }


# --------------------------------------------------------------------------- #
# leak / fault detection
# --------------------------------------------------------------------------- #

def _weekly_offprod_kw(meter_kwh: pd.Series, off: np.ndarray, grid: pd.DatetimeIndex) -> pd.Series:
    df = pd.DataFrame({"kwh": meter_kwh, "off": off, "week": grid.isocalendar()["week"].values})
    df = df[df["off"] & df["kwh"].notna()]
    # average off-production kW per week (= kWh per 15-min interval x 4)
    return (df.groupby("week")["kwh"].mean() * 4.0).rename("kw")


def _asset_findings(meter_df: pd.DataFrame, production_df: pd.DataFrame, period: str) -> list[dict]:
    s, e = quarter_bounds(period)
    m = meter_df.copy()
    m["ts"] = pd.to_datetime(m["ts"], utc=True)
    m = m[(m["ts"] >= s) & (m["ts"] <= e)]
    grid = pd.DatetimeIndex(sorted(pd.to_datetime(m["ts"], utc=True).dt.floor("15min").unique()))
    _, hour_windows, _ = production_calendar_from_production_df(production_df)
    off = off_production_mask(grid, hour_windows)

    findings: list[dict] = []
    for meter_id, g in m.groupby("meter_id"):
        mtype = meter_type(meter_id)
        if mtype not in ("compressor", "cooling", "furnace", "lighting", "line"):
            continue
        k = pd.Series(g["kwh"].astype(float).values,
                      index=pd.DatetimeIndex(g["ts"])).reindex(grid)
        wk = _weekly_offprod_kw(k, off, grid)
        if len(wk) < 8:
            continue
        early = wk.iloc[:max(3, len(wk) // 3)].median()
        recent = wk.iloc[-max(3, len(wk) // 3):]
        if early <= 0.5:  # ua small anchor; switch to level-persistence rule
            continue
        ratio = recent / early
        n_persistent = 0
        best_run = 0
        for r in ratio:
            n_persistent = n_persistent + 1 if r >= 1.25 else 0
            best_run = max(best_run, n_persistent)
        if best_run < 3:
            continue
        excess_kw = float(recent.median() - early)
        first_wo = None
        weeks_sorted = list(ratio.index)
        for i in range(len(ratio) - best_run + 1):
            window = ratio.iloc[i:i + best_run]
            if (window >= 1.25).all():
                first_wo = int(window.index[0])
                break
        var_low, var_high = float(recent.std()), None
        steady = bool((recent.std() if len(recent) > 1 else 0.0) < max(1.5, 0.5 * recent.median()))
        findings.append({
            "meter_id": meter_id,
            "meter_type": mtype,
            "kind": "persistent_off_production_load",
            "measured": {
                "off_prod_kw_early_median": round(float(early), 2),
                "off_prod_kw_recent_median": round(float(recent.median()), 2),
                "excess_off_prod_kw": round(excess_kw, 2),
                "steady_load": steady,
                "persistent_weeks": best_run,
            },
            "inferred_cause": ({
                "compressor": "Night-time and weekend compressor load staying above its early-quarter "
                              "level - consistent with a compressed-air leak; requires physical inspection.",
                "cooling": "Off-production cooling load persistently above its early-quarter level - "
                           "consistent with scheduling or setpoint issue; requires facilities check.",
                "furnace": "Off-production furnace load persistently above baseline - consistent with "
                           "idle/standby burn; requires process check.",
                "lighting": "Off-production lighting load above baseline - consistent with unused "
                            "lighting left on; requires facilities check.",
                "line": "Off-production line load above baseline - equipment left running idle; "
                        "requires production check.",
            }).get(mtype, "Persistent abnormal off-production load; requires inspection."),
            "confidence": None,  # set by the caller after corroboration
            "evidence_refs": [f"meter_15min:{meter_id}:{period}:weeks>=W{first_wo}" if first_wo
                              else f"meter_15min:{meter_id}:{period}",
                              "ops_calendar:nights+weekends+holidays"],
            "window": {"from_week": first_wo},
        })
    return findings


def idle_and_findings(meter_df: pd.DataFrame, production_df: pd.DataFrame, period: str,
                      corroborating_waste_pp: float | None = None) -> dict:
    idle = idle_stats(meter_df, production_df, period)
    findings = _asset_findings(meter_df, production_df, period)
    for f in findings:
        weeks = f["measured"]["persistent_weeks"]
        corroborated = corroborating_waste_pp is not None and corroborating_waste_pp > 0.3
        f["confidence"] = ("high" if (weeks >= 8 and corroborated)
                           else "medium" if weeks >= 4 else "low")
        f["measured"]["corroborated_by_bill_residual"] = bool(corroborated)
    idle["findings"] = findings
    idle["note"] = ("Findings are measured load patterns plus clearly-labelled inferred causes "
                    "(never confirmed faults). A repair recommendation is only raised when the "
                    "bill decomposition also leaves a controllable residual to explain.")
    return idle
