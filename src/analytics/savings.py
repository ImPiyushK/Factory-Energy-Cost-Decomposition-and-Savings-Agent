"""Savings, payback and action ranking (spec R8, R9).

Design rules:
  * Every recommendation draws from one DISJOINT evidence pool keyed by
    (meter, window); a ledger guarantees no double-counting of savings.
  * kWh ranges come from documented fractions applied to measured pools.
  * Money = pool kWh x the tariff ACTUALLY in effect for the pool's hours
    (usage-weighted from the tariff table — a measured rate, not an invented one).
  * CO2e = kWh x emission factor. The EF must come from the data card or config;
    if neither provides it, the field is omitted and the missing dependency is
    flagged (never invented).
  * Payback months = cost range / monthly saving range.
  * Ranking: payback midpoint ascending, then ease (easy first), then midpoint
    money saving descending.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import settings
from src.analytics.billing import bucket_series, build_price_schedule_ordered
from src.analytics.periods import quarter_bounds

# Implementation cost ranges: DOCUMENTED ASSUMPTIONS (Annex A may override).
COST_ASSUMPTIONS = {
    "leak_repair": (1200.0, 3500.0),
    "idle_shutdown": (0.0, 800.0),
    "cooling_setpoint": (0.0, 600.0),
}
REDUCTION_FRACTIONS = {"leak_repair": (0.8, 1.2), "idle_shutdown": (0.5, 0.9),
                       "cooling_setpoint": (0.10, 0.25)}
EASE = {"leak_repair": "medium", "idle_shutdown": "easy", "cooling_setpoint": "easy"}
APPROVAL = {"leak_repair": True, "idle_shutdown": False, "cooling_setpoint": False}
POOL_NOTES = {
    "leak_repair": "Measured off-production excess load extrapolated to all compressed-air hours "
                   "with a 0.8–1.2 uncertainty band (documented extrapolation).",
    "idle_shutdown": "Measured idle kWh annualized (13 observed weeks x 4); achievable shutdown "
                     "fraction 50–90% of the idle pool (documented).",
    "cooling_setpoint": "Cooling idle pool with achievable setpoint/network tuning 10–25% (documented).",
}


def resolve_emission_factor() -> tuple[float | None, str]:
    """EF from the data card in DATA_DIR, else from configuration."""
    card = Path(settings.resolve(settings.data_dir)) / "data_card.json"
    if card.exists():
        try:
            ef = json.loads(card.read_text()).get("emission_factor_kg_per_kwh")
            if ef:
                return float(ef), "data card (from the data pack)"
        except Exception:
            pass
    if settings.emission_factor_kg_per_kwh:
        return float(settings.emission_factor_kg_per_kwh), "configuration (.env)"
    return None, "NOT PROVIDED — dependency flagged"


def _series(meter_df: pd.DataFrame, meter_id: str, period: str) -> pd.Series:
    s, e = quarter_bounds(period)
    g = meter_df[meter_df["meter_id"] == meter_id]
    k = pd.Series(g["kwh"].astype(float).values, index=pd.to_datetime(g["ts"], utc=True))
    k = k.loc[(k.index >= s) & (k.index <= e)]
    return k[~k.index.duplicated()]


def usage_weighted_rate(kwh: pd.Series, price: pd.Series) -> float:
    people = pd.DataFrame({"kwh": kwh.fillna(0.0), "p": price.reindex(kwh.index)}).dropna()
    tot = float(people["kwh"].sum())
    return float((people["kwh"] * people["p"]).sum() / tot) if tot > 1e-9 else 0.0


def _annualize(quarter_kwh: float) -> float:
    return quarter_kwh * 52.0 / 13.0


def _assemble(kind: str, action: str, kwh_lo: float, kwh_hi: float, rate: float,
              evidence_refs: list[str], confidence: str, ef: float | None,
              rationale: str, extra_assumptions: list[str] | None = None) -> dict:
    kwh_lo, kwh_hi = max(0.0, kwh_lo), max(kwh_lo, kwh_hi)
    money_lo, money_hi = kwh_lo * rate, kwh_hi * rate
    co2e = None if ef is None else [round(kwh_lo * ef / 1000.0, 1), round(kwh_hi * ef / 1000.0, 1)]
    cost_lo, cost_hi = COST_ASSUMPTIONS[kind]
    pay_lo = cost_lo / (money_hi / 12.0) if money_hi > 0 else 240.0
    pay_hi = cost_hi / (money_lo / 12.0) if money_lo > 0 else 240.0
    pay_lo, pay_hi = min(pay_lo, 240.0), min(pay_hi, 240.0)
    if pay_hi < pay_lo:
        pay_lo, pay_hi = pay_hi, pay_lo
    frac_lo, frac_hi = REDUCTION_FRACTIONS[kind]
    impact = [f"{kwh_lo/1000:.0f}–{kwh_hi/1000:.0f} MWh/yr",
              f"€{money_lo/1000:.1f}k–€{money_hi/1000:.1f}k/yr"]
    impact.append(f"{co2e[0]}–{co2e[1]} t CO2e/yr" if co2e else "CO2e unavailable (no emission factor)")
    impact.append(f"payback ~{pay_lo:.0f}–{pay_hi:.0f} months")
    return {
        "action": action,
        "rationale": rationale,
        "evidence_refs": evidence_refs,
        "confidence": confidence,
        "requires_approval": APPROVAL[kind],
        "ease": EASE[kind],
        "payback_months": [round(pay_lo, 1), round(pay_hi, 1)],
        "saving_kwh_per_year": [round(kwh_lo, 0), round(kwh_hi, 0)],
        "saving_money_per_year": [round(money_lo, 0), round(money_hi, 0)],
        "saving_co2e_t_per_year": co2e,
        "estimated_impact": " · ".join(impact),
        "assumptions": [
            POOL_NOTES[kind],
            f"energy valued at the metered usage-weighted rate for the pool's hours (€{rate:.3f}/kWh)",
            f"implementation cost €{cost_lo:,.0f}–€{cost_hi:,.0f} (documented assumption; Annex A may override)",
        ] + (extra_assumptions or []),
        "limitations": [] if ef else
            ["emission factor not provided by the data pack or config — CO2e range omitted "
             "(missing dependency flagged, not invented)"],
    }


def build_recommendations(
    idle_report: dict,
    bridge: list[dict],
    tariff_df: pd.DataFrame,
    meter_df: pd.DataFrame,
    period: str,
) -> dict:
    """Turn measured findings into ranked savings recommendations with a non-overlap ledger."""
    ef, ef_source = resolve_emission_factor()
    waste_pp = next((b["pct_points"] for b in bridge if b["cause"] == "waste"), 0.0)
    by_meter = {row["meter_id"]: row for row in idle_report.get("by_meter", [])}

    price = build_price_schedule_ordered(tariff_df, _grid(meter_df, period)).series
    recommendations, used_pools = [], []

    # --- compressor leak repair (full pool: measured excess, extrapolated band) ---
    for f in idle_report.get("findings", []):
        meter = f["measured"]
        excess_kw = meter["excess_off_prod_kw"]
        if f["meter_type"] != "compressor" or excess_kw < 1.0 or (f["meter_id"], "all_hours") in used_pools:
            continue
        meter_kwh = _series(meter_df, f["meter_id"], period)
        rate = usage_weighted_rate(meter_kwh, price)
        pool_lo = excess_kw * 8760.0 * REDUCTION_FRACTIONS["leak_repair"][0]
        pool_hi = excess_kw * 8760.0 * REDUCTION_FRACTIONS["leak_repair"][1]
        corr = ("The unexplained bill residual (" + f"{waste_pp:.1f} pct points) also leaves room "
                "for this waste." if meter.get("corroborated_by_bill_residual") else "")
        rec = _assemble(
            "leak_repair",
            f"Repair suspected compressed-air leak on {f['meter_id']}",
            pool_lo, pool_hi, rate,
            list(f.get("evidence_refs", [])),
            f.get("confidence", "medium"),
            ef,
            (f"Night/weekend load on {f['meter_id']} stayed at ~{meter['off_prod_kw_recent_median']} kW "
             f"vs an early-quarter median of {meter['off_prod_kw_early_median']} kW for "
             f"{meter['persistent_weeks']}+ weeks (steady load = {meter['steady_load']}). " + corr).strip(),
            extra_assumptions=[
                "the inferred cause ('consistent with compressed-air leak') requires inspection - "
                "the saving pool assumes the leak is real and is fully fixed",
                f"bill residual corroboration: {'yes' if meter.get('corroborated_by_bill_residual') else 'no'}",
            ])
        rec["evidence_refs"] += [f"ops_calendar:{period}:off_production_windows"]
        used_pools.append((f["meter_id"], "all_hours"))
        recommendations.append(rec)

    # --- idle shutdown for lighting / cooling / lines (off-production pool) ---
    for meter_id, row in by_meter.items():
        if meter_id == "MAIN" or any(p[0] == meter_id for p in used_pools):
            continue
        if row["meter_type"] not in ("lighting", "cooling", "line") or row["idle_kwh"] <= 200:
            continue
        meta = row["meter_type"]
        kind = "cooling_setpoint" if meta == "cooling" else "idle_shutdown"
        meter_kwh = _series(meter_df, meter_id, period)
        rate = usage_weighted_rate(meter_kwh, price)
        annual = _annualize(row["idle_kwh"])
        rec = _assemble(
            kind,
            (f"Fix cooling scheduling / setpoints for {meter_id} outside production hours"
             if kind == "cooling_setpoint" else
             f"Switch off / schedule {meter_id} during non-production hours"),
            annual * REDUCTION_FRACTIONS[kind][0], annual * REDUCTION_FRACTIONS[kind][1], rate,
            [f"meter_15min:{meter_id}:{period}:off_production"],
            "medium" if row["idle_share_pct"] >= 30 else "low",
            ef,
            (f"Measured idle load: {row['idle_kwh']:,.0f} kWh over {row['idle_hours']:,.0f} idle hours "
             f"({row['idle_share_pct']}% of this meter's consumption) - nights, weekends and holidays."))
        used_pools.append((meter_id, "off_production"))
        recommendations.append(rec)

    def sort_key(r):
        ease_order = {"easy": 0, "medium": 1, "hard": 2}
        return ((r["payback_months"][0] + r["payback_months"][1]) / 2.0,
                ease_order.get(r["ease"], 3),
                -(r["saving_money_per_year"][0] + r["saving_money_per_year"][1]) / 2.0)

    recommendations.sort(key=sort_key)
    return {
        "recommendations": recommendations,
        "ledger": {
            "pools": [f"{m}:{w}" for m, w in used_pools],
            "note": "each recommendation draws from one disjoint (meter, window) evidence pool; "
                    "no pool funds two actions (no double-counting)",
        },
        "emission_factor": {"kg_per_kwh": ef, "source": ef_source if ef else ef_source,
                            "missing": ef is None},
    }


def _grid(meter_df: pd.DataFrame, period: str) -> pd.DatetimeIndex:
    s, e = quarter_bounds(period)
    g = pd.DatetimeIndex(sorted(pd.to_datetime(meter_df["ts"].unique(), utc=True)))
    return g[(g >= s) & (g <= e)]
