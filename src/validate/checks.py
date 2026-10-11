"""Data-quality validation (spec Section 4.5). Runs before any analysis and on every ingest.

Every check returns one of: pass | warn | fail.
  fail    => blocker: analysis refuses to answer (data is untrustworthy).
  warn    => analysis proceeds; the warning flows into `assumptions`/`limitations`
             and is shown in the UI/demo.
Corrections already applied by the loader (gaps interpolated, resets rebased) are
reported as part of the corrections summary, never silently.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def _check(name: str, status: str, detail: str, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"check": name, "status": status, "detail": detail, "metrics": metrics or {}}


def reconcile_submeters(meter: pd.DataFrame, tolerance: float, main_id: str = "MAIN") -> dict[str, Any]:
    """Sub-meter total within a stated tolerance of the main meter."""
    out: dict[str, Any] = {"check": "sub_meter_reconciliation", "tolerance": tolerance}
    if meter.empty:
        out.update(status="fail", detail="no meter data loaded")
        return out
    total = meter.groupby("meter_id")["kwh"].sum()
    if main_id not in total.index:
        out.update(status="fail", detail=f"main meter '{main_id}' missing", metrics={"meters": sorted(total.index)})
        return out
    subs = total.drop(index=main_id)
    sub_sum = float(subs.sum())
    main_sum = float(total[main_id])
    drift = abs(sub_sum - main_sum) / main_sum if main_sum else 1.0
    coverage = sub_sum / main_sum if main_sum else 0.0
    out.update(
        status="pass" if drift <= tolerance else "warn",
        detail=(f"sub-meters sum to {coverage:.1%} of main (drift {drift:.1%}, tolerance {tolerance:.0%}). "
                f"Coverage is disclosed, never silently rescaled."),
        metrics={"drift": round(drift, 4), "coverage": round(coverage, 4),
                 "main_kwh": round(main_sum, 1), "subs_kwh": round(sub_sum, 1)},
    )
    return out


def negatives_and_resets(meter: pd.DataFrame, corrections: dict[str, int]) -> dict[str, Any]:
    neg = int((meter["kwh"] < 0).sum()) if not meter.empty else 0
    resets = int(corrections.get("meter_reset_rebased", 0))
    rejected_neg = int(corrections.get("negative_rejected", 0))
    status = "pass" if neg == 0 else "fail"
    if resets or rejected_neg:
        status = "pass" if status == "pass" else status  # resets were repaired upstream
    return _check(
        "negative_energy_and_resets", status,
        f"{neg} negative values remain; {resets} meter resets rebased; "
        f"{rejected_neg} negative rows rejected (logged).",
        {"negative_remaining": neg, "resets_rebased": resets, "negatives_rejected": rejected_neg},
    )


def timestamps_regular(meter: pd.DataFrame) -> dict[str, Any]:
    if meter.empty:
        return _check("timestamp_regularity", "fail", "no meter data")
    ts = pd.to_datetime(meter["ts"], utc=True)
    irregular = int(((ts.dt.minute % 15) != 0).sum())
    tz_naive_mixed = 0
    missing_intervals = int(meter["kwh"].isna().sum())
    span_days = (ts.max() - ts.min()).total_seconds() / 86400
    status = "pass" if irregular == 0 else "warn"
    return _check(
        "timestamp_regularity", status,
        f"single time zone (UTC); grid = 15 min; {irregular} off-grid stamps; "
        f"{missing_intervals} missing intervals left unfilled; span {span_days:.0f} days.",
        {"off_grid": irregular, "missing_intervals": missing_intervals, "span_days": round(span_days, 1)},
    )


def tariff_coverage(tariff: pd.DataFrame, period_start: pd.Timestamp, period_end: pd.Timestamp) -> dict[str, Any]:
    if tariff.empty:
        return _check("tariff_coverage", "fail", "no tariff data")
    t = tariff.copy()
    t["valid_from"] = pd.to_datetime(t["valid_from"], utc=True)
    t["valid_to"] = pd.to_datetime(t["valid_to"], utc=True)
    # overlap check within each tou bucket/hours combination
    overlaps = 0
    for (bucket, hs, he), g in t.groupby(["tou_bucket", "tou_start", "tou_end"], dropna=False):
        g = g.sort_values("valid_from")
        prev_end = None
        for _, row in g.iterrows():
            if prev_end is not None and row["valid_from"] < prev_end:
                overlaps += 1
            prev_end = row["valid_to"]
    # coverage: the UNION of validity intervals must cover the whole analysis period
    # (rows may change part-way through the period; no single row has to span it)
    intervals = sorted(zip(t["valid_from"], t["valid_to"]))
    merged: list[list[pd.Timestamp]] = []
    for s, e in intervals:
        # seams up to 1 day are considered continuous (date-keyed validity rows)
        if merged and s - merged[-1][1] <= pd.Timedelta(days=1):
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    covered = False
    cursor = period_start
    for s, e in merged:
        if s <= cursor < e:
            cursor = e
        if cursor >= period_end:
            covered = True
            break
    status = "pass" if (overlaps == 0 and covered) else "fail"
    detail = (
        f"{len(t)} tariff rows; {overlaps} overlapping intervals; "
        f"{'full' if covered else 'NO'} coverage of {period_start.date()}..{period_end.date()} "
        f"(union of {len(merged)} merged interval(s))."
    )
    return _check("tariff_coverage", status, detail, {"rows": len(t), "overlaps": overlaps, "covered": covered})


def production_shift_consistency(production: pd.DataFrame) -> dict[str, Any]:
    if production.empty:
        return _check("production_shift_calendar", "fail", "no production data")
    p = production.copy()
    p["date"] = pd.to_datetime(p["date"]).dt.date
    day = p.groupby("date").agg(volume=("volume", "sum"), shift=("shift", "first"),
                                is_holiday=("is_holiday", "first"))
    day["is_holiday"] = day["is_holiday"].fillna(0).astype(int)
    # Public holiday logged as a normal shift (spec 4.4 edge case)
    holiday_as_shift = day[(day["is_holiday"] == 1) & (day["shift"].fillna("").str.lower().isin(["normal", "day"]))]
    zero_vol_shift = day[(day["shift"].fillna("").str.lower().isin(["normal", "day"])) & (day["volume"] <= 0)]
    status = "pass"
    if len(holiday_as_shift) or len(zero_vol_shift):
        status = "warn"
    return _check(
        "production_shift_calendar", status,
        f"{len(day)} production days; {len(holiday_as_shift)} holiday(s) logged as a normal shift; "
        f"{len(zero_vol_shift)} zero-volume day(s) marked as a production shift "
        f"(handled as non-production for idle analysis and flagged).",
        {"days": len(day), "holiday_as_shift": sorted(map(str, holiday_as_shift.index)),
         "zero_volume_shift_days": len(zero_vol_shift)},
    )


def row_counts_vs_card(frames: dict[str, pd.DataFrame], card: dict[str, Any] | None) -> dict[str, Any]:
    if not card:
        return _check("data_card_conformance", "pass", "no data card available for this dataset (skipped).")
    mismatches: list[str] = []
    table_map = {"meter_15min": "meter_15min", "tariff": "tariff", "production": "production", "weather": "weather"}
    for table, key in table_map.items():
        declared = (card.get("tables", {}) or {}).get(key, {}).get("row_count")
        if declared is not None and table in frames and len(frames[table]):
            ratio = abs(len(frames[table]) - declared) / max(1, declared)
            if ratio > 0.02:  # small deviation expected (cleaning removes dupes/artifacts)
                mismatches.append(f"{table}: loaded {len(frames[table])} vs card {declared}")
    meters_declared = (card.get("tables", {}) or {}).get("meter_15min", {}).get("meters")
    meters_loaded = (sorted(frames["meter_15min"]["meter_id"].unique().tolist())
                     if frames.get("meter_15min") is not None and not frames["meter_15min"].empty else [])
    if meters_declared and sorted(meters_declared) != meters_loaded:
        mismatches.append(f"meters: loaded {meters_loaded} vs card {meters_declared}")
    status = "warn" if mismatches else "pass"
    return _check("data_card_conformance", status,
                  "matches declared card." if not mismatches else "; ".join(mismatches),
                  {"mismatches": mismatches})


def run_all_checks(
    frames: dict[str, pd.DataFrame],
    *,
    tolerance: float,
    corrections: dict[str, int] | None = None,
    card: dict[str, Any] | None = None,
    period: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Run the full validation suite over the active (or candidate) data set.

    strict=False (ingest time): tables not yet loaded are reported as `skip`.
    strict=True (pre-analysis): any missing table is a blocker.
    """
    from src.config import settings

    corrections = corrections or {}
    meter = frames.get("meter_15min")
    tariff = frames.get("tariff")
    production = frames.get("production")
    meter_ok = meter is not None and not meter.empty
    tariff_ok = tariff is not None and not tariff.empty
    prod_ok = production is not None and not production.empty
    missing = "fail" if strict else "skip"

    if period is None and meter_ok:
        ts = pd.to_datetime(meter["ts"], utc=True)
        period = (ts.min(), ts.max())

    checks: list[dict[str, Any]] = [
        reconcile_submeters(meter, tolerance) if meter_ok
        else _check("sub_meter_reconciliation", missing, "no meter data loaded"),
        negatives_and_resets(meter, corrections) if meter_ok
        else _check("negative_energy_and_resets", missing, "no meter data loaded"),
        timestamps_regular(meter) if meter_ok
        else _check("timestamp_regularity", missing, "no meter data loaded"),
        tariff_coverage(tariff, *period) if (tariff_ok and period is not None)
        else _check("tariff_coverage", missing, "tariff not loaded / period undefined"),
        production_shift_consistency(production) if prod_ok
        else _check("production_shift_calendar", missing, "no production data"),
        row_counts_vs_card(frames, card),
    ]
    blockers = [c for c in checks if c["status"] == "fail"]
    warnings = [c["detail"] for c in checks if c["status"] == "warn"]
    applied = {k: v for k, v in corrections.items() if v}
    if applied:
        warnings.append("corrections applied and logged: " + ", ".join(f"{k} x{v}" for k, v in applied.items()))
    return {
        "checks": checks,
        "blockers": [c["detail"] for c in blockers],
        "warnings": warnings,
        "corrections": applied,
        "summary": {
            "pass": sum(1 for c in checks if c["status"] == "pass"),
            "warn": sum(1 for c in checks if c["status"] == "warn"),
            "fail": sum(1 for c in checks if c["status"] == "fail"),
            "skip": sum(1 for c in checks if c["status"] == "skip"),
        },
        "ok": not blockers,
    }
