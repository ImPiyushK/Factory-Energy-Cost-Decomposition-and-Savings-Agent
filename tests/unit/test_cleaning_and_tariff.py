"""Unit tests: data cleaning (gaps, resets, duplicates), tariff lookup & coverage checks."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.billing import bucket_series, build_price_schedule_ordered, legacy_schedule
from src.ingest import loader
from src.validate import checks as vchecks


def _meter_df():
    """Interval meter with: a reset on cumulative meter, negatives, duplicates, a gap."""
    ts = pd.date_range("2026-07-01", periods=64, freq="15min", tz="UTC")
    rows = []
    # cumulative MAIN counter (64 points) with reset at ts[20]
    lvl = 1000.0
    vals = []
    for i, t in enumerate(ts):
        lvl += 10.0
        if i == 20:
            lvl = 5.0  # reset
        vals.append(lvl)
    for t, v in zip(ts[:60], vals[:60]):
        rows.append({"ts": t, "meter_id": "MAIN", "kwh": v})
    # interval sub-meter with negatives + duplicates
    for i, t in enumerate(ts[:60]):
        v = 5.0 + (1.0 if i % 4 == 0 else 0.0)
        rows.append({"ts": t, "meter_id": "LINE", "kwh": v})
    rows.append({"ts": ts[5], "meter_id": "LINE", "kwh": -1.0})
    rows.append({"ts": ts[6], "meter_id": "LINE", "kwh": 5.0})  # duplicate of an earlier stamp
    # a 1-hour gap on LINE (rows 60..63 belong to COOL only; the loader reindexes both)
    for i, t in enumerate(ts[60:]):
        rows.append({"ts": t, "meter_id": "COOL", "kwh": 8.0})
    return pd.DataFrame(rows)


def test_clean_meter_handles_reset_negatives_gaps_dups():
    df, ops = loader.adapt_columns(_meter_df(), "meter_15min")
    cleaned, corrections, rejected = loader.clean_meter(df)
    assert isinstance(cleaned, pd.DataFrame) and {"ts", "meter_id", "kwh"} <= set(cleaned.columns)
    # cumulative conversion + reset rebase detected for MAIN
    assert any(k.startswith("cumulative_converted") for k in corrections)
    assert corrections.get("meter_reset_rebased:MAIN", 0) == 1
    assert corrections.get("negative_rejected:LINE", 0) == 1
    assert corrections.get("duplicates_dropped", 0) >= 1
    assert len(rejected) == 1 and rejected[0][1].startswith("negative")


def test_tariff_lookup_midperiod_change():
    t = pd.DataFrame([
        {"valid_from": "2026-01-01", "valid_to": "2026-06-30", "tou_bucket": "peak",
         "price_per_kwh": 0.20, "tou_start": "09:00", "tou_end": "17:00"},
        {"valid_from": "2026-01-01", "valid_to": "2026-06-30", "tou_bucket": "offpeak",
         "price_per_kwh": 0.10, "tou_start": "00:00", "tou_end": "09:00"},
        {"valid_from": "2026-01-01", "valid_to": "2026-06-30", "tou_bucket": "offpeak",
         "price_per_kwh": 0.10, "tou_start": "17:00", "tou_end": "24:00"},
        {"valid_from": "2026-07-01", "valid_to": "2026-09-30", "tou_bucket": "peak",
         "price_per_kwh": 0.30, "tou_start": "09:00", "tou_end": "17:00"},
        {"valid_from": "2026-07-01", "valid_to": "2026-09-30", "tou_bucket": "offpeak",
         "price_per_kwh": 0.12, "tou_start": "00:00", "tou_end": "09:00"},
        {"valid_from": "2026-07-01", "valid_to": "2026-09-30", "tou_bucket": "offpeak",
         "price_per_kwh": 0.12, "tou_start": "17:00", "tou_end": "24:00"},
    ])
    grid = pd.date_range("2026-06-29", periods=48 * 21, freq="15min", tz="UTC")
    sched = build_price_schedule_ordered(t, grid)
    assert abs(sched.series[(sched.index.date == pd.Timestamp("2026-06-30").date())
                            & (sched.index.hour == 10)].iloc[0] - 0.20) < 1e-9
    assert abs(sched.series[(sched.index.date == pd.Timestamp("2026-07-01").date())
                            & (sched.index.hour == 10)].iloc[0] - 0.30) < 1e-9
    # offpeak differs across the change date too (0.10 -> 0.12)
    assert abs(sched.series[(sched.index.date == pd.Timestamp("2026-06-30").date())
                            & (sched.index.hour == 1)].iloc[0] - 0.10) < 1e-9
    assert abs(sched.series[(sched.index.date == pd.Timestamp("2026-07-02").date())
                            & (sched.index.hour == 1)].iloc[0] - 0.12) < 1e-9
    # price at 08:00 (in 'offpeak 00:00-09:00' window) exists on both dates
    assert sched.series[(sched.index.date == pd.Timestamp("2026-07-02").date())
                        & (sched.index.hour == 8)].notna().all()


def test_buckets_series():
    t = pd.DataFrame([
        {"valid_from": "2026-01-01", "valid_to": "2026-12-31", "tou_bucket": "peak",
         "price_per_kwh": 0.30, "tou_start": "09:00", "tou_end": "17:00"},
    ])
    grid = pd.date_range("2026-07-01", periods=96, freq="15min", tz="UTC")
    buckets = bucket_series(t, grid)
    assert buckets[grid.hour == 10].iloc[0] == "peak"
    assert buckets[grid.hour == 3].isna().all()


def test_reconciliation_check_and_card():
    meter = pd.DataFrame({
        "ts": pd.date_range("2026-07-01", periods=96, freq="15min", tz="UTC").tolist() * 2,
        "meter_id": ["MAIN"] * 96 + ["S1"] * 96,
        "kwh": [1.0] * 96 + [0.90] * 96,
    })
    out = vchecks.reconcile_submeters(meter, tolerance=0.15)
    assert out["status"] == "pass" and abs(out["metrics"]["drift"] - 0.10) < 1e-6
    out = vchecks.reconcile_submeters(meter, tolerance=0.05)
    assert out["status"] == "warn"


def test_tariff_coverage_union_with_midperiod_change():
    tariff = pd.DataFrame([
        {"valid_from": pd.Timestamp("2026-07-19 23:59:59+00:00"),
         "valid_to": pd.Timestamp("2026-07-19 23:59:59+00:00"), "tou_bucket": "x",
         "price_per_kwh": 0.2, "tou_start": None, "tou_end": None},
    ])
    # mimic two continuous regimes (as produced by clean_tariff normalization)
    tariff = pd.DataFrame([
        {"valid_from": pd.Timestamp("2026-04-01 00:00:00+00:00"),
         "valid_to": pd.Timestamp("2026-07-19 23:59:59+00:00"), "tou_bucket": "flat",
         "price_per_kwh": 0.2, "tou_start": None, "tou_end": None},
        {"valid_from": pd.Timestamp("2026-07-20 00:00:00+00:00"),
         "valid_to": pd.Timestamp("2026-09-30 23:59:59+00:00"), "tou_bucket": "flat",
         "price_per_kwh": 0.25, "tou_start": None, "tou_end": None},
    ])
    out = vchecks.tariff_coverage(tariff, pd.Timestamp("2026-04-01", tz="UTC"),
                                  pd.Timestamp("2026-09-30 23:45", tz="UTC"))
    assert out["status"] == "pass"
