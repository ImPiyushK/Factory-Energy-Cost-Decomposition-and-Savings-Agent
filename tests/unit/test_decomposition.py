"""Decomposition correctness: exact reconciliation + injected-truth recovery."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analytics import baseline as bl
from src.analytics import decomposition as dc

ROOT = Path(__file__).resolve().parent.parent


# --------------------------- property test: exact reconciliation --------------------------- #

def _random_frames(rng: np.random.Generator, n_days: int = 80) -> dict[str, pd.DataFrame]:
    """Minimal two-quarter dataset; the second quarter has a mid-quarter price change."""

    def quarter(start: str, price_a: float, price_b: float | None = None,
                change_offset: int | None = None) -> tuple[pd.DataFrame, ...]:
        days = pd.date_range(start, periods=n_days, freq="D", tz="UTC")
        ts = pd.date_range(start, periods=n_days * 96, freq="15min", tz="UTC")
        wd = days.dayofweek
        vol = np.clip(rng.normal(1000, 80, len(days)), 0, None)
        vol[wd >= 5] *= 0.2
        units = {"A": vol * 0.5, "B": vol * 0.3, "C": vol * 0.2}
        cdd = np.clip(np.abs(rng.normal(8, 6, n_days)) - 2, 0, None)
        hdd = np.clip(np.abs(rng.normal(3, 3, n_days)) - 2, 0, None)
        meter_rows = []
        base = 500.0
        for d, day in enumerate(days):
            day_units = sum(units[p][d] for p in units)
            for q in range(96):
                frac = 1.0 if (6 <= (q // 4) < 22) else 0.0
                kwh = (base + 0.9 * day_units * frac + 6.0 * cdd[d] + rng.normal(0, 3)) * 0.25
                meter_rows.append({"ts": ts[d * 96 + q], "meter_id": "MAIN", "kwh": max(kwh, 0.0)})
                meter_rows.append({"ts": ts[d * 96 + q], "meter_id": "S1", "kwh": max(kwh, 0.0) * 0.96})
        meter = pd.DataFrame(meter_rows)
        rows = []
        if price_b is None:
            spans = [(days[0], days[-1], price_a)]
        else:
            cut = days[change_offset or n_days // 2]
            spans = [(days[0], cut - pd.Timedelta(days=1), price_a),
                     (cut, days[-1], price_b)]
        for vf, vt, pa in spans:
            for b, f, hs, he in (("peak", 1.0, "06:00", "22:00"),
                                 ("offpeak", 0.5, "00:00", "06:00"),
                                 ("offpeak", 0.5, "22:00", "24:00")):
                rows.append({"valid_from": vf.strftime("%Y-%m-%d"), "valid_to": vt.strftime("%Y-%m-%d"),
                             "tou_bucket": b, "price_per_kwh": pa * f,
                             "tou_start": hs, "tou_end": he})
        tariff = pd.DataFrame(rows)
        production = pd.DataFrame([
            {"date": day.strftime("%Y-%m-%d"), "product_id": p, "volume": float(units[p][d]),
             "shift": "normal" if units["A"][d] > 10 else "none", "is_holiday": 0}
            for d, day in enumerate(days) for p in ("A", "B", "C")
        ])
        weather = pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in days],
                                "temperature_c": 20 + cdd - hdd,
                                "cdd": cdd, "hdd": hdd})
        return meter, tariff, production, weather

    m1, t1, p1, w1 = quarter("2026-01-01", 0.20)
    m2, t2, p2, w2 = quarter("2026-04-01", 0.20, price_b=0.30, change_offset=n_days // 2)
    m1 = m1[m1["ts"] < pd.Timestamp("2026-04-01", tz="UTC")]
    meter = pd.concat([m1, m2], ignore_index=True)
    tariff = pd.concat([t1, t2], ignore_index=True)
    production = pd.concat([p1, p2], ignore_index=True)
    weather = pd.concat([w1, w2], ignore_index=True)
    return {"meter": meter, "tariff": tariff, "production": production, "weather": weather}


def test_reconciliation_is_exact_for_random_inputs():
    rng = np.random.default_rng(7)
    frames = _random_frames(rng)
    res = dc.decompose(frames["meter"], frames["tariff"], frames["production"], frames["weather"],
                       "2026-Q2", "2026-Q1", method="shapley", n_boot=20)
    total = res["bill"]["change_pct"]
    s = sum(b["pct_points"] for b in res["bridge"])
    assert abs(s - total) < 0.1, (s, total)
    for b in res["bridge"]:
        assert b["range"][0] - 1e-6 <= b["pct_points"] <= b["range"][1] + 1e-6, b


# --------------------------- synthetic-data truth checks --------------------------- #

@pytest.fixture(scope="module")
def synthetic_decomp(frames):
    return dc.decompose(frames["meter_15min"], frames["tariff"], frames["production"],
                        frames["weather"], "2026-Q3", "2026-Q2", method="shapley", n_boot=40)


def test_synthetic_reconciliation_exact(synthetic_decomp):
    res = synthetic_decomp
    total = res["bill"]["change_pct"]
    s = sum(b["pct_points"] for b in res["bridge"])
    assert abs(s - total) < 0.1


def test_tariff_is_the_largest_driver_on_synthetic_data(synthetic_decomp):
    bridge = {b["cause"]: b["pct_points"] for b in synthetic_decomp["bridge"]}
    assert bridge["tariff"] == max(bridge.values()), bridge


def test_injected_leak_appears_as_positive_waste(synthetic_decomp):
    bridge = {b["cause"]: b for b in synthetic_decomp["bridge"]}
    # injected leak ~9 kW (about +1.2..2 pp of the bill); model error must not swamp it
    assert 0.3 <= bridge["waste"]["pct_points"] <= 6.0, bridge["waste"]


def test_sequential_method_reconciles_too(frames):
    res = dc.decompose(frames["meter_15min"], frames["tariff"], frames["production"],
                       frames["weather"], "2026-Q3", "2026-Q2", method="sequential", n_boot=20)
    s = sum(b["pct_points"] for b in res["bridge"])
    assert abs(s - res["bill"]["change_pct"]) < 0.1


# --------------------------- baseline sanity --------------------------- #

def test_baseline_fits_well_on_synthetic_daily(frames):
    from src.analytics.periods import quarter_bounds
    s, e = quarter_bounds("2026-Q2")
    daily = bl.build_daily_frame(frames["meter_15min"], frames["production"], frames["weather"])
    qb = daily[(daily.index >= s) & (daily.index <= e)]
    spec, stats = bl.fit_baseline(qb)
    assert stats["r2"] > 0.9, stats
    # per-product intensities ordered A > B > C as designed in the generator
    i = spec.prod
    assert i["units_A"] > i["units_B"] > i["units_C"], i
