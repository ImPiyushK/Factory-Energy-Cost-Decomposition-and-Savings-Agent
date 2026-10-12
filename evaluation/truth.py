"""Independent ground-truth helpers for the evaluation harness.

These NEVER import the analytics modules: they recompute expected numbers straight
from the CSVs with independent code, so an analytical bug cannot hide behind the
implementation it is supposed to check. (Where the hidden answer key is not
accessible — per the competition rules — these analytic truths are what we can
defensibly report against.)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_truth(dataset_dir: Path) -> dict:
    return json.loads((dataset_dir / "ground_truth.json").read_text())


def _quarter_window(quarter: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    yy, qq = quarter.split("-Q")
    months = {"1": (1, 3), "2": (4, 6), "3": (7, 9), "4": (10, 12)}[str(qq)]
    start = pd.Timestamp(int(yy), months[0], 1, tz="UTC")
    end = start + pd.offsets.QuarterEnd(0) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return start, end


def _ts_col(df: pd.DataFrame) -> str:
    for c in ("timestamp", "ts", "datetime", "time"):
        if c in df.columns:
            return c
    raise ValueError(f"no timestamp column found in {df.columns.tolist()}")


def _raw_main_interval_kwh(dataset_dir: Path, quarter: str) -> tuple[pd.Series, pd.Series]:
    """Main-meter interval kWh computed independently from the raw CSV (cumulative)."""
    meter = pd.read_csv(dataset_dir / "meter_15min.csv")
    m = meter[meter["meter_id"] == "MAIN"].copy()
    m["ts"] = pd.to_datetime(m["ts"] if "ts" in meter.columns else m[_ts_col(meter)], utc=True)
    start, end = _quarter_window(quarter)
    m = m[(m["ts"] >= start) & (m["ts"] <= end)].sort_values("ts")
    kwh = m["kwh"].diff().clip(lower=0)
    # the first interval inside the window carries the raw cumulative level -> unknown;
    # exclude it (0) exactly like the loader's documented policy
    kwh.iloc[0] = 0.0
    kwh[pd.to_numeric(m["kwh"].diff(), errors="coerce") < 0] = None
    return pd.to_numeric(kwh, errors="coerce").fillna(0.0), m["ts"]


def _price_at(ts: pd.Timestamp, tariff: pd.DataFrame) -> float:
    date = ts.normalize()
    hour = ts.hour + ts.minute / 60.0
    for _, r in tariff.sort_values("valid_from").iterrows():
        vf = pd.Timestamp(r["valid_from"], tz="UTC").normalize()
        vt = pd.Timestamp(r["valid_to"], tz="UTC")
        if vf <= date and date <= vt:
            hs = str(r.get("tou_start") or "00:00")
            he = str(r.get("tou_end") or "24:00")
            h1 = int(hs.split(":")[0]) + int(hs.split(":")[1]) / 60.0
            h2 = int(he.split(":")[0]) + int(he.split(":")[1]) / 60.0
            in_hours = True if (hs == "00:00" and he == "24:00") else \
                (h1 <= hour < h2 if h1 <= h2 else hour >= h1 or hour < h2)
            if in_hours:
                return float(r["price_per_kwh"])
    raise ValueError(f"no tariff row covers {ts}")


def _legacy_rows(tariff: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    vf = pd.to_datetime(tariff["valid_from"], utc=True)
    eligible = tariff[vf <= start]
    if eligible.empty:
        raise ValueError("no tariff rows before the period start")
    last = eligible["valid_from"].max()
    return eligible[vf == last]


def _price_from(rows: pd.DataFrame, ts: pd.Timestamp) -> float:
    hour = ts.hour + ts.minute / 60.0
    for _, r in rows.iterrows():
        hs = str(r.get("tou_start") or "00:00")
        he = str(r.get("tou_end") or "24:00")
        h1 = int(hs.split(":")[0]) + int(hs.split(":")[1]) / 60.0
        h2 = int(he.split(":")[0]) + int(he.split(":")[1]) / 60.0
        in_hours = True if (hs == "00:00" and he == "24:00") else \
            (h1 <= hour < h2 if h1 <= h2 else hour >= h1 or hour < h2)
        if in_hours:
            return float(r["price_per_kwh"])
    raise ValueError("legacy price lookup hole")


def main_bill(dataset_dir: Path, quarter: str) -> dict:
    """Exact observed bill + energy for one quarter, from raw CSVs only."""
    kwh, ts = _raw_main_interval_kwh(dataset_dir, quarter)
    tariff = pd.read_csv(dataset_dir / "tariff.csv")
    prices = ts.map(lambda t: _price_at(pd.Timestamp(t), tariff))
    bill = float((kwh.values * prices.values).sum())
    energy = float(kwh.sum())
    return {"period": quarter, "bill": round(bill, 2), "energy_kwh": round(energy, 1),
            "avg_price": round(bill / max(energy, 1e-9), 4)}


def tariff_paasche_effect(dataset_dir: Path, current: str) -> float:
    """Tariff ground truth (Paasche-style): the quarter's actual kWh re-priced with the
    tariff regime in force at the quarter START, in pct points of the BASE bill."""
    prev = _previous_quarter(current)
    base = main_bill(dataset_dir, prev)["bill"]
    kwh, ts = _raw_main_interval_kwh(dataset_dir, current)
    tariff = pd.read_csv(dataset_dir / "tariff.csv")
    legacy = _legacy_rows(tariff, _quarter_window(current)[0])
    bill_actual = float((kwh.values * ts.map(lambda t: _price_at(pd.Timestamp(t), tariff)).values).sum())
    bill_old = float((kwh.values * ts.map(lambda t: _price_from(legacy, pd.Timestamp(t))).values).sum())
    return round((bill_actual - bill_old) / base * 100.0, 2)


def _previous_quarter(quarter: str) -> str:
    yy, q = int(quarter.split("-Q")[0]), int(quarter.split("-Q")[1])
    return f"{yy - 1}-Q4" if q == 1 else f"{yy}-Q{q - 1}"


def total_kwh(dataset_dir: Path, quarter: str, meter_id: str = "MAIN") -> float:
    if meter_id == "MAIN":
        kwh, _ = _raw_main_interval_kwh(dataset_dir, quarter)
        return round(float(kwh.sum()), 1)
    meter = pd.read_csv(dataset_dir / "meter_15min.csv")
    tcol = _ts_col(meter)
    m = meter[meter["meter_id"] == meter_id].copy()
    m["ts"] = pd.to_datetime(m[tcol], utc=True)
    start, end = _quarter_window(quarter)
    m = m[(m["ts"] >= start) & (m["ts"] <= end)]
    return round(float(pd.to_numeric(m["kwh"], errors="coerce").fillna(0.0).sum()), 1)


def leak_truth_kwh(dataset_dir: Path) -> float:
    """Injected leak: constant kW across all hours -> full-year kWh."""
    truth = load_truth(dataset_dir)
    leak = truth.get("leak") or {}
    return float(leak.get("kw", 0.0)) * 8760.0
