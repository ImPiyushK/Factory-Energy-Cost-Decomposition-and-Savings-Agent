"""Late data drop drill for the live demo (assumes the API on localhost:8000).

Creates stand-in 'corrected' files from the ACTIVE data, ingests them as new
versions, re-runs the canonical question, and prints the before/after diff.
With the real HCLTech drop: replace steps 2-3 with the provided files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
BASE = "http://localhost:8000"
TMP = ROOT / "data" / "late_drop"


def main() -> int:
    question = {"question": "Why did the bill go up this quarter compared with last quarter?",
                "period": "2026-Q3", "compare_to": "2026-Q2"}
    before = httpx.post(f"{BASE}/analyze", json=question, timeout=300).json()
    print("before:", before["run_id"], "change", before.get("computed_change_pct"))

    # build stand-in corrected tariff + a calibrated sub-meter export (from active data)
    import pandas as pd
    from src.ingest import loader
    frames = loader.load_active_frames()
    TMP.mkdir(parents=True, exist_ok=True)
    tariff = frames["tariff"].copy()
    tariff.loc[tariff["tou_bucket"].astype(str) == "peak", "price_per_kwh"] *= 0.97
    tariff["valid_from"] = pd.to_datetime(tariff["valid_from"], utc=True)
    tariff["valid_to"] = pd.to_datetime(tariff["valid_to"], utc=True)
    tariff_path = TMP / "tariff_corrected.csv"
    tariff.to_csv(tariff_path, index=False)
    meter = frames["meter_15min"].copy()
    sel = meter["meter_id"] == "compressors_2"
    meter.loc[sel, "kwh"] = meter.loc[sel, "kwh"] * 0.98
    meter["ts"] = pd.to_datetime(meter["ts"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    meter_path = TMP / "meter_corrected.csv"
    meter.to_csv(meter_path, index=False)

    for path, table in ((tariff_path, "tariff"), (meter_path, "meter_15min")):
        with open(path, "rb") as f:
            r = httpx.post(f"{BASE}/ingest", files={"file": (path.name, f, "text/csv")},
                           data={"meta": json.dumps({"table": table,
                                                     "note": "late drop correction"})},
                           timeout=300)
        print(table, "->", r.json()["status"])

    after = httpx.post(f"{BASE}/analyze", json=question, timeout=300).json()
    print("after:", after["run_id"], "change", after.get("computed_change_pct"))
    ta = httpx.get(f"{BASE}/trace/{before['run_id']}", timeout=60).json()
    tb = httpx.get(f"{BASE}/trace/{after['run_id']}", timeout=60).json()
    db_, da = ta["final_answer"]["vres"].get("bridge", []), tb["final_answer"]["vres"].get("bridge", [])
    db_map = {b["cause"]: b["pct_points"] for b in db_}
    da_map = {b["cause"]: b["pct_points"] for b in da}
    print("\nwhat changed and why (pp):")
    for cause in (db_map.keys() | da_map.keys()):
        delta = round(da_map.get(cause, 0) - db_map.get(cause, 0), 3)
        print(f"  {cause:12s} {db_map.get(cause):+7.2f} -> {da_map.get(cause):+7.2f}  (Δ {delta:+.2f})")
    print("\ntariff version before/after:",
          ta["data_fingerprints"]["tariff"]["version"], "->",
          tb["data_fingerprints"]["tariff"]["version"])
    print(f"both runs auditable: GET /trace/{before['run_id']} and /trace/{after['run_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
