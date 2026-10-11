"""Reproducible synthetic data generator (spec Section 4.2-4.4 minimums).

Emits the four Annex-B-shaped CSVs into data/synthetic/, a data card, and a
ground-truth file with the *injected* parameters (used only by the evaluation
harness — the analysis system never reads it).

Injected ground truth (documented here, and in the data card, because this is OUR
test data; in the official HCLTech pack such patterns are hidden):
  * tariff increase part-way through 2026-Q3 (effective 2026-07-20)
  * product mix shift toward the energy-intensive product A
  * volume +5%
  * summer heatwave (Aug 10-20)
  * compressor leak: +9 kW constant on `compressors_2` from 2026-08-10 (hidden waste)
  * one public holiday (2026-08-15) logged as a normal shift
  * one public holiday (2026-05-01) logged correctly
  * MAIN meter as a cumulative counter with a reset on 2026-07-15 02:00
  * ~0.3% missing intervals + one 8-hour block gap on furnace_1
  * 5 duplicated rows
  * sub-meter coverage ~96.5% of main (within stated tolerance, never exact)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

METERS: dict[str, dict] = {
    # meter: base kW (24h), kWh per unit by product, kWh per CDD
    "compressors_1": dict(base=12.0, coef={"A": 0.050, "B": 0.040, "C": 0.030}, cdd=0.0),
    "compressors_2": dict(base=10.0, coef={"A": 0.030, "B": 0.020, "C": 0.015}, cdd=0.0),
    "furnace_1":     dict(base=0.5,  coef={"A": 0.300, "B": 0.180, "C": 0.120}, cdd=0.0),
    "furnace_2":     dict(base=0.2,  coef={"A": 0.120, "B": 0.070, "C": 0.050}, cdd=2.0),
    "cooling":       dict(base=6.0,  coef={"A": 0.012, "B": 0.008, "C": 0.006}, cdd=8.0),
    "lighting":      dict(base=8.0,  coef={"A": 0.012, "B": 0.008, "C": 0.006}, cdd=0.0),
    "line_1":        dict(base=0.3,  coef={"A": 0.340, "B": 0.200, "C": 0.140}, cdd=0.0),
    "line_2":        dict(base=0.2,  coef={"A": 0.280, "B": 0.170, "C": 0.110}, cdd=0.0),
}
PRODUCTS = ["A", "B", "C"]
MIX_Q2 = {"A": 0.40, "B": 0.35, "C": 0.25}
MIX_Q3 = {"A": 0.52, "B": 0.30, "C": 0.18}
VOL_Q2_WEEKDAY, VOL_Q3_WEEKDAY = 1200.0, 1260.0  # +5% volume
LEAK_METER, LEAK_KW = "compressors_2", 9.0
LEAK_START = pd.Timestamp("2026-08-10 00:00", tz="UTC")
TARIFF_CHANGE_DATE = pd.Timestamp("2026-07-20", tz="UTC")
RESET_TS = pd.Timestamp("2026-07-15 02:00", tz="UTC")
HEATWAVE = (pd.Timestamp("2026-08-10", tz="UTC"), pd.Timestamp("2026-08-20 23:59", tz="UTC"))
HOLIDAY_CORRECT = pd.Timestamp("2026-05-01", tz="UTC")     # logged correctly
HOLIDAY_MISLABELED = pd.Timestamp("2026-08-15", tz="UTC")  # logged as a normal shift (edge case)
COVERAGE = 0.965  # sub-meters as share of main
EMISSION_FACTOR = 0.42  # kg CO2e / kWh (declared in the data card)

TARIFF_Q2 = [("offpeak", 0.120, "00:00", "07:00"), ("offpeak", 0.120, "19:00", "24:00"),
             ("shoulder", 0.180, "07:00", "09:00"), ("peak", 0.240, "09:00", "17:00"),
             ("shoulder", 0.180, "17:00", "19:00")]
TARIFF_Q3 = [("offpeak", 0.135, "00:00", "07:00"), ("offpeak", 0.135, "19:00", "24:00"),
             ("shoulder", 0.200, "07:00", "09:00"), ("peak", 0.270, "09:00", "17:00"),
             ("shoulder", 0.200, "17:00", "19:00")]

MONTH_TEMP = {4: 19.0, 5: 23.0, 6: 28.0, 7: 31.0, 8: 33.0, 9: 28.0}


def prod_frac(day: pd.Timestamp, hour: int) -> float:
    """Production intensity per hour for the day (weekday 06-22 with lunch dip, Sat half)."""
    wd = day.dayofweek  # 0=Mon
    if wd == 6 or day == HOLIDAY_CORRECT or day == HOLIDAY_MISLABELED:
        return 0.0
    if wd == 5:  # Saturday half shift
        return 0.6 if 6 <= hour < 14 else 0.0
    if 6 <= hour < 22:
        return 0.7 if hour == 12 else 1.0
    return 0.0


def generate(seed: int, outdir: Path) -> dict:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2026-04-01", "2026-09-30", freq="D", tz="UTC")

    # ---------------- production ----------------
    prod_rows = []
    for d in days:
        wd = d.dayofweek
        holiday = int(d in (HOLIDAY_CORRECT, HOLIDAY_MISLABELED))
        is_q2 = d < pd.Timestamp("2026-07-01", tz="UTC")
        if wd == 6 or d in (HOLIDAY_CORRECT, HOLIDAY_MISLABELED):
            shift, vol = "none", 0.0
        elif wd == 5:
            shift, vol = "weekend", (VOL_Q2_WEEKDAY if is_q2 else VOL_Q3_WEEKDAY) * 0.4
        else:
            shift, vol = "normal", (VOL_Q2_WEEKDAY if is_q2 else VOL_Q3_WEEKDAY)
        vol *= float(rng.normal(1.0, 0.03))
        vol = max(0.0, round(vol, 0))
        mix = MIX_Q2 if is_q2 else MIX_Q3
        # integer unit split with largest-remainder rounding
        raw = np.array([vol * mix[p] for p in PRODUCTS])
        units = np.floor(raw).astype(int)
        rem = int(round(vol - units.sum()))
        for i in np.argsort(-(raw - units))[:rem]:
            units[i] += 1
        for p, u in zip(PRODUCTS, units):
            prod_rows.append({"date": d.strftime("%Y-%m-%d"), "product_id": p,
                              "volume": float(u), "shift": shift,
                              "is_holiday": holiday if shift == "normal" and vol == 0 else holiday})
    production = pd.DataFrame(prod_rows)

    # ---------------- weather ----------------
    temps = []
    for d in days:
        t = MONTH_TEMP[d.month] + float(rng.normal(0, 2.0))
        if HEATWAVE[0] <= d <= HEATWAVE[1].normalize():
            t += 8.0
        temps.append(t)
    weather = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in days],
        "temperature_c": [round(t, 1) for t in temps],
    })
    weather["cdd"] = (weather["temperature_c"] - 18).clip(lower=0).round(1)
    weather["hdd"] = (18 - weather["temperature_c"]).clip(lower=0).round(1)

    # ---------------- meter 15-min ----------------
    rows: list[dict] = []
    for d in days:
        date_s = d.strftime("%Y-%m-%d")
        day_prod = production[production["date"] == date_s]
        units = {p: float(day_prod[day_prod["product_id"] == p]["volume"].sum()) for p in PRODUCTS}
        cdd = float(weather.loc[weather["date"] == date_s, "cdd"].iloc[0])
        sub_kwh: dict[str, list[float]] = {}
        for meter, cfg in METERS.items():
            hourly = []
            for h in range(24):
                frac = prod_frac(d, h)
                e = (cfg["base"] * (1.0 if frac > 0 or d.dayofweek < 5 else 0.95)
                     + sum(cfg["coef"][p] * units[p] for p in PRODUCTS) * frac
                     + cfg["cdd"] * cdd * (0.6 + 0.4 * frac)) * (1.0 + float(rng.normal(0, 0.05, 1)[0]))
                hourly.append(max(0.0, e))
            for h, e in enumerate(hourly):
                per_iv = e / 4.0
                if meter == LEAK_METER and d >= LEAK_START.normalize():
                    per_iv += LEAK_KW * 0.25  # leak adds constant load, most visible off-production
                for q in range(4):
                    ts = d + pd.Timedelta(hours=h, minutes=15 * q)
                    rows.append({"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S%z").replace("+0000", "Z"),
                                 "meter_id": meter, "kwh": round(per_iv, 5)})
    meter_df = pd.DataFrame(rows)

    # injected missing intervals (0.3% of sub-meter rows) + one 8h block gap on furnace_1
    drop_idx = meter_df[meter_df["meter_id"] != "MAIN"].sample(frac=0.003, random_state=seed).index
    meter_df = meter_df.drop(index=drop_idx)
    block = ((meter_df["meter_id"] == "furnace_1")
             & (meter_df["timestamp"] >= "2026-08-05 08:00")
             & (meter_df["timestamp"] < "2026-08-05 16:00"))
    meter_df = meter_df[~block]

    # MAIN as a cumulative counter with a reset (edge case for the loader)
    sub_sum = meter_df.groupby("timestamp")["kwh"].sum()
    ts_index = sorted(sub_sum.index)
    cov = {t: float(np.clip(rng.normal(COVERAGE, 0.004), 0.94, 0.99)) for t in ts_index}
    main_iv = {t: sub_sum[t] / cov[t] for t in ts_index}
    counter = 0.0
    main_rows = []
    for t in ts_index:
        tpd = pd.Timestamp(t).tz_convert("UTC")
        counter += main_iv[t]
        if tpd == RESET_TS:  # counter reset -> drops to ~0 (loader must rebase)
            counter = 0.0
        main_rows.append({"timestamp": t, "meter_id": "MAIN", "kwh": round(counter, 3)})
    main_df = pd.DataFrame(main_rows)
    meter_df = pd.concat([meter_df, main_df], ignore_index=True)
    # 5 duplicated rows (dedup edge case)
    dup = meter_df.sample(5, random_state=seed)
    meter_df = pd.concat([meter_df, dup], ignore_index=True)
    meter_df = meter_df.sort_values(["meter_id", "timestamp"]).reset_index(drop=True)
    meter_df["kwh"] = meter_df["kwh"].round(5)

    # ---------------- tariff ----------------
    tariff_rows = []
    for (bucket, price, hs, he) in TARIFF_Q2:
        tariff_rows.append({"valid_from": "2026-04-01", "valid_to": "2026-07-19",
                            "tou_bucket": bucket, "price_per_kwh": price,
                            "tou_start": hs, "tou_end": he})
    for (bucket, price, hs, he) in TARIFF_Q3:
        tariff_rows.append({"valid_from": "2026-07-20", "valid_to": "2026-09-30",
                            "tou_bucket": bucket, "price_per_kwh": price,
                            "tou_start": hs, "tou_end": he})
    tariff = pd.DataFrame(tariff_rows)

    # ---------------- write outputs ----------------
    outdir.mkdir(parents=True, exist_ok=True)
    meter_df.to_csv(outdir / "meter_15min.csv", index=False)
    tariff.to_csv(outdir / "tariff.csv", index=False)
    production.to_csv(outdir / "production.csv", index=False)
    weather.to_csv(outdir / "weather.csv", index=False)

    truth = {
        "seed": seed,
        "periods": {"baseline": "2026-Q2", "current": "2026-Q3"},
        "meters": {"main": "MAIN", "sub": sorted(METERS)},
        "leak": {"meter": LEAK_METER, "start": LEAK_START.isoformat(), "kw": LEAK_KW},
        "tariff_change": {"effective": TARIFF_CHANGE_DATE.strftime("%Y-%m-%d"),
                          "prices": {q2[0]: [q2[1], q3[1]] for q2, q3 in zip(TARIFF_Q2, TARIFF_Q3)}},
        "mix_shift": {"2026-Q2": MIX_Q2, "2026-Q3": MIX_Q3},
        "volume_ratio_q3_vs_q2_weekday": VOL_Q3_WEEKDAY / VOL_Q2_WEEKDAY,
        "heatwave": [HEATWAVE[0].strftime("%Y-%m-%d"), HEATWAVE[1].strftime("%Y-%m-%d")],
        "holidays": {"correct": HOLIDAY_CORRECT.strftime("%Y-%m-%d"),
                     "mislabeled_as_normal_shift": HOLIDAY_MISLABELED.strftime("%Y-%m-%d")},
        "meter_reset": RESET_TS.strftime("%Y-%m-%d %H:%M"),
        "sub_meter_coverage_target": COVERAGE,
        "missing_intervals_fraction": 0.003,
        "emission_factor_kg_per_kwh": EMISSION_FACTOR,
        "note": "Leak kW is constant across all hours; full-year leak = kw * 8760 kWh. "
                "Off-production leak is measured from data by the evaluation harness.",
    }
    (outdir / "ground_truth.json").write_text(json.dumps(truth, indent=2))

    card = {
        "dataset": "synthetic factory energy data (generated by scripts/generate_synthetic.py)",
        "seed": seed,
        "generated_by": "deterministic generator; NOT official hackathon data",
        "tables": {
            "meter_15min": {"row_count": int(len(meter_df)), "meters": sorted(meter_df["meter_id"].unique().tolist()),
                            "granularity": "15 min; MAIN is a cumulative counter, sub-meters are interval kWh"},
            "tariff": {"row_count": int(len(tariff))},
            "production": {"row_count": int(len(production))},
            "weather": {"row_count": int(len(weather))},
        },
        "periods": ["2026-Q2", "2026-Q3"],
        "declared_edge_cases": [
            "tariff change mid-Q3 (2026-07-20)",
            "product mix shift (A share 0.40 -> 0.52)",
            "heatwave 2026-08-10..2026-08-20",
            "one waste pattern present (a compressor leak)",
            "public holiday 2026-08-15 logged as a normal shift",
            "meter reset on MAIN at 2026-07-15 02:00",
            "~0.3% missing intervals + one 8-hour gap on furnace_1",
            "5 duplicated rows",
            "sub-meter coverage ~96.5% of main",
        ],
        "emission_factor_kg_per_kwh": EMISSION_FACTOR,
        "tolerances": {"sub_meter_reconciliation": 0.05},
    }
    (outdir / "data_card.json").write_text(json.dumps(card, indent=2))
    shutil.copy(outdir / "data_card.json", ROOT / "data" / "data_card.json")
    return truth


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate reproducible synthetic factory data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "data" / "synthetic"))
    args = ap.parse_args()
    truth = generate(args.seed, Path(args.out))
    print(json.dumps({"written": args.out, "seed": args.seed,
                      "leak": truth["leak"]["meter"], "emission_factor": truth["emission_factor_kg_per_kwh"]}, indent=2))


if __name__ == "__main__":
    main()
