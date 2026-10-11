"""What-if counterfactuals and numeric questions (answer_type 'calculated').

What-ifs are answered only when computable from data already loaded:
  * 'what if we had last quarter's tariff'  -> reprice current kWh with the old regime
  * 'what if <bucket> price were x% lower'  -> rescale that bucket's price, re-bill
  * 'what if volume were x% higher/lower'   -> counterfactual bill with scaled volume
Anything needing the future (next year's tariff) or assets not in the data is
declared out of scope by the classifier -> 'not_in_data', never guessed.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.analytics import baseline as bl
from src.analytics.billing import bucket_series, build_price_schedule_ordered, legacy_schedule
from src.analytics.decomposition import _alloc_shares, Blocks, _grid_for, _product_list, _tou_profile, kwh_series
from src.analytics.periods import quarter_bounds, quarter_of


def apply_tariff_pct(tariff_df: pd.DataFrame, meter_df: pd.DataFrame, period: str,
                     bucket: str | None, pct: float) -> dict:
    """Re-bill the current period with one bucket's price scaled by pct (negative = lower)."""
    s, e = quarter_bounds(period)
    grid = _grid_for(meter_df, s, e)
    price = build_price_schedule_ordered(tariff_df, grid)
    buckets = bucket_series(tariff_df, grid)
    kwh = kwh_series(meter_df, "MAIN", grid).fillna(0.0)
    scaled = price.series.copy()
    if bucket is None:
        mask = pd.Series(True, index=grid)
        applied = "ALL buckets"
    else:
        mask = buckets.astype(str).str.lower() == bucket.lower()
        applied = f"bucket '{bucket}'"
    pct_v = float(pct)
    scaled[mask] = scaled[mask] * (1.0 + pct_v)
    base_bill = float((kwh * price.series).sum())
    new_bill = float((kwh * scaled).sum())
    return {
        "question_kind": "what_if_tariff_pct",
        "period": period,
        "scenario": f"{applied} price{' ' if bucket else ''}{'scaled by ' + format(pct_v, '+.0%')}",
        "value": round(new_bill - base_bill, 2),
        "unit": "currency",
        "detail": {
            "bill_base": round(base_bill, 2), "bill_new": round(new_bill, 2),
            "energy_kwh": round(float(kwh.sum()), 0),
        },
        "assumptions": ["consumption held constant; only prices changes (mechanical re-billing)"],
        "evidence_refs": [f"meter_15min:MAIN:{period}", f"tariff:{period}"],
    }


def apply_base_tariff(tariff_df: pd.DataFrame, meter_df: pd.DataFrame,
                      period: str, compare_to: str) -> dict:
    """Current period's consumption priced with the compare_to quarter's tariff regime."""
    s, e = quarter_bounds(period)
    grid = _grid_for(meter_df, s, e)
    actual = build_price_schedule_ordered(tariff_df, grid)
    old = legacy_schedule(tariff_df, grid)
    kwh = kwh_series(meter_df, "MAIN", grid).fillna(0.0)
    new_bill = float((kwh * old.series).sum())
    base_bill = float((kwh * actual.series).sum())
    return {
        "question_kind": "what_if_old_tariff",
        "period": period,
        "scenario": f"bill {period} consumption at {compare_to}'s tariff (old regime continued)",
        "value": round(new_bill - base_bill, 2),
        "unit": "currency",
        "detail": {"bill_actual": round(base_bill, 2), "bill_at_old_tariff": round(new_bill, 2)},
        "assumptions": ["the historical tariff regime is projected forward unchanged"],
        "evidence_refs": [f"meter_15min:MAIN:{period}", f"tariff:{compare_to}"],
    }


def apply_volume_pct(tariff_df: pd.DataFrame, meter_df: pd.DataFrame,
                     production_df: pd.DataFrame, weather_df: pd.DataFrame,
                     period: str, compare_to: str, pct: float) -> dict:
    """Counterfactual bill with production volume scaled by pct using the baseline model."""
    s, e = quarter_bounds(period)
    grid = _grid_for(meter_df, s, e)
    daily_all = bl.build_daily_frame(meter_df, production_df, weather_df, "MAIN")
    daily_base = daily_all[(daily_all.index >= quarter_bounds(compare_to)[0])
                           & (daily_all.index <= quarter_bounds(compare_to)[1])]
    daily_cur = daily_all[(daily_all.index >= s) & (daily_all.index <= e)]
    products = _product_list(daily_base, daily_cur)
    spec, _ = bl.fit_baseline(daily_base, products)
    blk = Blocks.from_frames(daily_base, daily_cur, products)
    # counterfactual stack: everything current, volume scaled by (1 + pct)
    vol_f = 1.0 + float(pct)
    vol = daily_cur[bl._units_cols(daily_cur)].fillna(0.0).sum(axis=1) * vol_f
    cdd_cur = daily_cur["cdd"].fillna(0.0)
    pred = spec.predict(vol, blk.mix_cur, cdd_cur, daily_cur["hdd"].fillna(0.0),
                        daily_cur["dow"], daily_cur["is_holiday"])
    share_ts = _alloc_shares(grid, bucket_series(tariff_df, grid),
                             _tou_profile(meter_df, grid, bucket_series(tariff_df, grid), daily_base),
                             daily_cur)
    per_day = pred.groupby(pred.index.normalize()).sum()
    per_ts = pd.Series(per_day.reindex(grid.normalize()).values, index=grid)
    price = build_price_schedule_ordered(tariff_df, grid)
    kwh_cur = kwh_series(meter_df, "MAIN", grid).fillna(0.0)
    base_bill = float((kwh_cur * price.series).sum())
    cf_bill = float((per_ts * share_ts * price.series).sum())
    return {
        "question_kind": "what_if_volume_pct",
        "period": period,
        "scenario": f"production volume {'scaled by ' + format(float(pct), '+.0%')} (mix shares held)",
        "value": round(cf_bill - base_bill, 2),
        "unit": "currency",
        "detail": {"bill_actual": round(base_bill, 2),
                   "bill_counterfactual": round(cf_bill, 2),
                   "model_expected_energy_vs_actual_note":
                       "counterfactual uses the baseline model, so it includes model error - "
                       "the comparison below is between two model states"},
        "assumptions": ["baseline model (fit on the compare_to quarter) predicts the energy response",
                        "product mix shares held at the current period's values"],
        "evidence_refs": [f"baseline_model:{compare_to}", f"meter_15min:MAIN:{period}"],
    }


# --------------------------------------------------------------------------- #
# numeric questions
# --------------------------------------------------------------------------- #

METER_ALIASES = {
    "compressor": "compressors", "air compressor": "compressors",
    "cooling": "cooling", "hvac": "cooling", "air conditioning": "cooling",
    "furnace": "furnace", "lighting": "lighting", "light": "lighting",
    "line": "line", "main": "MAIN", "total": "ALL",
}


def resolve_meter_words(question: str) -> list[str]:
    """Return matched meter groups; empty means 'asked nothing specific'."""
    q = question.lower()
    found = []
    for name in ("main", "cooling", "hvac", "air conditioning", "air compressor",
                 "compressor", "furnace", "lighting", "light", "line"):
        if re.search(r"\b" + re.escape(name) + r"s?\b", q) and name not in found:
            found.append(name)
    return found


def numeric_answer(question: str, meter_df: pd.DataFrame, tariff_df: pd.DataFrame,
                   period: str | None) -> dict | None:
    """Deterministic calculators; returns None when the question isn't a numeric one."""
    q = question.lower()
    if period is None:
        return None
    s, e = quarter_bounds(period)
    words = resolve_meter_words(q)
    meter_names = []
    for w in words:
        if w in ("main",):
            meter_names.append("MAIN")
        elif w in ("compressor", "air compressor"):
            meter_names += ["compressors_1", "compressors_2"]
        elif w == "furnace":
            meter_names += ["furnace_1", "furnace_2"]
        elif w == "line":
            meter_names += ["line_1", "line_2"]
        elif w in ("cooling", "hvac", "air conditioning"):
            meter_names.append("cooling")
        elif w in ("lighting", "light"):
            meter_names.append("lighting")

    targets = meter_names or ["MAIN"]

    if re.search(r"\b(total|overall|how much)\b.*\b(kwh|energy|consum)\b|\b(kwh|energy|consum)\b.*\b(total|used|consume)", q) or \
       (words and re.search(r"\b(kwh|energy|consum)\b", q) and not re.search(r"price|rate", q)):
        grid = _grid_for(meter_df, s, e)
        vals = {}
        for mid in sorted(set(targets)):
            k = kwh_series(meter_df, mid, grid)
            vals[mid] = round(float(k.sum(skipna=True)), 1)
        tot = round(sum(vals.values()), 1)
        return {
            "question_kind": "energy_in_period",
            "period": period,
            "scenario": f"{' + '.join(sorted(set(targets)))} kWh in {period}",
            "value": tot, "unit": "kWh",
            "detail": {"by_meter": vals},
            "assumptions": ["missing intervals are excluded (documented data-quality policy)"],
            "evidence_refs": [f"meter_15min:{period}"],
        }

    if re.search(r"(average|mean|blended|what).*(price|rate)|price.*(paid|average)|average.*price", q):
        grid = _grid_for(meter_df, s, e)
        price = build_price_schedule_ordered(tariff_df, grid).series
        kwh = kwh_series(meter_df, "MAIN", grid).fillna(0.0)
        bill = float((kwh * price).sum())
        energy = float(kwh.sum())
        return {
            "question_kind": "average_price",
            "period": period, "scenario": f"average price paid in {period}",
            "value": round(bill / energy, 4) if energy > 0 else 0.0, "unit": "currency_per_kwh",
            "detail": {"bill": round(bill, 2), "energy_kwh": round(energy, 1)},
            "assumptions": ["average = bill / energy (usage-weighted)"],
            "evidence_refs": [f"meter_15min:MAIN:{period}", f"tariff:{period}"],
        }

    if re.search(r"hott|warm|hot week", q):
        # meter energy for the hottest week of the period vs the median week
        grid = _grid_for(meter_df, s, e)
        df = pd.DataFrame({"kwh": kwh_series(meter_df, "MAIN", grid).fillna(0.0)},
                          index=grid)
        df["week"] = grid.isocalendar()["week"].values
        wk_energy = df.groupby("week")["kwh"].sum()
        wdf = pd.DataFrame({"week": wk_energy.index})
        wdf["start"] = [grid[grid.isocalendar()["week"].values == w].min().date() for w in wk_energy.index]
        hottest = int(wk_energy.idxmax())
        mid = float(wk_energy.median())
        return {
            "question_kind": "hottest_week_main",
            "period": period,
            "scenario": f"main-meter energy in ISO week {hottest} vs the median week of {period}",
            "value": round(float(wk_energy[hottest]) / mid, 3), "unit": "ratio",
            "detail": {"hottest_week": hottest, "hottest_week_kwh": round(float(wk_energy[hottest]), 1),
                       "median_week_kwh": round(mid, 1)},
            "assumptions": ["weeks defined by ISO calendar within the quarter"],
            "evidence_refs": [f"meter_15min:MAIN:{period}"],
        }

    if re.search(r"how many (meters|sub.meters|assets)", q):
        mids = sorted(meter_df["meter_id"].unique().tolist())
        return {
            "question_kind": "meter_count",
            "period": period, "scenario": "meters present in the loaded data",
            "value": len(mids), "unit": "count",
            "detail": {"meters": mids},
            "assumptions": [], "evidence_refs": ["meter_15min"],
        }
    return None
