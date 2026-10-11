"""Asset breakdown (spec R5): energy by sub-meter, ranking, shares, discrepancy.

Discrepancies are DISCLOSED, never silently rescaled: the coverage of the main
meter by the sub-meters is reported, and if it is outside tolerance the finding
is surfaced as a data-quality warning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.decomposition import kwh_series
from src.analytics.periods import quarter_bounds
from src.config import settings


def asset_breakdown(meter_df: pd.DataFrame, period: str) -> dict:
    """Rank sub-meters by consumption share for one quarter; expose main/sub discrepancy."""
    s, e = quarter_bounds(period)
    totals: dict[str, float] = {}
    for meter_id, g in meter_df.groupby("meter_id"):
        k = pd.Series(g["kwh"].astype(float).values, index=pd.to_datetime(g["ts"], utc=True))
        k = k.loc[(k.index >= s) & (k.index <= e)]
        k = k[~k.index.duplicated()]
        totals[meter_id] = float(k.sum(skipna=True))

    main = totals.get("MAIN", np.nan)
    sub_ids = sorted(m for m in totals if m != "MAIN")
    sub_sum = float(sum(totals[m] for m in sub_ids))
    coverage = sub_sum / main if main else np.nan
    drift = abs(coverage - 1.0) if coverage == coverage else 1.0
    ranking = []
    for meter_id in sorted(sub_ids, key=lambda m: -totals[m]):
        ranking.append({
            "meter_id": meter_id,
            "kwh": round(totals[meter_id], 1),
            "share_pct": round(totals[meter_id] / main * 100, 2) if main else 0.0,
        })
    # rank on share of sub-meter total too (shares of main sum to `coverage*100`)
    return {
        "period": period,
        "ranking": ranking,
        "main_meter": {"meter_id": "MAIN", "kwh": round(main, 1)},
        "sub_meter_coverage": round(coverage, 4) if coverage == coverage else None,
        "reconciliation_drift": round(drift, 4) if coverage == coverage else None,
        "within_tolerance": bool(drift <= settings.recon_tolerance),
        "tolerance": settings.recon_tolerance,
        "assumption": "Coverage <100% is disclosed as a measurement-coverage fact; "
                      "sub-meter totals are not rescaled to force a match.",
    }
