"""Price lookup and billing.

The tariff table may change part-way through a period (spec R4). A single vectorized
price series is built over the 15-minute grid so both actual bills and counterfactual
bills reuse the exact same lookup (no boundary pro-rating approximation anywhere).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PriceSchedule:
    """Vectorized price lookup over a 15-minute grid."""

    index: pd.DatetimeIndex  # 15-min grid, UTC
    series: pd.Series        # price per kWh at each interval
    buckets: pd.Series | None = None  # TOU bucket label at each interval (optional)

    def price_at(self, ts_index: pd.DatetimeIndex) -> pd.Series:
        return self.series.reindex(ts_index)

    def billed(self, kwh: pd.Series) -> float:
        """Bill for interval kWh aligned to self.index (NaN kwh excluded)."""
        aligned = kwh.reindex(self.index)
        return float((aligned * self.series).sum(skipna=True))


def _hour_in_window(hour: int, start: str | None, end: str | None) -> bool:
    if not start or not end or start == "00:00" and end == "24:00":
        return True
    hs, he = int(start.split(":")[0]), int(end.split(":")[0])
    if he == 24:
        return hour >= hs
    if hs <= he:
        return hs <= hour < he
    return hour >= hs or hour < he  # window crossing midnight


def build_price_schedule(tariff: pd.DataFrame, grid: pd.DatetimeIndex) -> PriceSchedule:
    """Map each grid interval to its price using (date range, bucket hours) tariff rows."""
    n = len(grid)
    price = pd.Series(np.nan, index=grid, dtype=float)
    if tariff.empty or n == 0:
        return PriceSchedule(grid, price)

    t = tariff.copy()
    t["valid_from"] = pd.to_datetime(t["valid_from"], utc=True)
    t["valid_to"] = pd.to_datetime(t["valid_to"], utc=True)
    hours = grid.hour.values
    dates = grid.normalize()  # midnight stamps

    for _, row in t.iterrows():
        in_date = (dates >= row["valid_from"].normalize()) & (dates <= row["valid_to"].normalize())
        mask = in_date
        if row.get("tou_start") is not None or row.get("tou_end") is not None:
            hmask = np.fromiter(
                (_hour_in_window(int(h), row.get("tou_start"), row.get("tou_end")) for h in hours),
                dtype=bool, count=n,
            )
            mask = mask & hmask
        # rows applied in ascending valid_from order => the latest applicable regime wins
        price[mask.values if hasattr(mask, "values") else mask] = float(row["price_per_kwh"])
    if price.isna().any():
        raise ValueError("tariff does not cover the full meter grid (missing intervals present)")
    return PriceSchedule(grid, price.sort_index())


def build_price_schedule_ordered(tariff: pd.DataFrame, grid: pd.DatetimeIndex) -> PriceSchedule:
    """As build_price_schedule, but applies rows in valid_from order so later regimes win."""
    if tariff.empty:
        return PriceSchedule(grid, pd.Series(np.nan, index=grid, dtype=float))
    t = tariff.sort_values("valid_from")
    return build_price_schedule(t, grid)


def legacy_schedule(tariff: pd.DataFrame, grid: pd.DatetimeIndex) -> PriceSchedule:
    """The tariff regime(s) in force at grid start, extended over the whole grid.

    Used for the counterfactual 'what if the old tariff had continued' — the base
    regime rows are projected forward to cover the current period.
    """
    if tariff.empty:
        return PriceSchedule(grid, pd.Series(np.nan, index=grid, dtype=float))
    start = grid.min()
    keep = tariff[pd.to_datetime(tariff["valid_from"], utc=True) <= start].copy()
    keep["valid_from"] = pd.to_datetime(keep["valid_from"], utc=True)
    keep["valid_to"] = pd.to_datetime(keep["valid_to"], utc=True)
    if keep.empty:
        keep = tariff[tariff["valid_from"] == tariff["valid_from"].min()].copy()
    last_from = keep["valid_from"].max()
    keep.loc[keep["valid_from"] == last_from, "valid_to"] = grid.max()
    return build_price_schedule_ordered(keep, grid)


def bucket_series(tariff: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.Series:
    """TOU bucket label (as object) at each grid interval, same matching rules."""
    n = len(grid)
    buckets = pd.Series([None] * n, index=grid, dtype=object)
    if tariff.empty or n == 0:
        return buckets
    t = tariff.copy()
    t["valid_from"] = pd.to_datetime(t["valid_from"], utc=True)
    t["valid_to"] = pd.to_datetime(t["valid_to"], utc=True)
    t = t.sort_values("valid_from")
    hours = grid.hour.values
    dates = grid.normalize()
    for _, row in t.iterrows():
        in_date = (dates >= row["valid_from"].normalize()) & (dates <= row["valid_to"].normalize())
        mask = in_date.to_numpy() if hasattr(in_date, "to_numpy") else np.asarray(in_date)
        if row.get("tou_start") is not None or row.get("tou_end") is not None:
            hmask = np.fromiter(
                (_hour_in_window(int(h), row.get("tou_start"), row.get("tou_end")) for h in hours),
                dtype=bool, count=n,
            )
            mask = mask & hmask
        buckets[mask] = str(row["tou_bucket"])
    return buckets
