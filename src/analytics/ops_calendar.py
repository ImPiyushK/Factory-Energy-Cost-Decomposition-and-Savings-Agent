"""Operational calendar: derives production / off-production windows from the shift data.

Documented default (used when the source data has no explicit hour columns):
  * weekday production: 06:00-22:00 (two shifts)
  * Saturday production (volume > 0): 06:00-14:00
  * nights: 22:00-06:00 every day
  * non-production days (volume == 0 or holiday): whole day is off-production
  * public holidays logged as a normal shift are STILL off-production (edge case 4.4)
"""
from __future__ import annotations

from datetime import time as dtime

import numpy as np
import pandas as pd

from src.config import settings

WEEKDAY_START, WEEKDAY_END = dtime(6, 0), dtime(22, 0)
SATURDAY_START, SATURDAY_END = dtime(6, 0), dtime(14, 0)
NIGHT_START, NIGHT_END = dtime(settings.night_start), dtime(settings.night_end)


def daytype_of(date: pd.Timestamp, holiday: bool) -> str:
    if holiday:
        return "holiday"
    return "weekend" if date.dayofweek >= 5 else "weekday"


def is_production_day(volume_day: float, holiday: bool) -> bool:
    return (not holiday) and volume_day > 0


def day_hours(daytype: str, volume_day: float) -> tuple[dtime, dtime] | None:
    """Production hour window for a day, or None if not a production day."""
    if not is_production_day(volume_day, daytype == "holiday"):
        return None
    if daytype == "weekend":  # Saturday (Sun handled as non-production above)
        return (SATURDAY_START, SATURDAY_END)
    return (WEEKDAY_START, WEEKDAY_END)


def off_production_mask(ts_index: pd.DatetimeIndex,
                        hour_windows: dict[pd.Timestamp, tuple[dtime, dtime] | None]) -> np.ndarray:
    """Boolean mask over the 15-min grid: True where NO production is happening.

    A day without a production window is off-production all day; within production
    days, the nights/outside-shift hours are off-production.
    """
    h_frac = ts_index.hour + ts_index.minute / 60.0
    out = np.ones(len(ts_index), dtype=bool)
    for date, win in hour_windows.items():
        idx = ts_index.normalize() == date  # dates are UTC-midnight stamps
        if not idx.any():
            continue
        if win is None:
            out[idx] = True
            continue
        lo = win[0].hour + win[0].minute / 60.0
        hi = win[1].hour + win[1].minute / 60.0
        out[idx] = ~((h_frac[idx] >= lo) & (h_frac[idx] < hi))
    return out


def production_calendar_from_production_df(production_df: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Per-date flags: (prod_flags, hour_windows, daytypes) from the production table."""
    if production_df is None or production_df.empty:
        return {}, {}, {}
    p = production_df.copy()
    p["date"] = pd.to_datetime(p["date"], utc=True).dt.normalize()
    per_day = p.groupby("date").agg(volume=("volume", "sum"), shift=("shift", "first"),
                                    is_holiday=("is_holiday", "first"))
    per_day["is_holiday"] = per_day["is_holiday"].fillna(0).astype(int) > 0
    prod_flags: dict[pd.Timestamp, bool] = {}
    hour_windows: dict[pd.Timestamp, tuple | None] = {}
    daytypes: dict[pd.Timestamp, str] = {}
    for date, row in per_day.iterrows():
        dt = daytype_of(date, bool(row["is_holiday"]))
        daytypes[date] = dt
        prod_flags[date] = is_production_day(float(row["volume"]), bool(row["is_holiday"]))
        hour_windows[date] = day_hours(dt, float(row["volume"]))
    return prod_flags, hour_windows, daytypes
