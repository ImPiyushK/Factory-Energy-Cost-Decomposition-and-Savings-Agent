"""Period helpers and quarter utilities."""
from __future__ import annotations

import re

import pandas as pd


def quarter_bounds(quarter: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """'2026-Q3' -> (2026-07-01 00:00 UTC, 2026-09-30 23:59:59 UTC)."""
    m = re.fullmatch(r"(\d{4})-Q([1-4])", str(quarter).strip())
    if not m:
        raise ValueError(f"invalid quarter label: {quarter!r} (expected e.g. '2026-Q3')")
    year, q = int(m.group(1)), int(m.group(2))
    start_month = 3 * (q - 1) + 1
    start = pd.Timestamp(year=year, month=start_month, day=1, tz="UTC")
    end_month = start_month + 3
    if end_month > 12:
        end = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC") - pd.Timedelta(seconds=1)
    else:
        end = pd.Timestamp(year=year, month=end_month, day=1, tz="UTC") - pd.Timedelta(seconds=1)
    return start, end


def quarter_of(ts: pd.Timestamp) -> str:
    return f"{ts.year}-Q{((ts.month - 1) // 3) + 1}"


def available_quarters(meter_ts: pd.Series) -> list[str]:
    """Quarters with substantial meter coverage (>=20 distinct days), ascending."""
    ts = pd.to_datetime(meter_ts, utc=True).dt.tz_convert("UTC")
    df = pd.DataFrame({"q": ts.map(quarter_of), "d": ts.dt.date})
    per_q = df.groupby("q")["d"].nunique()
    return sorted(per_q[per_q >= 20].index.tolist())
