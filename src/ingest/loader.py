"""CSV ingestion for Annex-B-schema files with a thin schema-adapter layer.

Judge-provided files (Annex B) or our synthetic files flow through the same path:
  read CSV -> map columns (aliases) -> coerce types -> clean -> validate -> version-scoped write.

Cleaning policy (documented, logged, never silent):
  * duplicates                -> dropped, logged
  * off-grid timestamps       -> warned; grid reindexed to 15 min
  * gaps <= 2 intervals (30m) -> linearly interpolated, logged
  * larger gaps               -> left missing; excluded from totals (count logged)
  * cumulative meters         -> auto-detected, converted to interval kWh
  * meter resets (counter)    -> rebased (carry-forward), logged
  * negative interval kWh     -> row rejected with reason (logged)
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import settings
from src.store import db
from src.validate import checks as vchecks

ALIASES: dict[str, dict[str, list[str]]] = {
    "meter_15min": {
        "ts": ["timestamp", "ts", "datetime", "time", "interval_start", "reading_time"],
        "meter_id": ["meter_id", "meter", "meter_name", "asset", "asset_id"],
        "kwh": ["kwh", "energy_kwh", "consumption_kwh", "energy", "kwh_interval", "value"],
    },
    "tariff": {
        "valid_from": ["valid_from", "effective_from", "from_date", "start"],
        "valid_to": ["valid_to", "effective_to", "to_date", "end"],
        "tou_bucket": ["tou_bucket", "tou", "bucket", "period", "rate_period", "time_of_use"],
        "price_per_kwh": ["price_per_kwh", "price", "rate", "kwh_price", "unit_price"],
        "tou_start": ["tou_start", "start_time", "from_time", "hours_from"],
        "tou_end": ["tou_end", "end_time", "to_time", "hours_to"],
    },
    "production": {
        "date": ["date", "day", "production_date"],
        "product_id": ["product_id", "product", "sku", "product_code", "item"],
        "volume": ["volume", "units", "quantity", "units_produced", "production_volume"],
        "shift": ["shift", "shift_code", "shift_type"],
        "is_holiday": ["is_holiday", "holiday"],
    },
    "weather": {
        "date": ["date", "day"],
        "temperature_c": ["temperature_c", "temp", "temperature", "temp_c", "avg_temp", "avg_temp_c"],
        "cdd": ["cdd", "cooling_degree_days", "cdd_18"],
        "hdd": ["hdd", "heating_degree_days", "hdd_18"],
    },
}
TABLES = list(ALIASES)


def adapt_columns(df: pd.DataFrame, table: str) -> tuple[pd.DataFrame, list[str]]:
    """Map judge/official column names to canonical names. Missing required cols => error."""
    errors: list[str] = []
    lower = {c.strip().lower().replace(" ", "_"): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    for canon, aliases in ALIASES[table].items():
        found = next((lower[a] for a in aliases if a in lower), None)
        if found:
            out[canon] = df[found]
        elif canon in ("shift", "is_holiday", "tou_start", "tou_end", "cdd", "hdd", "temperature_c"):
            out[canon] = np.nan  # optional column
        else:
            errors.append(f"missing required column '{canon}' (aliases: {aliases})")
    return (out, errors) if not errors else (out, errors)


# ------------------------------- meter cleaning -------------------------------

def _detect_cumulative(v: pd.Series) -> bool:
    if len(v) < 50:
        return False
    d = v.diff().dropna()
    nonneg = (d >= 0).mean()
    return bool(nonneg >= 0.995 and (v.max() > 10 * d.abs().median() + 1))


def clean_meter(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], list[tuple[int, str, str]]]:
    """Returns (clean_df, corrections_counter, rejected_rows[(idx, reason, raw)])."""
    corrections: dict[str, int] = {}
    rejected: list[tuple[int, str, str]] = []

    ts = pd.to_datetime(df["ts"], utc=True, errors="coerce", format="mixed")
    bad_ts = ts.isna()
    rejected.extend((int(i), "unparseable timestamp", str(r)) for i, r in zip(df.index[bad_ts], df.loc[bad_ts, "ts"]))
    df = df[~bad_ts].copy()
    df["ts"] = ts[~bad_ts]
    df["ts"] = df["ts"].dt.floor("15min")
    off_grid = int(((ts.dt.minute % 15 != 0)).sum())
    if off_grid:
        corrections["off_grid_floored"] = off_grid

    df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce")
    df["meter_id"] = df["meter_id"].astype(str).str.strip()

    before = len(df)
    df = df.sort_values(["meter_id", "ts"]).drop_duplicates(subset=["meter_id", "ts"], keep="last")
    corrections["duplicates_dropped"] = before - len(df)

    cleaned_parts: list[pd.DataFrame] = []
    for meter_id, g in df.groupby("meter_id"):
        g = g.sort_values("ts").copy()
        v = g["kwh"]

        if _detect_cumulative(v):
            corrections[f"cumulative_converted:{meter_id}"] = len(g)
            raw = v.copy()
            offset = 0.0
            intervals: list[float | None] = [None]  # first interval unknown
            prev_raw = float(raw.iloc[0])
            for i in range(1, len(raw)):
                r_i = float(raw.iloc[i])
                interval = r_i - prev_raw
                if interval < 0:  # meter reset: counter assumed restarted at ~0
                    offset += prev_raw
                    interval = r_i
                    corrections[f"meter_reset_rebased:{meter_id}"] = corrections.get(f"meter_reset_rebased:{meter_id}", 0) + 1
                intervals.append(interval)
                prev_raw = r_i
            g["kwh"] = pd.Series([float(x) if x is not None else np.nan for x in intervals], index=g.index)
        else:
            neg = v < 0
            for i, raw in zip(g.index[neg], v[neg]):
                rejected.append((int(i), "negative interval kWh", f"{raw}"))
            g.loc[neg, "kwh"] = np.nan
            if neg.any():
                corrections[f"negative_rejected:{meter_id}"] = int(neg.sum())

        # regular 15-min grid
        full = pd.date_range(g["ts"].min(), g["ts"].max(), freq="15min", tz="UTC")
        g = g.set_index("ts").reindex(full)
        g.index.name = "ts"
        missing = g["kwh"].isna()
        g["meter_id"] = meter_id
        # small gaps: interpolate runs of length <= 2
        run_id = (missing != missing.shift()).cumsum()
        run_len = missing.groupby(run_id).transform("sum")
        small = missing & (run_len <= 2)
        n_small = int(small.sum())
        if n_small:
            g["kwh"] = g["kwh"].interpolate(method="time", limit=2)
            corrections[f"gap_interpolated:{meter_id}"] = n_small
        still_missing = int(g["kwh"].isna().sum())
        if still_missing:
            corrections[f"gap_left_missing:{meter_id}"] = still_missing
        cleaned_parts.append(g.reset_index()[["ts", "meter_id", "kwh"]])

    if not cleaned_parts:
        return pd.DataFrame(columns=["ts", "meter_id", "kwh"]), corrections, rejected
    out = pd.concat(cleaned_parts, ignore_index=True)
    return out, corrections, rejected


def clean_tariff(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], list[tuple[int, str, str]]]:
    corrections: dict[str, int] = {}
    rejected: list[tuple[int, str, str]] = []
    # Dates are normalized to UTC. A date-only valid_to means end of that day (inclusive).
    def _to_utc(s: pd.Series) -> pd.Series:
        return pd.to_datetime(s, utc=True, errors="coerce", format="mixed")
    vf, vt = _to_utc(df["valid_from"]), _to_utc(df["valid_to"])
    midnight = (vt.dt.hour == 0) & (vt.dt.minute == 0) & (vt.dt.second == 0)
    df["valid_from"] = vf
    df["valid_to"] = vt.where(~midnight, vt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    df["price_per_kwh"] = pd.to_numeric(df["price_per_kwh"], errors="coerce")
    bad = df["valid_from"].isna() | df["valid_to"].isna() | df["price_per_kwh"].isna() | (df["price_per_kwh"] <= 0)
    rejected.extend((int(i), "invalid tariff row (dates/price)", str(r)) for i, r in zip(df.index[bad], df.loc[bad].to_dict("records")))
    df = df[~bad].copy()
    if df.empty:
        return df, corrections, rejected
    if df["tou_bucket"].isna().all():
        df["tou_bucket"] = "flat"
    df["tou_bucket"] = df["tou_bucket"].astype(str).str.lower()
    if df["tou_start"].isna().all() and df["tou_end"].isna().all():
        df["tou_start"], df["tou_end"] = "00:00", "24:00"
        corrections["tariff_flat_hours_assumed"] = len(df)
    return df, corrections, rejected


def clean_production(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], list[tuple[int, str, str]]]:
    corrections: dict[str, int] = {}
    rejected: list[tuple[int, str, str]] = []
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_date = df["date"].isna()
    rejected.extend((int(i), "unparseable date", str(r)) for i, r in zip(df.index[bad_date], df.loc[bad_date, "date"]))
    df = df[~bad_date].copy()
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    neg = df["volume"] < 0
    rejected.extend((int(i), "negative volume", str(v)) for i, v in zip(df.index[neg], df.loc[neg, "volume"]))
    df = df[~neg].copy()
    if "shift" in df:
        df["shift"] = df["shift"].astype("object").where(df["shift"].notna(), None)
        df["shift"] = df["shift"].apply(lambda s: str(s).lower().strip() if s is not None else None)
    if "is_holiday" in df:
        df["is_holiday"] = pd.to_numeric(df["is_holiday"], errors="coerce").fillna(0).astype(int)
    else:
        df["is_holiday"] = 0
    if "product_id" in df:
        df["product_id"] = df["product_id"].astype(str).str.strip()
    return df, corrections, rejected


def clean_weather(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], list[tuple[int, str, str]]]:
    corrections: dict[str, int] = {}
    rejected: list[tuple[int, str, str]] = []
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[~df["date"].isna()].copy()
    df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")
    base = settings.cdd_base
    if "cdd" not in df or df["cdd"].isna().all():
        df["cdd"] = (df["temperature_c"] - base).clip(lower=0)
        corrections["cdd_computed_from_temperature"] = int(df["cdd"].notna().sum())
    if "hdd" not in df or df["hdd"].isna().all():
        df["hdd"] = (base - df["temperature_c"]).clip(lower=0)
        corrections["hdd_computed_from_temperature"] = int(df["hdd"].notna().sum())
    return df, corrections, rejected


CLEANERS = {"meter_15min": clean_meter, "tariff": clean_tariff, "production": clean_production, "weather": clean_weather}


# ------------------------------- ingest pipeline -------------------------------

def ingest_table(
    table: str,
    file_path: str | Path,
    *,
    source: str = "upload",
    set_active: bool = True,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Full ingest pipeline. Returns an IngestResponse-shaped dict."""
    table = table.strip().lower()
    if table not in TABLES:
        return {"table": table, "rows_loaded": 0, "rows_rejected": 0, "status": "rejected",
                "version_id": None, "warnings": [], "errors": [f"unknown table '{table}'"],
                "validation": {}}
    path = Path(file_path)
    if not path.exists():
        return {"table": table, "rows_loaded": 0, "rows_rejected": 0, "status": "rejected",
                "version_id": None, "warnings": [], "errors": [f"file not found: {path}"], "validation": {}}

    raw = pd.read_csv(path)
    df, errors = adapt_columns(raw, table)
    if errors:
        return {"table": table, "rows_loaded": 0, "rows_rejected": 0, "status": "rejected",
                "version_id": None, "warnings": [], "errors": errors, "validation": {}}

    cleaned, corrections, rejected = CLEANERS[table](df)
    sha = db.sha256_file(path)

    own_conn = conn is None
    conn = conn or db.get_conn()
    try:
        # Build the candidate frame set (active versions + the new candidate) for cross-table checks
        if set_active:
            frames = {t: read_active_table(conn, t) for t in TABLES if t != table}
            frames[table] = cleaned
        else:
            frames = {t: read_active_table(conn, t) for t in TABLES}

        card = _load_card(path)
        report = vchecks.run_all_checks(frames, tolerance=settings.recon_tolerance,
                                        corrections=corrections, card=card)
        status = "rejected" if report["blockers"] or errors else (
            "loaded_with_warnings" if report["warnings"] else "loaded")

        vid = None
        if not report["blockers"]:
            with conn:
                vid = db.register_version(conn, table, str(path), sha, len(cleaned),
                                          len(rejected), status, report["warnings"], errors,
                                          set_active=set_active)
                cleaned["version_id"] = vid
                _write_rows(conn, table, cleaned)
                for idx, reason, rawrow in rejected:
                    conn.execute("INSERT INTO rejected_rows (version_id, row_index, reason, raw) VALUES (?,?,?,?)",
                                 (vid, idx, reason, rawrow))
                db.load_log_corrections(conn, vid, {k.split(":")[0]: v for k, v in corrections.items()})
        # save report on a separate connection AFTER the ingest transaction commits
        db.save_validation_report(report)

        result = {
            "table": table, "rows_loaded": len(cleaned) if vid is not None else 0,
            "rows_rejected": len(rejected), "status": status, "version_id": vid,
            "warnings": report["warnings"], "errors": errors + report["blockers"],
            "validation": {"summary": report["summary"], "checks": report["checks"], "corrections": report["corrections"]},
        }
        return result
    finally:
        if own_conn:
            conn.close()


def _write_rows(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> None:
    cols = {
        "meter_15min": ["ts", "meter_id", "kwh", "version_id"],
        "tariff": ["valid_from", "valid_to", "tou_bucket", "price_per_kwh", "tou_start", "tou_end", "version_id"],
        "production": ["date", "product_id", "volume", "shift", "is_holiday", "version_id"],
        "weather": ["date", "temperature_c", "cdd", "hdd", "version_id"],
    }
    use = [c for c in cols[table] if c in df.columns]
    df2 = df[use].copy()
    for c in ("ts", "valid_from", "valid_to", "date"):
        if c in df2:
            df2[c] = pd.to_datetime(df2[c], utc=True).astype(str)
    conn.executemany(
        f"INSERT INTO {table} ({','.join(use)}) VALUES ({','.join('?' * len(use))})",
        df2.itertuples(index=False, name=None),
    )


def _load_card(file_path: Path) -> dict[str, Any] | None:
    """Look for a data card next to the file, then in DATA_DIR."""
    for cand in (file_path.parent / "data_card.json", Path(settings.resolve(settings.data_dir)) / "data_card.json"):
        if cand.exists():
            try:
                return json.loads(cand.read_text())
            except Exception:
                return None
    return None


# ------------------------------- read side -------------------------------

def read_active_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    cols = {
        "meter_15min": "ts, meter_id, kwh",
        "tariff": "valid_from, valid_to, tou_bucket, price_per_kwh, tou_start, tou_end",
        "production": "date, product_id, volume, shift, is_holiday",
        "weather": "date, temperature_c, cdd, hdd",
    }
    vid = conn.execute("SELECT id FROM data_versions WHERE table_name=? AND active=1", (table,)).fetchone()
    if vid is None:
        return pd.DataFrame()
    df = pd.read_sql_query(f"SELECT {cols[table]} FROM {table} WHERE version_id=?", conn, params=(vid["id"],))
    return df


def load_active_frames(conn: sqlite3.Connection | None = None) -> dict[str, pd.DataFrame]:
    own = conn is None
    conn = conn or db.get_conn()
    try:
        return {t: read_active_table(conn, t) for t in TABLES}
    finally:
        if own:
            conn.close()
