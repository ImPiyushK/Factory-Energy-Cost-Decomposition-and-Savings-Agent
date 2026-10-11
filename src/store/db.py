"""SQLite persistence: canonical tables, data-version registry, run/trace store.

Design (per plan):
  * Every ingest writes a NEW version of a table (never overwrites); one version per
    table is `active` at a time. /analyze resolves active versions at run time and
    stamps fingerprints into the trace -> late data drop is automatically reflected.
  * Derived results are computed per run (no derived cache) so version swaps are
    always reflected without invalidation logic.
  * `runs` holds the full trace for GET /trace/{run_id}.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS meter_15min (
    ts TEXT NOT NULL, meter_id TEXT NOT NULL, kwh REAL, version_id INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS ix_meter_ts ON meter_15min (version_id, meter_id, ts);

CREATE TABLE IF NOT EXISTS tariff (
    valid_from TEXT NOT NULL, valid_to TEXT NOT NULL, tou_bucket TEXT NOT NULL,
    price_per_kwh REAL NOT NULL, tou_start TEXT, tou_end TEXT, version_id INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS production (
    date TEXT NOT NULL, product_id TEXT NOT NULL, volume REAL NOT NULL,
    shift TEXT, is_holiday INTEGER, version_id INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS ix_prod_date ON production (version_id, date);

CREATE TABLE IF NOT EXISTS weather (
    date TEXT NOT NULL, temperature_c REAL, cdd REAL, hdd REAL, version_id INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS data_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name TEXT NOT NULL, version INTEGER NOT NULL,
    sha256 TEXT NOT NULL, row_count INTEGER NOT NULL, rows_rejected INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL, loaded_at TEXT NOT NULL,
    status TEXT NOT NULL, warnings_json TEXT NOT NULL DEFAULT '[]', errors_json TEXT NOT NULL DEFAULT '[]',
    active INTEGER NOT NULL DEFAULT 0);

CREATE TABLE IF NOT EXISTS rejected_rows (
    version_id INTEGER NOT NULL, row_index INTEGER, reason TEXT, raw TEXT);

CREATE TABLE IF NOT EXISTS corrections_log (
    version_id INTEGER NOT NULL, kind TEXT NOT NULL, detail TEXT NOT NULL, count INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS validation_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, report_json TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
    question TEXT NOT NULL, params_json TEXT NOT NULL, data_fingerprints_json TEXT NOT NULL,
    validation_json TEXT NOT NULL, steps_json TEXT NOT NULL, tool_calls_json TEXT NOT NULL,
    model_json TEXT NOT NULL, llm_calls INTEGER NOT NULL DEFAULT 0, tokens_json TEXT NOT NULL,
    errors_json TEXT NOT NULL, faithfulness_json TEXT NOT NULL,
    final_answer_json TEXT NOT NULL, latency_ms INTEGER NOT NULL DEFAULT 0);
"""


def db_path() -> Path:
    p = Path(settings.resolve(settings.db_path))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path()), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]  # short fingerprint (full hash not needed for trace)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id() -> str:
    return "en-" + uuid.uuid4().hex[:6]


# ------------------------- data-version registry -------------------------

def register_version(
    conn: sqlite3.Connection,
    table: str,
    source: str,
    sha: str,
    row_count: int,
    rows_rejected: int,
    status: str,
    warnings: list[str],
    errors: list[str],
    set_active: bool = True,
) -> int:
    """Insert a new version of `table` and (optionally) make it the active one."""
    latest = conn.execute(
        "SELECT COALESCE(MAX(version),0) v FROM data_versions WHERE table_name=?", (table,)
    ).fetchone()["v"]
    cur = conn.execute(
        """INSERT INTO data_versions
           (table_name, version, sha256, row_count, rows_rejected, source, loaded_at,
            status, warnings_json, errors_json, active)
           VALUES (?,?,?,?,?,?,?,?,?,?,0)""",
        (table, latest + 1, sha, row_count, rows_rejected, source, now_iso(), status,
         json.dumps(warnings), json.dumps(errors)),
    )
    vid = cur.lastrowid
    if set_active and status != "rejected":
        conn.execute("UPDATE data_versions SET active=0 WHERE table_name=?", (table,))
        conn.execute("UPDATE data_versions SET active=1 WHERE id=?", (vid,))
    return vid


def get_active_versions(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, table_name, version, sha256, row_count, loaded_at, source, status"
        " FROM data_versions WHERE active=1"
    ).fetchall()
    return {
        r["table_name"]: {
            "version_id": r["id"], "version": r["version"], "sha256": r["sha256"],
            "row_count": r["row_count"], "loaded_at": r["loaded_at"], "source": r["source"],
            "status": r["status"],
        }
        for r in rows
    }


def load_log_corrections(conn: sqlite3.Connection, version_id: int, corrections: dict[str, int]) -> None:
    for kind, count in corrections.items():
        if count:
            conn.execute(
                "INSERT INTO corrections_log (version_id, kind, detail, count) VALUES (?,?,?,?)",
                (version_id, kind, "", count),
            )


# ------------------------- run / trace store -------------------------

def save_run(record: dict[str, Any]) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO runs
               (run_id, created_at, question, params_json, data_fingerprints_json, validation_json,
                steps_json, tool_calls_json, model_json, llm_calls, tokens_json, errors_json,
                faithfulness_json, final_answer_json, latency_ms)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record["run_id"], record["created_at"], record["question"],
                json.dumps(record["params"]), json.dumps(record["data_fingerprints"]),
                json.dumps(record["validation"]), json.dumps(record["steps"]),
                json.dumps(record["tool_calls"]), json.dumps(record["model"]),
                record["llm_calls"], json.dumps(record["tokens"]), json.dumps(record["errors"]),
                json.dumps(record["faithfulness"]), json.dumps(record["final_answer"]),
                record["latency_ms"],
            ),
        )


def get_run(run_id: str) -> dict[str, Any] | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if row is None:
        return None
    return {
        "run_id": row["run_id"], "created_at": row["created_at"], "question": row["question"],
        "params": json.loads(row["params_json"]),
        "data_fingerprints": json.loads(row["data_fingerprints_json"]),
        "validation": json.loads(row["validation_json"]),
        "steps": json.loads(row["steps_json"]),
        "tool_calls": json.loads(row["tool_calls_json"]),
        "model": json.loads(row["model_json"]), "llm_calls": row["llm_calls"],
        "tokens": json.loads(row["tokens_json"]), "errors": json.loads(row["errors_json"]),
        "faithfulness": json.loads(row["faithfulness_json"]),
        "final_answer": json.loads(row["final_answer_json"]), "latency_ms": row["latency_ms"],
    }


def list_runs(limit: int = 50) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT run_id, created_at, question, answer_type FROM ("
            "  SELECT run_id, created_at, question, json_extract(final_answer_json,'$.answer_type') answer_type"
            "  FROM runs ORDER BY created_at DESC LIMIT ?)",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def save_validation_report(report: dict[str, Any]) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO validation_reports (created_at, report_json) VALUES (?,?)",
            (now_iso(), json.dumps(report)),
        )
        return cur.lastrowid


def last_validation_report() -> dict[str, Any] | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT report_json FROM validation_reports ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return json.loads(row["report_json"]) if row else None
