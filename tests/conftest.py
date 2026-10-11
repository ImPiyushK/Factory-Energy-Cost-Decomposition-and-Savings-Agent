"""Shared test configuration.

Runs against a dedicated test database (fresh), the repository's committed synthetic
data, and MOCK_LLM=true so tests are fast, deterministic and network-free.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["DB_PATH"] = str(ROOT / "data" / "test_energy_agent.db")
os.environ["MOCK_LLM"] = "true"

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def loaded_db():
    """Load the synthetic pack into the test DB once for the session."""
    from src.ingest import loader
    from src.store import db

    db.init_db()
    with db.get_conn() as conn:
        have = db.get_active_versions(conn)
    if "meter_15min" not in have:
        for table in ("tariff", "production", "weather", "meter_15min"):
            loader.ingest_table(table, ROOT / "data" / "synthetic" / f"{table}.csv",
                                source="tests")
    return ROOT / "data" / "test_energy_agent.db"


@pytest.fixture(scope="session")
def frames(loaded_db):
    from src.ingest import loader
    return loader.load_active_frames()


@pytest.fixture(scope="session")
def ctx(frames):
    from src.graph import engine
    from src.validate import checks as vchecks
    report = vchecks.run_all_checks(frames, tolerance=0.05, strict=True)
    assert report["ok"], "synthetic data must pass strict validation"
    return {"frames": frames,
            "quarters": engine.available_quarters(frames["meter_15min"]["ts"])}
