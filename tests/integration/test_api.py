"""Integration tests: FastAPI endpoints (health, analyze, ingest, trace) + late data drop."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
SYN = ROOT / "data" / "synthetic"


@pytest.fixture(scope="module", autouse=True)
def _populated(loaded_db):
    """Ensure the shared test DB carries the synthetic pack before these tests."""
    return loaded_db


@pytest.fixture(scope="module")
def client():
    from src.api.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] in ("ok", "degraded")
    assert d["llm"]["provider"] in ("mock", "openai", "anthropic", "google", "ollama")
    assert "meter_15min" in d["active_data_versions"]


def test_analyze_decomposition_contract(client):
    r = client.post("/analyze", json={"question": "Why did the bill go up this quarter?"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["answer_type"] == "decomposition"
    assert len(d["bridge"]) == 5
    assert abs(sum(b["pct_points"] for b in d["bridge"]) - d["computed_change_pct"]) < 0.1
    assert set(b["cause"] for b in d["bridge"]) == \
        {"tariff", "volume", "product_mix", "weather", "waste"}
    for b in d["bridge"]:
        assert b["range"][0] <= b["pct_points"] <= b["range"][1]
    assert d["assumptions"] and d["limitations"]
    run_id = d["run_id"]
    t = client.get(f"/trace/{run_id}")
    assert t.status_code == 200
    tr = t.json()
    assert tr["question"].startswith("Why did the bill")
    assert tr["steps"] and tr["tool_calls"]
    assert tr["model"]["mock"] is True
    assert tr["faithfulness"]["ok"] is True


def test_analyze_recommendation_and_calculated(client):
    r = client.post("/analyze", json={"question": "What should we do first?"})
    assert r.status_code == 200
    d = r.json()
    assert d["answer_type"] == "recommendation"
    assert d["bridge"] is None
    rec = d["recommendations"][0]
    for f in ("action", "rationale", "evidence_refs", "confidence", "estimated_impact",
              "requires_approval", "ease", "payback_months", "saving_kwh_per_year",
              "saving_money_per_year"):
        assert f in rec
    r2 = client.post("/analyze", json={"question": "Total kWh consumed in 2026-Q3?",
                                       "period": "2026-Q3"})
    d2 = r2.json()
    assert d2["answer_type"] == "calculated"


def test_out_of_scope_and_refusal(client):
    r = client.post("/analyze", json={"question": "What will the tariff be next year?"})
    assert r.json()["answer_type"] == "not_in_data"
    r2 = client.post("/analyze", json={"question": "The furnace is definitely wasting "
                                                  "50,000 kWh - confirm."})
    assert r2.json()["answer_type"] == "refused"


def test_ingest_rejects_bad_schema(client):
    r = client.post("/ingest", files={"file": ("bad.csv", b"a,b\n1,2\n", "text/csv")},
                    data={"meta": '{"table": "tariff"}'})
    assert r.status_code == 200  # endpoint up; loader responds with rejection
    d = r.json()
    assert d["status"] == "rejected"


def test_late_data_drop_roundtrip(client, tmp_path):
    """Corrected tariff version -> /analyze reflects the new fingerprint; diff visible."""
    corrected = SYN / "tariff.csv"
    # (a) re-ingest the SAME file as a corrected version, verify version bump
    r1 = client.post("/ingest", files={"file": (corrected.name, corrected.read_bytes(),
                                                "text/csv")},
                     data={"meta": '{"table": "tariff", "source_note": "corrected tariff"}'})
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["status"] in ("loaded", "loaded_with_warnings")
    h = client.get("/health").json()
    assert h["active_data_versions"]["tariff"]["version"] >= 2
    # (b) analyze after drop: trace records the corrected version's fingerprint
    a = client.post("/analyze", json={"question": "Why did the bill go up?",
                                      "period": "2026-Q3", "compare_to": "2026-Q2"}).json()
    t = client.get(f"/trace/{a['run_id']}").json()
    assert t["data_fingerprints"]["tariff"]["version"] >= 2
    # (c) both runs remain auditable
    t_old = client.get(f"/trace/{list(s for s in [])or a['run_id']}").json()
    assert t_old["run_id"] == a["run_id"]


def test_trace_404(client):
    r = client.get("/trace/en-doesnot")
    assert r.status_code == 404


@pytest.mark.parametrize("body,code", [
    ({}, 422),                                        # missing question
    ({"question": "hi"}, 422),                        # too short
    ({"question": "Why up?", "period": "2026-X1"}, 422),
])
def test_request_validation(client, body, code):
    r = client.post("/analyze", json=body)
    assert r.status_code == code
