"""FastAPI app: POST /analyze, POST /ingest, GET /health, GET /trace/{run_id}."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.graph.workflow import GRAPH, run as run_graph
from src.ingest import loader
from src.schemas import (
    AnalyzeRequest, AnalyzeResponse, HealthResponse, IngestResponse, TraceResponse,
)
from src.store import db

app = FastAPI(title="Factory Energy Cost Decomposition & Savings Agent",
              version=settings.app_version,
              description="Bill decomposition, waste detection and savings ranking - "
                          "all numbers computed by verified tools.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

db.init_db()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    from src.llm.provider import llm_reachable

    store_ok, versions = True, {}
    try:
        conn = db.get_conn()
        try:
            conn.execute("SELECT 1").fetchall()
            versions = db.get_active_versions(conn)
        finally:
            conn.close()
    except Exception as e:  # pragma: no cover
        store_ok = False
        versions = {"error": str(e)}

    llm = dict(llm_reachable()) if not isinstance(llm_reachable(), tuple) else llm_reachable()[0]
    degraded = (not store_ok) or (not llm.get("reachable", True))
    last_val = db.last_validation_report()
    return HealthResponse(
        status="degraded" if degraded else "ok",
        api={"status": "ok", "version": settings.app_version},
        data_store={"ok": store_ok, "path": str(settings.resolve(settings.db_path)),
                    "tables_loaded": sorted(versions)},
        llm={"provider": settings.llm_provider, "model": settings.llm_model,
             "mock": settings.mock_llm, **llm},
        active_data_versions=versions,
        last_validation=last_val,
        version=settings.app_version,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    try:
        payload = run_graph(req.question, req.period, req.compare_to)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"analysis failed: {type(e).__name__}: {e}") from e
    # the API response carries the contract fields; verified internals stay in the trace
    api_payload = {k: v for k, v in payload.items() if k != "vres"}
    try:
        return AnalyzeResponse(**api_payload)
    except Exception as e:  # contract violation -> internal error with the run_id for audit
        raise HTTPException(status_code=500, detail=f"response contract violation: {e}; "
                                                    f"run_id={payload.get('run_id')} (trace available)") from e


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...), meta: str = Form(...)) -> IngestResponse:
    """Add data while running. `meta` is a JSON naming the table (+optional options)."""
    import json

    try:
        meta_obj = json.loads(meta) if isinstance(meta, str) else meta
    except Exception:
        raise HTTPException(status_code=422, detail="meta must be valid JSON naming the table")
    table = str(meta_obj.get("table", "")).strip().lower()
    if table not in loader.TABLES:
        raise HTTPException(status_code=422,
                            detail=f"table must be one of {loader.TABLES} (got '{table}')")
    suffix = Path(file.filename or "upload.csv").suffix or ".csv"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        set_active = bool(meta_obj.get("set_active", True))
        result = loader.ingest_table(table, tmp_path, source=f"api:{file.filename}",
                                     set_active=set_active)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return IngestResponse(**result)


@app.get("/trace/{run_id}", response_model=TraceResponse)
def trace(run_id: str) -> TraceResponse:
    record = db.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found")
    return TraceResponse(**record)
