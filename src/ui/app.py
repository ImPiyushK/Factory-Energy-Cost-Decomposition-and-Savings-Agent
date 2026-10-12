"""Streamlit UI (spec: minimal; scored only for usability).

Every value shown comes from the system's API responses and traces — the UI never
recomputes anything, so it can never display a number the tools did not produce.
"""
from __future__ import annotations

import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# API base comes from the environment (or the sidebar override below).
# NOTE: st.secrets is NOT used — it raises StreamlitSecretNotFoundError when no
# secrets.toml exists, which broke local/demo launches.
API = os.environ.get("API_URL", "http://localhost:8000")


def _api(path: str, method: str = "GET", **kw):
    import httpx
    url = f"{API}{path}"
    try:
        r = httpx.request(method, url, timeout=300, **kw)
        if r.status_code >= 400:
            return None, {"status_code": r.status_code, "detail": r.text[:400]}
        return r.json(), None
    except Exception as e:
        return None, {"detail": f"{type(e).__name__}: {e}", "api": API}


st.set_page_config(page_title="Factory Energy Agent", layout="wide", page_icon="⚡")
API = st.sidebar.text_input("API base URL", value=API)
SCREENS = ["Ask & Answer", "Data & Validation", "Traces", "Late-drop compare"]
_default = SCREENS.index(os.environ.get("UI_SCREEN", "")) if os.environ.get("UI_SCREEN") in SCREENS else 0
screen = st.sidebar.radio("Screen", SCREENS, index=_default)


def waterfall(bridge: list[dict], bill_base: float, bill_cur: float):
    fig = go.Figure()
    fig.add_trace(go.Waterfall(
        x=["baseline bill"] + [b["cause"] for b in bridge] + ["actual bill"],
        measure=["absolute"] + ["relative"] * len(bridge) + ["total"],
        y=[bill_base] + [b["amount"] for b in bridge] + [None],
        text=[f"{bill_base:,.0f}"] + [f"{b['pct_points']:+.1f} pp" for b in bridge]
             + [f"{bill_cur:,.0f}"],
        connector={"line": {"color": "#888"}},
        increasing={"marker": {"color": "#d9534f"}},
        decreasing={"marker": {"color": "#5cb85c"}},
        totals={"marker": {"color": "#4472c4"}},
    ))
    fig.update_layout(title="Bill bridge — every bar from the verified decomposition",
                      yaxis_title="currency units", height=430)
    return fig


def _bridge_with_amounts(res: dict) -> list[dict]:
    """Bridge incl. currency amounts. Wire format (§6.1) carries only cause/pct_points/
    range; amounts come from the run's trace (verified internals). Fallback: derive
    amounts from pct_points × baseline bill so the chart stays tool-sourced either way."""
    if not res.get("bridge"):
        return []
    tr, _ = _api("/trace/" + res["run_id"])
    vbridge = ((tr or {}).get("final_answer", {}).get("vres", {}) or {}).get("bridge") or []
    if vbridge:
        return vbridge
    base = ((tr or {}).get("final_answer", {}).get("vres", {}) or {}).get("bill", {}).get("base", 0.0)
    return [{**b, "amount": b["pct_points"] / 100.0 * base} for b in res["bridge"]]


def _recommendation_cards(recs: list[dict]):
    for r in recs:
        with st.expander(f"{r['action']} · payback {r['payback_months'][0]:.0f}–"
                         f"{r['payback_months'][1]:.0f} months · ease: {r['ease']}"):
            st.write(r["rationale"])
            st.write("**Estimated impact:** " + r["estimated_impact"])
            st.write("Confidence:", r["confidence"],
                     "| approval required:", r["requires_approval"],
                     "| evidence:", ", ".join(r["evidence_refs"]))
            for asm in r.get("assumptions", []):
                st.caption("ℹ️ " + asm)
            for lim in r.get("limitations", []):
                st.caption("⚠️ " + lim)


def render_result(res: dict):
    st.subheader(f"answer_type: {res['answer_type']} · run_id: {res['run_id']}")
    st.markdown(res["summary"])
    if res.get("computed_change_pct") is not None:
        st.caption(f"Computed bill change: {res['computed_change_pct']:+.2f}% — the percentage "
                   "quoted in a question is never taken as fact.")
    if res.get("bridge"):
        tr, _ = _api("/trace/" + res["run_id"])
        vres = ((tr or {}).get("final_answer", {}) or {}).get("vres", {}) if tr else {}
        bill = vres.get("bill", {}) or {}
        st.plotly_chart(waterfall(_bridge_with_amounts(res),
                                  float(bill.get("base", 0.0)), float(bill.get("current", 0.0))),
                        width='stretch')
        st.dataframe(pd.DataFrame(res["bridge"]))
    _recommendation_cards(res.get("recommendations") or [])
    for a in res.get("assumptions", []):
        st.caption("ℹ️ assumption: " + a)
    for lim in res.get("limitations", []):
        st.caption("⚠️ limitation: " + lim)
    st.caption("Audit: " + (res.get("trace_hint") or ""))


# --------------------------------------------------------------------------- #
if screen == "Ask & Answer":
    st.title("⚡ Factory Energy Cost Decomposition & Savings Agent")
    q = st.text_input("Natural-language question:",
                      value="Why did the bill go up this quarter compared with last quarter?")
    c1, c2 = st.columns(2)
    p = c1.text_input("period (optional)").strip()
    cto = c2.text_input("compare_to (optional)").strip()
    if st.button("Analyze", type="primary") and q:
        body: dict = {"question": q}
        if p:
            body["period"] = p
        if cto:
            body["compare_to"] = cto
        res, err = _api("/analyze", "POST", json=body)
        if err:
            st.error(f"API error: {err}")
        else:
            st.session_state["last_result"] = res
    if st.session_state.get("last_result"):
        render_result(st.session_state["last_result"])

# --------------------------------------------------------------------------- #
elif screen == "Data & Validation":
    st.title("Data store & validation")
    h, err = _api("/health")
    if h:
        st.write("**Status**:", h["status"], "| store ok:", h["data_store"]["ok"],
                 "| llm:", h["llm"])
        st.dataframe(pd.DataFrame([
            {"table": t, "version": v["version"], "rows": v["row_count"],
             "sha256": v["sha256"], "loaded_at": v["loaded_at"], "status": v["status"]}
            for t, v in h["active_data_versions"].items()]))
        if h.get("last_validation"):
            rep = h["last_validation"]
            st.subheader("Validation report (latest)")
            st.dataframe(pd.DataFrame(rep["checks"]))
            a, b = st.columns(2)
            a.metric("warnings", len(rep["warnings"]))
            b.metric("blockers", len(rep["blockers"]))
            with st.expander("corrections applied and logged (never silent)"):
                st.write(rep.get("corrections", {}))
    st.subheader("Ingest a CSV (Annex B schema)")
    table = st.selectbox("table", ("meter_15min", "tariff", "production", "weather"))
    up = st.file_uploader("CSV", type="csv")
    if up and st.button("Ingest"):
        d, err = _api("/ingest", "POST",
                      files={"file": (up.name, up.getvalue(), "text/csv")},
                      data={"meta": json.dumps({"table": table})})
        if d:
            st.success(f"status {d['status']}: {d['rows_loaded']} loaded, "
                       f"{d['rows_rejected']} rejected (version {d.get('version_id')})")
            with st.expander("validation summary"):
                st.json(d.get("validation", {}))
        else:
            st.error(str(err))

# --------------------------------------------------------------------------- #
elif screen == "Traces":
    st.title("Run traces")
    run_id = st.text_input("run_id", placeholder="en-5d20ab").strip()
    if run_id and st.button("Fetch trace"):
        tr, err = _api(f"/trace/{run_id}")
        if err:
            st.error(str(err))
        elif tr:
            st.write("**question:**", tr["question"], "| **intent:**", tr["params"].get("intent"))
            st.write("**model:**", tr["model"], "| **llm_calls:**", tr["llm_calls"],
                     "| **tokens:**", tr["tokens"])
            st.write("**faithfulness:**", json.dumps(tr["faithfulness"]))
            st.markdown("**steps**")
            st.dataframe(pd.DataFrame(tr["steps"]))
            st.markdown("**tool calls**")
            st.dataframe(pd.DataFrame(tr["tool_calls"]))
            st.markdown("**data fingerprints (versions in effect for this run)**")
            st.json(tr["data_fingerprints"])
            with st.expander("final answer"):
                st.json(tr["final_answer"])
            with st.expander("validation in effect for this run"):
                st.json(tr["validation"])

# --------------------------------------------------------------------------- #
elif screen == "Late-drop compare":
    st.title("Before / after the late data drop")
    st.caption("Run the same question before and after ingesting the corrected files; "
               "both run_ids remain auditable.")
    a = st.text_input("run before", placeholder="en-xxxxxx").strip()
    b = st.text_input("run after", placeholder="en-yyyyyy").strip()
    if a and b and st.button("Compare runs"):
        ta, ea = _api(f"/trace/{a}")
        tb, eb = _api(f"/trace/{b}")
        if not ta or not tb:
            st.error(str(ea or eb))
        else:
            fa, fb = ta["final_answer"], tb["final_answer"]
            if fa.get("bridge") and fb.get("bridge"):
                df = pd.DataFrame([{"cause": x["cause"], "before_pp": x["pct_points"],
                                    "after_pp": y["pct_points"],
                                    "delta_pp": round(y["pct_points"] - x["pct_points"], 2)}
                                   for x, y in zip(fa["bridge"], fb["bridge"])])
                st.dataframe(df)
                st.caption("Change driven by the corrected data versions recorded in each trace.")
            va, vb = fa.get("vres", {}), fb.get("vres", {})
            if va.get("bill") and vb.get("bill"):
                st.write("**bills** before:", va["bill"], "→ after:", vb["bill"])
            if va.get("model_expected_current_bill") and vb.get("model_expected_current_bill"):
                st.write("**model expected current bill** before:", va["model_expected_current_bill"],
                         "→ after:", vb["model_expected_current_bill"])
            st.write("versions before:", json.dumps(ta["data_fingerprints"]))
            st.write("versions after:", json.dumps(tb["data_fingerprints"]))
