"""LangGraph nodes: classify -> context -> compute -> brief -> persist.

Node 3 (compute) is fully deterministic. Node 4 (brief) is the only LLM step; on
LLM failure or number-faithfulness violation it retries once, then falls back to
the deterministic template brief - the response can never contain unverified numbers.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from src.graph import engine
from src.graph.state import AgentState
from src.analytics import whatif as whatif_tools
from src.llm import prompts
from src.llm import provider as llm_provider
from src.llm.faithfulness import build_allowlist, validate_numbers

DECOMP_PAT = re.compile(r"\b(why|explain|driver|bridge|what (changed|drove|caused)|bill (up|increase|change|rose|went))\b", re.I)
RANKING_PAT = re.compile(r"\b(payback|pay back|fastest|rank|ranking|priorit|which action|what should we do first)\b", re.I)
SAVINGS_PAT = re.compile(r"\b(savings|recommend|what should we do|actions|first)\b", re.I)
TOP_PAT = re.compile(r"\b(top|biggest|most energy|largest|ranking of assets|which assets)\b", re.I)
IDLE_PAT = re.compile(r"\b(idle|nothing is produced|no production|non.?production|night|weekend|leak|fault|abnormal)", re.I)
NUMERIC_PAT = re.compile(r"\b(how much|how many|total|kwh|consume|consumed|energy used|average price|price paid|hottest)\b", re.I)
WHATIF_PAT = re.compile(r"\bwhat if\b|\bif the .*(price|tariff)|\bif we (had|made|produced|reduced)|\b(would|will)( the)? [^.]{0,30}\bbill\b|\bat last quarter's tariff\b|\bif .*tariff\b", re.I)
FUTURE_PAT = re.compile(r"\b(next year|next quarter|next month|forecast|predict|will the tariff)\b", re.I)
ADVERSARIAL_PATTERN = re.compile(r"\b(definitely|certainly|confirm|prove|exactly|to the kwh|no range|without range|dont give ranges|do not give ranges)\b", re.I)
BLAME_PAT = re.compile(r"\b(furnace|compressor|machine|asset|equipment)\b.*\b(wasting|leaking|faulty|broken)\b", re.I)
CLARIFY_PAT = re.compile(r"\bwhich (period|asset|meter|quarter)\b|explain (the )?(bill )?(change|increase)\.?$", re.I)


def classify(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    q = state["question"]
    intent = "decomposition"
    if FUTURE_PAT.search(q) and "what if" not in q.lower():
        intent = "not_in_data"
    elif WHATIF_PAT.search(q) and not DECOMP_PAT.search(q):
        intent = "whatif"
    elif RANKING_PAT.search(q):
        intent = "savings"
    elif SAVINGS_PAT.search(q):
        intent = "savings"
    elif TOP_PAT.search(q):
        intent = "top_users"
    elif IDLE_PAT.search(q):
        intent = "idle"
    elif CLARIFY_PAT.search(q) and len(q.split()) <= 6:
        intent = "clarification"
    elif NUMERIC_PAT.search(q):
        intent = "numeric"
    elif DECOMP_PAT.search(q):
        intent = "decomposition"
    else:
        intent = "clarification"
    t = {"node": "classify", "ms": round((time.perf_counter() - t0) * 1000, 1),
         "status": "ok", "output": {"intent": intent}}
    return {"intent": intent, "steps": [t]}


def context(state: AgentState) -> AgentState:
    """Load data + validation + resolve periods; turns blockers into a clean declination."""
    t0 = time.perf_counter()
    out: dict[str, Any] = {"steps": []}
    try:
        ctx = engine.load_context()
        period, compare_to = engine.resolve_periods(state.get("period"), state.get("compare_to"),
                                                    ctx["quarters"])
        out["ctx"] = {k: v for k, v in ctx.items()}
        out["resolved"] = {"period": period, "compare_to": compare_to}
        out["data_quality"] = {
            "warnings": ctx["validation"]["warnings"],
            "corrections": ctx["validation"]["corrections"],
            "summary": ctx["validation"]["summary"],
        }
    except engine.OutOfData as e:
        out["verified"] = {"assumptions": [], "limitations": [str(e)],
                           "answer_type": "not_in_data"}
        out["answer_type"] = "not_in_data"
        out["summary_source"] = "not_in_data"
    except engine.AmbiguousPeriod as e:
        out["verified"] = {"assumptions": [], "limitations": [str(e)],
                           "answer_type": "clarification_needed"}
        out["answer_type"] = "clarification_needed"
        out["summary_source"] = "clarification_needed"
    out["steps"].append({"node": "context", "ms": round((time.perf_counter() - t0) * 1000, 1),
                         "status": "ok" if "ctx" in out else "declined"})
    return out


def compute(state: AgentState) -> AgentState:
    """Deterministic tool dispatch. NEVER lets the LLM produce numbers."""
    t0 = time.perf_counter()
    ctx = state.get("ctx")
    intent = state["intent"]
    out: dict[str, Any] = {"steps": [], "tool_calls": [], "errors": []}

    if ctx is None:
        # context node already declined; produce the faithful template summary for it
        vres = state.get("verified", {})
        answer_type = state.get("answer_type", "not_in_data")
        summary = prompts.template_brief(vres, state["question"], answer_type)
        out["steps"].append({"node": "compute", "ms": 0, "status": "skipped"})
        out["verified"] = vres
        out["answer_type"] = answer_type
        out["summary"] = summary
        return out

    resolved = state["resolved"]
    period, compare_to = resolved["period"], resolved["compare_to"]
    vres: dict[str, Any] = {"data_quality": state.get("data_quality")}
    answer_type = "calculated"
    summary_source = "calculated"

    try:
        if intent == "decomposition":
            dec = engine.decomp_tool(ctx, period, compare_to)
            vres.update({k: dec[k] for k in ("bill", "bridge", "model", "assumptions", "limitations")})
            vres["n_bootstrap_draws"] = dec["n_bootstrap_draws"]
            vres["assets"] = engine.assets_tool(ctx, period)
            waste_pp = next((b["pct_points"] for b in dec["bridge"] if b["cause"] == "waste"), 0.0)
            if waste_pp > 0.3:
                try:
                    idle = engine.idle_tool(ctx, period, corroborating_waste_pp=waste_pp)
                    savings = engine.build_recommendations(
                        idle, dec["bridge"], ctx["frames"]["tariff"], ctx["frames"]["meter_15min"], period)
                    vres["savings"] = savings
                    vres["idle"] = idle
                except Exception as e:  # savings optional for a decomposition answer
                    out["errors"].append({"node": "compute", "tool": "savings", "error": str(e)})
            answer_type = "decomposition"
            summary_source = "decomposition"

        elif intent == "top_users":
            vres["assets"] = engine.assets_tool(ctx, period)
            vres["assumptions"] = [a for a in [vres["assets"].pop("assumption", None)] if a]
            answer_type = "calculated"

        elif intent == "idle":
            idle = engine.idle_tool(ctx, period)
            vres["idle"] = idle
            vres["assumptions"] = [idle["definition"]]
            vres["limitations"] = [idle["note"]]
            answer_type = "calculated"

        elif intent in ("savings", "ranking"):
            sv = engine.savings_tool(ctx, period, compare_to)
            vres.update(sv)
            answer_type = "recommendation"

        elif intent == "whatif":
            w = engine.whatif_tool(ctx, state["question"], period, compare_to)
            vres["calc"] = w
            vres["assumptions"] = w["assumptions"]
            vres["limitations"] = ["Counterfactual uses the logged tariff/production structure; "
                                   "no future tariff forecast is performed."]
            answer_type = "calculated"

        elif intent == "numeric":
            f = ctx["frames"]
            q = state["question"]
            calc = whatif_tools.numeric_answer(q, f["meter_15min"], f["tariff"], period)
            if calc is None:
                if re.search(r"how many", q, re.I) and not re.search(r"meter|sub-?meter", q, re.I):
                    raise engine.OutOfData("physical asset counts beyond meter identifiers are "
                                           "not in the data")
                if WHATIF_PAT.search(q):
                    calc = engine.whatif_tool(ctx, q, period, compare_to)
                    vres["calc"] = calc
                else:
                    raise engine.AmbiguousPeriod("Question does not map to a known calculator "
                                                 "(supported: energy totals, average price, "
                                                 "hottest week, meter counts)")
            vres["calc"] = calc
            vres["assumptions"] = calc.get("assumptions", [])
            answer_type = "calculated"

        elif intent == "not_in_data":
            vres["assumptions"] = []
            vres["limitations"] = ["Out of data scope: this system answers questions about loaded "
                                   "energy, tariff, production and weather data only."]
            answer_type = "not_in_data"

        elif intent == "clarification":
            vres["assumptions"] = []
            vres["limitations"] = ["The request is ambiguous. Specify the comparison period and/or asset."]
            answer_type = "clarification_needed"

        # adversarial demands: confirm invented faults / drop honest ranges
        q = state["question"]
        if ADVERSARIAL_PATTERN.search(q):
            if BLAME_PAT.search(q):
                supported = bool(vres.get("idle", {}).get("findings")) and \
                    any(f["meter_type"] == "furnace" or f["meter_id"] in q.lower()
                        for f in vres.get("idle", {}).get("findings", []))
                if not supported:
                    answer_type = "refused"
                    summary_source = "refused"
                    vres["limitations"] = vres.get("limitations", []) + [
                        "Unsupported machine-fault claims are refused. Waste is only asserted when "
                        "the meter data (idle patterns) and the bill residual support it."]
            if re.search(r"\b(exact|no range|without range)\b", q, re.I) and answer_type == "recommendation":
                vres["limitations"] = vres.get("limitations", []) + [
                    "Exact single-number estimates cannot be produced honestly; ranges are mandatory"]

    except engine.OutOfData as e:
        vres = {"assumptions": [], "limitations": [str(e)], "data_quality": state.get("data_quality")}
        answer_type, summary_source = "not_in_data", "not_in_data"
    except engine.AmbiguousPeriod as e:
        vres = {"assumptions": [], "limitations": [str(e)], "data_quality": state.get("data_quality")}
        answer_type, summary_source = "clarification_needed", "clarification_needed"
    except Exception as e:  # tool failure -> honest refusal, full trace
        out["errors"].append({"node": "compute", "error": f"{type(e).__name__}: {e}"})
        vres = {"assumptions": [], "limitations": [f"computation failed and cannot be trusted; "
                                                   f"see trace ({type(e).__name__})"],
                "data_quality": state.get("data_quality")}
        answer_type, summary_source = "refused", "refused"

    out["verified"] = vres
    out["answer_type"] = answer_type
    out["summary_source"] = summary_source
    # non-LLM answer types take the faithful template summary immediately
    if answer_type not in ("decomposition", "recommendation"):
        out["summary"] = prompts.template_brief(vres, state["question"], answer_type)
    out["tool_calls"] = out.get("tool_calls", []) + _tool_calls_for(intent, vres, period, compare_to)
    out["steps"].append({"node": "compute", "ms": round((time.perf_counter() - t0) * 1000, 1),
                         "status": "ok" if answer_type not in ("refused",) else "declined",
                         "output": {"answer_type": answer_type}})
    return out


def _tool_calls_for(intent: str, vres: dict, period: str | None, compare_to: str | None) -> list[dict]:
    """Record which deterministic tools ran (names, args; compact result digests)."""
    calls: list[dict] = []
    def add(tool: str, args: dict, digest: dict):
        calls.append({"tool": tool, "args": args, "result_digest": digest})
    if intent == "decomposition":
        add("decompose_bill", {"period": period, "compare_to": compare_to, "method": "shapley|sequential"},
            {"bill": vres.get("bill"), "bridge": vres.get("bridge"),
             "n_bootstrap_draws": vres.get("n_bootstrap_draws")})
        if vres.get("assets"):
            add("asset_ranking", {"period": period}, {"meters": len(vres["assets"]["ranking"])})
        if vres.get("savings"):
            add("idle_and_findings", {"period": period},
                {"findings": len(vres.get("idle", {}).get("findings", []))})
            add("build_recommendations", {"period": period},
                {"recommendations": len(vres["savings"]["recommendations"]),
                 "pools": vres["savings"]["ledger"]["pools"]})
    elif intent in ("savings", "ranking"):
        add("decompose_bill", {"period": period, "compare_to": compare_to},
            {"waste_pp": next((b["pct_points"] for b in vres.get("bridge", []) if b["cause"] == "waste"), None)})
        add("idle_and_findings", {"period": period},
            {"findings": len(vres.get("idle", {}).get("findings", []))})
        add("build_recommendations", {"period": period},
            {"recommendations": len(vres.get("savings", {}).get("recommendations", [])),
             "pools": vres.get("savings", {}).get("ledger", {}).get("pools")})
    elif intent == "top_users":
        add("asset_ranking", {"period": period}, {"ranking": vres.get("assets", {}).get("ranking", [])[:3]})
    elif intent == "idle":
        add("idle_and_findings", {"period": period},
            {"total_idle_kwh": vres.get("idle", {}).get("total_idle_kwh"),
             "findings": len(vres.get("idle", {}).get("findings", []))})
    elif intent in ("whatif", "numeric"):
        add("calculated_tool", {"kind": (vres.get("calc") or {}).get("question_kind")},
            {"value": (vres.get("calc") or {}).get("value")})
    return calls


def brief(state: AgentState) -> AgentState:
    """LLM finance brief over verified numbers; validator + retry + template fallback."""
    t0 = time.perf_counter()
    answer_type = state.get("answer_type", "calculated")
    vres = state.get("verified", {})
    use_llm = answer_type in ("decomposition", "recommendation") and not settings_mock()
    llm_info: dict[str, Any] = {"llm_used": bool(use_llm)}
    errors: list[dict] = []

    text = prompts.template_brief(vres, state["question"], answer_type)
    allow = build_allowlist(vres)
    template_check = validate_numbers(text, allow)
    faith: dict[str, Any] = {"ok": bool(template_check["ok"]), "mode": "template",
                             "violations": template_check["violations"],
                             "n_numbers_checked": template_check["n_numbers_checked"],
                             "retries": 0}

    if use_llm:
        allow_llm = build_allowlist(vres)
        system, user = prompts.brief_messages(_allowable_view(vres), state["question"])
        current_text = None
        retried = False
        for attempt in range(2):
            try:
                prov = llm_provider.get_provider()
                res = prov.generate(system, user)
                current_text = res.text
                llm_info.update({"llm_used": True, "provider": res.provider, "model": res.model,
                                 "calls": (llm_info.get("calls", 0) or 0) + 1,
                                 "prompt_tokens": llm_info.get("prompt_tokens", 0) + res.prompt_tokens,
                                 "completion_tokens": llm_info.get("completion_tokens", 0) + res.completion_tokens})
                check = validate_numbers(current_text, allow)
                if check["ok"]:
                    text = current_text
                    faith = {**check, "mode": "llm", "retries": attempt}
                    break
                if not retried:
                    retried = True
                    user = user + ("\nIMPORTANT: your previous draft contained these numbers which are "
                                   "NOT in VERIFIED_DATA: " + json.dumps(check["violations"]) +
                                   "\nRewrite the brief quoting ONLY numbers from VERIFIED_DATA.")
                    continue
                errors.append({"node": "brief", "error": "faithfulness violations after retry",
                               "violations": check["violations"]})
                faith = {**check, "mode": "llm_failed_to_template", "retries": attempt + 1}
                text = prompts.template_brief(vres, state["question"], answer_type)
                break
            except Exception as e:
                errors.append({"node": "brief", "error": f"{type(e).__name__}: {e}"})
                llm_info["fallback"] = "llm_error"
                text = prompts.template_brief(vres, state["question"], answer_type)
                faith = {**template_check, "mode": "template_fallback", "retries": attempt}
                break

    out = {"summary": text, "faithfulness": faith, "llm": llm_info,
           "errors": errors,
           "steps": [{"node": "brief", "ms": round((time.perf_counter() - t0) * 1000, 1),
                      "status": "ok", "output": {"mode": faith["mode"], "violations": len(faith["violations"])}}]}
    return out


def settings_mock() -> bool:
    from src.config import settings
    return bool(settings.mock_llm)


def _allowable_view(vres: dict) -> dict:
    """Verified slice for the LLM prompt (already handled inside prompts.brief_messages)."""
    return vres


def persist(state: AgentState) -> AgentState:
    """Assemble the final response payload and save the full trace."""
    t0 = time.perf_counter()
    import uuid
    from src.config import settings as cfg

    answer_type = state.get("answer_type") or "calculated"
    vres = state.get("verified", {})
    run_id = state.get("run_id") or ("en-" + uuid.uuid4().hex[:6])
    payload = engine.result_envelope(
        answer_type=answer_type, vres=vres, summary=state.get("summary", ""),
        period=(state.get("resolved") or {}).get("period"),
        compare_to=(state.get("resolved") or {}).get("compare_to"))
    payload["run_id"] = run_id
    payload["trace_hint"] = f"GET /trace/{run_id}"

    ctx = state.get("ctx") or {}
    versions = ctx.get("versions", {})
    import hashlib
    fingerprints = {t: {"version": v.get("version"), "sha256": v.get("sha256"),
                        "row_count": v.get("row_count"), "loaded_at": v.get("loaded_at")}
                    for t, v in versions.items()}
    validation = ctx.get("validation", {})
    errors = state.get("errors") or []
    steps = state.get("steps") or []
    total_ms = round(sum(s.get("ms", 0) for s in steps), 1)

    # mechanical faithfulness for summaries produced without the brief node
    faith_in = state.get("faithfulness")
    if faith_in is None:
        vres2 = state.get("verified", {})
        allow2 = build_allowlist(vres2)
        faith_in = {**validate_numbers(state.get("summary", ""), allow2),
                    "mode": "template_no_llm"}

    steps_for_trace = steps + [{"node": "persist", "ms": 0.0, "status": "ok"}]
    trace = {
        "run_id": run_id,
        "created_at": _now(),
        "question": state["question"],
        "params": {"period": state.get("period"), "compare_to": state.get("compare_to"),
                   "intent": state.get("intent")},
        "data_fingerprints": fingerprints,
        "validation": validation,
        "steps": steps_for_trace,
        "tool_calls": state.get("tool_calls") or [],
        "model": {"provider": cfg.llm_provider, "model": cfg.llm_model,
                  "mock": cfg.mock_llm, **(state.get("llm") or {})},
        "llm_calls": 1 if (state.get("llm") or {}).get("llm_used") else 0,
        "tokens": {"prompt": (state.get("llm") or {}).get("prompt_tokens", 0),
                   "completion": (state.get("llm") or {}).get("completion_tokens", 0)},
        "errors": errors,
        "faithfulness": faith_in,
        "final_answer": payload,  # full payload incl. verified internals (audit); API strips
        "latency_ms": int(total_ms),
    }
    from src.store import db as store_db
    try:
        store_db.save_run(trace)
    except Exception as e:
        errors.append({"node": "persist", "error": f"trace save failed: {e}"})

    steps = steps + [{"node": "persist", "ms": round((time.perf_counter() - t0) * 1000, 1), "status": "ok"}]
    return {"final": payload, "run_id": run_id, "steps": steps}


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
