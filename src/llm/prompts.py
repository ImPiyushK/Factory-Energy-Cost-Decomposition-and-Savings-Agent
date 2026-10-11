"""Deterministic template brief (the faithful fallback) + LLM prompt construction.

The template is used by:
  * the mock provider (MOCK_LLM=true)
  * the automatic fallback when the real LLM fails or violates number faithfulness
It quotes only numbers from the verified results - by construction.
"""
from __future__ import annotations

import json
from typing import Any


def _fmt_pct(x: float) -> str:
    return f"{x:+.1f} pp"


def template_brief(vres: dict, question: str, answer_type: str) -> str:
    """Number-faithful brief assembled from verified results (no LLM required)."""
    if answer_type == "decomposition" and vres.get("bill"):
        bill, bridge = vres["bill"], vres.get("bridge", [])
        causes = {b["cause"]: b for b in bridge}
        biggest = max(bridge, key=lambda b: b["amount"]) if bridge else None
        parts = []
        for name, label in (("tariff", "tariff/price"), ("volume", "production volume"),
                            ("product_mix", "product mix"), ("weather", "weather"),
                            ("waste", "unexplained waste")):
            c = causes.get(name)
            if c:
                parts.append(f"{label} {c['pct_points']:+.1f} pp (range {c['range'][0]:+.1f} to "
                             f"{c['range'][1]:+.1f})")
        waste_note = (f" Controllable waste is small but real (+{causes['waste']['pct_points']:.1f} pp) "
                      "and is being quantified below." if causes.get("waste", {}).get("pct_points", 0) > 0.3
                      else " No material unexplained waste was detected - the increase is explained by "
                           "price, production and conditions, so no asset should be blamed.")
        return (f"The bill changed {bill['change_pct']:+.1f}% ({bill['base']:,.0f} -> {bill['current']:,.0f} "
                f"in currency terms). Split: " + "; ".join(parts) + ". "
                + (f"The largest single driver is {biggest['cause'].replace('_', ' ')}." if biggest else "")
                + waste_note)
    if answer_type == "recommendation" and vres.get("savings"):
        recs = vres["savings"].get("recommendations", [])
        if recs:
            first = recs[0]
            bullet = (f" Start with: {first['action']} - {first['estimated_impact']}.")
            listing = "; ".join(f"{r['action']} ({r['estimated_impact']})" for r in recs[:3])
            return (f"{len(recs)} ranked actions pay back between "
                    f"{min(r['payback_months'][0] for r in recs):.0f} and "
                    f"{max(r['payback_months'][1] for r in recs):.0f} months." + bullet
                    + f" Full ranking: {listing}.")
        return "No cost-justified actions were identified from measured evidence."
    if answer_type == "calculated":
        if vres.get("assets"):
            a = vres["assets"]
            top = a["ranking"][:3]
            cov = a.get("sub_meter_coverage")
            firsts = "; ".join(f"{r['meter_id']} {r['kwh']:,.0f} kWh ({r['share_pct']}% of main)"
                               for r in top)
            return (f"Top energy users {a['period']}: {firsts}. Sub-meters cover "
                    f"{cov * 100:.1f}% of the main meter" + ("." if (
                        a.get("within_tolerance")) else
                        f" — outside the {a['tolerance'] * 100:.0f}% tolerance, disclosed not rescaled."))
        if vres.get("idle"):
            i = vres["idle"]
            lines = [f"{r['meter_id']}: {r['idle_kwh']:,.0f} kWh over {r['idle_hours']:,.0f} idle h"
                     for r in i.get("by_meter", [])[:4] if r["meter_id"] != "MAIN"]
            total = i.get("total_idle_kwh", 0)
            findings = i.get("findings") or []
            ftxt = ""
            if findings:
                f0 = findings[0]
                ftxt = (f" One measured finding: {f0['meter_id']} off-production load "
                        f"{f0['measured']['off_prod_kw_recent_median']} kW vs "
                        f"{f0['measured']['off_prod_kw_early_median']} kW early-quarter median, "
                        f"persistent {f0['measured']['persistent_weeks']}+ weeks "
                        f"({f0['inferred_cause'][:60]}...).")
            return (f"Energy used when nothing is produced totals {total:,.0f} kWh "
                    f"(sub-meters) for {i['period']}. Largest: " + "; ".join(lines) + "." + ftxt)
        if vres.get("calc"):
            c = vres["calc"]
            unit = c.get("unit", "")
            label = {"currency": "currency units", "kWh": "kWh",
                     "currency_per_kwh": "per kWh", "ratio": "x", "count": "meters"}.get(unit, unit)
            val = c.get("value", 0)
            num = f"{val:,.4f}" if abs(val) < 10 else f"{val:,.1f}"
            return (f"{c.get('scenario', 'Computed')}: {num} {label}. "
                    f"Details: " + json.dumps(c.get("detail", {}), default=str))
        return "Verified results are available for this question type."
    if answer_type == "not_in_data":
        return ("The data available to this system does not contain the answer to that question, "
                "so I will not guess. Available data: meter readings (15-min), tariffs, production, "
                "weather for the loaded quarters - ask about bills, assets, waste or savings.")
    if answer_type == "clarification_needed":
        return "The question is ambiguous (which period? which asset?). Please specify the comparison period."
    if answer_type == "refused":
        return ("I cannot comply with the request as stated: this system only claims numbers that "
                "verified tools computed, and does not assert unsupported machine faults. "
                "Reformulate the question and I will calculate from the data.")
    return "Verified results are available for this question type."


def system_prompt() -> str:
    return (
        "You are a finance-brief writer for a factory energy system. You receive ONLY verified "
        "tool results (JSON). Rules: (1) quote numbers exactly as given, with their units, or "
        "round them to at most 2 decimals; (2) do NOT compute, convert or invent any new number; "
        "(3) do not state machine faults beyond 'findings' given to you; do not name an asset as "
        "'faulty' unless a finding with evidence is provided for it; (4) keep the brief under 150 "
        "words; (5) separate price/volume/mix/weather effects from controllable waste; mention "
        "key assumptions/limitations briefly. Output plain text."
    )


def user_prompt(verified_json: str, question: str) -> str:
    return (
        f"Manager question: {question}\n\nVERIFIED_DATA (the ONLY numbers you may quote):\n{verified_json}"
    )


def brief_messages(vres: dict, question: str) -> tuple[str, str]:
    """Only the verified slice needed for prose - not the raw internal structures."""
    slice_: dict[str, Any] = {}
    if vres.get("bill"):
        slice_["bill"] = vres["bill"]
        slice_["bridge"] = vres.get("bridge")
        slice_["model_holdout_mape_pct"] = (vres.get("model") or {}).get("holdout_mape_pct")
    if vres.get("assets"):
        slice_["assets"] = {"ranking": vres["assets"]["ranking"][:5],
                            "sub_meter_coverage": vres["assets"].get("sub_meter_coverage")}
    if vres.get("idle"):
        slice_["idle"] = {"total_idle_kwh": vres["idle"].get("total_idle_kwh"),
                          "findings": vres["idle"].get("findings"),
                          "top_idle": vres["idle"].get("by_meter", [])[:4]}
    if vres.get("savings"):
        slice_["savings"] = vres["savings"]
    slice_["assumptions"] = vres.get("assumptions", [])[:4]
    slice_["limitations"] = vres.get("limitations", [])[:4]
    slice_["data_quality"] = vres.get("data_quality")
    return system_prompt(), user_prompt(json.dumps(slice_, default=str), question)
