"""Number-faithfulness validation (spec R10): the LLM may quote ONLY computed numbers.

Algorithm:
  1. Extract every numeric token from the generated text (handling 1,234.5 forms).
  2. A token PASSES if it matches an allowlisted value within rounding tolerance,
     or is a simple arithmetic derivation (a+b, a-b, a/b, a/b*100) of allowlisted
     values, or is a small bare prose integer (<= 12) not followed by a unit/%.
  3. Any other token is a VIOLATION. Violations trigger one corrective retry,
     then a deterministic template brief (always faithful).

The allowlist is built mechanically from the verified-results JSON: every number a
tool produced (bridge points and ranges, savings, rankings, idle stats, calc values).
The question's own numbers (e.g. "15%") are NOT exempt - if the tools did not compute
15, quoting "15%" is a violation and is corrected to the computed figure.
"""
from __future__ import annotations

import re

NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
UNIT_AFTER_RE = re.compile(r"\s*(?:%|kwh|mwh|€|\$|£|kw|tonn|kg|t\b|pp)", re.IGNORECASE)


def _norm_token(raw: str) -> float | None:
    v = raw.replace(",", "")
    try:
        return float(v)
    except ValueError:
        return None


def extract_numbers(text: str) -> list[tuple[str, float]]:
    out = []
    for m in NUMBER_RE.finditer(text):
        val = _norm_token(m.group(0))
        if val is not None:
            out.append((m.group(0), val))
    return out


def _is_derived(cand: float, allow: list[float], tol_fn) -> bool:
    n = len(allow)
    for i in range(n):
        a = allow[i]
        for j in range(n):
            b = allow[j]
            if abs(cand - (a + b)) <= tol_fn(max(abs(a), abs(b))):
                return True
            if abs(cand - (a - b)) <= tol_fn(max(abs(a), abs(b))):
                return True
            if a != 0 and abs(cand - a / b) <= tol_fn(max(abs(a), abs(b))):
                return True
            if b != 0 and abs(cand - a / b * 100.0) <= tol_fn(max(abs(a), abs(b), 1.0)):
                return True
    return False


def first_char_after(text: str, idx: int, raw: str) -> str:
    j = idx + len(raw)
    while j < len(text) and text[j].isspace():
        j += 1
    return text[j] if j < len(text) else ""


def validate_numbers(text: str, allowlist: list[float]) -> dict:
    """Validate every numeric token in `text` against the allowlist."""
    tol_fn = lambda v: max(0.06, 0.005 * abs(v))
    violations, checked, derived_used = [], 0, 0
    for raw, val in extract_numbers(text):
        checked += 1
        exact = any(abs(val - a) <= tol_fn(a) for a in allowlist)
        if exact:
            continue
        if _is_derived(val, allowlist, tol_fn):
            derived_used += 1
            continue
        if val <= 12 and float(val).is_integer():
            pos = None
            for m in NUMBER_RE.finditer(text):
                if m.group(0) == raw:
                    pos = m.end()
                    break
            if pos is not None:
                while pos < len(text) and text[pos].isspace():
                    pos += 1
                nxt = text[pos] if pos < len(text) else ""
                if nxt and not UNIT_AFTER_RE.match(nxt) and not nxt.isdigit():
                    continue  # prose count like "5 causes" - allowed
        violations.append({"token": raw, "value": val})
    return {
        "ok": not violations,
        "n_numbers_checked": checked,
        "n_derived_ok": derived_used,
        "violations": violations,
    }


def build_allowlist(vres: dict) -> list[float]:
    """Collect every number produced by tools from the verified-results structure."""
    vals: list[float] = []

    def walk(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            if abs(float(x)) < 1e12:
                vals.append(float(x))
            return
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            for _, v in extract_numbers(x):
                vals.append(v)

    walk(vres)
    # round forms the brief may quote: keep base values; validator handles rounding tolerance
    uniq = sorted(set(vals))
    return uniq
