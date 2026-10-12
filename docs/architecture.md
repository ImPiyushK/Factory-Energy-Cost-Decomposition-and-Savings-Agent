# Architecture & methodology

## Stack
Streamlit · FastAPI+Uvicorn+Pydantic v2 · LangGraph · SQLite · pandas/numpy ·
statsmodels (robust Huber RLM) · Plotly · matplotlib-free · configurable LLM
(mock | openai | anthropic | google | ollama) · Docker compose.

## LangGraph workflow (minimal, 5 nodes)
| Node | Type | Responsibility |
|---|---|---|
| classify | deterministic rules (+optional LLM assist point) | intent → one of 7 internal routes, periods defaulting |
| context | deterministic | load active data versions, strict validation, period resolution; declines cleanly |
| compute | **deterministic tools only** | decomposition / assets / idle / savings / what-if / calculators |
| brief | LLM (mock-capable) | finance brief from verified numbers; 1 corrective retry; template fallback |
| persist | deterministic | number-faithfulness validation + full trace save |

Conditional edge: only decomposition/recommendation answers pass through the brief
node; everything else is answered from tools with the faithful template summary.

## Analytical method (why it reconciles)
Let `B(q)` be the exact bill of quarter q (interval kWh × per-interval price).
Counterfactual `Bill(stack)` evaluates the fitted two-part baseline over the CURRENT
period's calendar with drivers at base or current levels:
- volume: current daily volumes scaled by the total-units ratio,
- mix: quarterly mix shares,
- weather: current CDD/HDD shifted by the quarter-mean difference,
- price: exact per-interval lookup (old regime projected forward).

Shapley over the four factors gives φ; reconciliation identity:
`B1 − B0 = Σφ + waste` where `waste = B1 − B0 − Σφ` (exact by construction).
Baseline model error is included in waste and reported in limitations
(`model_expected_current_bill` vs actual bill in the trace).

Two-part baseline (identifiability rationale):
- off-production daily kWh ~ intercept + CDD + HDD + day-type (robust Huber RLM),
- production surplus = whole-day − off-production ~ per-product unit volumes + CDD + HDD + Saturday effect.

Uncertainty: weekly block bootstrap per quarter (weeks drawn with replacement, model
refit, blocks re-derived, actual bills weighted per week) → 10–90% bands reported
per cause.

## Number faithfulness
`allowlist = every number in the verified results` (bridge ± ranges, savings, idle,
calcs; plus numbers embedded in string fields). The LLM brief and even template
summaries are mechanically validated: each numeric token must equal an allowlisted
value within rounding tolerance, be a simple derivation (±/ratio/%), or be a small
prose count not attached to a unit. Violation → one corrective retry → deterministic
template fallback. The validator state is recorded per run in the trace.

## Trace
runs table stores: question, params/intent, data fingerprints (per-table version,
sha256, rows), validation report, steps (node/ms), tool calls (name/args/digest),
model+tokens, errors/retries, faithfulness, final answer (incl. verified internals),
latency. `GET /trace/{run_id}` returns the whole record. Sample exports:
`docs/sample_traces/*.json`.

## Design decisions we deliberately did NOT make
- **No vector store/embeddings**: no retrieval requirement exists in the problem;
  adding one would add a failure mode without earning marks.
- **No multi-agent topology**: agents are illustrative only per the statement; a
  single fixed pipeline is easier to verify.
- **SQLite over DuckDB/Parquet**: single-file persistence, judge-inspectable,
  volume-mountable; analytics happen in pandas in-memory.
- **Hosted flash LLM first, Ollama as offline fallback, mock for tests**.
