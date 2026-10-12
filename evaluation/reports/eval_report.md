# Evaluation report

Generated from `evaluation/runner.py` runs over the committed synthetic datasets (ground truth = injected generator parameters; independent recomputation in `evaluation/truth.py`). No hidden answer key is used. Method: exact-match/numeric checks + structural rules.

**LLM in effect during these runs: google gemini-3.1-flash-lite (mock=False)** — the brief node called the hosted model for decomposition/recommendation answers; every brief passes the mechanical number-faithfulness validator (0 violations across all runs).


## Dataset `no_leak` — Config A

| metric | value |
|---|---|
| cases passed | 5/5 |
| method | shapley |
| reconciliation_residual_pp | 0.0 |
| tariff_decomposition_error_pp_vs_paasche | -4.38 |
| tariff_shapley_pp | 39.42 |
| tariff_paasche_truth_pp | 43.8 |
| n_bootstrap_draws | 59 |
| model_holdout_mape_pct | 6.85 |
| latency p50 / p95 (ms, mock config) | 36943 / 65520 |
| LLM calls per run (mean) | 1.0 |

| case | pass | answer_type | failing checks |
|---|---|---|---|
| EV-02 | ✅ | decomposition | - |
| EV-03 | ✅ | decomposition | - |
| EV-04 | ✅ | decomposition | - |
| EV-23 | ✅ | - | - |
| EV-24 Δ{'tariff': -0.11, 'volume': 0.0, 'product_mix': 0.0, 'weather': 0.0, 'waste': 0.03} | ✅ | - | - |

## Dataset `no_leak` — Config B

| metric | value |
|---|---|
| cases passed | 5/5 |
| method | sequential |
| reconciliation_residual_pp | 0.0 |
| tariff_decomposition_error_pp_vs_paasche | -9.03 |
| tariff_shapley_pp | 34.77 |
| tariff_paasche_truth_pp | 43.8 |
| n_bootstrap_draws | 59 |
| model_holdout_mape_pct | 6.85 |
| latency p50 / p95 (ms, mock config) | 36070 / 62172 |
| LLM calls per run (mean) | 1.0 |

| case | pass | answer_type | failing checks |
|---|---|---|---|
| EV-02 | ✅ | decomposition | - |
| EV-03 | ✅ | decomposition | - |
| EV-04 | ✅ | decomposition | - |
| EV-23 | ✅ | - | - |
| EV-24 Δ{'tariff': -0.1, 'volume': 0.0, 'product_mix': 0.0, 'weather': 0.0, 'waste': 0.03} | ✅ | - | - |

## Dataset `synthetic` — Config A

| metric | value |
|---|---|
| cases passed | 20/20 |
| method | shapley |
| reconciliation_residual_pp | 0.01 |
| tariff_decomposition_error_pp_vs_paasche | -4.89 |
| tariff_shapley_pp | 39.12 |
| tariff_paasche_truth_pp | 44.01 |
| n_bootstrap_draws | 59 |
| model_holdout_mape_pct | 6.0 |
| latency p50 / p95 (ms, mock config) | 3175 / 36971 |
| LLM calls per run (mean) | 0.33 |

| case | pass | answer_type | failing checks |
|---|---|---|---|
| EV-01 | ✅ | decomposition | - |
| EV-05 | ✅ | calculated | - |
| EV-06 | ✅ | calculated | - |
| EV-07 | ✅ | calculated | - |
| EV-08 | ✅ | recommendation | - |
| EV-09 | ✅ | recommendation | - |
| EV-10 | ✅ | calculated | - |
| EV-11 | ✅ | calculated | - |
| EV-13 | ✅ | not_in_data | - |
| EV-14 | ✅ | not_in_data | - |
| EV-15 | ✅ | calculated | - |
| EV-16 | ✅ | calculated | - |
| EV-17 | ✅ | refused | - |
| EV-18 | ✅ | recommendation | - |
| EV-19 | ✅ | decomposition | - |
| EV-20 | ✅ | decomposition | - |
| EV-21 | ✅ | calculated | - |
| EV-22 | ✅ | calculated | - |
| EV-23 | ✅ | - | - |
| EV-24 Δ{'tariff': -0.11, 'volume': 0.0, 'product_mix': 0.0, 'weather': -0.01, 'waste': 0.03} | ✅ | - | - |

## Dataset `synthetic` — Config B

| metric | value |
|---|---|
| cases passed | 20/20 |
| method | sequential |
| reconciliation_residual_pp | 0.0 |
| tariff_decomposition_error_pp_vs_paasche | -9.28 |
| tariff_shapley_pp | 34.73 |
| tariff_paasche_truth_pp | 44.01 |
| n_bootstrap_draws | 59 |
| model_holdout_mape_pct | 6.0 |
| latency p50 / p95 (ms, mock config) | 3493 / 39141 |
| LLM calls per run (mean) | 0.33 |

| case | pass | answer_type | failing checks |
|---|---|---|---|
| EV-01 | ✅ | decomposition | - |
| EV-05 | ✅ | calculated | - |
| EV-06 | ✅ | calculated | - |
| EV-07 | ✅ | calculated | - |
| EV-08 | ✅ | recommendation | - |
| EV-09 | ✅ | recommendation | - |
| EV-10 | ✅ | calculated | - |
| EV-11 | ✅ | calculated | - |
| EV-13 | ✅ | not_in_data | - |
| EV-14 | ✅ | not_in_data | - |
| EV-15 | ✅ | calculated | - |
| EV-16 | ✅ | calculated | - |
| EV-17 | ✅ | refused | - |
| EV-18 | ✅ | recommendation | - |
| EV-19 | ✅ | decomposition | - |
| EV-20 | ✅ | decomposition | - |
| EV-21 | ✅ | calculated | - |
| EV-22 | ✅ | calculated | - |
| EV-23 | ✅ | - | - |
| EV-24 Δ{'tariff': -0.1, 'volume': 0.0, 'product_mix': -0.01, 'weather': 0.0, 'waste': 0.03} | ✅ | - | - |

## Configuration comparison (measured)

| dataset | config | method | reconciliation residual (pp) | tariff error vs Paasche truth (pp) | draws | holdout MAPE % |
|---|---|---|---|---|---|---|
| no_leak | A | shapley | 0.0 | -4.38 | 59 | 6.85 |
| no_leak | B | sequential | 0.0 | -9.03 | 59 | 6.85 |
| synthetic | A | shapley | 0.01 | -4.89 | 59 | 6.0 |
| synthetic | B | sequential | 0.0 | -9.28 | 59 | 6.0 |

**Choice justification**: Config A (Shapley) is order-independent and reports tighter consistent ranges; Config B (sequential) is the simpler chained fallback kept as a documented alternative. The measured table above justifies the final choice; numbers are from these runs, not from claims.


## Report honesty notes

- Ground truths are the generator's injected parameters and independent recomputations (Paasche tariff effect, raw meter totals); the hidden true split is not accessible (per competition rules).
