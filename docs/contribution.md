# Contribution statement

| Member | Owned | Key deliverables |
|---|---|---|
| [M1] Data engineering | schema adapter & loader, versions, validation, synthetic generator, late-drop playbook | src/ingest, src/validate, scripts/generate_synthetic.py |
| [M2] Analytics | baseline model, decomposition (Shapley/sequential), asset/idle/leak rules, savings & payback | src/analytics/* |
| [M3] Backend/AI | API contract, LangGraph pipeline, LLM switch + faithfulness, traces | src/api, src/graph, src/llm |
| [M4] UI/Eval/DevOps | Streamlit screens, evaluation harness & reports, Docker, docs/demo | src/ui, evaluation, docker-compose.yml, README |

Replace with real names and adjust percentages before submission; every member has
reviewed the complete architecture, the decomposition method, the faithfulness
mechanism and the evaluation results (see docs/architecture.md).
