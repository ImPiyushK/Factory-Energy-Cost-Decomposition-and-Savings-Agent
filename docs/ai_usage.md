# AI usage disclosure (competition integrity statement)

This project was developed by the team with AI assistance. Specifically:

* **GLM / OpenCode coding agent** was used for scaffolding, implementation, debugging
  and test-writing; all committed code has been reviewed and executed by the team,
  and every analytical formula is documented in `docs/architecture.md` and `README.md`.
* No proprietary or confidential company data was used anywhere. All datasets are
  synthetic (committed generator + seeds, see `docs/data_card.md`), plus the
  hackathon data pack once provided.
* The LLM inside the PRODUCT (finance brief) is configured via
  `LLM_PROVIDER`/`LLM_MODEL` and is strictly restricted by a mechanical
  number-faithfulness validator: it may only quote numbers produced by verified
  analytical tools, and falls back to a deterministic template.

Signed: [team members list — replace with real names before submission]
