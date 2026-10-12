"""Export sample trace records for the submission (>= 3 answer types)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "sample_traces"


def pick(runs: list[dict], order: list[str]) -> dict[str, dict]:
    got: dict[str, dict] = {}
    for answer_type in order:
        for r in runs:
            if r.get("answer_type") == answer_type and answer_type not in got:
                got[answer_type] = r
    return got


def main() -> int:
    from src.llm.faithfulness import validate_numbers
    from src.store import db

    db.init_db()
    runs = db.list_runs(300)
    if not runs:
        print("no runs to export — run some /analyze calls first")
        return 1
    targets = pick(runs, ["decomposition", "recommendation", "calculated",
                          "not_in_data", "refused", "clarification_needed"])
    OUT.mkdir(parents=True, exist_ok=True)
    exported = []
    for answer_type, run_row in targets.items():
        tr = db.get_run(run_row["run_id"])
        path = OUT / f"trace_{answer_type}_{tr['run_id']}.json"
        path.write_text(json.dumps(tr, indent=2, default=str))
        exported.append(str(path.relative_to(ROOT)))
        ok = tr["faithfulness"].get("ok")
        v = validate_numbers(tr["final_answer"].get("summary", ""),
                             []) if ok is None else None
        print(f"exported {path.name} (type={answer_type}, faithfulness ok={ok})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
