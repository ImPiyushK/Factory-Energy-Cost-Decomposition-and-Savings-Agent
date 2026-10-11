#!/usr/bin/env python3
"""Administrative data loader (spec Section 4.7 / Section 6 "Data loader").

Loads CSV files in the Annex B schema (adapter maps common column aliases).

Examples:
  python -m src.ingest.cli load --table meter_15min --file data/synthetic/meter_15min.csv
  python -m src.ingest.cli load-pack --dir data/synthetic
  python -m src.ingest.cli status
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.store import db
from src.ingest import loader


def _load_one(table: str, file: str) -> int:
    result = loader.ingest_table(table, file, source="cli")
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["status"] != "rejected" else 1


def main() -> int:
    db.init_db()
    ap = argparse.ArgumentParser(prog="load_data", description="Load Annex B CSVs into the data store")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("load", help="load a single CSV")
    p.add_argument("--table", required=True, choices=loader.TABLES)
    p.add_argument("--file", required=True)

    pack = sub.add_parser("load-pack", help="load a directory containing the four standard CSVs")
    pack.add_argument("--dir", required=True)

    sub.add_parser("status", help="show active data versions")
    sub.add_parser("revalidate", help="re-run validation over the active data set")

    args = ap.parse_args()

    if args.cmd == "load":
        return _load_one(args.table, args.file)

    if args.cmd == "load-pack":
        d = Path(args.dir)
        rc = 0
        for table in ("tariff", "production", "weather", "meter_15min"):
            f = d / f"{table}.csv"
            if not f.exists():
                print(f"SKIP {table}: file not found ({f})")
                continue
            rc |= _load_one(table, str(f))
        return rc

    if args.cmd == "status":
        with db.get_conn() as conn:
            print(json.dumps(db.get_active_versions(conn), indent=2, default=str))
        return 0

    # revalidate
    import pandas as pd
    from src.validate import checks as vchecks
    from src.config import settings

    frames = loader.load_active_frames()
    card_path = Path(settings.resolve(settings.data_dir)) / "data_card.json"
    card = json.loads(card_path.read_text()) if card_path.exists() else None
    report = vchecks.run_all_checks(frames, tolerance=settings.recon_tolerance, card=card)
    print(json.dumps(report, indent=2, default=str))
    db.save_validation_report(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
