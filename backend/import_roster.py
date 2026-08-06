from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.database import SessionLocal, init_db
from backend.app.roster_import import import_roster_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import a stable-ID school roster CSV.")
    parser.add_argument("roster_path", type=Path)
    parser.add_argument("--strict", action="store_true", help="Abort when any row is malformed.")
    parser.add_argument(
        "--allow-legacy-text",
        action="store_true",
        help="Migration only: accept the old class-header/name text format.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    init_db()
    with SessionLocal() as db:
        summary = import_roster_file(
            db,
            args.roster_path,
            allow_partial=not args.strict,
            allow_legacy_text=args.allow_legacy_text,
        )
    print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2))
    return 1 if summary.errors and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
