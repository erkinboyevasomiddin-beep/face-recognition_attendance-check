from __future__ import annotations

import argparse

from backend.app.database import SessionLocal, init_db
from backend.seed_data import seed_synthetic_data_if_requested


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize an empty attendance database.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Also insert a small, clearly synthetic roster and attendance sample.",
    )
    args = parser.parse_args()
    init_db()
    if args.synthetic:
        with SessionLocal() as db:
            seed_synthetic_data_if_requested(db)
    print("Attendance database initialized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
