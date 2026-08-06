from __future__ import annotations

import argparse

from sqlalchemy import delete, select

from backend.app.database import SessionLocal, init_db
from backend.app.models import (
    AttendanceRecord,
    ManualCorrection,
    RecognitionEvent,
    SchoolClass,
    Student,
    User,
    Worker,
    WorkerCategory,
)
from backend.app.settings import get_settings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove only synthetic DEMO-* data and demo.* accounts."
    )
    parser.parse_args()
    if get_settings().app_env == "production":
        raise RuntimeError("Synthetic-data reset is disabled in production")

    init_db()
    with SessionLocal() as db:
        attendance_ids = list(
            db.scalars(
                select(AttendanceRecord.id).where(AttendanceRecord.person_id.startswith("DEMO-"))
            )
        )
        if attendance_ids:
            db.execute(
                delete(ManualCorrection).where(
                    ManualCorrection.attendance_record_id.in_(attendance_ids)
                )
            )
        db.execute(delete(AttendanceRecord).where(AttendanceRecord.person_id.startswith("DEMO-")))
        db.execute(
            delete(RecognitionEvent).where(
                RecognitionEvent.person_id.startswith("DEMO-")
                | RecognitionEvent.matched_person_id.startswith("DEMO-")
            )
        )
        db.execute(delete(Student).where(Student.person_id.startswith("DEMO-")))
        db.execute(delete(Worker).where(Worker.person_id.startswith("DEMO-")))
        db.execute(delete(User).where(User.username.startswith("demo.")))
        db.execute(
            delete(SchoolClass).where(~SchoolClass.id.in_(select(Student.class_id).distinct()))
        )
        db.execute(
            delete(WorkerCategory).where(
                ~WorkerCategory.id.in_(select(Worker.category_id).distinct())
            )
        )
        db.commit()
    print("Removed synthetic DEMO-* records and demo.* accounts; no database file was deleted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
