from __future__ import annotations

import argparse

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database import SessionLocal, init_db
from backend.app.models import AttendanceRecord, SchoolClass, Student, Worker, WorkerCategory
from backend.app.schemas import AttendanceStatus, PersonType
from backend.app.services import current_local_date

SYNTHETIC_CLASSES = ((5, "01"), (5, "02"), (6, "01"))


def seed_synthetic_data_if_requested(db: Session) -> None:
    """Create clearly synthetic roster data; never create public default accounts."""
    if db.scalar(select(Student.id).limit(1)) is not None:
        return

    student_number = 1
    for grade, section in SYNTHETIC_CLASSES:
        class_name = f"{grade}-{section}"
        school_class = SchoolClass(grade=grade, section_code=section, name=class_name)
        db.add(school_class)
        db.flush()
        for _ in range(4):
            person_id = f"DEMO-STU-{student_number:04d}"
            db.add(
                Student(
                    person_id=person_id,
                    display_name=f"Demo Student {student_number:03d}",
                    grade=grade,
                    class_name=class_name,
                    class_id=school_class.id,
                )
            )
            if student_number % 3:
                db.add(
                    AttendanceRecord(
                        person_type=PersonType.STUDENT,
                        person_id=person_id,
                        date=current_local_date(),
                        status=(
                            AttendanceStatus.PRESENT
                            if student_number % 4
                            else AttendanceStatus.LATE
                        ),
                        source="synthetic_seed",
                        note="Synthetic demonstration record.",
                    )
                )
            student_number += 1

    category = WorkerCategory(name="Demo Staff", sort_order=1)
    db.add(category)
    db.flush()
    for number in range(1, 4):
        db.add(
            Worker(
                person_id=f"DEMO-WRK-{number:04d}",
                display_name=f"Demo Worker {number:03d}",
                category_id=category.id,
            )
        )
    db.commit()


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed synthetic demonstration roster data.")
    parser.parse_args()
    init_db()
    with SessionLocal() as session:
        seed_synthetic_data_if_requested(session)
    print("Synthetic demonstration data initialized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
