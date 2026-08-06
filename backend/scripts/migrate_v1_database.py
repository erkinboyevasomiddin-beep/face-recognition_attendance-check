from __future__ import annotations

import argparse
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.database import Base
from backend.app.models import (
    AttendanceRecord,
    RecognitionEvent,
    SchoolClass,
    Student,
    User,
    Worker,
    WorkerCategory,
)
from backend.app.schemas import AttendanceStatus, EventOutcome, EventType, PersonType, UserRole

MIGRATION_NAMESPACE = uuid.UUID("894491ce-36ff-4209-9ec2-a162686e80b6")


def _aware(value: str | None, legacy_zone: ZoneInfo) -> datetime:
    if not value:
        return datetime.now(UTC)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=legacy_zone)
    return parsed.astimezone(UTC)


def migrate(source: Path, destination: Path, legacy_timezone: str) -> None:
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    zone = ZoneInfo(legacy_timezone)
    source_db = sqlite3.connect(f"file:{source.resolve().as_posix()}?mode=ro", uri=True)
    source_db.row_factory = sqlite3.Row
    target_engine = create_engine(
        f"sqlite:///{destination.as_posix()}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(target_engine)

    student_ids: dict[int, str] = {}
    worker_ids: dict[int, str] = {}
    try:
        with Session(target_engine) as target:
            for row in source_db.execute("SELECT * FROM users ORDER BY id"):
                target.add(
                    User(
                        id=row["id"],
                        username=row["username"],
                        password_hash=row["password_hash"],
                        role=UserRole(row["role"]),
                        display_name=row["full_name"],
                        created_at=_aware(row["created_at"], zone),
                    )
                )

            for row in source_db.execute("SELECT * FROM school_classes ORDER BY id"):
                target.add(
                    SchoolClass(
                        id=row["id"],
                        grade=row["grade"],
                        section_code=row["section_code"],
                        name=row["name"],
                    )
                )
            target.flush()

            for row in source_db.execute("SELECT * FROM students ORDER BY id"):
                stable_id = f"legacy-student-{int(row['id']):06d}"
                student_ids[int(row["id"])] = stable_id
                target.add(
                    Student(
                        id=row["id"],
                        person_id=stable_id,
                        display_name=row["full_name"],
                        grade=row["grade"],
                        class_name=row["class_name"],
                        class_id=row["class_id"],
                    )
                )

            for row in source_db.execute("SELECT * FROM worker_categories ORDER BY id"):
                target.add(
                    WorkerCategory(id=row["id"], name=row["name"], sort_order=row["sort_order"])
                )
            target.flush()
            for row in source_db.execute("SELECT * FROM workers ORDER BY id"):
                stable_id = f"legacy-worker-{int(row['id']):06d}"
                worker_ids[int(row["id"])] = stable_id
                target.add(
                    Worker(
                        id=row["id"],
                        person_id=stable_id,
                        display_name=row["full_name"],
                        category_id=row["category_id"],
                    )
                )

            for row in source_db.execute("SELECT * FROM attendance_records ORDER BY id"):
                person_type = PersonType(row["person_type"])
                lookup = student_ids if person_type == PersonType.STUDENT else worker_ids
                attendance_person_id = lookup.get(int(row["person_id"]))
                if attendance_person_id is None:
                    continue
                recorded = _aware(row["recorded_at"], zone)
                status = AttendanceStatus(row["status"])
                target.add(
                    AttendanceRecord(
                        id=row["id"],
                        person_type=person_type,
                        person_id=attendance_person_id,
                        date=datetime.fromisoformat(row["date"]).date(),
                        status=status,
                        arrival_at=recorded if status != AttendanceStatus.ABSENT else None,
                        source=row["source"],
                        note=row["note"],
                        absence_reason=row["absence_reason"],
                        recorded_at=recorded,
                        updated_at=recorded,
                    )
                )

            for row in source_db.execute("SELECT * FROM recognition_events ORDER BY id"):
                timestamp = _aware(row["event_timestamp"], zone)
                matched_id = None
                # Version 1 did not capture camera direction. Preserve every legacy
                # detection as a recorded ENTRY observation instead of inventing EXIT
                # semantics from odd/even counts.
                event_type = EventType.ENTRY
                if row["matched"] and row["matched_person_type"] == PersonType.STUDENT.value:
                    matched_id = student_ids.get(int(row["matched_person_id"]))
                target.add(
                    RecognitionEvent(
                        id=row["id"],
                        event_id=str(uuid.uuid5(MIGRATION_NAMESPACE, f"legacy-event-{row['id']}")),
                        person_id=matched_id or f"legacy-unmatched-{int(row['id']):06d}",
                        display_name=row["person_name"],
                        normalized_display_name=row["normalized_person_name"],
                        class_name=row["class_name"],
                        similarity=row["similarity"] if row["similarity"] is not None else 0.0,
                        event_timestamp=timestamp,
                        event_type=event_type,
                        source=row["source"],
                        matched=bool(row["matched"] and matched_id),
                        matched_person_type=PersonType.STUDENT if matched_id else None,
                        matched_person_id=matched_id,
                        matched_display_name=row["matched_person_name"] if matched_id else None,
                        outcome=EventOutcome.RECORDED,
                        received_at=_aware(row["received_at"], zone),
                    )
                )
            target.commit()
    finally:
        source_db.close()
        target_engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy a version-1 attendance database into the stable-ID schema."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--legacy-timezone", default="Asia/Tashkent")
    args = parser.parse_args()
    migrate(args.source, args.destination, args.legacy_timezone)
    print(f"Migrated database to {args.destination}; the source was not modified.")
    print("Legacy SHA-256 users were copied. Recreate them before production use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
