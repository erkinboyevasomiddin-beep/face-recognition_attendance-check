from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, date, datetime, time, timedelta
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from backend.app.models import (
    AttendanceRecord,
    ManualCorrection,
    RecognitionEvent,
    SchoolClass,
    Student,
    User,
    Worker,
    WorkerCategory,
    utc_now,
)
from backend.app.schemas import (
    AttendanceStatus,
    EventOutcome,
    EventType,
    ManualCorrectionCreate,
    PersonType,
    RecognitionEventIngest,
    RecognitionEventIngestResponse,
)
from backend.app.settings import get_settings

STATUS_ORDER = [AttendanceStatus.PRESENT, AttendanceStatus.LATE, AttendanceStatus.ABSENT]


class StaleRecognitionEventError(ValueError):
    pass


def status_counts(statuses: list[str]) -> dict[str, int]:
    counter = Counter(statuses)
    return {status.value: counter.get(status.value, 0) for status in STATUS_ORDER}


def normalize_status(record: AttendanceRecord | None) -> str:
    return record.status.value if record else AttendanceStatus.ABSENT.value


def normalize_lookup_text(value: str | None) -> str:
    return " ".join(value.split()).casefold() if value else ""


def attendance_cutoff_time() -> time:
    settings = get_settings()
    return time(hour=settings.attendance_on_time_hour, minute=settings.attendance_on_time_minute)


def attendance_lookup(
    db: Session,
    person_type: PersonType,
    person_ids: list[str],
    target_date: date,
) -> dict[str, AttendanceRecord]:
    if not person_ids:
        return {}
    records = db.scalars(
        select(AttendanceRecord).where(
            AttendanceRecord.person_type == person_type,
            AttendanceRecord.person_id.in_(person_ids),
            AttendanceRecord.date == target_date,
        )
    ).all()
    return {record.person_id: record for record in records}


def get_director_overview(db: Session, target_date: date) -> dict[str, object]:
    total_students = db.scalar(select(func.count()).select_from(Student)) or 0
    total_classes = db.scalar(select(func.count()).select_from(SchoolClass)) or 0
    total_workers = db.scalar(select(func.count()).select_from(Worker)) or 0
    student_ids = list(db.scalars(select(Student.person_id).order_by(Student.person_id)).all())
    worker_ids = list(db.scalars(select(Worker.person_id).order_by(Worker.person_id)).all())
    student_records = attendance_lookup(db, PersonType.STUDENT, student_ids, target_date)
    worker_records = attendance_lookup(db, PersonType.WORKER, worker_ids, target_date)
    return {
        "today": target_date,
        "totals": {
            "students": total_students,
            "classes": total_classes,
            "workers": total_workers,
        },
        "student_counts": status_counts(
            [normalize_status(student_records.get(item)) for item in student_ids]
        ),
        "worker_counts": status_counts(
            [normalize_status(worker_records.get(item)) for item in worker_ids]
        ),
    }


def get_students_overview(db: Session, target_date: date) -> dict[int, list[dict[str, object]]]:
    classes = db.scalars(
        select(SchoolClass)
        .options(selectinload(SchoolClass.students))
        .order_by(SchoolClass.grade, SchoolClass.section_code)
    ).all()
    person_ids = [student.person_id for item in classes for student in item.students]
    records = attendance_lookup(db, PersonType.STUDENT, person_ids, target_date)
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for school_class in classes:
        statuses = [
            normalize_status(records.get(student.person_id)) for student in school_class.students
        ]
        grouped[school_class.grade].append(
            {
                "id": school_class.id,
                "name": school_class.name,
                "total_students": len(school_class.students),
                "counts": status_counts(statuses),
            }
        )
    return dict(grouped)


def get_class_detail(db: Session, class_id: int, target_date: date) -> dict[str, object] | None:
    school_class = db.scalar(
        select(SchoolClass)
        .options(selectinload(SchoolClass.students))
        .where(SchoolClass.id == class_id)
    )
    if not school_class:
        return None
    students = sorted(school_class.students, key=lambda item: item.display_name.casefold())
    records = attendance_lookup(
        db, PersonType.STUDENT, [student.person_id for student in students], target_date
    )
    rows = []
    for student in students:
        record = records.get(student.person_id)
        rows.append(
            {
                "person_id": student.person_id,
                "full_name": student.display_name,
                "status": normalize_status(record),
                "absence_reason": record.absence_reason if record else None,
                "note": record.note if record else None,
            }
        )
    summary = status_counts([str(row["status"]) for row in rows])
    summary["total_students"] = len(rows)
    return {
        "school_class": school_class,
        "students": rows,
        "summary": summary,
        "today": target_date,
    }


def get_worker_categories_overview(db: Session, target_date: date) -> list[dict[str, object]]:
    categories = db.scalars(
        select(WorkerCategory)
        .options(selectinload(WorkerCategory.workers))
        .order_by(WorkerCategory.sort_order, WorkerCategory.name)
    ).all()
    person_ids = [worker.person_id for item in categories for worker in item.workers]
    records = attendance_lookup(db, PersonType.WORKER, person_ids, target_date)
    return [
        {
            "id": category.id,
            "name": category.name,
            "total_workers": len(category.workers),
            "counts": status_counts(
                [normalize_status(records.get(worker.person_id)) for worker in category.workers]
            ),
        }
        for category in categories
    ]


def get_worker_category_detail(
    db: Session, category_id: int, target_date: date
) -> dict[str, object] | None:
    category = db.scalar(
        select(WorkerCategory)
        .options(selectinload(WorkerCategory.workers))
        .where(WorkerCategory.id == category_id)
    )
    if not category:
        return None
    workers = sorted(category.workers, key=lambda item: item.display_name.casefold())
    records = attendance_lookup(
        db, PersonType.WORKER, [worker.person_id for worker in workers], target_date
    )
    rows = [
        {
            "person_id": worker.person_id,
            "full_name": worker.display_name,
            "status": normalize_status(records.get(worker.person_id)),
            "note": records[worker.person_id].note if worker.person_id in records else None,
        }
        for worker in workers
    ]
    summary = status_counts([str(row["status"]) for row in rows])
    summary["total_workers"] = len(rows)
    return {"category": category, "workers": rows, "summary": summary, "today": target_date}


def get_recent_recognition_events(db: Session, limit: int = 100) -> list[dict[str, object]]:
    safe_limit = min(max(1, limit), 500)
    events = db.scalars(
        select(RecognitionEvent)
        .order_by(RecognitionEvent.event_timestamp.desc(), RecognitionEvent.id.desc())
        .limit(safe_limit)
    ).all()
    return [
        {
            "id": event.id,
            "event_id": event.event_id,
            "person_id": event.person_id,
            "person_name": event.display_name,
            "class_name": event.class_name,
            "similarity": event.similarity,
            "timestamp": event.event_timestamp,
            "status": event.event_type.value,
            "source": event.source,
            "matched": event.matched,
            "matched_person_type": (
                event.matched_person_type.value if event.matched_person_type else None
            ),
            "matched_person_id": event.matched_person_id,
            "matched_person_name": event.matched_display_name,
            "outcome": event.outcome.value,
        }
        for event in events
    ]


def find_matching_student(db: Session, event: RecognitionEventIngest) -> Student | None:
    stable_match = db.scalar(select(Student).where(Student.person_id == event.person_id))
    if stable_match is not None:
        return stable_match
    if not get_settings().allow_legacy_name_fallback:
        return None

    normalized_name = normalize_lookup_text(event.display_name)
    students = db.scalars(select(Student).order_by(Student.id)).all()
    candidates = [
        student
        for student in students
        if normalize_lookup_text(student.display_name) == normalized_name
        and (
            event.class_name is None
            or normalize_lookup_text(student.class_name) == normalize_lookup_text(event.class_name)
        )
    ]
    return candidates[0] if len(candidates) == 1 else None


def _event_response(
    db: Session,
    event: RecognitionEvent,
    *,
    duplicate: bool = False,
) -> RecognitionEventIngestResponse:
    attendance = None
    if event.matched_person_id:
        attendance = db.scalar(
            select(AttendanceRecord).where(
                AttendanceRecord.person_type == PersonType.STUDENT,
                AttendanceRecord.person_id == event.matched_person_id,
                AttendanceRecord.date == _local_date(event.event_timestamp),
            )
        )
    return RecognitionEventIngestResponse(
        event_id=UUID(event.event_id),
        matched=event.matched,
        matched_person_type=event.matched_person_type,
        matched_person_id=event.matched_person_id,
        matched_display_name=event.matched_display_name,
        attendance_date=attendance.date if attendance else None,
        attendance_status=attendance.status if attendance else None,
        outcome=EventOutcome.DUPLICATE if duplicate else event.outcome,
        duplicate=duplicate,
    )


def _local_date(value: datetime) -> date:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(get_settings().local_zone).date()


def current_local_date() -> date:
    return datetime.now(UTC).astimezone(get_settings().local_zone).date()


def _is_on_time(value: datetime) -> bool:
    local_value = value.astimezone(get_settings().local_zone)
    return local_value.time().replace(tzinfo=None) < attendance_cutoff_time()


def _validate_event_age(event: RecognitionEventIngest, received_at: datetime) -> None:
    event_utc = event.timestamp.astimezone(UTC)
    delta = abs((received_at - event_utc).total_seconds())
    if delta > get_settings().recognition_max_event_age_seconds:
        raise StaleRecognitionEventError(
            "Recognition event timestamp is outside the allowed window"
        )


def _cooldown_active(db: Session, event: RecognitionEventIngest, student: Student) -> bool:
    seconds = get_settings().recognition_cooldown_seconds
    if seconds <= 0:
        return False
    threshold = event.timestamp.astimezone(UTC) - timedelta(seconds=seconds)
    previous = db.scalar(
        select(RecognitionEvent.id)
        .where(
            RecognitionEvent.matched_person_id == student.person_id,
            RecognitionEvent.event_type == event.event_type,
            RecognitionEvent.source == event.source,
            RecognitionEvent.outcome.in_([EventOutcome.APPLIED, EventOutcome.RECORDED]),
            RecognitionEvent.event_timestamp >= threshold,
            RecognitionEvent.event_timestamp <= event.timestamp,
        )
        .limit(1)
    )
    return previous is not None


def _apply_entry(
    db: Session, student: Student, event: RecognitionEventIngest
) -> tuple[AttendanceRecord, EventOutcome]:
    target_date = _local_date(event.timestamp)
    record = db.scalar(
        select(AttendanceRecord).where(
            AttendanceRecord.person_type == PersonType.STUDENT,
            AttendanceRecord.person_id == student.person_id,
            AttendanceRecord.date == target_date,
        )
    )
    proposed = AttendanceStatus.PRESENT if _is_on_time(event.timestamp) else AttendanceStatus.LATE
    if record is None:
        record = AttendanceRecord(
            person_type=PersonType.STUDENT,
            person_id=student.person_id,
            date=target_date,
            status=proposed,
            arrival_at=event.timestamp.astimezone(UTC),
            source=event.source or "face_recognition",
            note="First valid ENTRY event for this attendance day.",
            recorded_at=event.timestamp.astimezone(UTC),
            updated_at=utc_now(),
        )
        db.add(record)
        db.flush()
        return record, EventOutcome.APPLIED

    has_manual_correction = db.scalar(
        select(ManualCorrection.id)
        .where(ManualCorrection.attendance_record_id == record.id)
        .limit(1)
    )
    if has_manual_correction:
        return record, EventOutcome.RECORDED

    if record.arrival_at is None:
        record.arrival_at = event.timestamp.astimezone(UTC)
        record.status = proposed
        record.source = event.source or "face_recognition"
        record.updated_at = utc_now()
        return record, EventOutcome.APPLIED
    return record, EventOutcome.RECORDED


def _apply_exit(
    db: Session, student: Student, event: RecognitionEventIngest
) -> tuple[AttendanceRecord | None, EventOutcome]:
    record = db.scalar(
        select(AttendanceRecord).where(
            AttendanceRecord.person_type == PersonType.STUDENT,
            AttendanceRecord.person_id == student.person_id,
            AttendanceRecord.date == _local_date(event.timestamp),
        )
    )
    if record is None:
        return None, EventOutcome.RECORDED
    if record.departure_at is None:
        record.departure_at = event.timestamp.astimezone(UTC)
        record.updated_at = utc_now()
        return record, EventOutcome.APPLIED
    return record, EventOutcome.RECORDED


def ingest_recognition_event(
    db: Session, event: RecognitionEventIngest
) -> RecognitionEventIngestResponse:
    duplicate = db.scalar(
        select(RecognitionEvent).where(RecognitionEvent.event_id == str(event.event_id))
    )
    if duplicate is not None:
        return _event_response(db, duplicate, duplicate=True)

    received_at = utc_now()
    _validate_event_age(event, received_at)
    identity_candidate = find_matching_student(db, event)
    matched_student = None
    outcome = EventOutcome.UNMATCHED

    if event.similarity < get_settings().recognition_min_similarity:
        outcome = EventOutcome.REJECTED
    elif identity_candidate is not None:
        matched_student = identity_candidate
        if _cooldown_active(db, event, matched_student):
            outcome = EventOutcome.COOLDOWN
        elif event.event_type == EventType.ENTRY:
            _, outcome = _apply_entry(db, matched_student, event)
        elif event.event_type == EventType.EXIT:
            _, outcome = _apply_exit(db, matched_student, event)

    stored = RecognitionEvent(
        event_id=str(event.event_id),
        person_id=event.person_id,
        display_name=event.display_name,
        normalized_display_name=normalize_lookup_text(event.display_name),
        class_name=event.class_name,
        similarity=event.similarity,
        event_timestamp=event.timestamp.astimezone(UTC),
        event_type=event.event_type,
        source=event.source,
        matched=matched_student is not None,
        matched_person_type=PersonType.STUDENT if matched_student else None,
        matched_person_id=matched_student.person_id if matched_student else None,
        matched_display_name=matched_student.display_name if matched_student else None,
        outcome=outcome,
        received_at=received_at,
    )
    db.add(stored)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        concurrent_duplicate = db.scalar(
            select(RecognitionEvent).where(RecognitionEvent.event_id == str(event.event_id))
        )
        if concurrent_duplicate is not None:
            return _event_response(db, concurrent_duplicate, duplicate=True)
        raise
    db.refresh(stored)
    return _event_response(db, stored)


def apply_manual_correction(
    db: Session, correction: ManualCorrectionCreate, actor: User
) -> AttendanceRecord:
    model = Student if correction.person_type == PersonType.STUDENT else Worker
    if db.scalar(select(model).where(model.person_id == correction.person_id)) is None:
        raise ValueError("Unknown stable person ID")
    record = db.scalar(
        select(AttendanceRecord).where(
            AttendanceRecord.person_type == correction.person_type,
            AttendanceRecord.person_id == correction.person_id,
            AttendanceRecord.date == correction.attendance_date,
        )
    )
    previous = record.status if record else None
    if record is None:
        record = AttendanceRecord(
            person_type=correction.person_type,
            person_id=correction.person_id,
            date=correction.attendance_date,
            status=correction.status,
            source="manual",
            note=correction.reason,
            recorded_at=utc_now(),
            updated_at=utc_now(),
        )
        db.add(record)
        db.flush()
    else:
        record.status = correction.status
        record.source = "manual"
        record.note = correction.reason
        record.updated_at = utc_now()

    if correction.status == AttendanceStatus.ABSENT:
        record.arrival_at = None
        record.departure_at = None

    db.add(
        ManualCorrection(
            attendance_record_id=record.id,
            actor_user_id=actor.id,
            previous_status=previous,
            new_status=correction.status,
            reason=correction.reason,
        )
    )
    db.commit()
    db.refresh(record)
    return record
