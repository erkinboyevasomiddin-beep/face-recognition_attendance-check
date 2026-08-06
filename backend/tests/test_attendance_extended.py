from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from backend.app import services
from backend.app.models import (
    AttendanceRecord,
    ManualCorrection,
    RecognitionEvent,
    Worker,
    WorkerCategory,
)
from backend.app.schemas import (
    AttendanceStatus,
    EventOutcome,
    EventType,
    ManualCorrectionCreate,
    PersonType,
)
from backend.app.services import (
    StaleRecognitionEventError,
    _local_date,
    apply_manual_correction,
    get_class_detail,
    get_director_overview,
    get_recent_recognition_events,
    get_students_overview,
    get_worker_categories_overview,
    get_worker_category_detail,
    ingest_recognition_event,
)
from backend.tests.helpers import add_student, add_user
from backend.tests.test_attendance import event


def test_entry_cutoff_boundary_repeated_entry_and_repeated_exit(db, monkeypatch):
    add_student(db)
    local_zone = __import__("zoneinfo").ZoneInfo("Asia/Tashkent")
    local_day = datetime.now(local_zone).date()
    cutoff = datetime.combine(local_day, datetime.min.time(), tzinfo=local_zone).replace(
        hour=8, minute=30
    )
    monkeypatch.setattr(services, "utc_now", lambda: cutoff.astimezone(UTC))

    first = ingest_recognition_event(db, event(timestamp=cutoff.astimezone(UTC)))
    assert first.attendance_status == AttendanceStatus.LATE

    repeated = ingest_recognition_event(
        db, event(timestamp=(cutoff + timedelta(minutes=2)).astimezone(UTC))
    )
    assert repeated.outcome == EventOutcome.RECORDED

    departure_time = (cutoff + timedelta(minutes=3)).astimezone(UTC)
    departure = ingest_recognition_event(
        db, event(event_type=EventType.EXIT, timestamp=departure_time)
    )
    assert departure.outcome == EventOutcome.APPLIED
    repeated_departure = ingest_recognition_event(
        db,
        event(event_type=EventType.EXIT, timestamp=departure_time + timedelta(seconds=31)),
    )
    assert repeated_departure.outcome == EventOutcome.RECORDED


def test_exit_without_entry_is_an_audited_noop(db):
    add_student(db)
    result = ingest_recognition_event(db, event(event_type=EventType.EXIT))
    assert result.outcome == EventOutcome.RECORDED
    assert result.attendance_status is None
    assert db.scalar(select(AttendanceRecord)) is None
    stored = db.scalar(select(RecognitionEvent))
    assert stored is not None and stored.matched is True


def test_stale_event_is_rejected_without_storage(db):
    add_student(db)
    with pytest.raises(StaleRecognitionEventError, match="outside the allowed window"):
        ingest_recognition_event(db, event(timestamp=datetime.now(UTC) - timedelta(hours=1)))
    assert db.scalar(select(RecognitionEvent)) is None


def test_timezone_day_boundary_is_explicit():
    assert _local_date(datetime(2026, 1, 1, 18, 59, tzinfo=UTC)) == date(2026, 1, 1)
    assert _local_date(datetime(2026, 1, 1, 19, 0, tzinfo=UTC)) == date(2026, 1, 2)
    assert _local_date(datetime(2026, 1, 2, 0, 0)) == date(2026, 1, 2)


def test_manual_worker_correction_and_unknown_id(db):
    actor = add_user(db)
    category = WorkerCategory(name="Synthetic Staff", sort_order=1)
    db.add(category)
    db.flush()
    worker = Worker(
        person_id="DEMO-WRK-0099",
        display_name="Demo Worker 099",
        category_id=category.id,
    )
    db.add(worker)
    db.commit()

    correction = ManualCorrectionCreate(
        person_type=PersonType.WORKER,
        person_id=worker.person_id,
        attendance_date=date(2026, 1, 1),
        status=AttendanceStatus.PRESENT,
        reason="Synthetic manual entry",
    )
    record = apply_manual_correction(db, correction, actor)
    assert record.person_type == PersonType.WORKER
    assert db.scalar(select(ManualCorrection)) is not None

    with pytest.raises(ValueError, match="Unknown stable person ID"):
        apply_manual_correction(
            db,
            correction.model_copy(update={"person_id": "DEMO-WRK-4040"}),
            actor,
        )


def test_manual_correction_updates_existing_record_without_losing_audit(db):
    add_student(db)
    actor = add_user(db)
    arrival = ingest_recognition_event(db, event())
    assert arrival.attendance_date is not None
    record = apply_manual_correction(
        db,
        ManualCorrectionCreate(
            person_type=PersonType.STUDENT,
            person_id="DEMO-STU-0001",
            attendance_date=arrival.attendance_date,
            status=AttendanceStatus.PRESENT,
            reason="Reviewed against synthetic register",
        ),
        actor,
    )
    assert record.status == AttendanceStatus.PRESENT
    assert record.arrival_at is not None
    assert db.query(ManualCorrection).count() == 1


def test_dashboard_queries_and_event_limit_are_stable(db):
    student = add_student(db)
    actor = add_user(db)
    ingest_recognition_event(db, event(person_id=student.person_id))
    category = WorkerCategory(name="Synthetic Staff", sort_order=1)
    db.add(category)
    db.flush()
    worker = Worker(person_id="DEMO-WRK-0001", display_name="Demo Worker", category_id=category.id)
    db.add(worker)
    db.commit()
    apply_manual_correction(
        db,
        ManualCorrectionCreate(
            person_type=PersonType.WORKER,
            person_id=worker.person_id,
            attendance_date=datetime.now(UTC).astimezone().date(),
            status=AttendanceStatus.PRESENT,
            reason="Synthetic test correction",
        ),
        actor,
    )

    target = _local_date(datetime.now(UTC))
    overview = get_director_overview(db, target)
    assert overview["totals"] == {"students": 1, "classes": 1, "workers": 1}
    assert get_students_overview(db, target)[5][0]["total_students"] == 1
    assert get_class_detail(db, student.class_id, target) is not None
    assert get_class_detail(db, 9999, target) is None
    assert get_worker_categories_overview(db, target)[0]["total_workers"] == 1
    assert get_worker_category_detail(db, category.id, target) is not None
    assert get_worker_category_detail(db, 9999, target) is None
    assert len(get_recent_recognition_events(db, limit=0)) == 1
    assert len(get_recent_recognition_events(db, limit=9999)) == 1


def test_api_schema_rejects_manual_naive_and_invalid_payload(client):
    key = {"X-API-Key": "test-only-ingest-key-with-more-than-32-characters"}
    payload = event().model_dump(mode="json")
    payload["event_type"] = "MANUAL_CORRECTION"
    assert client.post("/api/recognition-event", json=payload, headers=key).status_code == 422
    payload["event_type"] = "ENTRY"
    payload["timestamp"] = "2026-01-01T00:00:00"
    assert client.post("/api/recognition-event", json=payload, headers=key).status_code == 422
    payload["timestamp"] = datetime.now(UTC).isoformat()
    payload["person_id"] = "bad id"
    assert client.post("/api/recognition-event", json=payload, headers=key).status_code == 422


def test_stable_id_database_constraints_reject_case_reuse_and_bad_format(db):
    add_student(db, person_id="DEMO-CASE-001")
    with pytest.raises(IntegrityError):
        add_student(db, person_id="demo-case-001", class_name="6-01")
    db.rollback()
    with pytest.raises(IntegrityError):
        add_student(db, person_id="bad id", class_name="6-01")
    db.rollback()
