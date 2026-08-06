from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from backend.app.models import AttendanceRecord, ManualCorrection, RecognitionEvent
from backend.app.schemas import (
    AttendanceStatus,
    EventOutcome,
    EventType,
    ManualCorrectionCreate,
    PersonType,
    RecognitionEventIngest,
)
from backend.app.services import apply_manual_correction, ingest_recognition_event
from backend.tests.helpers import add_student, add_user


def event(
    *,
    person_id: str = "DEMO-STU-0001",
    event_type: EventType = EventType.ENTRY,
    event_id: uuid.UUID | None = None,
    timestamp: datetime | None = None,
    similarity: float = 0.91,
) -> RecognitionEventIngest:
    return RecognitionEventIngest(
        event_id=event_id or uuid.uuid4(),
        person_id=person_id,
        display_name="Payload Display Name",
        class_name="5-01",
        similarity=similarity,
        timestamp=timestamp or datetime.now(UTC),
        event_type=event_type,
        source="test-entry-camera",
    )


def test_entry_cooldown_idempotency_and_exit_are_deterministic(db):
    add_student(db)
    now = datetime.now(UTC)
    event_id = uuid.uuid4()

    first = ingest_recognition_event(db, event(event_id=event_id, timestamp=now))
    assert first.outcome == EventOutcome.APPLIED
    assert first.attendance_status in {AttendanceStatus.PRESENT, AttendanceStatus.LATE}

    repeated_request = ingest_recognition_event(db, event(event_id=event_id, timestamp=now))
    assert repeated_request.duplicate is True
    assert repeated_request.outcome == EventOutcome.DUPLICATE

    cooldown = ingest_recognition_event(db, event(timestamp=now + timedelta(seconds=2)))
    assert cooldown.outcome == EventOutcome.COOLDOWN

    departure = ingest_recognition_event(
        db,
        event(event_type=EventType.EXIT, timestamp=now + timedelta(seconds=31)),
    )
    assert departure.outcome == EventOutcome.APPLIED
    record = db.scalar(select(AttendanceRecord))
    assert record is not None and record.arrival_at is not None
    assert record.departure_at is not None
    assert record.status != AttendanceStatus.ABSENT


def test_low_similarity_and_unknown_ids_do_not_match(db):
    add_student(db)
    rejected = ingest_recognition_event(db, event(similarity=0.20))
    unknown = ingest_recognition_event(db, event(person_id="DEMO-STU-9999"))
    assert rejected.outcome == EventOutcome.REJECTED
    assert rejected.matched is False
    assert unknown.outcome == EventOutcome.UNMATCHED
    assert db.scalar(select(AttendanceRecord)) is None


def test_duplicate_display_names_use_stable_ids(db):
    add_student(db, person_id="DEMO-STU-0001", display_name="Same Demo Name")
    add_student(db, person_id="DEMO-STU-0002", display_name="Same Demo Name")
    result = ingest_recognition_event(db, event(person_id="DEMO-STU-0002"))
    assert result.matched_person_id == "DEMO-STU-0002"


def test_manual_correction_is_audited_and_absent_clears_times(db):
    add_student(db)
    actor = add_user(db)
    ingest_recognition_event(db, event())
    corrected = apply_manual_correction(
        db,
        ManualCorrectionCreate(
            person_type=PersonType.STUDENT,
            person_id="DEMO-STU-0001",
            attendance_date=datetime.now(UTC).date(),
            status=AttendanceStatus.ABSENT,
            reason="Synthetic test correction",
        ),
        actor,
    )
    assert corrected.status == AttendanceStatus.ABSENT
    assert corrected.arrival_at is None and corrected.departure_at is None
    audit = db.scalar(select(ManualCorrection))
    assert audit is not None and audit.actor_user_id == actor.id


def test_api_auth_validation_and_idempotency_header(client, db):
    add_student(db)
    payload_event = event()
    payload = payload_event.model_dump(mode="json")

    assert client.post("/api/recognition-event", json=payload).status_code == 401
    headers = {"X-API-Key": "test-only-ingest-key-with-more-than-32-characters"}
    mismatch = client.post(
        "/api/recognition-event",
        json=payload,
        headers={**headers, "Idempotency-Key": str(uuid.uuid4())},
    )
    assert mismatch.status_code == 400
    accepted = client.post(
        "/api/recognition-event",
        json=payload,
        headers={**headers, "Idempotency-Key": str(payload_event.event_id)},
    )
    assert accepted.status_code == 200
    assert accepted.json()["outcome"] == "applied"
    duplicate = client.post(
        "/api/recognition-event",
        json=payload,
        headers={**headers, "Idempotency-Key": str(payload_event.event_id)},
    )
    assert duplicate.json()["duplicate"] is True
    assert db.query(RecognitionEvent).count() == 1
