from __future__ import annotations

import hashlib
import sqlite3

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from backend.app.models import AttendanceRecord, RecognitionEvent, Student, User
from backend.app.schemas import EventType
from backend.scripts.migrate_v1_database import migrate


def test_v1_backend_migration_is_copy_based_and_deterministic(tmp_path):
    source = tmp_path / "legacy.db"
    destination = tmp_path / "migrated.db"
    with sqlite3.connect(source) as db:
        db.executescript(
            """
            CREATE TABLE users (id INTEGER, username TEXT, password_hash TEXT, role TEXT,
              full_name TEXT, created_at TEXT);
            CREATE TABLE school_classes (id INTEGER, grade INTEGER, section_code TEXT, name TEXT);
            CREATE TABLE students (id INTEGER, full_name TEXT, grade INTEGER, class_name TEXT,
              class_id INTEGER);
            CREATE TABLE worker_categories (id INTEGER, name TEXT, sort_order INTEGER);
            CREATE TABLE workers (id INTEGER, full_name TEXT, category_id INTEGER);
            CREATE TABLE attendance_records (id INTEGER, person_type TEXT, person_id INTEGER,
              date TEXT, status TEXT, source TEXT, note TEXT, absence_reason TEXT, recorded_at TEXT);
            CREATE TABLE recognition_events (id INTEGER, person_name TEXT,
              normalized_person_name TEXT, class_name TEXT, similarity REAL,
              event_timestamp TEXT, source TEXT, matched INTEGER, matched_person_type TEXT,
              matched_person_id INTEGER, matched_person_name TEXT, received_at TEXT);
            """
        )
        db.execute(
            "INSERT INTO users VALUES (1, ?, ?, 'director', ?, '2026-01-01T08:00:00')",
            ("legacy.demo", hashlib.sha256(b"legacy password").hexdigest(), "Legacy Demo"),
        )
        db.execute("INSERT INTO school_classes VALUES (1, 5, '01', '5-01')")
        db.execute("INSERT INTO students VALUES (1, 'Demo Student', 5, '5-01', 1)")
        db.execute("INSERT INTO worker_categories VALUES (1, 'Demo Staff', 1)")
        db.execute("INSERT INTO workers VALUES (1, 'Demo Worker', 1)")
        db.execute(
            "INSERT INTO attendance_records VALUES "
            "(1, 'student', 1, '2026-01-01', 'present', 'legacy', NULL, NULL, "
            "'2026-01-01T08:15:00')"
        )
        db.execute(
            "INSERT INTO recognition_events VALUES "
            "(1, 'Demo Student', 'demo student', '5-01', 0.9, "
            "'2026-01-01T08:15:00', 'legacy-camera', 1, 'student', 1, "
            "'Demo Student', '2026-01-01T08:15:01')"
        )

    source_size = source.stat().st_size
    migrate(source, destination, "Asia/Tashkent")
    assert source.stat().st_size == source_size

    engine = create_engine(f"sqlite:///{destination.as_posix()}")
    with Session(engine) as db:
        student = db.scalar(select(Student))
        event = db.scalar(select(RecognitionEvent))
        attendance = db.scalar(select(AttendanceRecord))
        user = db.scalar(select(User))
        assert student is not None and student.person_id == "legacy-student-000001"
        assert attendance is not None and attendance.person_id == student.person_id
        assert event is not None and event.event_type == EventType.ENTRY
        assert event.matched_person_id == student.person_id
        assert user is not None and len(user.password_hash) == 64
    engine.dispose()
