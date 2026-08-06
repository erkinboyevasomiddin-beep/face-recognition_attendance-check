from __future__ import annotations

from datetime import UTC, date, datetime
from enum import Enum

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
    func,
)
from sqlalchemy import (
    Enum as SqlEnum,
)
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database import Base
from backend.app.schemas import AttendanceStatus, EventOutcome, EventType, PersonType, UserRole

STABLE_ID_SQL_CHECK = (
    "length(person_id) BETWEEN 2 AND 64 "
    "AND substr(person_id, 1, 1) GLOB '[A-Za-z0-9]' "
    "AND person_id NOT GLOB '*[^A-Za-z0-9_.:-]*'"
)


def enum_values(enum_cls: type[Enum]) -> list[str]:
    return [member.value for member in enum_cls]


def utc_now() -> datetime:
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator[datetime]):
    """Persist UTC and restore timezone information lost by SQLite."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Datetime values must include a UTC offset")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    role: Mapped[UserRole] = mapped_column(
        SqlEnum(UserRole, native_enum=False, values_callable=enum_values),
        nullable=False,
        index=True,
    )
    display_name: Mapped[str] = mapped_column("full_name", String(120), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(timezone=True), default=utc_now, nullable=False
    )

    @property
    def full_name(self) -> str:
        return self.display_name


class SchoolClass(Base):
    __tablename__ = "school_classes"
    __table_args__ = (
        UniqueConstraint("grade", "section_code", name="uq_grade_section"),
        CheckConstraint("grade BETWEEN 1 AND 12", name="ck_school_class_grade"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    grade: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    section_code: Mapped[str] = mapped_column(String(2), nullable=False)
    name: Mapped[str] = mapped_column(String(20), unique=True, index=True, nullable=False)
    students: Mapped[list[Student]] = relationship(
        back_populates="school_class", cascade="all, delete-orphan"
    )


class Student(Base):
    __tablename__ = "students"
    __table_args__ = (
        UniqueConstraint("person_id", name="uq_students_person_id"),
        CheckConstraint(STABLE_ID_SQL_CHECK, name="ck_students_person_id_format"),
        Index("ix_students_class_display_name", "class_id", "full_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    person_id: Mapped[str] = mapped_column(
        String(64, collation="NOCASE"), nullable=False, index=True
    )
    display_name: Mapped[str] = mapped_column("full_name", String(120), nullable=False, index=True)
    grade: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    class_name: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    class_id: Mapped[int] = mapped_column(
        ForeignKey("school_classes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    school_class: Mapped[SchoolClass] = relationship(back_populates="students")

    @property
    def full_name(self) -> str:
        return self.display_name

    @full_name.setter
    def full_name(self, value: str) -> None:
        self.display_name = value


class WorkerCategory(Base):
    __tablename__ = "worker_categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    sort_order: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    workers: Mapped[list[Worker]] = relationship(
        back_populates="category", cascade="all, delete-orphan"
    )


class Worker(Base):
    __tablename__ = "workers"
    __table_args__ = (
        UniqueConstraint("person_id", name="uq_workers_person_id"),
        CheckConstraint(STABLE_ID_SQL_CHECK, name="ck_workers_person_id_format"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    person_id: Mapped[str] = mapped_column(
        String(64, collation="NOCASE"), nullable=False, index=True
    )
    display_name: Mapped[str] = mapped_column("full_name", String(120), nullable=False, index=True)
    category_id: Mapped[int] = mapped_column(
        ForeignKey("worker_categories.id", ondelete="CASCADE"), nullable=False, index=True
    )
    category: Mapped[WorkerCategory] = relationship(back_populates="workers")

    @property
    def full_name(self) -> str:
        return self.display_name


class AttendanceRecord(Base):
    __tablename__ = "attendance_records"
    __table_args__ = (
        UniqueConstraint("person_type", "person_id", "date", name="uq_person_day"),
        CheckConstraint(STABLE_ID_SQL_CHECK, name="ck_attendance_person_id_format"),
        Index("ix_attendance_lookup", "person_type", "person_id", "date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    person_type: Mapped[PersonType] = mapped_column(
        SqlEnum(PersonType, native_enum=False, values_callable=enum_values),
        nullable=False,
        index=True,
    )
    person_id: Mapped[str] = mapped_column(
        String(64, collation="NOCASE"), nullable=False, index=True
    )
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    status: Mapped[AttendanceStatus] = mapped_column(
        SqlEnum(AttendanceStatus, native_enum=False, values_callable=enum_values),
        nullable=False,
        index=True,
    )
    arrival_at: Mapped[datetime | None] = mapped_column(UTCDateTime(timezone=True))
    departure_at: Mapped[datetime | None] = mapped_column(UTCDateTime(timezone=True))
    source: Mapped[str | None] = mapped_column(String(100))
    note: Mapped[str | None] = mapped_column(String(500))
    absence_reason: Mapped[str | None] = mapped_column(String(255))
    recorded_at: Mapped[datetime] = mapped_column(
        UTCDateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )


class RecognitionEvent(Base):
    __tablename__ = "recognition_events"
    __table_args__ = (
        CheckConstraint("similarity BETWEEN -1.0 AND 1.0", name="ck_event_similarity"),
        CheckConstraint(STABLE_ID_SQL_CHECK, name="ck_event_person_id_format"),
        Index("ix_recognition_events_lookup", "matched", "matched_person_id"),
        Index(
            "ix_recognition_events_cooldown",
            "matched_person_id",
            "event_type",
            "source",
            "event_timestamp",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[str] = mapped_column(String(36), unique=True, index=True, nullable=False)
    person_id: Mapped[str] = mapped_column(
        String(64, collation="NOCASE"), nullable=False, index=True
    )
    display_name: Mapped[str] = mapped_column("person_name", String(120), nullable=False)
    normalized_display_name: Mapped[str] = mapped_column(
        "normalized_person_name", String(120), nullable=False, index=True
    )
    class_name: Mapped[str | None] = mapped_column(String(20), index=True)
    similarity: Mapped[float] = mapped_column(Float, nullable=False)
    event_timestamp: Mapped[datetime] = mapped_column(UTCDateTime(timezone=True), index=True)
    event_type: Mapped[EventType] = mapped_column(
        SqlEnum(EventType, native_enum=False, values_callable=enum_values), nullable=False
    )
    source: Mapped[str | None] = mapped_column(String(100))
    matched: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    matched_person_type: Mapped[PersonType | None] = mapped_column(
        SqlEnum(PersonType, native_enum=False, values_callable=enum_values), index=True
    )
    matched_person_id: Mapped[str | None] = mapped_column(
        String(64, collation="NOCASE"), index=True
    )
    matched_display_name: Mapped[str | None] = mapped_column("matched_person_name", String(120))
    outcome: Mapped[EventOutcome] = mapped_column(
        SqlEnum(EventOutcome, native_enum=False, values_callable=enum_values), nullable=False
    )
    received_at: Mapped[datetime] = mapped_column(
        UTCDateTime(timezone=True), default=utc_now, server_default=func.now(), nullable=False
    )


class ManualCorrection(Base):
    __tablename__ = "manual_corrections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    attendance_record_id: Mapped[int] = mapped_column(
        ForeignKey("attendance_records.id", ondelete="CASCADE"), nullable=False, index=True
    )
    actor_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    previous_status: Mapped[AttendanceStatus | None] = mapped_column(
        SqlEnum(AttendanceStatus, native_enum=False, values_callable=enum_values)
    )
    new_status: Mapped[AttendanceStatus] = mapped_column(
        SqlEnum(AttendanceStatus, native_enum=False, values_callable=enum_values), nullable=False
    )
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(timezone=True), default=utc_now, nullable=False
    )
