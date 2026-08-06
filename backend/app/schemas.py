from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class UserRole(str, Enum):
    DIRECTOR = "director"
    TEACHER = "teacher"


class AttendanceStatus(str, Enum):
    PRESENT = "present"
    LATE = "late"
    ABSENT = "absent"


class PersonType(str, Enum):
    STUDENT = "student"
    WORKER = "worker"


class EventType(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    MANUAL_CORRECTION = "MANUAL_CORRECTION"


class EventOutcome(str, Enum):
    APPLIED = "applied"
    RECORDED = "recorded"
    DUPLICATE = "duplicate"
    COOLDOWN = "cooldown"
    UNMATCHED = "unmatched"
    REJECTED = "rejected"


def collapse_spaces(value: str) -> str:
    return " ".join(value.split())


class RecognitionEventIngest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    event_id: UUID
    person_id: str = Field(min_length=2, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]+$")
    display_name: str = Field(min_length=1, max_length=120)
    class_name: str | None = Field(default=None, max_length=20)
    similarity: float = Field(ge=-1.0, le=1.0)
    timestamp: datetime
    event_type: EventType
    source: str | None = Field(default=None, max_length=100)

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, value: str) -> str:
        collapsed = collapse_spaces(value)
        if not collapsed:
            raise ValueError("display_name is required")
        return collapsed

    @field_validator("class_name", "source", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> str | None:
        if value is None:
            return None
        collapsed = collapse_spaces(str(value))
        return collapsed or None

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a UTC offset")
        return value

    @field_validator("event_type")
    @classmethod
    def reject_manual_api_events(cls, value: EventType) -> EventType:
        if value == EventType.MANUAL_CORRECTION:
            raise ValueError("MANUAL_CORRECTION events require an authenticated director")
        return value


class RecognitionEventIngestResponse(BaseModel):
    event_id: UUID
    matched: bool
    matched_person_type: PersonType | None = None
    matched_person_id: str | None = None
    matched_display_name: str | None = None
    attendance_date: date | None = None
    attendance_status: AttendanceStatus | None = None
    outcome: EventOutcome
    duplicate: bool = False


class ManualCorrectionCreate(BaseModel):
    person_type: PersonType
    person_id: str = Field(min_length=2, max_length=64)
    attendance_date: date
    status: AttendanceStatus
    reason: str = Field(min_length=3, max_length=500)
