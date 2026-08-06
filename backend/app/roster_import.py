from __future__ import annotations

import csv
import io
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import SchoolClass, Student
from backend.app.settings import get_settings

REQUIRED_COLUMNS = {"student_id", "display_name", "class_name"}
CLASS_NAME_PATTERN = re.compile(r"^(?P<grade>\d{1,2})-(?P<section>\d{2})$")
STUDENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{1,63}$")
LEGACY_NAMESPACE = uuid.UUID("03dc49bb-36c5-4aed-b952-b17e673b8f25")
FORMULA_PREFIXES = ("=", "+", "-", "@")


@dataclass(frozen=True)
class RosterRow:
    line_number: int
    student_id: str
    display_name: str
    class_name: str


@dataclass
class RosterImportSummary:
    rows_read: int = 0
    rows_valid: int = 0
    students_created: int = 0
    students_updated: int = 0
    duplicates_skipped: int = 0
    classes_created: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, int | list[str]]:
        return {
            "rows_read": self.rows_read,
            "rows_valid": self.rows_valid,
            "students_created": self.students_created,
            "students_updated": self.students_updated,
            "duplicates_skipped": self.duplicates_skipped,
            "classes_created": self.classes_created,
            "errors": list(self.errors),
        }


def normalize_roster_text(value: str | None) -> str:
    return " ".join(value.split()) if value else ""


def parse_class_identity(class_name: str) -> tuple[int, str]:
    match = CLASS_NAME_PATTERN.fullmatch(class_name)
    if not match:
        raise ValueError(f"Invalid class name format: {class_name!r}; expected 5-03")
    grade = int(match.group("grade"))
    if not 1 <= grade <= 12:
        raise ValueError(f"Grade must be between 1 and 12: {class_name!r}")
    return grade, match.group("section")


def _read_limited(path: Path) -> bytes:
    if path.suffix.casefold() not in {".csv", ".txt"}:
        raise ValueError("Roster must be a .csv file; .txt is accepted only for legacy migration")
    size = path.stat().st_size
    if size > get_settings().roster_max_bytes:
        raise ValueError(f"Roster is too large ({size} bytes)")
    return path.read_bytes()


def _decode(data: bytes) -> str:
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError("Roster encoding must be UTF-8, UTF-8 with BOM, or CP1251") from last_error


def _validate_cell(value: str, *, field_name: str) -> str:
    cleaned = normalize_roster_text(value)
    if not cleaned:
        raise ValueError(f"{field_name} is required")
    if cleaned.startswith(FORMULA_PREFIXES):
        raise ValueError(f"{field_name} may not begin with spreadsheet formula syntax")
    if any(character in cleaned for character in ("\x00", "\r", "\n", "\t")):
        raise ValueError(f"{field_name} contains control characters")
    maximum = {"student_id": 64, "display_name": 120, "class_name": 20}[field_name]
    if len(cleaned) > maximum:
        raise ValueError(f"{field_name} exceeds {maximum} characters")
    return cleaned


def parse_csv_roster(path: Path) -> tuple[list[RosterRow], RosterImportSummary]:
    text = _decode(_read_limited(path))
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames:
        reader.fieldnames = [item.strip() for item in reader.fieldnames]
    headers = set(reader.fieldnames or [])
    missing = REQUIRED_COLUMNS - headers
    if missing:
        raise ValueError(f"Missing required roster columns: {', '.join(sorted(missing))}")

    summary = RosterImportSummary()
    rows: list[RosterRow] = []
    seen_ids: set[str] = set()
    for line_number, raw in enumerate(reader, start=2):
        summary.rows_read += 1
        if summary.rows_read > get_settings().roster_max_rows:
            raise ValueError("Roster row limit exceeded")
        try:
            if None in raw:
                raise ValueError("row contains more values than the header")
            student_id = _validate_cell(raw.get("student_id", ""), field_name="student_id")
            if not STUDENT_ID_PATTERN.fullmatch(student_id):
                raise ValueError("student_id contains unsupported characters")
            display_name = _validate_cell(raw.get("display_name", ""), field_name="display_name")
            class_name = _validate_cell(raw.get("class_name", ""), field_name="class_name")
            parse_class_identity(class_name)
            normalized_id = student_id.casefold()
            if normalized_id in seen_ids:
                summary.duplicates_skipped += 1
                raise ValueError(f"duplicate student_id {student_id!r} in upload")
            seen_ids.add(normalized_id)
        except ValueError as exc:
            summary.errors.append(f"line {line_number}: {exc}")
            continue
        rows.append(RosterRow(line_number, student_id, display_name, class_name))
    summary.rows_valid = len(rows)
    return rows, summary


def parse_legacy_text_roster(path: Path) -> tuple[list[RosterRow], RosterImportSummary]:
    """Migration-only parser for the old class-header/name format.

    IDs are deterministically frozen from class and normalized name. New rosters should
    provide authoritative school IDs in CSV instead.
    """
    text = _decode(_read_limited(path))
    summary = RosterImportSummary()
    rows: list[RosterRow] = []
    current_class: str | None = None
    seen: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith(":"):
            current_class = line[:-1]
            parse_class_identity(current_class)
            continue
        summary.rows_read += 1
        if current_class is None:
            summary.errors.append(f"line {line_number}: student appears before class header")
            continue
        display_name = _validate_cell(line, field_name="display_name")
        identity_source = f"{current_class}:{display_name.casefold()}"
        student_id = f"legacy-{uuid.uuid5(LEGACY_NAMESPACE, identity_source)}"
        if student_id in seen:
            summary.duplicates_skipped += 1
            summary.errors.append(f"line {line_number}: duplicate class/name pair")
            continue
        seen.add(student_id)
        rows.append(RosterRow(line_number, student_id, display_name, current_class))
    summary.rows_valid = len(rows)
    return rows, summary


def _get_or_create_class(db: Session, class_name: str) -> tuple[SchoolClass, bool]:
    grade, section = parse_class_identity(class_name)
    school_class = db.scalar(select(SchoolClass).where(SchoolClass.name == class_name))
    if school_class:
        return school_class, False
    school_class = SchoolClass(grade=grade, section_code=section, name=class_name)
    db.add(school_class)
    db.flush()
    return school_class, True


def import_roster_file(
    db: Session,
    roster_path: Path,
    *,
    allow_partial: bool = True,
    allow_legacy_text: bool = False,
) -> RosterImportSummary:
    path = Path(roster_path)
    if not path.is_file():
        raise FileNotFoundError(f"Roster file not found: {path}")
    if path.suffix.casefold() == ".txt":
        if not allow_legacy_text:
            raise ValueError("Legacy .txt import requires the explicit --allow-legacy-text flag")
        rows, summary = parse_legacy_text_roster(path)
    else:
        rows, summary = parse_csv_roster(path)
    if summary.errors and not allow_partial:
        raise ValueError("Roster contains invalid rows: " + "; ".join(summary.errors[:5]))

    classes: dict[str, SchoolClass] = {}
    for row in rows:
        school_class = classes.get(row.class_name)
        if school_class is None:
            school_class, created = _get_or_create_class(db, row.class_name)
            classes[row.class_name] = school_class
            summary.classes_created += int(created)

        student = db.scalar(select(Student).where(Student.person_id == row.student_id))
        if student is None:
            db.add(
                Student(
                    person_id=row.student_id,
                    display_name=row.display_name,
                    grade=school_class.grade,
                    class_name=school_class.name,
                    class_id=school_class.id,
                )
            )
            summary.students_created += 1
            continue

        changed = False
        for attribute, value in (
            ("display_name", row.display_name),
            ("grade", school_class.grade),
            ("class_name", school_class.name),
            ("class_id", school_class.id),
        ):
            if getattr(student, attribute) != value:
                setattr(student, attribute, value)
                changed = True
        summary.students_updated += int(changed)
        summary.duplicates_skipped += int(not changed)

    db.commit()
    return summary
