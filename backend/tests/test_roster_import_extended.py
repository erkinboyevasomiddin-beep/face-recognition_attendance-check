from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import select

from backend.app import roster_import
from backend.app.models import Student
from backend.app.roster_import import (
    import_roster_file,
    parse_class_identity,
    parse_csv_roster,
    parse_legacy_text_roster,
)


@pytest.mark.parametrize(
    "class_name",
    ["", "5A", "0-01", "13-01", "5-1", "5-AAA"],
)
def test_class_identity_validation(class_name):
    with pytest.raises(ValueError):
        parse_class_identity(class_name)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("bad id,Demo Student,5-01", "unsupported characters"),
        ("DEMO-001,,5-01", "display_name is required"),
        ("DEMO-001,@formula,5-01", "formula syntax"),
        ("DEMO-001,Demo Student,+5-01", "formula syntax"),
        ("DEMO-001,Demo Student,99-01", "Grade must"),
        (f"DEMO-001,{'x' * 121},5-01", "exceeds 120"),
        (f"{'x' * 65},Demo Student,5-01", "exceeds 64"),
    ],
)
def test_each_roster_cell_rule(tmp_path, row, message):
    path = tmp_path / "invalid.csv"
    path.write_text(f"student_id,display_name,class_name\n{row}\n", encoding="utf-8")
    _, summary = parse_csv_roster(path)
    assert message in summary.errors[0]


def test_duplicate_ids_are_case_insensitive(tmp_path):
    path = tmp_path / "duplicates.csv"
    path.write_text(
        "student_id,display_name,class_name\nDemo-001,First Demo,5-01\ndemo-001,Second Demo,5-01\n",
        encoding="utf-8",
    )
    rows, summary = parse_csv_roster(path)
    assert len(rows) == 1
    assert summary.duplicates_skipped == 1


def test_file_type_size_row_and_encoding_limits(tmp_path, monkeypatch):
    wrong = tmp_path / "roster.json"
    wrong.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a .csv"):
        parse_csv_roster(wrong)

    csv_path = tmp_path / "large.csv"
    csv_path.write_text(
        "student_id,display_name,class_name\nDEMO-001,Demo Student,5-01\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        roster_import,
        "get_settings",
        lambda: SimpleNamespace(roster_max_bytes=10, roster_max_rows=1),
    )
    with pytest.raises(ValueError, match="too large"):
        parse_csv_roster(csv_path)

    invalid = tmp_path / "invalid.csv"
    invalid.write_bytes(b"\x98\x98\x98")
    monkeypatch.setattr(
        roster_import,
        "get_settings",
        lambda: SimpleNamespace(roster_max_bytes=1000, roster_max_rows=1),
    )
    with pytest.raises(ValueError, match="encoding"):
        parse_csv_roster(invalid)

    rows = tmp_path / "rows.csv"
    rows.write_text(
        "student_id,display_name,class_name\n"
        "DEMO-001,Demo Student,5-01\nDEMO-002,Demo Student,5-01\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="row limit"):
        parse_csv_roster(rows)


def test_cp1251_and_utf8_bom_are_supported(tmp_path):
    body = "student_id,display_name,class_name\nDEMO-001,Синтетический ученик,5-01\n"
    cp_path = tmp_path / "cp1251.csv"
    cp_path.write_bytes(body.encode("cp1251"))
    assert parse_csv_roster(cp_path)[0][0].display_name == "Синтетический ученик"
    bom_path = tmp_path / "bom.csv"
    bom_path.write_bytes(body.encode("utf-8-sig"))
    assert parse_csv_roster(bom_path)[0][0].student_id == "DEMO-001"


def test_legacy_import_requires_flag_and_generates_deterministic_ids(db, tmp_path):
    path = tmp_path / "legacy.txt"
    path.write_text("Orphan Demo\n5-01:\nDemo Name\nDemo Name\n", encoding="utf-8")
    rows, summary = parse_legacy_text_roster(path)
    assert len(rows) == 1
    assert rows[0].student_id.startswith("legacy-")
    assert summary.duplicates_skipped == 1
    assert len(summary.errors) == 2
    with pytest.raises(ValueError, match="explicit"):
        import_roster_file(db, path)
    imported = import_roster_file(db, path, allow_legacy_text=True)
    assert imported.students_created == 1


def test_update_existing_student_and_missing_file(db, tmp_path):
    with pytest.raises(FileNotFoundError):
        import_roster_file(db, tmp_path / "missing.csv")
    first = tmp_path / "first.csv"
    first.write_text(
        "student_id,display_name,class_name\nDEMO-001,First Demo,5-01\n",
        encoding="utf-8",
    )
    second = tmp_path / "second.csv"
    second.write_text(
        "student_id,display_name,class_name\nDEMO-001,Updated Demo,6-01\n",
        encoding="utf-8",
    )
    import_roster_file(db, first)
    summary = import_roster_file(db, second)
    student = db.scalar(select(Student).where(Student.person_id == "DEMO-001"))
    assert summary.students_updated == 1
    assert student is not None and student.display_name == "Updated Demo"
    assert student.class_name == "6-01"
