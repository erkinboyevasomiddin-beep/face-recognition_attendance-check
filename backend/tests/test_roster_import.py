from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select

from backend.app.models import Student
from backend.app.roster_import import import_roster_file, parse_csv_roster


def write_roster(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_valid_roster_duplicate_names_and_repeat_import(db, tmp_path):
    path = write_roster(
        tmp_path / "roster.csv",
        "student_id,display_name,class_name\n"
        "DEMO-001,Same Demo Name,5-01\n"
        "DEMO-002,Same Demo Name,5-01\n",
    )
    first = import_roster_file(db, path)
    second = import_roster_file(db, path)
    assert first.students_created == 2
    assert second.duplicates_skipped == 2
    assert len(db.scalars(select(Student)).all()) == 2


def test_malformed_partial_and_strict_import(db, tmp_path):
    path = write_roster(
        tmp_path / "partial.csv",
        "student_id,display_name,class_name\nDEMO-001,Demo Student,5-01\nbad id,=FORMULA,99-01\n",
    )
    partial = import_roster_file(db, path, allow_partial=True)
    assert partial.students_created == 1
    assert partial.errors

    with pytest.raises(ValueError, match="invalid rows"):
        import_roster_file(db, path, allow_partial=False)


@pytest.mark.parametrize(
    "body,message",
    [
        ("display_name,class_name\nDemo Student,5-01\n", "Missing required"),
        (
            "student_id,display_name,class_name\nDEMO-001,Demo Student,5-01,extra\n",
            "more values",
        ),
    ],
)
def test_roster_shape_validation(tmp_path, body, message):
    path = write_roster(tmp_path / "bad.csv", body)
    if message == "Missing required":
        with pytest.raises(ValueError, match=message):
            parse_csv_roster(path)
    else:
        _, summary = parse_csv_roster(path)
        assert message in summary.errors[0]
