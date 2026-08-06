from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.models import Student, User
from backend.app.settings import AttendanceSettings
from backend.seed_data import seed_synthetic_data_if_requested


def test_synthetic_initialization_is_idempotent_and_creates_no_account(db):
    seed_synthetic_data_if_requested(db)
    first_count = db.query(Student).count()
    seed_synthetic_data_if_requested(db)
    assert first_count > 0
    assert db.query(Student).count() == first_count
    assert db.query(User).count() == 0


def test_production_rejects_placeholder_secret_and_insecure_cookie():
    with pytest.raises(ValidationError):
        AttendanceSettings(
            app_env="production",
            session_secret="replace-with-a-placeholder-value-longer-than-32-characters",
            session_cookie_secure=False,
            allowed_hosts=["attendance.example.invalid"],
        )
