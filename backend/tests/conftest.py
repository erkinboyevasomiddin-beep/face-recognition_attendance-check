from __future__ import annotations

import os

import pytest

os.environ.update(
    {
        "ATTENDANCE_APP_ENV": "test",
        "ATTENDANCE_DATABASE_URL": "sqlite:///:memory:",
        "ATTENDANCE_SESSION_SECRET": "test-only-session-secret-with-more-than-32-characters",
        "ATTENDANCE_API_INGEST_ENABLED": "true",
        "ATTENDANCE_API_INGEST_KEY": "test-only-ingest-key-with-more-than-32-characters",
        "ATTENDANCE_RECOGNITION_COOLDOWN_SECONDS": "30",
        "ATTENDANCE_AUTO_SEED_DEMO": "false",
    }
)

from fastapi.testclient import TestClient

from backend.app.database import Base, SessionLocal, engine
from backend.app.security import login_rate_limiter
from backend.main import app


@pytest.fixture(autouse=True)
def clean_database():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    login_rate_limiter._attempts.clear()
    yield


@pytest.fixture
def db():
    with SessionLocal() as session:
        yield session


@pytest.fixture
def client():
    with TestClient(app, follow_redirects=False) as test_client:
        yield test_client
