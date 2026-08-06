from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from backend.app import auth, security
from backend.app.auth import authenticate_user, verify_password
from backend.app.models import User
from backend.app.schemas import UserRole
from backend.app.security import LoginRateLimiter
from backend.app.settings import AttendanceSettings
from backend.tests.helpers import add_user, login


def production_settings(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "app_env": "production",
        "debug": False,
        "database_url": "sqlite:////var/lib/attendance/attendance.db",
        "allowed_hosts": ["attendance.example.invalid"],
        "public_base_url": "https://attendance.example.invalid",
        "session_secret": "a-secure-session-secret-with-more-than-thirty-two-characters",
        "session_cookie_secure": True,
        "api_ingest_enabled": True,
        "api_ingest_key": "a-secure-ingest-secret-with-more-than-thirty-two-characters",
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"debug": True}, "DEBUG"),
        ({"session_secret": None}, "SESSION_SECRET"),
        ({"session_secret": "short"}, "SESSION_SECRET"),
        ({"session_cookie_secure": False}, "COOKIE_SECURE"),
        ({"allowed_hosts": ["*.example.invalid"]}, "Wildcard"),
        ({"public_base_url": None}, "PUBLIC_BASE_URL"),
        ({"public_base_url": "http://attendance.example.invalid"}, "HTTPS"),
        ({"database_url": "sqlite:///./relative.db"}, "absolute"),
        ({"database_url": "sqlite:///:memory:"}, "persistent"),
        ({"api_ingest_key": "replace-with-production-key-xxxxxxxx"}, "API_INGEST_KEY"),
    ],
)
def test_production_security_rejects_unsafe_values(override, message):
    with pytest.raises(ValidationError, match=message):
        AttendanceSettings(**production_settings(**override))


def test_production_security_accepts_explicit_https_configuration():
    settings = AttendanceSettings(**production_settings())
    assert settings.app_env == "production"
    assert settings.public_base_url == "https://attendance.example.invalid"


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"timezone": "Invalid/Timezone"}, "Unknown IANA"),
        ({"allowed_hosts": []}, "at least one"),
        ({"database_url": "mysql://localhost/example"}, "SQLite or PostgreSQL"),
        ({"public_base_url": "not-a-url"}, "absolute HTTP"),
        ({"public_base_url": "https://user:pass@example.invalid"}, "credentials"),
        ({"recognition_min_similarity": 1.1}, "less than or equal to 1"),
    ],
)
def test_general_setting_validation(values, message):
    with pytest.raises(ValidationError, match=message):
        AttendanceSettings(**values)


def test_unknown_user_and_invalid_hash_have_safe_failures(db):
    assert authenticate_user(db, "missing.user", "some incorrect password") is None
    assert verify_password("some incorrect password", "not-an-argon2-hash") is False


def test_weak_legacy_password_is_not_migrated(db):
    import hashlib

    user = User(
        username="legacy.weak",
        display_name="Synthetic Legacy User",
        role=UserRole.DIRECTOR,
        password_hash=hashlib.sha256(b"weakpass").hexdigest(),
    )
    db.add(user)
    db.commit()
    assert authenticate_user(db, "legacy.weak", "weakpass") is None


def test_production_never_accepts_legacy_hash(db, monkeypatch):
    import hashlib

    user = User(
        username="legacy.production",
        display_name="Synthetic Legacy User",
        role=UserRole.DIRECTOR,
        password_hash=hashlib.sha256(b"legacy production password").hexdigest(),
    )
    db.add(user)
    db.commit()
    monkeypatch.setattr(auth, "get_settings", lambda: SimpleNamespace(app_env="production"))
    assert authenticate_user(db, "legacy.production", "legacy production password") is None


def test_login_throttling_and_expiry(monkeypatch):
    limiter = LoginRateLimiter()
    monkeypatch.setattr(
        security,
        "get_settings",
        lambda: SimpleNamespace(login_window_seconds=10, login_max_attempts=2),
    )
    limiter.record_failure("client:user", now=1)
    limiter.record_failure("client:user", now=2)
    with pytest.raises(Exception) as captured:
        limiter.check("client:user", now=3)
    assert getattr(captured.value, "status_code", None) == 429
    limiter.check("client:user", now=20)
    limiter.clear("client:user")


def test_session_rotation_logout_and_protected_redirect(client, db):
    add_user(db)
    client.get("/login")
    before = client.cookies.get("school_attendance_session")
    assert before

    response = login(client, "director.test", "correct horse battery staple")
    after = client.cookies.get("school_attendance_session")
    assert response.status_code == 303
    assert after and after != before
    set_cookie = response.headers["set-cookie"].lower()
    assert "httponly" in set_cookie
    assert "samesite=lax" in set_cookie

    page = client.get("/director")
    token_marker = 'name="csrf_token" value="'
    start = page.text.index(token_marker) + len(token_marker)
    token = page.text[start : page.text.index('"', start)]
    logout = client.post("/logout", data={"csrf_token": token})
    assert logout.status_code == 303
    assert client.get("/director").headers["location"] == "/login"


def test_health_endpoint_and_security_headers(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_deleted_user_session_cannot_authorize_reused_database_id(client, db):
    original = add_user(db)
    assert login(client, original.username, "correct horse battery staple").status_code == 303
    original_id = original.id
    db.delete(original)
    db.commit()
    replacement = add_user(db, username="replacement.director")
    assert replacement.id == original_id
    response = client.get("/director")
    assert response.status_code == 303
    assert response.headers["location"] == "/login"
