from __future__ import annotations

import hashlib

from backend.app.auth import authenticate_user, hash_password, verify_password
from backend.app.models import User
from backend.app.schemas import UserRole
from backend.tests.helpers import add_user, login


def test_argon2_password_hash_and_verification():
    encoded = hash_password("correct horse battery staple")
    assert encoded.startswith("$argon2id$")
    assert verify_password("correct horse battery staple", encoded)
    assert not verify_password("incorrect password", encoded)
    assert encoded != hash_password("correct horse battery staple")


def test_short_password_is_rejected():
    try:
        hash_password("too-short")
    except ValueError as exc:
        assert "12" in str(exc)
    else:
        raise AssertionError("short password unexpectedly accepted")


def test_development_legacy_hash_is_upgraded(db):
    legacy = hashlib.sha256(b"legacy development password").hexdigest()
    user = User(
        username="legacy.user",
        password_hash=legacy,
        role=UserRole.DIRECTOR,
        display_name="Synthetic Legacy User",
    )
    db.add(user)
    db.commit()

    authenticated = authenticate_user(db, "legacy.user", "legacy development password")
    assert authenticated is not None
    assert authenticated.password_hash.startswith("$argon2id$")


def test_login_success_failure_and_logout_csrf(client, db):
    add_user(db)
    failed = login(client, "director.test", "wrong password")
    assert failed.status_code == 400
    assert "Login ma'lumotlari" in failed.text or "Login ma&#39;lumotlari" in failed.text

    response = login(client, "director.test", "correct horse battery staple")
    assert response.status_code == 303
    assert response.headers["location"] == "/director"

    forbidden_logout = client.post("/logout", data={"csrf_token": "wrong"})
    assert forbidden_logout.status_code == 403
