from __future__ import annotations

import hashlib
import re
import secrets
from collections.abc import Callable
from datetime import UTC

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import User
from backend.app.schemas import UserRole
from backend.app.settings import get_settings

SESSION_USER_ID = "user_id"
SESSION_USER_CREATED_AT = "user_created_at"
LEGACY_SHA256_PATTERN = re.compile(r"^[a-fA-F0-9]{64}$")
PASSWORD_HASHER = PasswordHasher()
DUMMY_PASSWORD_HASH = PASSWORD_HASHER.hash(secrets.token_urlsafe(32))


class RedirectException(Exception):
    def __init__(self, destination: str):
        self.destination = destination


def hash_password(password: str) -> str:
    settings = get_settings()
    if not settings.password_min_length <= len(password) <= settings.password_max_length:
        raise ValueError(
            f"Passwords must contain {settings.password_min_length} to "
            f"{settings.password_max_length} characters"
        )
    if password.casefold() in {"admin", "password", "123456", "change-me", "changeme"}:
        raise ValueError("Password is a known unsafe default")
    return PASSWORD_HASHER.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return PASSWORD_HASHER.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def _verify_legacy_sha256(password: str, password_hash: str) -> bool:
    if not LEGACY_SHA256_PATTERN.fullmatch(password_hash):
        return False
    candidate = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return secrets.compare_digest(candidate, password_hash.lower())


def get_dashboard_path(role: UserRole) -> str:
    return "/director" if role == UserRole.DIRECTOR else "/teacher"


def authenticate_user(db: Session, username: str, password: str) -> User | None:
    user = db.scalar(select(User).where(User.username == username.strip()))
    if user is None:
        # Keep nonexistent-user failures closer in cost to incorrect-password
        # failures so the login endpoint does not become a cheap username oracle.
        verify_password(password, DUMMY_PASSWORD_HASH)
        return None
    if verify_password(password, user.password_hash):
        if PASSWORD_HASHER.check_needs_rehash(user.password_hash):
            user.password_hash = hash_password(password)
            db.commit()
        return user

    # Development-only migration path for old SHA-256 demo accounts. Production
    # operators must recreate accounts with the documented user-management script.
    if get_settings().app_env != "production" and _verify_legacy_sha256(
        password, user.password_hash
    ):
        try:
            user.password_hash = hash_password(password)
        except ValueError:
            return None
        db.commit()
        return user
    return None


def login_user(request: Request, user: User) -> None:
    request.session.clear()
    request.session[SESSION_USER_ID] = user.id
    request.session[SESSION_USER_CREATED_AT] = user.created_at.astimezone(UTC).isoformat()


def logout_user(request: Request) -> None:
    request.session.clear()


def get_current_user(request: Request, db: Session) -> User | None:
    user_id = request.session.get(SESSION_USER_ID)
    created_at = request.session.get(SESSION_USER_CREATED_AT)
    if not isinstance(user_id, int) or not isinstance(created_at, str):
        return None
    user = db.scalar(select(User).where(User.id == user_id))
    if user is None:
        return None
    expected = user.created_at.astimezone(UTC).isoformat()
    return user if secrets.compare_digest(created_at, expected) else None


def optional_user(request: Request, db: Session = Depends(get_db)) -> User | None:
    return get_current_user(request, db)


def require_role(role: UserRole) -> Callable[..., User]:
    def dependency(request: Request, db: Session = Depends(get_db)) -> User:
        user = get_current_user(request, db)
        if not user:
            raise RedirectException("/login")
        if user.role != role:
            raise RedirectException(get_dashboard_path(user.role))
        return user

    return dependency


require_director = require_role(UserRole.DIRECTOR)
require_teacher = require_role(UserRole.TEACHER)


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    settings = get_settings()
    if not settings.api_ingest_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    expected = settings.api_ingest_key
    if (
        expected is None
        or not x_api_key
        or not secrets.compare_digest(x_api_key, expected.get_secret_value())
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
