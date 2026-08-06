from __future__ import annotations

import secrets
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status

from backend.app.settings import get_settings

CSRF_SESSION_KEY = "csrf_token"


def get_csrf_token(request: Request) -> str:
    token = request.session.get(CSRF_SESSION_KEY)
    if not isinstance(token, str) or len(token) < 32:
        token = secrets.token_urlsafe(get_settings().csrf_token_bytes)
        request.session[CSRF_SESSION_KEY] = token
    return token


def validate_csrf_token(request: Request, submitted: str | None) -> None:
    expected = request.session.get(CSRF_SESSION_KEY)
    if (
        not isinstance(expected, str)
        or not submitted
        or not secrets.compare_digest(expected, submitted)
    ):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid CSRF token")


class LoginRateLimiter:
    """Small in-process limiter suitable for a single-process prototype."""

    def __init__(self) -> None:
        self._attempts: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> None:
        settings = get_settings()
        moment = time.monotonic() if now is None else now
        cutoff = moment - settings.login_window_seconds
        with self._lock:
            attempts = self._attempts[key]
            while attempts and attempts[0] < cutoff:
                attempts.popleft()
            if len(attempts) >= settings.login_max_attempts:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many login attempts. Try again later.",
                    headers={"Retry-After": str(settings.login_window_seconds)},
                )

    def record_failure(self, key: str, now: float | None = None) -> None:
        with self._lock:
            self._attempts[key].append(time.monotonic() if now is None else now)

    def clear(self, key: str) -> None:
        with self._lock:
            self._attempts.pop(key, None)


login_rate_limiter = LoginRateLimiter()


def login_rate_key(request: Request, username: str) -> str:
    client = request.client.host if request.client else "unknown"
    return f"{client}:{username.strip().casefold()}"
