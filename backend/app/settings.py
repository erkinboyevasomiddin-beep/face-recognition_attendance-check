from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
SECRET_PLACEHOLDER_MARKERS = ("replace-with", "change-me", "example", "placeholder")


def _is_placeholder(value: SecretStr | None) -> bool:
    if value is None:
        return True
    lowered = value.get_secret_value().casefold()
    return any(marker in lowered for marker in SECRET_PLACEHOLDER_MARKERS)


class AttendanceSettings(BaseSettings):
    """Validated settings for the attendance web application."""

    model_config = SettingsConfigDict(
        env_prefix="ATTENDANCE_",
        env_file=(BACKEND_DIR.parent / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["development", "test", "production"] = "development"
    debug: bool = False
    database_url: str = f"sqlite:///{(BACKEND_DIR / 'data' / 'attendance.db').as_posix()}"
    timezone: str = "Asia/Tashkent"
    allowed_hosts: list[str] = ["localhost", "127.0.0.1", "testserver"]
    public_base_url: str | None = None

    session_secret: SecretStr | None = None
    session_cookie_name: str = "school_attendance_session"
    session_cookie_secure: bool = False
    session_cookie_same_site: Literal["lax", "strict"] = "lax"
    session_max_age_seconds: int = Field(default=28_800, ge=300, le=2_592_000)
    csrf_token_bytes: int = Field(default=32, ge=24, le=64)
    password_min_length: int = Field(default=12, ge=12, le=128)
    password_max_length: int = Field(default=256, ge=64, le=1024)

    api_ingest_enabled: bool = False
    api_ingest_key: SecretStr | None = None
    recognition_min_similarity: float = Field(default=0.60, ge=-1.0, le=1.0)
    recognition_cooldown_seconds: int = Field(default=30, ge=0, le=86_400)
    recognition_max_event_age_seconds: int = Field(default=300, ge=30, le=86_400)
    allow_legacy_name_fallback: bool = False

    attendance_on_time_hour: int = Field(default=8, ge=0, le=23)
    attendance_on_time_minute: int = Field(default=30, ge=0, le=59)
    roster_max_bytes: int = Field(default=1_000_000, ge=1_024, le=10_000_000)
    roster_max_rows: int = Field(default=10_000, ge=1, le=100_000)

    login_window_seconds: int = Field(default=900, ge=60, le=86_400)
    login_max_attempts: int = Field(default=5, ge=1, le=100)
    auto_seed_demo: bool = False

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown IANA timezone: {value}") from exc
        return value

    @field_validator("allowed_hosts")
    @classmethod
    def validate_allowed_hosts(cls, value: list[str]) -> list[str]:
        cleaned = [host.strip() for host in value if host.strip()]
        if not cleaned:
            raise ValueError("allowed_hosts must contain at least one host")
        return cleaned

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        if not value.startswith(("sqlite://", "postgresql://", "postgresql+")):
            raise ValueError("database_url must use SQLite or PostgreSQL")
        return value

    @field_validator("public_base_url")
    @classmethod
    def validate_public_base_url(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("public_base_url must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("public_base_url must not contain credentials")
        return normalized

    @model_validator(mode="after")
    def validate_production_security(self) -> AttendanceSettings:
        if self.app_env != "production":
            return self
        if self.debug:
            raise ValueError("ATTENDANCE_DEBUG cannot be enabled in production")
        if (
            self.session_secret is None
            or _is_placeholder(self.session_secret)
            or len(self.session_secret.get_secret_value()) < 32
        ):
            raise ValueError(
                "ATTENDANCE_SESSION_SECRET must contain at least 32 characters in production"
            )
        if not self.session_cookie_secure:
            raise ValueError("ATTENDANCE_SESSION_COOKIE_SECURE must be true in production")
        if any("*" in host for host in self.allowed_hosts):
            raise ValueError("Wildcard allowed hosts are forbidden in production")
        if self.public_base_url is None or not self.public_base_url.startswith("https://"):
            raise ValueError("ATTENDANCE_PUBLIC_BASE_URL must use HTTPS in production")
        if self.database_url.startswith("sqlite:///"):
            parsed_path = self.database_url.removeprefix("sqlite:///")
            if parsed_path in {"", ":memory:"}:
                raise ValueError("Production SQLite requires an explicit persistent database path")
            is_absolute = parsed_path.startswith("/") or (
                len(parsed_path) >= 3 and parsed_path[1:3] in {":/", ":\\"}
            )
            if not is_absolute:
                raise ValueError("Production SQLite requires an absolute database path")
        if self.api_ingest_enabled and (
            self.api_ingest_key is None
            or _is_placeholder(self.api_ingest_key)
            or len(self.api_ingest_key.get_secret_value()) < 32
        ):
            raise ValueError(
                "ATTENDANCE_API_INGEST_KEY must contain at least 32 characters when API "
                "ingestion is enabled in production"
            )
        return self

    @property
    def local_zone(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


@lru_cache
def get_settings() -> AttendanceSettings:
    return AttendanceSettings()
