from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

RECOGNITION_DIR = Path(__file__).resolve().parents[1]
SECRET_PLACEHOLDER_MARKERS = ("replace-with", "change-me", "example", "placeholder")


class RecognitionSettings(BaseSettings):
    """Validated recognition-client settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="RECOGNITION_",
        env_file=(RECOGNITION_DIR.parent / ".env", RECOGNITION_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["development", "test", "production"] = "development"
    data_dir: Path = RECOGNITION_DIR / "data"
    models_dir: Path = RECOGNITION_DIR / "models"
    model_name: str = "buffalo_l"
    preferred_providers: list[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    match_threshold: float = Field(default=0.45, ge=-1.0, le=1.0)
    display_threshold: float = Field(default=0.60, ge=-1.0, le=1.0)
    confirm_threshold: float = Field(default=0.68, ge=-1.0, le=1.0)
    api_min_similarity: float = Field(default=0.60, ge=-1.0, le=1.0)
    detection_size: tuple[int, int] = (640, 640)
    webcam_detection_size: tuple[int, int] = (320, 320)
    min_face_size: int = Field(default=40, ge=0, le=4096)

    camera_id: int = Field(default=0, ge=0)
    camera_mode: Literal["ENTRY", "EXIT"] = "ENTRY"
    video_source: str | int = 0
    video_source_type: Literal["auto", "camera", "url", "file"] = "auto"
    video_source_name: str = ""
    webcam_frame_width: int = Field(default=960, ge=160, le=7680)
    webcam_frame_height: int = Field(default=540, ge=120, le=4320)
    webcam_process_every_n_frames: int = Field(default=4, ge=1, le=120)
    webcam_recognition_scale: float = Field(default=0.5, gt=0, le=1.0)
    webcam_result_ttl_seconds: float = Field(default=0.5, ge=0, le=30)
    webcam_max_queue_size: int = Field(default=16, ge=1, le=1000)
    webcam_enable_threaded_capture: bool = True
    webcam_start_fullscreen: bool = False
    webcam_overlay_mode: Literal["minimal", "full"] = "minimal"
    max_active_faces: int = Field(default=5, ge=1, le=100)
    track_match_distance_threshold: float = Field(default=100, gt=0)
    track_max_missing_frames: int = Field(default=10, ge=0)
    track_max_age_seconds: float = Field(default=2.0, gt=0)
    camera_auto_reconnect: bool = True
    camera_reconnect_interval_seconds: float = Field(default=2.0, gt=0)
    camera_max_read_failures_before_reconnect: int = Field(default=5, ge=1)
    camera_read_failure_retry_delay_seconds: float = Field(default=0.1, gt=0)

    temporal_smoothing: bool = True
    smoothing_window: int = Field(default=5, ge=1, le=100)
    required_consecutive_hits: int = Field(default=3, ge=1, le=100)
    max_misses_before_reset: int = Field(default=2, ge=0, le=100)
    stability_time_window: float = Field(default=2.0, gt=0)
    recognition_position_tolerance: float = Field(default=80, gt=0)

    api_enabled: bool = False
    api_url: str = "http://127.0.0.1:8000/api/recognition-event"
    api_timeout_seconds: float = Field(default=5.0, gt=0, le=120)
    api_cooldown_seconds: float = Field(default=30.0, ge=0)
    api_key: SecretStr | None = None
    api_max_retries: int = Field(default=2, ge=0, le=10)

    snapshot_enabled: bool = False
    snapshot_dir: Path | None = None
    snapshot_jpeg_quality: int = Field(default=90, ge=1, le=100)
    snapshot_face_padding: float = Field(default=0.25, ge=0, le=3)
    snapshot_cooldown_seconds: float = Field(default=60, ge=0)

    frame_resize_scale: float = Field(default=1.0, gt=0, le=4)
    frame_skip: int = Field(default=0, ge=0, le=120)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    quiet_mode: bool = True

    @field_validator("api_url")
    @classmethod
    def validate_api_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("api_url must use http:// or https://")
        parsed = urlsplit(value)
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("api_url must not contain embedded credentials")
        if not parsed.hostname:
            raise ValueError("api_url must include a host")
        return value.rstrip("/")

    @field_validator("detection_size", "webcam_detection_size")
    @classmethod
    def validate_detection_size(cls, value: tuple[int, int]) -> tuple[int, int]:
        if any(dimension <= 0 or dimension > 4096 for dimension in value):
            raise ValueError("detection dimensions must be greater than zero and at most 4096")
        return value

    @model_validator(mode="after")
    def validate_security_and_thresholds(self) -> RecognitionSettings:
        if self.display_threshold < self.match_threshold:
            raise ValueError("display_threshold must be at least match_threshold")
        if self.confirm_threshold < self.display_threshold:
            raise ValueError("confirm_threshold must be at least display_threshold")
        if self.app_env == "production" and self.api_enabled:
            secret = self.api_key.get_secret_value() if self.api_key else ""
            if len(secret) < 32 or any(
                marker in secret.casefold() for marker in SECRET_PLACEHOLDER_MARKERS
            ):
                raise ValueError(
                    "RECOGNITION_API_KEY must contain at least 32 characters when the API "
                    "is enabled in production"
                )
            if not self.api_url.startswith("https://"):
                raise ValueError("RECOGNITION_API_URL must use HTTPS in production")
        if self.snapshot_dir is None:
            self.snapshot_dir = self.data_dir / "snapshots"
        return self

    @property
    def known_faces_dir(self) -> Path:
        return self.data_dir / "known_faces"

    @property
    def embeddings_dir(self) -> Path:
        return self.data_dir / "embeddings"

    @property
    def database_path(self) -> Path:
        return self.embeddings_dir / "face_embeddings.db"


@lru_cache
def get_settings() -> RecognitionSettings:
    return RecognitionSettings()
