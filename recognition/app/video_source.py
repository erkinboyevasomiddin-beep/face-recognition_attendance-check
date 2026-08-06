from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import cv2

from recognition.app.utils import open_webcam

LOGGER = logging.getLogger(__name__)

SUPPORTED_SOURCE_TYPES = {"auto", "camera", "url", "file"}
URL_SCHEMES = {"rtsp", "http", "https"}


@dataclass(frozen=True)
class ResolvedVideoSource:
    raw_value: str | int
    capture_value: str | int
    source_type: str
    description: str
    event_source: str
    is_live: bool


def _looks_like_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme.lower() in URL_SCHEMES


def _looks_like_integer(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    if stripped[0] in "+-":
        stripped = stripped[1:]
    return stripped.isdigit()


def _resolve_camera_source(value: str | int) -> ResolvedVideoSource:
    try:
        camera_index = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid camera index: {value}") from exc

    return ResolvedVideoSource(
        raw_value=value,
        capture_value=camera_index,
        source_type="camera",
        description=f"camera index {camera_index}",
        event_source=f"camera_{camera_index}",
        is_live=True,
    )


def _resolve_url_source(value: str) -> ResolvedVideoSource:
    if not value.strip():
        raise ValueError("Video source URL cannot be empty.")

    parsed = urlparse(value)
    host = parsed.hostname or "stream"
    port = f"_{parsed.port}" if parsed.port is not None else ""
    event_source = _sanitize_source_name(f"{parsed.scheme}_{host}{port}")
    display_port = f":{parsed.port}" if parsed.port is not None else ""

    return ResolvedVideoSource(
        raw_value=value,
        capture_value=value,
        source_type="url",
        # Never place URL credentials, query strings, or private stream paths in logs/UI.
        description=f"{parsed.scheme.lower()} stream at {host}{display_port}",
        event_source=event_source,
        is_live=True,
    )


def _resolve_file_source(value: str | Path) -> ResolvedVideoSource:
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return ResolvedVideoSource(
        raw_value=str(value),
        capture_value=str(path),
        source_type="file",
        description=f"video file {path.name}",
        event_source=_sanitize_source_name(path.stem) or "video_file",
        is_live=False,
    )


def _sanitize_source_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    return cleaned.strip("_").lower()


def resolve_video_source(
    source: str | int | Path,
    source_type: str = "auto",
) -> ResolvedVideoSource:
    """Resolve a configured source into camera, URL, or local file."""
    normalized_type = source_type.lower().strip()
    if normalized_type not in SUPPORTED_SOURCE_TYPES:
        raise ValueError(
            f"Unsupported VIDEO_SOURCE_TYPE '{source_type}'. "
            f"Expected one of: {sorted(SUPPORTED_SOURCE_TYPES)}"
        )

    if normalized_type == "camera":
        return _resolve_camera_source(str(source) if isinstance(source, Path) else source)

    if isinstance(source, Path):
        return _resolve_file_source(source)

    if isinstance(source, int):
        return _resolve_camera_source(source)

    value = str(source).strip()
    if not value:
        raise ValueError("Video source cannot be empty.")

    if normalized_type == "url":
        return _resolve_url_source(value)

    if normalized_type == "file":
        return _resolve_file_source(value)

    if _looks_like_url(value):
        return _resolve_url_source(value)

    if _looks_like_integer(value):
        return _resolve_camera_source(value)

    return _resolve_file_source(value)


def open_video_capture(
    source: str | int | Path,
    source_type: str = "auto",
) -> tuple[cv2.VideoCapture, ResolvedVideoSource]:
    """Open an OpenCV capture for a resolved camera, stream URL, or file."""
    resolved = resolve_video_source(source=source, source_type=source_type)
    LOGGER.info(
        "Opening video source | type=%s | value=%s",
        resolved.source_type,
        resolved.description,
    )

    if resolved.source_type == "camera":
        capture = open_webcam(int(resolved.capture_value))
    else:
        capture = cv2.VideoCapture(str(resolved.capture_value))

    if capture.isOpened():
        LOGGER.info("Video source opened successfully: %s", resolved.description)
        return capture, resolved

    capture.release()
    if resolved.source_type == "camera":
        raise RuntimeError(f"Failed to open video source: {resolved.description}")
    if resolved.source_type == "url":
        raise RuntimeError(f"Failed to open RTSP/HTTP stream: {resolved.description}")
    raise RuntimeError(f"Failed to open video file: {resolved.description}")
