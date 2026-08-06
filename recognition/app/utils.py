from __future__ import annotations

import logging
import platform
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from recognition import config

LOGGER = logging.getLogger(__name__)


def setup_logging(level: str = config.LOG_LEVEL) -> None:
    """Configure application logging."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()

    if not root_logger.handlers:
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
        return

    root_logger.setLevel(numeric_level)
    for handler in root_logger.handlers:
        handler.setLevel(numeric_level)


def ensure_runtime_directories() -> None:
    """Ensure the expected local project folders exist."""
    config.ensure_project_directories()


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp without microseconds."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in config.VIDEO_EXTENSIONS


def iter_image_paths(directory: Path) -> list[Path]:
    """Return all supported image files under a directory."""
    if not directory.exists():
        return []

    return sorted(path for path in directory.rglob("*") if is_image_file(path))


def read_image(path: Path) -> np.ndarray | None:
    """
    Read an image from disk.

    np.fromfile + cv2.imdecode is used for better Windows path compatibility.
    """
    try:
        buffer = np.fromfile(path, dtype=np.uint8)
    except OSError:
        LOGGER.exception("Failed to read image bytes from %s", path)
        return None

    if buffer.size == 0:
        LOGGER.warning("Image file is empty: %s", path)
        return None

    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        LOGGER.warning("OpenCV could not decode image: %s", path)
    return image


def write_image(path: Path, image: np.ndarray, jpeg_quality: int | None = None) -> None:
    """Write an image to disk with Windows-safe path handling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".jpg"

    encode_params: list[int] = []
    if jpeg_quality is not None and suffix.lower() in {".jpg", ".jpeg"}:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]

    success, encoded = cv2.imencode(suffix, image, encode_params)
    if not success:
        raise ValueError(f"Failed to encode image for output path: {path}")

    encoded.tofile(path)


def resize_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    """Resize a frame using a uniform scale factor."""
    if scale == 1.0:
        return frame

    if scale <= 0:
        raise ValueError("Resize scale must be greater than zero.")

    new_width = max(1, int(frame.shape[1] * scale))
    new_height = max(1, int(frame.shape[0] * scale))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


def open_webcam(camera_index: int) -> cv2.VideoCapture:
    """Open a webcam with a Windows-friendly fallback path."""
    if platform.system().lower() == "windows":
        capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if capture.isOpened():
            return capture
        capture.release()

    return cv2.VideoCapture(camera_index)


def format_supported_extensions(extensions: Iterable[str]) -> str:
    return ", ".join(sorted(extensions))
