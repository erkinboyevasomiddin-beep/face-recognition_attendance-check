from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from recognition import config
from recognition.app.video_source import ResolvedVideoSource, open_video_capture

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamFrame:
    frame: np.ndarray
    timestamp: float
    frame_id: int


class ThreadedVideoStream:
    """Continuously read camera frames in a background thread."""

    def __init__(
        self,
        source: str | int | Path,
        frame_width: int,
        frame_height: int,
        source_type: str = "auto",
        threaded: bool = True,
        auto_reconnect: bool = config.CAMERA_AUTO_RECONNECT,
        reconnect_interval_seconds: float = config.CAMERA_RECONNECT_INTERVAL_SECONDS,
        max_read_failures_before_reconnect: int = config.CAMERA_MAX_READ_FAILURES_BEFORE_RECONNECT,
        read_failure_retry_delay_seconds: float = config.CAMERA_READ_FAILURE_RETRY_DELAY_SECONDS,
        quiet_mode: bool = config.QUIET_MODE,
    ) -> None:
        self.source = source
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.source_type = source_type
        self.threaded = threaded
        self.auto_reconnect = bool(auto_reconnect)
        self.reconnect_interval_seconds = max(0.1, float(reconnect_interval_seconds))
        self.max_read_failures_before_reconnect = max(1, int(max_read_failures_before_reconnect))
        self.read_failure_retry_delay_seconds = max(0.01, float(read_failure_retry_delay_seconds))
        self.quiet_mode = bool(quiet_mode)

        self._capture: cv2.VideoCapture | None = None
        self._resolved_source: ResolvedVideoSource | None = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._ended = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_timestamp = 0.0
        self._latest_frame_id = 0
        self._frame_interval_seconds = 0.0
        self._consecutive_read_failures = 0
        self._last_failure_log_at = 0.0
        self._last_reconnect_attempt_at = 0.0

    @property
    def resolved_source(self) -> ResolvedVideoSource | None:
        return self._resolved_source

    @property
    def ended(self) -> bool:
        return self._ended.is_set()

    def start(self) -> bool:
        """Open the camera and start the capture thread."""
        if not self._open_capture():
            return False

        self._ended.clear()
        with self._lock:
            self._latest_frame = None
            self._latest_timestamp = 0.0
            self._latest_frame_id = 0

        self._running.set()
        if self.threaded:
            self._thread = threading.Thread(
                target=self._read_loop,
                name="ThreadedVideoStream",
                daemon=True,
            )
            self._thread.start()
        return True

    def _read_loop(self) -> None:
        while self._running.is_set():
            if self._capture is None:
                if self._should_attempt_reconnect() and self._attempt_reconnect():
                    continue
                time.sleep(self.read_failure_retry_delay_seconds)
                continue

            success, frame = self._capture.read()
            if not success:
                if self._handle_failed_read():
                    continue
                continue

            self._consecutive_read_failures = 0
            with self._lock:
                self._latest_frame = frame
                self._latest_timestamp = time.perf_counter()
                self._latest_frame_id += 1

            if self._frame_interval_seconds > 0:
                time.sleep(self._frame_interval_seconds)

    def read(self) -> StreamFrame | None:
        """Return a copy of the latest camera frame, if one is available."""
        if not self.threaded:
            return self._read_direct()

        with self._lock:
            if self._latest_frame is None:
                return None

            return StreamFrame(
                frame=self._latest_frame.copy(),
                timestamp=self._latest_timestamp,
                frame_id=self._latest_frame_id,
            )

    def _read_direct(self) -> StreamFrame | None:
        if not self._running.is_set():
            return None

        if self._capture is None:
            if self._should_attempt_reconnect():
                self._attempt_reconnect()
            return None

        success, frame = self._capture.read()
        if not success:
            self._handle_failed_read()
            return None

        self._consecutive_read_failures = 0
        with self._lock:
            self._latest_frame = frame
            self._latest_timestamp = time.perf_counter()
            self._latest_frame_id += 1
            return StreamFrame(
                frame=frame.copy(),
                timestamp=self._latest_timestamp,
                frame_id=self._latest_frame_id,
            )

    def stop(self) -> None:
        """Stop capture and release the camera."""
        self._running.clear()

        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._capture is not None:
            self._capture.release()
            self._capture = None

        self._resolved_source = None
        with self._lock:
            self._latest_frame = None
            self._latest_timestamp = 0.0
            self._latest_frame_id = 0

    def _open_capture(self, *, clear_state_on_failure: bool = True) -> bool:
        try:
            capture, resolved_source = open_video_capture(
                source=self.source,
                source_type=self.source_type,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            if clear_state_on_failure:
                LOGGER.error("%s", exc)
                self._capture = None
                self._resolved_source = None
            else:
                self._log_warning(
                    "Video source recovery attempt failed: %s",
                    exc,
                    interval_seconds=self.reconnect_interval_seconds,
                )
            return False

        self._capture = capture
        self._resolved_source = resolved_source
        self._configure_capture()
        self._consecutive_read_failures = 0
        return True

    def _configure_capture(self) -> None:
        if self._capture is None or self._resolved_source is None:
            return

        if self._resolved_source.source_type == "camera":
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        if self._resolved_source.is_live:
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        source_fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if self._resolved_source.source_type == "file":
            self._frame_interval_seconds = 1.0 / source_fps if source_fps > 1e-6 else 1.0 / 25.0
            LOGGER.info(
                "Video file playback pacing enabled at %.2f FPS for %s",
                1.0 / self._frame_interval_seconds,
                self._resolved_source.capture_value,
            )
        else:
            self._frame_interval_seconds = 0.0

    def _handle_failed_read(self) -> bool:
        if self._resolved_source is not None and self._resolved_source.source_type == "file":
            LOGGER.info("Reached end of video file: %s", self._resolved_source.capture_value)
            self._ended.set()
            self._running.clear()
            return False

        self._consecutive_read_failures += 1
        self._log_read_failure()

        if self._should_attempt_reconnect() and self._attempt_reconnect():
            return True

        time.sleep(self.read_failure_retry_delay_seconds)
        return False

    def _should_attempt_reconnect(self) -> bool:
        if not self.auto_reconnect:
            return False
        if self._resolved_source is None or not self._resolved_source.is_live:
            return False
        return self._consecutive_read_failures >= self.max_read_failures_before_reconnect

    def _attempt_reconnect(self) -> bool:
        now = time.perf_counter()
        if now - self._last_reconnect_attempt_at < self.reconnect_interval_seconds:
            return False
        self._last_reconnect_attempt_at = now

        source_description = (
            self._resolved_source.description
            if self._resolved_source is not None
            else str(self.source)
        )
        self._log_warning(
            "Attempting to recover video source: %s",
            source_description,
            interval_seconds=self.reconnect_interval_seconds,
        )

        if self._capture is not None:
            self._capture.release()
            self._capture = None

        if not self._open_capture(clear_state_on_failure=False):
            time.sleep(self.read_failure_retry_delay_seconds)
            return False

        if self.quiet_mode:
            LOGGER.debug("Video source recovered: %s", source_description)
        else:
            LOGGER.info("Video source recovered: %s", source_description)
        return True

    def _log_read_failure(self) -> None:
        source_description = (
            self._resolved_source.description
            if self._resolved_source is not None
            else str(self.source)
        )
        self._log_warning(
            "Failed to read a frame from %s (failures=%d).",
            source_description,
            self._consecutive_read_failures,
            interval_seconds=5.0,
        )

    def _log_warning(
        self,
        message: str,
        *args,
        interval_seconds: float,
    ) -> None:
        if not self.quiet_mode:
            LOGGER.warning(message, *args)
            return

        now = time.perf_counter()
        if now - self._last_failure_log_at >= interval_seconds:
            self._last_failure_log_at = now
            LOGGER.warning(message, *args)
        else:
            LOGGER.debug(message, *args)
