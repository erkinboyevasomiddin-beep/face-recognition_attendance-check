from __future__ import annotations

import logging
import queue
import re
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from recognition import config
from recognition.app.api_client import RecognitionEventClient
from recognition.app.utils import write_image

LOGGER = logging.getLogger(__name__)

_SHUTDOWN = object()


@dataclass(frozen=True)
class ConfirmedEventTask:
    person_id: str
    display_name: str
    class_name: str | None
    similarity: float
    source: str | None
    created_at: datetime
    snapshot_image: np.ndarray | None


class AsyncConfirmedEventDispatcher:
    """Handle confirmed-event side effects away from the webcam loop."""

    def __init__(
        self,
        api_client: RecognitionEventClient | None = None,
        max_queue_size: int = config.WEBCAM_MAX_QUEUE_SIZE,
        async_api_send: bool = config.ASYNC_API_SEND,
        async_snapshot_save: bool = config.ASYNC_SNAPSHOT_SAVE,
        snapshot_enabled: bool = config.SNAPSHOT_ENABLED,
        snapshot_dir: Path | None = config.SNAPSHOT_DIR,
        snapshot_extension: str = config.SNAPSHOT_IMAGE_EXTENSION,
        snapshot_jpeg_quality: int = config.SNAPSHOT_JPEG_QUALITY,
        snapshot_face_padding: float = config.SNAPSHOT_FACE_PADDING,
        snapshot_cooldown_seconds: float = config.SNAPSHOT_COOLDOWN_SECONDS,
    ) -> None:
        self.api_client = api_client or RecognitionEventClient()
        self.max_queue_size = max(1, int(max_queue_size))
        self.async_api_send = bool(async_api_send)
        self.async_snapshot_save = bool(async_snapshot_save)
        self.snapshot_enabled = bool(snapshot_enabled)
        self.snapshot_dir = Path(snapshot_dir or (config.DATA_DIR / "snapshots"))
        self.snapshot_extension = (
            snapshot_extension if snapshot_extension.startswith(".") else f".{snapshot_extension}"
        )
        self.snapshot_jpeg_quality = int(snapshot_jpeg_quality)
        self.snapshot_face_padding = max(0.0, float(snapshot_face_padding))
        self.snapshot_cooldown_seconds = max(0.0, float(snapshot_cooldown_seconds))
        self.quiet_mode = bool(config.QUIET_MODE)

        self._queue: queue.Queue[ConfirmedEventTask | object] = queue.Queue(
            maxsize=self.max_queue_size
        )
        self._lock = threading.Lock()
        self._pending_keys: set[str] = set()
        self._last_snapshot_at: dict[str, float] = {}
        self._last_queue_full_log_at = 0.0
        self._closed = False
        self._use_background_worker = (self.api_client.enabled and self.async_api_send) or (
            self.snapshot_enabled and self.async_snapshot_save
        )
        self._thread: threading.Thread | None = None

        if self._use_background_worker:
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="confirmed-event-worker",
                daemon=True,
            )
            self._thread.start()

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    def submit_confirmed_event(
        self,
        person_id: str | None,
        display_name: str,
        similarity: float,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int] | None = None,
        class_name: str | None = None,
        source: str | None = None,
    ) -> bool:
        if self._closed:
            LOGGER.debug("Skipping confirmed event for '%s': dispatcher is closed.", person_id)
            return False

        if not person_id:
            return False

        if not self.api_client.enabled and not self.snapshot_enabled:
            return False

        snapshot_image = (
            self._extract_snapshot_image(frame, bbox) if self.snapshot_enabled else None
        )
        task = ConfirmedEventTask(
            person_id=person_id,
            display_name=display_name,
            class_name=class_name,
            similarity=float(similarity),
            source=source,
            created_at=datetime.now(UTC),
            snapshot_image=snapshot_image,
        )

        if not self._use_background_worker:
            self._process_task(task)
            return True

        person_key = self._person_key(person_id)
        with self._lock:
            if person_key in self._pending_keys:
                LOGGER.debug(
                    "Skipping confirmed event queue for '%s': a task is already pending.",
                    person_id,
                )
                return False
            self._pending_keys.add(person_key)

        try:
            self._queue.put_nowait(task)
        except queue.Full:
            with self._lock:
                self._pending_keys.discard(person_key)
            self._log_queue_full(person_id)
            return False

        LOGGER.debug(
            "Queued confirmed event | person=%s similarity=%.4f queue_size=%d",
            person_id,
            similarity,
            self._queue.qsize(),
        )
        return True

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        if self._thread is not None:
            self._queue.put(_SHUTDOWN)
            self._thread.join(timeout=5.0)
            self._thread = None
        self.api_client.close()

    def _worker_loop(self) -> None:
        while True:
            task = self._queue.get()
            if task is _SHUTDOWN:
                self._queue.task_done()
                break

            assert isinstance(task, ConfirmedEventTask)
            try:
                self._process_task(task)
            except Exception as exc:  # pragma: no cover - fail-safe runtime guard
                LOGGER.warning(
                    "Confirmed event worker failed for '%s': %s",
                    task.person_id,
                    exc,
                )
            finally:
                with self._lock:
                    self._pending_keys.discard(self._person_key(task.person_id))
                self._queue.task_done()

    def _process_task(self, task: ConfirmedEventTask) -> None:
        if self.snapshot_enabled:
            self._save_snapshot(task)

        if self.api_client.enabled:
            self.api_client.send_recognition_event(
                person_id=task.person_id,
                display_name=task.display_name,
                similarity=task.similarity,
                apply_cooldown=True,
                once_per_run=False,
                class_name=task.class_name,
                source=task.source,
                occurred_at=task.created_at,
            )

    def _save_snapshot(self, task: ConfirmedEventTask) -> str | None:
        if task.snapshot_image is None or task.snapshot_image.size == 0:
            return None

        person_key = self._person_key(task.person_id)
        now = time.perf_counter()
        last_snapshot_at = self._last_snapshot_at.get(person_key)
        if (
            last_snapshot_at is not None
            and (now - last_snapshot_at) < self.snapshot_cooldown_seconds
        ):
            LOGGER.debug(
                "Skipping snapshot for '%s': cooldown active (%.1fs remaining).",
                task.person_id,
                self.snapshot_cooldown_seconds - (now - last_snapshot_at),
            )
            return None

        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"{task.created_at.strftime('%Y%m%d_%H%M%S_%f')}_"
            f"{self._sanitize_name(task.person_id)}"
            f"{self.snapshot_extension}"
        )
        destination = self.snapshot_dir / filename

        try:
            write_image(
                destination,
                task.snapshot_image,
                jpeg_quality=self.snapshot_jpeg_quality,
            )
        except Exception as exc:
            LOGGER.warning(
                "Failed to save snapshot for '%s' to %s: %s",
                task.person_id,
                destination,
                exc,
            )
            return None

        self._last_snapshot_at[person_key] = now
        if self.quiet_mode:
            LOGGER.debug("Snapshot saved | person_id=%s path=%s", task.person_id, destination)
        else:
            LOGGER.info("Snapshot saved | person_id=%s path=%s", task.person_id, destination)
        return str(destination)

    def _log_queue_full(self, person_name: str) -> None:
        if not self.quiet_mode:
            LOGGER.warning("Confirmed event queue is full. Dropping event for '%s'.", person_name)
            return

        now = time.perf_counter()
        if now - self._last_queue_full_log_at >= 5.0:
            self._last_queue_full_log_at = now
            LOGGER.warning("Confirmed event queue is full. Dropping event for '%s'.", person_name)
        else:
            LOGGER.debug("Suppressed repeated queue-full warning for '%s'.", person_name)

    def _extract_snapshot_image(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int] | None,
    ) -> np.ndarray | None:
        if frame is None or frame.size == 0:
            return None

        if bbox is None:
            return frame.copy()

        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        face_width = max(1, x2 - x1)
        face_height = max(1, y2 - y1)
        pad_x = int(round(face_width * self.snapshot_face_padding))
        pad_y = int(round(face_height * self.snapshot_face_padding))

        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(frame_width, x2 + pad_x)
        crop_y2 = min(frame_height, y2 + pad_y)

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return frame.copy()

        return frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()

    @staticmethod
    def _person_key(person_name: str) -> str:
        return person_name.strip().casefold()

    @staticmethod
    def _sanitize_name(person_name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", person_name.strip())
        return cleaned.strip("_") or "person"
