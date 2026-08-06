from __future__ import annotations

import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from recognition import config

LOGGER = logging.getLogger(__name__)
EventType = Literal["ENTRY", "EXIT"]


class RecognitionEventClient:
    """Send stable-ID recognition events with bounded retries and idempotency."""

    def __init__(
        self,
        enabled: bool = config.API_ENABLED,
        url: str = config.API_URL,
        timeout: float = config.API_TIMEOUT,
        min_similarity: float = config.API_MIN_SIMILARITY,
        cooldown_seconds: float = config.API_COOLDOWN_SECONDS,
        api_key: str = config.API_KEY,
        camera_mode: EventType = config.CAMERA_MODE,
        max_retries: int = config.API_MAX_RETRIES,
        session: requests.Session | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.url = url
        self.timeout = float(timeout)
        self.min_similarity = float(min_similarity)
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.api_key = api_key
        self.camera_mode = camera_mode
        self.session = session or requests.Session()
        if session is None:
            retry = Retry(
                total=max(0, int(max_retries)),
                connect=max(0, int(max_retries)),
                read=max(0, int(max_retries)),
                status=max(0, int(max_retries)),
                allowed_methods=frozenset({"POST"}),
                status_forcelist=(429, 500, 502, 503, 504),
                backoff_factor=0.25,
                respect_retry_after_header=True,
                raise_on_status=False,
            )
            self.session.mount("http://", HTTPAdapter(max_retries=retry))
            self.session.mount("https://", HTTPAdapter(max_retries=retry))

        self._last_sent_at: dict[tuple[str, EventType], float] = {}
        self._sent_once_per_run: set[tuple[str, EventType]] = set()

    def _headers(self, event_id: str) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Idempotency-Key": event_id}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def _passes_filters(self, person_id: str | None, similarity: float) -> bool:
        if not person_id:
            return False
        if similarity < self.min_similarity:
            LOGGER.debug(
                "Skipping API event for %s: similarity %.4f is below %.4f",
                person_id,
                similarity,
                self.min_similarity,
            )
            return False
        return True

    def _passes_cooldown(self, key: tuple[str, EventType], now: float) -> bool:
        previous = self._last_sent_at.get(key)
        return previous is None or now - previous >= self.cooldown_seconds

    def build_payload(
        self,
        *,
        person_id: str,
        display_name: str,
        similarity: float,
        event_type: EventType,
        event_id: str,
        class_name: str | None,
        source: str | None,
        occurred_at: datetime,
    ) -> dict[str, Any]:
        if occurred_at.tzinfo is None or occurred_at.utcoffset() is None:
            raise ValueError("occurred_at must include a UTC offset")
        timestamp = occurred_at.astimezone(UTC)
        payload: dict[str, Any] = {
            "event_id": event_id,
            "person_id": person_id,
            "display_name": display_name,
            "similarity": float(similarity),
            "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
            "event_type": event_type,
        }
        if class_name and class_name.strip():
            payload["class_name"] = class_name.strip()
        if source and source.strip():
            payload["source"] = source.strip()
        return payload

    def send_recognition_event(
        self,
        *,
        person_id: str | None,
        display_name: str,
        similarity: float,
        event_type: EventType | None = None,
        event_id: str | None = None,
        apply_cooldown: bool = True,
        once_per_run: bool = False,
        class_name: str | None = None,
        source: str | None = None,
        occurred_at: datetime | None = None,
    ) -> bool:
        if not self.enabled or not self._passes_filters(person_id, similarity):
            return False
        assert person_id is not None

        resolved_type = event_type or self.camera_mode
        key = (person_id.casefold(), resolved_type)
        monotonic_now = time.monotonic()
        if once_per_run and key in self._sent_once_per_run:
            return False
        if apply_cooldown and not self._passes_cooldown(key, monotonic_now):
            return False

        resolved_event_id = event_id or str(uuid.uuid4())
        payload = self.build_payload(
            person_id=person_id,
            display_name=display_name,
            similarity=similarity,
            event_type=resolved_type,
            event_id=resolved_event_id,
            class_name=class_name,
            source=source,
            occurred_at=occurred_at or datetime.now(UTC),
        )
        try:
            response = self.session.post(
                self.url,
                json=payload,
                timeout=self.timeout,
                headers=self._headers(resolved_event_id),
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            LOGGER.warning(
                "Recognition API unavailable for stable ID %s; event %s was not accepted: %s",
                person_id,
                resolved_event_id,
                exc,
            )
            return False

        self._last_sent_at[key] = monotonic_now
        if once_per_run:
            self._sent_once_per_run.add(key)
        LOGGER.info(
            "Recognition event accepted | person_id=%s event_id=%s event_type=%s",
            person_id,
            resolved_event_id,
            resolved_type,
        )
        return True

    def close(self) -> None:
        self.session.close()
