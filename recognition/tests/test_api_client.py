from __future__ import annotations

from datetime import UTC, datetime

import pytest
import requests

from recognition.app.api_client import RecognitionEventClient


class Response:
    def __init__(self, error=False):
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise requests.HTTPError("synthetic failure")


class Session:
    def __init__(self, error=False):
        self.error = error
        self.calls = []
        self.closed = False

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return Response(self.error)

    def close(self):
        self.closed = True


def test_api_payload_success_cooldown_and_failure():
    session = Session()
    client = RecognitionEventClient(
        enabled=True,
        min_similarity=0.6,
        cooldown_seconds=60,
        api_key="synthetic-key",
        session=session,
    )
    sent = client.send_recognition_event(
        person_id="DEMO-1",
        display_name="Demo Person",
        similarity=0.8,
        event_id="00000000-0000-0000-0000-000000000001",
        class_name="5-01",
        source="test-camera",
        occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert sent
    assert not client.send_recognition_event(
        person_id="DEMO-1", display_name="Demo Person", similarity=0.8
    )
    payload = session.calls[0][1]["json"]
    assert payload["person_id"] == "DEMO-1"
    assert payload["event_type"] == "ENTRY"
    assert session.calls[0][1]["headers"]["Idempotency-Key"] == payload["event_id"]
    client.close()
    assert session.closed

    failing = RecognitionEventClient(enabled=True, session=Session(error=True))
    assert not failing.send_recognition_event(
        person_id="DEMO-2", display_name="Demo Person", similarity=0.9
    )


def test_api_filters_unknown_and_weak_matches():
    session = Session()
    client = RecognitionEventClient(enabled=True, min_similarity=0.7, session=session)
    assert not client.send_recognition_event(
        person_id=None, display_name="Unknown", similarity=0.99
    )
    assert not client.send_recognition_event(
        person_id="DEMO-1", display_name="Demo", similarity=0.69
    )
    assert session.calls == []


def test_default_client_has_bounded_post_retries():
    client = RecognitionEventClient(enabled=True, max_retries=3)
    retries = client.session.get_adapter("https://").max_retries
    assert retries.total == 3
    assert retries.allowed_methods == frozenset({"POST"})
    assert 503 in retries.status_forcelist
    client.close()


def test_disabled_once_per_run_event_override_and_naive_timestamp():
    disabled = RecognitionEventClient(enabled=False, session=Session())
    assert not disabled.send_recognition_event(
        person_id="DEMO-1", display_name="Demo", similarity=1.0
    )

    session = Session()
    client = RecognitionEventClient(
        enabled=True, camera_mode="EXIT", cooldown_seconds=0, session=session
    )
    assert client.send_recognition_event(
        person_id="DEMO-1",
        display_name=" Demo Person ",
        similarity=0.9,
        once_per_run=True,
        class_name=" 5-01 ",
        source=" camera ",
    )
    assert not client.send_recognition_event(
        person_id="demo-1",
        display_name="Demo Person",
        similarity=0.9,
        once_per_run=True,
    )
    assert session.calls[0][1]["json"]["event_type"] == "EXIT"
    assert session.calls[0][1]["json"]["class_name"] == "5-01"
    with pytest.raises(ValueError, match="UTC offset"):
        client.build_payload(
            person_id="DEMO-1",
            display_name="Demo",
            similarity=0.9,
            event_type="ENTRY",
            event_id="00000000-0000-0000-0000-000000000001",
            class_name=None,
            source=None,
            occurred_at=datetime(2026, 1, 1),
        )
