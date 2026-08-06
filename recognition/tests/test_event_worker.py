from __future__ import annotations

import queue

import numpy as np

from recognition.app.webcam_event_worker import AsyncConfirmedEventDispatcher


class FakeClient:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.calls = []
        self.closed = False

    def send_recognition_event(self, **kwargs):
        self.calls.append(kwargs)
        return True

    def close(self):
        self.closed = True


def test_synchronous_snapshot_crop_cooldown_and_shutdown(tmp_path):
    client = FakeClient(enabled=True)
    dispatcher = AsyncConfirmedEventDispatcher(
        api_client=client,
        async_api_send=False,
        async_snapshot_save=False,
        snapshot_enabled=True,
        snapshot_dir=tmp_path,
        snapshot_cooldown_seconds=60,
    )
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    assert not dispatcher.submit_confirmed_event(None, "Unknown", 0.1, frame)
    assert dispatcher.submit_confirmed_event(
        "DEMO-001", "Demo Person", 0.9, frame, bbox=(5, 5, 10, 10)
    )
    assert len(list(tmp_path.glob("*.jpg"))) == 1
    assert dispatcher.submit_confirmed_event("DEMO-001", "Demo Person", 0.9, frame)
    assert len(list(tmp_path.glob("*.jpg"))) == 1
    assert len(client.calls) == 2
    dispatcher.close()
    dispatcher.close()
    assert client.closed
    assert not dispatcher.submit_confirmed_event("DEMO-001", "Demo", 0.9, frame)


def test_queue_full_duplicate_pending_and_background_worker(tmp_path):
    disabled = FakeClient(enabled=False)
    dispatcher = AsyncConfirmedEventDispatcher(
        api_client=disabled,
        max_queue_size=1,
        async_api_send=False,
        snapshot_enabled=False,
        snapshot_dir=tmp_path,
    )
    assert not dispatcher.submit_confirmed_event(
        "DEMO-001", "Demo", 0.9, np.zeros((2, 2, 3), dtype=np.uint8)
    )

    dispatcher._use_background_worker = True
    dispatcher._queue.put_nowait(object())
    assert not dispatcher.submit_confirmed_event(
        "DEMO-001", "Demo", 0.9, np.zeros((2, 2, 3), dtype=np.uint8)
    )
    with dispatcher._lock:
        dispatcher._pending_keys.add("demo-002")
    assert not dispatcher.submit_confirmed_event(
        "DEMO-002", "Demo", 0.9, np.zeros((2, 2, 3), dtype=np.uint8)
    )
    assert dispatcher.queue_size == 1
    assert isinstance(dispatcher._queue, queue.Queue)
    dispatcher.close()

    client = FakeClient(enabled=True)
    background = AsyncConfirmedEventDispatcher(
        api_client=client,
        max_queue_size=2,
        async_api_send=True,
        snapshot_enabled=False,
        snapshot_dir=tmp_path,
    )
    assert background.submit_confirmed_event(
        "DEMO-003", "Demo", 0.9, np.zeros((2, 2, 3), dtype=np.uint8)
    )
    background.close()
    assert client.calls and client.closed
