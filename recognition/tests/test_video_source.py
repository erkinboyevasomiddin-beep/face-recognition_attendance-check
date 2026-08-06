from __future__ import annotations

import pytest

from recognition.app import video_source
from recognition.app.video_source import open_video_capture, resolve_video_source


def test_stream_description_redacts_credentials_and_path():
    resolved = resolve_video_source(
        "rtsp://camera-user:camera-password@192.0.2.10:8554/private/path?token=secret"
    )
    assert resolved.capture_value != resolved.description
    assert "camera-user" not in resolved.description
    assert "camera-password" not in resolved.description
    assert "private" not in resolved.description
    assert "token" not in resolved.description
    assert resolved.event_source == "rtsp_192_0_2_10_8554"


def test_camera_and_file_source_resolution(tmp_path):
    camera = resolve_video_source("2")
    assert camera.source_type == "camera" and camera.capture_value == 2
    video = tmp_path / "synthetic.mp4"
    video.write_bytes(b"placeholder")
    file_source = resolve_video_source(video)
    assert file_source.source_type == "file" and not file_source.is_live


@pytest.mark.parametrize(
    ("source", "source_type", "exception"),
    [
        ("", "auto", ValueError),
        ("abc", "camera", ValueError),
        ("anything", "unsupported", ValueError),
        ("missing.mp4", "file", FileNotFoundError),
        ("", "url", ValueError),
    ],
)
def test_invalid_video_sources(source, source_type, exception):
    with pytest.raises(exception):
        resolve_video_source(source, source_type)


class FakeCapture:
    def __init__(self, opened: bool):
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def release(self):
        self.released = True


def test_open_capture_success_and_redacted_failures(tmp_path, monkeypatch):
    capture = FakeCapture(True)
    monkeypatch.setattr(video_source, "open_webcam", lambda index: capture)
    opened, resolved = open_video_capture(1)
    assert opened is capture and resolved.event_source == "camera_1"

    failed = FakeCapture(False)
    monkeypatch.setattr(video_source.cv2, "VideoCapture", lambda value: failed)
    stream = "rtsp://user:pass@192.0.2.20/private?token=secret"
    with pytest.raises(RuntimeError, match="RTSP/HTTP") as captured:
        open_video_capture(stream)
    assert failed.released
    assert "user" not in str(captured.value)
    assert "pass" not in str(captured.value)
    assert "private" not in str(captured.value)
