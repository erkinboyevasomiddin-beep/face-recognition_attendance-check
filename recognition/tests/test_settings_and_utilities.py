from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from recognition.app import utils
from recognition.app.preprocessing import (
    average_normalized_embeddings,
    bbox_area,
    bbox_to_int_tuple,
    face_dimensions,
    is_face_large_enough,
    select_largest_face,
)
from recognition.app.settings import RecognitionSettings


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"api_url": "ftp://example.invalid"}, "http"),
        ({"api_url": "https://user:pass@example.invalid/api"}, "credentials"),
        ({"api_url": "https:///missing-host"}, "host"),
        ({"match_threshold": 0.7, "display_threshold": 0.6}, "display_threshold"),
        ({"display_threshold": 0.8, "confirm_threshold": 0.7}, "confirm_threshold"),
        ({"detection_size": [0, 640]}, "greater than"),
        ({"camera_id": -1}, "greater than or equal"),
    ],
)
def test_recognition_settings_validation(values, message):
    with pytest.raises(ValidationError, match=message):
        RecognitionSettings(_env_file=None, **values)


def test_production_api_requires_key_and_https(tmp_path):
    with pytest.raises(ValidationError, match="API_KEY"):
        RecognitionSettings(
            _env_file=None,
            app_env="production",
            api_enabled=True,
            api_url="https://example.invalid",
        )
    with pytest.raises(ValidationError, match="HTTPS"):
        RecognitionSettings(
            _env_file=None,
            app_env="production",
            api_enabled=True,
            api_url="http://example.invalid",
            api_key="secure-recognition-key-with-more-than-thirty-two-characters",
        )
    settings = RecognitionSettings(
        _env_file=None,
        app_env="production",
        api_enabled=True,
        api_url="https://example.invalid/api/",
        api_key="secure-recognition-key-with-more-than-thirty-two-characters",
        data_dir=tmp_path,
    )
    assert settings.api_url == "https://example.invalid/api"
    assert settings.snapshot_dir == tmp_path / "snapshots"
    assert settings.database_path == tmp_path / "embeddings" / "face_embeddings.db"


def test_face_geometry_and_average_validation():
    small = SimpleNamespace(bbox=np.array([1.9, 2.1, 11.8, 22.9]))
    large = SimpleNamespace(bbox=np.array([0, 0, 40, 50]))
    assert bbox_to_int_tuple(small.bbox) == (1, 2, 11, 22)
    assert face_dimensions(small) == (10, 20)
    assert bbox_area(small) == 200.0
    assert is_face_large_enough(large, 40)
    assert not is_face_large_enough(small, 30)
    assert select_largest_face([small, large]) is large
    assert select_largest_face([]) is None
    with pytest.raises(ValueError, match="At least one"):
        average_normalized_embeddings([])
    with pytest.raises(ValueError, match="same dimension"):
        average_normalized_embeddings([np.ones(2), np.ones(3)])


def test_file_and_image_utilities(tmp_path, monkeypatch):
    image = tmp_path / "demo.JPG"
    text = tmp_path / "notes.txt"
    image.write_bytes(b"placeholder")
    text.write_text("synthetic", encoding="utf-8")
    assert utils.is_image_file(image)
    assert not utils.is_image_file(text)
    assert utils.iter_image_paths(tmp_path) == [image]
    assert "jpg" in utils.format_supported_extensions({".jpg", ".png"})

    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    assert utils.resize_frame(frame, 1.0) is frame
    assert utils.resize_frame(frame, 0.5).shape == (5, 10, 3)
    with pytest.raises(ValueError, match="greater than zero"):
        utils.resize_frame(frame, 0)

    monkeypatch.setattr(utils.cv2, "imencode", lambda *args, **kwargs: (False, None))
    with pytest.raises(ValueError, match="Failed to encode"):
        utils.write_image(tmp_path / "failed.jpg", frame)


def test_runtime_directories_and_logging(tmp_path, monkeypatch):
    monkeypatch.setattr(utils.config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(utils.config, "KNOWN_FACES_DIR", tmp_path / "known")
    monkeypatch.setattr(utils.config, "EMBEDDINGS_DIR", tmp_path / "embeddings")
    monkeypatch.setattr(utils.config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(utils.config, "SNAPSHOT_DIR", tmp_path / "snapshots")
    monkeypatch.setattr(utils.config, "SNAPSHOT_ENABLED", True)
    utils.ensure_runtime_directories()
    assert (tmp_path / "snapshots").is_dir()
    utils.setup_logging("DEBUG")
