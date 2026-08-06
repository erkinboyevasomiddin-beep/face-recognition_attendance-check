from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from recognition.app.database import IdentityRecord
from recognition.app.recognizer import FaceRecognizer, TemporalSmoother


@dataclass
class FakeFace:
    bbox: np.ndarray
    embedding: np.ndarray
    det_score: float = 0.99


class FakeEngine:
    def __init__(self, faces):
        self.faces = faces

    def detect_faces(self, image):
        return self.faces

    def extract_normalized_embedding(self, face):
        vector = face.embedding.astype(np.float32)
        return vector / np.linalg.norm(vector)


class FakeDatabase:
    db_path = "synthetic.db"

    def __init__(self, identities):
        self.identities = identities

    def load_identities(self):
        return self.identities


def identity(person_id, name, embedding):
    return IdentityRecord(
        person_id=person_id,
        display_name=name,
        class_name="5-01",
        embedding=np.asarray(embedding, dtype=np.float32),
        embedding_index=0,
        usable_images=1,
        created_at="now",
        updated_at="now",
        source_images=[],
    )


def test_known_unknown_and_threshold_boundary():
    gallery = [identity("DEMO-1", "Demo One", [1.0, 0.0])]
    face = FakeFace(np.array([0, 0, 50, 50]), np.array([1.0, 0.0]))
    recognizer = FaceRecognizer(
        FakeEngine([face]), FakeDatabase(gallery), threshold=1.0, smoothing_enabled=False
    )
    known = recognizer.recognize(np.ones((2, 2, 3), dtype=np.uint8))[0]
    assert known.person_id == "DEMO-1" and known.is_known

    recognizer.threshold = 1.0001
    unknown = recognizer.recognize(np.ones((2, 2, 3), dtype=np.uint8))[0]
    assert unknown.person_id is None and not unknown.is_known


def test_multiple_templates_choose_best_identity():
    gallery = [
        identity("DEMO-1", "Same Name", [1.0, 0.0]),
        identity("DEMO-2", "Same Name", [0.0, 1.0]),
    ]
    face = FakeFace(np.array([0, 0, 50, 50]), np.array([0.0, 1.0]))
    result = FaceRecognizer(
        FakeEngine([face]), FakeDatabase(gallery), threshold=0.5, smoothing_enabled=False
    ).recognize(np.ones((2, 2, 3), dtype=np.uint8))[0]
    assert result.person_id == "DEMO-2"


def test_empty_gallery_and_no_faces_reset_state():
    with pytest.raises(RuntimeError, match="No enrolled identities"):
        FaceRecognizer(FakeEngine([]), FakeDatabase([]))

    gallery = [identity("DEMO-1", "Demo One", [1.0, 0.0])]
    recognizer = FaceRecognizer(
        FakeEngine([]), FakeDatabase(gallery), smoothing_enabled=True, smoothing_window=2
    )
    assert recognizer.recognize(np.ones((2, 2, 3), dtype=np.uint8)) == []
    assert recognizer.smoother is not None
    assert recognizer.smoother._histories == []


def test_temporal_smoother_majority_average_and_face_count_change():
    smoother = TemporalSmoother(window_size=3)
    first = identity("DEMO-1", "Demo One", [1.0, 0.0])
    second = identity("DEMO-2", "Demo Two", [0.0, 1.0])
    gallery = [first, second]
    face_one = FakeFace(np.array([50, 0, 100, 50]), np.array([1.0, 0.0]))
    face_two = FakeFace(np.array([0, 0, 40, 40]), np.array([0.0, 1.0]))
    recognizer = FaceRecognizer(
        FakeEngine([face_one, face_two]),
        FakeDatabase(gallery),
        threshold=0.5,
        smoothing_enabled=False,
    )
    raw = recognizer.recognize(np.ones((2, 2, 3), dtype=np.uint8))
    stabilized = smoother.apply(raw)
    assert [item.person_id for item in stabilized] == ["DEMO-2", "DEMO-1"]
    stabilized = smoother.apply([raw[0]])
    assert len(stabilized) == 1
    smoother.reset()
    assert smoother._histories == []
