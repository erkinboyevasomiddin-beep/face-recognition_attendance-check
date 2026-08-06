from __future__ import annotations

import logging
from collections import Counter, deque
from dataclasses import dataclass, replace

import numpy as np

from recognition import config
from recognition.app.database import FaceEmbeddingsDatabase, IdentityRecord
from recognition.app.face_engine import FaceEngine
from recognition.app.preprocessing import bbox_to_int_tuple

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecognitionResult:
    bbox: tuple[int, int, int, int]
    person_id: str | None
    label: str
    similarity: float
    is_known: bool
    detection_score: float
    class_name: str | None = None
    display_label: str | None = None
    stability_state: str = "raw"


class TemporalSmoother:
    """
    Simple temporal smoothing keyed by left-to-right face order.

    This is intentionally lightweight and works best for webcam scenes with a
    small, stable number of faces.
    """

    def __init__(self, window_size: int) -> None:
        self.window_size = max(1, window_size)
        self._histories: list[deque[tuple[str | None, str, float, str | None]]] = []

    def reset(self) -> None:
        self._histories.clear()

    def apply(self, results: list[RecognitionResult]) -> list[RecognitionResult]:
        ordered_results = sorted(results, key=lambda result: (result.bbox[0], result.bbox[1]))

        while len(self._histories) < len(ordered_results):
            self._histories.append(deque(maxlen=self.window_size))

        if len(self._histories) > len(ordered_results):
            self._histories = self._histories[: len(ordered_results)]

        stabilized: list[RecognitionResult] = []
        for index, result in enumerate(ordered_results):
            history = self._histories[index]
            history.append((result.person_id, result.label, result.similarity, result.class_name))

            stable_key = Counter(
                (person_id, label) for person_id, label, _, _ in history
            ).most_common(1)[0][0]
            stable_person_id, stable_label = stable_key
            stable_matches = [
                (score, class_name)
                for person_id, label, score, class_name in history
                if (person_id, label) == stable_key
            ]
            stable_scores = [score for score, _ in stable_matches]
            stable_similarity = float(sum(stable_scores) / len(stable_scores))
            stable_class_name = next(
                (
                    class_name
                    for _, class_name in reversed(stable_matches)
                    if class_name is not None
                ),
                None,
            )

            stabilized.append(
                replace(
                    result,
                    person_id=stable_person_id,
                    label=stable_label,
                    similarity=stable_similarity,
                    is_known=stable_person_id is not None,
                    class_name=stable_class_name if stable_person_id is not None else None,
                )
            )

        return stabilized


class FaceRecognizer:
    """Compare normalized query embeddings against enrolled local identities."""

    def __init__(
        self,
        engine: FaceEngine,
        database: FaceEmbeddingsDatabase,
        threshold: float = config.RECOGNITION_THRESHOLD,
        smoothing_enabled: bool = config.TEMPORAL_SMOOTHING,
        smoothing_window: int = config.SMOOTHING_WINDOW,
    ) -> None:
        self.engine = engine
        self.database = database
        self.threshold = threshold
        self.smoother = TemporalSmoother(smoothing_window) if smoothing_enabled else None
        self._identities: list[IdentityRecord] = []
        self._embedding_matrix: np.ndarray | None = None
        self.refresh_gallery()

    def reset_state(self) -> None:
        if self.smoother is not None:
            self.smoother.reset()

    def refresh_gallery(self) -> None:
        identities = self.database.load_identities()
        if not identities:
            raise RuntimeError(
                "No enrolled identities found. Add images under 'data/known_faces/<person_name>/' "
                "or 'data/known_faces/<class_name>/<person_name>/' "
                "and run `python enroll.py` first."
            )

        self._identities = identities
        self._embedding_matrix = np.vstack(
            [identity.embedding.astype(np.float32) for identity in identities]
        )
        LOGGER.info(
            "Loaded %d enrolled identities from %s",
            len(self._identities),
            self.database.db_path,
        )

    def _best_match(self, embedding: np.ndarray) -> tuple[IdentityRecord, float]:
        if self._embedding_matrix is None or not self._identities:
            raise RuntimeError("Recognition gallery is empty. Run enrollment first.")

        scores = np.dot(self._embedding_matrix, embedding)
        best_index = int(np.argmax(scores))
        return self._identities[best_index], float(scores[best_index])

    def recognize(self, image: np.ndarray) -> list[RecognitionResult]:
        faces = self.engine.detect_faces(image)
        if not faces:
            self.reset_state()
            return []

        results: list[RecognitionResult] = []
        for face in sorted(faces, key=lambda item: (float(item.bbox[0]), float(item.bbox[1]))):
            embedding = self.engine.extract_normalized_embedding(face)
            best_identity, best_score = self._best_match(embedding)
            is_known = best_score >= self.threshold
            label = best_identity.display_name if is_known else config.UNKNOWN_LABEL
            bbox = bbox_to_int_tuple(face.bbox)

            LOGGER.debug(
                "Face %s best match=%s similarity=%.4f threshold=%.2f label=%s",
                bbox,
                best_identity.person_id,
                best_score,
                self.threshold,
                label,
            )

            results.append(
                RecognitionResult(
                    bbox=bbox,
                    person_id=best_identity.person_id if is_known else None,
                    label=label,
                    similarity=best_score,
                    is_known=label != config.UNKNOWN_LABEL,
                    detection_score=float(getattr(face, "det_score", 0.0)),
                    class_name=best_identity.class_name if is_known else None,
                )
            )

        if self.smoother is not None:
            return self.smoother.apply(results)

        return results
