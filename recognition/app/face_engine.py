from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from recognition import config
from recognition.app.preprocessing import (
    is_face_large_enough,
    normalize_embedding,
    select_largest_face,
)

LOGGER = logging.getLogger(__name__)
_INSIGHTFACE_IMPORT_ERROR: ImportError | None
_ORT_IMPORT_ERROR: ImportError | None

try:
    from insightface.app import FaceAnalysis
except ImportError as exc:  # pragma: no cover - exercised at runtime when missing deps
    FaceAnalysis = None
    _INSIGHTFACE_IMPORT_ERROR = exc
else:
    _INSIGHTFACE_IMPORT_ERROR = None

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover - exercised at runtime when missing deps
    ort = None
    _ORT_IMPORT_ERROR = exc
else:
    _ORT_IMPORT_ERROR = None


class FaceEngine:
    """Thin wrapper around InsightFace FaceAnalysis with local-only configuration."""

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        preferred_providers: Sequence[str] = config.PREFERRED_PROVIDERS,
        detection_size: tuple[int, int] = config.DETECTION_SIZE,
        min_face_size: int = config.MIN_FACE_SIZE,
    ) -> None:
        self.model_name = model_name
        self.preferred_providers = list(preferred_providers)
        self.detection_size = detection_size
        self.min_face_size = min_face_size
        self._app: FaceAnalysis | None = None
        self._providers: list[str] = []

    @property
    def expected_model_dir(self) -> Path:
        return config.MODELS_DIR / self.model_name

    @property
    def providers(self) -> list[str]:
        self.ensure_loaded()
        return list(self._providers)

    def resolve_providers(self) -> list[str]:
        if ort is None:
            raise RuntimeError(
                "ONNX Runtime is not installed. Install `onnxruntime` for CPU or "
                "`onnxruntime-gpu` for GPU support."
            ) from _ORT_IMPORT_ERROR

        available_providers = ort.get_available_providers()
        matched = [
            provider for provider in self.preferred_providers if provider in available_providers
        ]

        if not matched and "CPUExecutionProvider" in available_providers:
            matched = ["CPUExecutionProvider"]

        if not matched:
            raise RuntimeError(
                f"No compatible ONNX Runtime execution provider found. Available providers: "
                f"{available_providers}"
            )

        return matched

    def ensure_loaded(self) -> None:
        if self._app is not None:
            return

        if FaceAnalysis is None:
            raise RuntimeError(
                "InsightFace is not installed. Install it with `pip install insightface`."
            ) from _INSIGHTFACE_IMPORT_ERROR

        providers = self.resolve_providers()
        LOGGER.info(
            "Loading InsightFace model pack '%s' with providers %s",
            self.model_name,
            providers,
        )

        try:
            app = FaceAnalysis(
                name=self.model_name,
                root=str(config.INSIGHTFACE_ROOT),
                providers=providers,
                allowed_modules=["detection", "recognition"],
            )
            app.prepare(ctx_id=0, det_size=self.detection_size)
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize InsightFace. If the first-run model download is blocked, "
                f"manually place the '{self.model_name}' model pack under "
                f"'{self.expected_model_dir}'. Original error: {exc}"
            ) from exc

        self._app = app
        self._providers = providers

    def detect_faces(self, image: np.ndarray) -> list[Any]:
        """Detect faces in an image and apply a minimum face-size filter."""
        if image is None or image.size == 0:
            raise ValueError("Input image is empty.")

        self.ensure_loaded()
        assert self._app is not None

        faces = list(self._app.get(image))
        filtered_faces: list[Any] = []

        for face in faces:
            if self.min_face_size > 0 and not is_face_large_enough(face, self.min_face_size):
                LOGGER.debug(
                    "Ignoring face smaller than the configured minimum size of %d px",
                    self.min_face_size,
                )
                continue
            filtered_faces.append(face)

        return filtered_faces

    def detect_largest_face(self, image: np.ndarray) -> Any | None:
        return select_largest_face(self.detect_faces(image))

    def extract_normalized_embedding(self, face: Any) -> np.ndarray:
        """Return a normalized embedding from a detected InsightFace face object."""
        embedding = getattr(face, "embedding", None)
        if embedding is None:
            raise RuntimeError("InsightFace did not return an embedding for the detected face.")
        return normalize_embedding(embedding)
