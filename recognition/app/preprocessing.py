from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np


def bbox_to_int_tuple(bbox: Sequence[float]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox[:4]
    return int(x1), int(y1), int(x2), int(y2)


def face_dimensions(face: Any) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox_to_int_tuple(face.bbox)
    return max(0, x2 - x1), max(0, y2 - y1)


def bbox_area(face: Any) -> float:
    width, height = face_dimensions(face)
    return float(width * height)


def is_face_large_enough(face: Any, min_face_size: int) -> bool:
    width, height = face_dimensions(face)
    return width >= min_face_size and height >= min_face_size


def select_largest_face(faces: Sequence[Any]) -> Any | None:
    if not faces:
        return None
    return max(faces, key=bbox_area)


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """L2-normalize a face embedding."""
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("Embedding must contain finite values.")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero-length embedding.")
    return vector / norm


def average_normalized_embeddings(embeddings: Iterable[np.ndarray]) -> np.ndarray:
    """Average normalized embeddings, then normalize the mean vector."""
    vectors = [normalize_embedding(embedding) for embedding in embeddings]
    if not vectors:
        raise ValueError("At least one embedding is required to compute a template.")

    dimensions = {vector.size for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError("All embeddings must use the same dimension.")

    mean_embedding = np.mean(np.vstack(vectors), axis=0)
    return normalize_embedding(mean_embedding)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return cosine similarity after validating and normalizing both vectors."""
    return float(np.dot(normalize_embedding(left), normalize_embedding(right)))
