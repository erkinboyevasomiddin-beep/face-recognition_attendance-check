from __future__ import annotations

import numpy as np
import pytest

from recognition.app.preprocessing import (
    average_normalized_embeddings,
    cosine_similarity,
    normalize_embedding,
)


def test_embedding_normalization_and_cosine_similarity():
    normalized = normalize_embedding(np.array([3.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(normalized, [0.6, 0.8], atol=1e-6)
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_invalid_and_average_embeddings():
    with pytest.raises(ValueError, match="zero-length"):
        normalize_embedding(np.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        normalize_embedding(np.array([np.nan, 1.0]))
    averaged = average_normalized_embeddings([np.array([1.0, 0.0]), np.array([0.0, 1.0])])
    np.testing.assert_allclose(averaged, [2**-0.5, 2**-0.5], atol=1e-6)
