"""Tests for spectral module."""

import numpy as np

from spectral_submersion.spectral import spectral_embedding, effective_rank


def test_spectral_embedding_shape():
    M = np.random.rand(10, 10)
    E, S, Vt = spectral_embedding(M, k=5)
    assert E.shape == (10, 5)
    assert S.shape == (5,)
    assert Vt.shape == (5, 10)


def test_effective_rank():
    # One dominant singular value -> effective rank close to 1
    S = np.array([100.0, 1.0, 1.0])
    r = effective_rank(S)
    assert 1.0 <= r < 2.0

    # Uniform singular values -> effective rank close to length
    S = np.ones(5)
    r = effective_rank(S)
    assert np.isclose(r, 5.0, rtol=0.01)
