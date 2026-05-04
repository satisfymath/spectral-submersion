"""Tests for alignment module."""

import numpy as np

from spectral_submersion.alignment import (
    orthogonal_procrustes,
    pairwise_squared_distances,
    soft_dictionary,
)


def test_procrustes_exact_rotation():
    # Create Y by rotating X
    X = np.random.rand(10, 5)
    theta = np.deg2rad(30)
    Q_true = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0, 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ]
    )
    Y = X @ Q_true
    Q_est = orthogonal_procrustes(X, Y)
    # Q_est should recover Q_true (up to sign ambiguities on singular vectors)
    resid = np.linalg.norm(X @ Q_est - Y, "fro")
    assert resid < 1e-10


def test_pairwise_squared_distances():
    X = np.array([[0, 0], [1, 1]])
    Y = np.array([[1, 0], [0, 1]])
    D = pairwise_squared_distances(X, Y)
    expected = np.array([[1, 1], [1, 1]])
    assert np.allclose(D, expected)


def test_soft_dictionary_stochastic():
    D = np.array([[0, 1], [2, 0]], dtype=float)
    Pi = soft_dictionary(D, temperature=1.0)
    assert np.allclose(Pi.sum(axis=1), 1.0)
    assert np.all(Pi >= 0)
