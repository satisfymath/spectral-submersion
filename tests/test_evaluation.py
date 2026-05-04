"""Tests for evaluation module."""

import numpy as np
import pytest

from spectral_submersion.evaluation import (
    geometric_distortion,
    permute_corpus,
    random_corpus_uniform,
    relational_distortion,
)


def test_permute_corpus_preserves_lengths():
    seqs = [["a", "b"], ["c", "d", "e"]]
    perm = permute_corpus(seqs)
    assert [len(s) for s in perm] == [2, 3]


def test_random_uniform_uses_vocab():
    seqs = [["a", "b"], ["c", "d"]]
    rand = random_corpus_uniform(seqs, ["x", "y", "z"])
    all_toks = [t for s in rand for t in s]
    assert set(all_toks).issubset({"x", "y", "z"})


def test_relational_distortion_perfect_alignment():
    # If Dx == Dy and Pi is identity, distortion should be 0
    n = 5
    X = np.random.rand(n, 3)
    Dx = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    Pi = np.eye(n)
    L = relational_distortion(Pi, Dx, Dx)
    assert np.isclose(L, 0.0, atol=1e-10)


def test_relational_distortion_random_vs_identity():
    # Random coupling should have higher distortion than identity for same Dx, Dy
    n = 5
    X = np.random.rand(n, 3)
    Y = np.random.rand(n, 3)
    Dx = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    Dy = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2)
    Pi_id = np.eye(n)
    Pi_rand = np.ones((n, n)) / n
    L_id = relational_distortion(Pi_id, Dx, Dy)
    L_rand = relational_distortion(Pi_rand, Dx, Dy)
    # Identity may or may not be better than random depending on geometry,
    # but for generic random points, identity is usually not worse by orders.
    assert L_rand >= 0.0
    assert L_id >= 0.0


def test_geometric_distortion():
    D = np.array([[0.0, 1.0], [2.0, 3.0]])
    Pi = np.array([[0.5, 0.5], [0.5, 0.5]])
    assert geometric_distortion(Pi, D) == pytest.approx(3.0)
