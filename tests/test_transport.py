"""Tests for transport module."""
import numpy as np
import pytest

from spectral_submersion.transport import (
    optimal_transport_matrix,
    gromov_wasserstein_matrix,
)


def test_optimal_transport_uniform_marginals():
    cost = np.array([[0.0, 1.0], [1.0, 0.0]])
    Pi = optimal_transport_matrix(cost, reg=0.1)
    assert np.allclose(Pi.sum(axis=1), 0.5)
    assert np.allclose(Pi.sum(axis=0), 0.5)
    assert np.all(Pi >= 0)


def test_optimal_transport_diagonal_cost():
    # If cost is zero on diagonal, OT should concentrate there
    cost = np.array([[0.0, 10.0], [10.0, 0.0]])
    Pi = optimal_transport_matrix(cost, reg=0.01)
    # With low reg, should be close to identity (up to scaling)
    assert Pi[0, 0] > Pi[0, 1]
    assert Pi[1, 1] > Pi[1, 0]


def test_gromov_wasserstein_same_space():
    # When Dx == Dy and n == m, GW should recover near-identity
    n = 5
    X = np.random.rand(n, 3)
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    Pi = gromov_wasserstein_matrix(D, D, reg=0.05, max_iter=100)
    assert Pi.shape == (n, n)
    assert np.allclose(Pi.sum(axis=1), 1.0 / n, atol=1e-3)
    assert np.allclose(Pi.sum(axis=0), 1.0 / n, atol=1e-3)
    # Diagonal should be larger than off-diagonal
    diag_mean = np.diag(Pi).mean()
    offdiag_mean = Pi[~np.eye(n, dtype=bool)].mean()
    assert diag_mean > offdiag_mean


def test_gromov_wasserstein_vs_ot_different_sizes():
    # GW should work for different-size spaces where direct OT also works
    n_x, n_y = 4, 6
    X = np.random.rand(n_x, 3)
    Y = np.random.rand(n_y, 3)
    Dx = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    Dy = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2)
    Pi = gromov_wasserstein_matrix(Dx, Dy, reg=0.5, max_iter=20)
    assert Pi.shape == (n_x, n_y)
    assert np.allclose(Pi.sum(axis=1), 1.0 / n_x, atol=1e-3)
    assert np.all(Pi >= 0)
