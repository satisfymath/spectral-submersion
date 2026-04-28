"""Tests for transport module."""
import numpy as np
import pytest

from spectral_submersion.transport import (
    optimal_transport_matrix,
    gromov_wasserstein_matrix,
    multi_marginal_gw,
    consensus_from_multi_gw,
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


def test_multi_marginal_gw_shapes():
    m = 3
    sizes = [4, 5, 6]
    rng = np.random.default_rng(42)
    embeddings = [rng.random((n, 3)) for n in sizes]
    dist_mats = []
    for E in embeddings:
        D = np.linalg.norm(E[:, None, :] - E[None, :, :], axis=2)
        dist_mats.append(D)

    couplings = multi_marginal_gw(dist_mats, reg=1.0, max_iter=3, sinkhorn_iter=50)

    for i in range(m):
        assert couplings[i][i].shape == (sizes[i], sizes[i])
        for j in range(m):
            if i != j:
                assert couplings[i][j].shape == (sizes[i], sizes[j])
                assert couplings[j][i].shape == (sizes[j], sizes[i])
                np.testing.assert_allclose(couplings[i][j], couplings[j][i].T, atol=1e-6)


def test_multi_marginal_gw_coupling_properties():
    m = 3
    sizes = [4, 5, 6]
    rng = np.random.default_rng(42)
    embeddings = [rng.random((n, 3)) for n in sizes]
    dist_mats = []
    for E in embeddings:
        D = np.linalg.norm(E[:, None, :] - E[None, :, :], axis=2)
        dist_mats.append(D)

    couplings = multi_marginal_gw(dist_mats, reg=1.0, max_iter=3, sinkhorn_iter=50)

    for i in range(m):
        for j in range(m):
            if i != j:
                Pi = couplings[i][j]
                assert np.all(Pi >= 0)
                assert pi_sum_rows_close(Pi, 1.0 / sizes[i])


def pi_sum_rows_close(Pi, expected, atol=1e-3):
    return np.allclose(Pi.sum(axis=1), expected, atol=atol)


def test_consensus_from_multi_gw():
    m = 3
    sizes = [4, 5, 6]
    d = 3
    rng = np.random.default_rng(42)
    embeddings = [rng.random((n, d)) for n in sizes]
    dist_mats = []
    for E in embeddings:
        D = np.linalg.norm(E[:, None, :] - E[None, :, :], axis=2)
        dist_mats.append(D)

    couplings = multi_marginal_gw(dist_mats, reg=1.0, max_iter=3, sinkhorn_iter=50)
    consensus, projections = consensus_from_multi_gw(couplings, embeddings, dist_mats)

    assert consensus.shape[1] == d
    assert len(projections) == m
    for P in projections:
        assert P.shape == (d, d)
