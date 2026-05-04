"""Tests for auditable transport: cost decomposition, stability."""

import numpy as np
from spectral_submersion.auditable_transport import decompose_transport_cost


class TestDecomposeTransportCost:
    def test_basic_decomposition(self):
        rng = np.random.RandomState(42)
        n, m = 5, 4
        Pi = np.abs(rng.randn(n, m))
        Pi /= Pi.sum()
        Dx = np.abs(rng.randn(n, n))
        Dx = (Dx + Dx.T) / 2
        Dy = np.abs(rng.randn(m, m))
        Dy = (Dy + Dy.T) / 2
        E_src = rng.randn(n, 3)
        E_tgt = rng.randn(m, 3)

        from spectral_submersion.alignment import orthogonal_procrustes

        n_anch = min(n, m)
        Q = orthogonal_procrustes(E_src[:n_anch], E_tgt[:n_anch])

        result = decompose_transport_cost(
            Pi,
            Dx,
            Dy,
            E_src[:n, :3],
            E_tgt[:m, :3],
            Q=Q,
            lambda_g=1.0,
            lambda_r=1.0,
            lambda_p=1.0,
            epsilon=0.1,
        )
        assert "L_geometric" in result
        assert "L_relational" in result
        assert "L_total" in result
        assert result["L_total"] > 0
        assert result["fraction_geometric"] >= 0
        assert result["fraction_relational"] >= 0

    def test_zero_prior(self):
        rng = np.random.RandomState(42)
        n, m = 4, 3
        Pi = np.abs(rng.randn(n, m))
        Pi /= Pi.sum()
        Dx = np.abs(rng.randn(n, n))
        Dx = (Dx + Dx.T) / 2
        Dy = np.abs(rng.randn(m, m))
        Dy = (Dy + Dy.T) / 2
        E_src = rng.randn(n, 3)
        E_tgt = rng.randn(m, 3)

        result = decompose_transport_cost(
            Pi,
            Dx,
            Dy,
            E_src,
            E_tgt,
            Q=None,
            prior=None,
        )
        assert result["L_geometric"] == 0.0
        assert result["L_prior"] == 0.0

    def test_with_prior(self):
        rng = np.random.RandomState(42)
        n, m = 4, 4
        Pi = np.ones((n, m)) / (n * m)
        Dx = np.eye(n)
        Dy = np.eye(m)
        E_src = rng.randn(n, 3)
        E_tgt = rng.randn(m, 3)
        prior = np.ones((n, m)) * 0.01

        result = decompose_transport_cost(
            Pi,
            Dx,
            Dy,
            E_src,
            E_tgt,
            prior=prior,
        )
        assert result["L_prior"] > 0
