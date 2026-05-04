"""Tests for identifiability module: no-free-decipherment, anchor power, Procrustes stability."""

import pytest
import numpy as np
from spectral_submersion.identifiability import (
    verify_non_identifiability,
    orbit_size,
    anchor_power,
    compute_automorphism_size_upper_bound,
    anchor_condition_number,
    leave_one_anchor_out_stability,
)


class TestNonIdentifiability:
    def test_svd_is_structural(self):
        rng = np.random.RandomState(42)
        n = 10
        corpus = rng.randint(0, n, size=100)

        def stat(c):
            from spectral_submersion.cooccurrence import (
                cooccurrence_matrix_from_sequences,
            )

            C = cooccurrence_matrix_from_sequences([c.tolist()], n, window_size=2)
            s = np.linalg.svd(C, compute_uv=False)
            return s[:5]

        result = verify_non_identifiability(n, stat, corpus, n_permutations=20, seed=42)
        assert result["is_invariant"]
        assert result["max_deviation"] < 1e-10

    def test_mean_is_structural(self):
        rng = np.random.RandomState(42)
        n = 5
        corpus = rng.randint(0, n, size=50)

        def stat(c):
            from spectral_submersion.cooccurrence import (
                cooccurrence_matrix_from_sequences,
            )

            C = cooccurrence_matrix_from_sequences([c.tolist()], n, window_size=2)
            return np.linalg.svd(C, compute_uv=False)[:3]

        result = verify_non_identifiability(n, stat, corpus, n_permutations=20, seed=42)
        assert result["is_invariant"]
        assert result["max_deviation"] < 1e-10


class TestOrbitSize:
    def test_no_anchors(self):
        assert orbit_size(10, 0) == 3628800

    def test_all_anchored(self):
        assert orbit_size(5, 5) == 1

    def test_some_anchored(self):
        assert orbit_size(5, 2) == 6

    def test_negative_anchored_raises(self):
        with pytest.raises(ValueError):
            orbit_size(3, 5)


class TestAnchorPower:
    def test_full_symmetry(self):
        ap = anchor_power(120, 120)
        assert ap == pytest.approx(0.0, abs=1e-10)

    def test_trivial_group(self):
        ap = anchor_power(120, 1)
        assert ap > 0.8

    def test_monotonicity(self):
        ap1 = anchor_power(120, 24)
        ap2 = anchor_power(120, 1)
        assert ap2 > ap1


class TestAnchorConditionNumber:
    def test_perfect_anchors(self):
        rng = np.random.RandomState(42)
        X = rng.randn(10, 5)
        Q, _ = np.linalg.qr(rng.randn(5, 5))
        Y = X @ Q
        cond = anchor_condition_number(X, Y)
        assert cond > 0.1

    def test_degenerate_anchors(self):
        X = np.ones((5, 3))
        Y = np.ones((5, 3))
        cond = anchor_condition_number(X, Y)
        assert cond < 1e-6


class TestAutomorphismBound:
    def test_identity_like_weight_matrix(self):
        W = np.eye(5)
        aut_size = compute_automorphism_size_upper_bound(W)
        assert aut_size == 120  # Sym(5)

    def test_all_same_weight_matrix(self):
        W = np.ones((4, 4))
        aut_size = compute_automorphism_size_upper_bound(W)
        assert aut_size == 24  # Sym(4)


class TestLOOStability:
    def test_enough_anchors(self):
        rng = np.random.RandomState(42)
        X = rng.randn(8, 3)
        Q, _ = np.linalg.qr(rng.randn(3, 3))
        Y = X @ Q + 0.01 * rng.randn(8, 3)
        result = leave_one_anchor_out_stability(X, Y, n_bootstrap=10, seed=42)
        assert "q_stability" in result
        assert "loo_mean_deviation" in result
        assert result["loo_mean_deviation"] < 1.0

    def test_too_few_anchors(self):
        X = np.array([[1, 0], [0, 1]])
        Y = np.array([[0, 1], [1, 0]])
        result = leave_one_anchor_out_stability(X, Y)
        assert np.isnan(result["q_stability"])
