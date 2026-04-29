"""Tests for stability module: spectral reliability, SPPMI, co-occurrence coverage."""
import pytest
import numpy as np
from spectral_submersion.stability import (
    spectral_gap,
    spectral_reliability,
    cooccurrence_coverage,
    expected_pair_count,
    sceptmi_matrix,
    pmi_sensitivity,
    spectral_rejection_rule,
)


class TestSpectralGap:
    def test_clear_gap(self):
        sv = np.array([10, 5, 3, 0.1, 0.05])
        gap = spectral_gap(sv, 3)
        assert gap == pytest.approx(3.0 - 0.1)

    def test_no_gap(self):
        sv = np.array([5, 5, 5, 5])
        gap = spectral_gap(sv, 2)
        assert gap == pytest.approx(0.0)

    def test_k_beyond_length(self):
        sv = np.array([10, 5])
        gap = spectral_gap(sv, 5)
        assert gap == 0.0


class TestSpectralReliability:
    def test_high_reliability(self):
        sv = np.array([100, 90, 50, 10, 1])
        reliability = spectral_reliability(sv, 2, bootstrap_error=5.0)
        assert reliability > 0.5

    def test_low_reliability(self):
        sv = np.array([10, 9.5, 9.0, 8.5, 8.0])
        reliability = spectral_reliability(sv, 2, bootstrap_error=5.0)
        assert reliability < 0.3

    def test_zero_gap(self):
        sv = np.array([5, 5, 5, 5])
        reliability = spectral_reliability(sv, 2, bootstrap_error=0.5)
        assert reliability == 0.0

    def test_zero_error(self):
        sv = np.array([10, 5, 3, 1, 0.1])
        reliability = spectral_reliability(sv, 2, bootstrap_error=0.0)
        assert reliability == 1.0


class TestCooccurrenceCoverage:
    def test_full_coverage(self):
        C = np.ones((5, 5))
        cov = cooccurrence_coverage(C)
        assert cov == pytest.approx(1.0)

    def test_sparse_coverage(self):
        C = np.zeros((10, 10))
        C[0, 1] = 1
        cov = cooccurrence_coverage(C)
        assert cov == pytest.approx(1 / 100)

    def test_empty_matrix(self):
        C = np.zeros((3, 3))
        assert cooccurrence_coverage(C) == 0.0


class TestExpectedPairCount:
    def test_basic(self):
        epc = expected_pair_count(total_tokens=10000, window_size=3, vocab_size=100)
        assert epc > 0

    def test_large_vocab_low_count(self):
        epc = expected_pair_count(total_tokens=100, window_size=2, vocab_size=500)
        assert epc < 1.0

    def test_small_vocab_high_count(self):
        epc = expected_pair_count(total_tokens=100000, window_size=5, vocab_size=50)
        assert epc > 10.0


class TestSPPMI:
    def test_output_non_negative(self):
        C = np.random.RandomState(42).randn(10, 10)
        C = np.abs(C) + 0.1
        S = sceptmi_matrix(C, epsilon=0.1, prior_type="uniform")
        assert (S >= 0).all()

    def test_marginal_prior_preserves_structure(self):
        rng = np.random.RandomState(42)
        C = rng.randint(1, 10, size=(5, 5))
        S_uni = sceptmi_matrix(C, epsilon=0.1, prior_type="uniform")
        S_mar = sceptmi_matrix(C, epsilon=0.1, prior_type="marginal_product")
        assert S_uni.shape == S_mar.shape
        assert not np.allclose(S_uni, S_mar)

    def test_diagonal_prior(self):
        C = np.random.RandomState(42).randn(5, 5)
        C = np.abs(C) + 0.1
        S = sceptmi_matrix(C, epsilon=0.5, prior_type="diagonal")
        assert S.shape == (5, 5)

    def test_unknown_prior_raises(self):
        C = np.eye(3)
        with pytest.raises(ValueError):
            sceptmi_matrix(C, prior_type="unknown")

    def test_sppmi_different_from_ppmi(self):
        from spectral_submersion.pmi import ppmi_matrix
        C = np.random.RandomState(42).randint(1, 10, size=(8, 8)).astype(float)
        ppmi = ppmi_matrix(C)
        sppmi = sceptmi_matrix(C, epsilon=1.0)
        assert not np.allclose(ppmi, sppmi)


class TestPMISensitivity:
    def test_high_sensitivity_sparse(self):
        p_ij = np.full((5, 5), 0.01)
        p_ij[0, 0] = 0.9
        p_ij /= p_ij.sum()
        p_i = p_ij.sum(axis=1)
        p_j = p_ij.sum(axis=0)
        result = pmi_sensitivity(p_ij, p_i, p_j)
        assert result["max_sensitivity"] > 10.0

    def test_low_sensitivity_dense(self):
        p_ij = np.ones((3, 3)) / 9.0
        p_i = p_ij.sum(axis=1)
        p_j = p_ij.sum(axis=0)
        result = pmi_sensitivity(p_ij, p_i, p_j)
        assert result["max_sensitivity"] < 20.0


class TestSpectralRejectionRule:
    def test_stable_dimensions(self):
        sv = np.array([100, 80, 50, 20, 5, 1, 0.5, 0.1])
        errors = np.ones(8) * 2.0
        results = spectral_rejection_rule(sv, errors, k_values=[2, 4])
        assert results[0]["stable"]
        assert results[0]["delta_k"] > 0

    def test_unstable_dimensions(self):
        sv = np.array([5, 4.5, 4.0, 3.8, 3.6, 3.4, 3.2, 3.0])
        errors = np.ones(8) * 1.0
        results = spectral_rejection_rule(sv, errors, k_values=[4])
        assert not results[0]["stable"]