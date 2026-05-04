"""Tests for audit metrics: NegCtrlGap, ECE, HypothesisLedger."""

import pytest
import numpy as np
from spectral_submersion.audit_metrics import (
    negative_control_gap,
    bootstrap_stability,
    bootstrap_coupling_stability,
    expected_calibration_error,
    HypothesisLedger,
)


class TestNegativeControlGap:
    def test_strong_gap(self):
        result = negative_control_gap(5.0, np.array([0.0, 0.1, 0.2, -0.1, 0.3]))
        assert result["gap"] > 3.0
        assert result["interpretation"] in ("strong", "very_strong_check_leakage")

    def test_no_gap(self):
        result = negative_control_gap(0.5, np.array([0.4, 0.6, 0.3, 0.7, 0.5]))
        assert result["gap"] < 1.0
        assert result["interpretation"] == "no_evidence"

    def test_moderate_gap(self):
        result = negative_control_gap(
            2.5, np.random.RandomState(42).normal(0, 0.5, 100)
        )
        assert 1.0 <= result["gap"]

    def test_zero_std(self):
        result = negative_control_gap(5.0, np.ones(10))
        assert result["gap"] == float("inf")


class TestBootstrapStability:
    def test_identical_scores(self):
        scores = np.ones(10)
        result = bootstrap_stability(scores)
        assert result == pytest.approx(1.0)

    def test_varying_scores(self):
        rng = np.random.RandomState(42)
        scores = rng.uniform(0.5, 0.9, 20)
        result = bootstrap_stability(scores)
        assert 0.0 < result < 1.0


class TestBootstrapCouplingStability:
    def test_identical_couplings(self):
        C = np.eye(5) / 5
        result = bootstrap_coupling_stability([C, C, C])
        assert result["mean_l1_distance"] == pytest.approx(0.0)
        assert result["pairwise_stability"] == pytest.approx(1.0)

    def test_different_couplings(self):
        rng = np.random.RandomState(42)
        couplings = [rng.dirichlet(np.ones(5), size=5) for _ in range(5)]
        for c in couplings:
            c /= c.sum()
        result = bootstrap_coupling_stability(couplings)
        assert result["mean_l1_distance"] > 0


class TestECE:
    def test_perfect_calibration(self):
        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        labels = (probs > 0.5).astype(int)
        result = expected_calibration_error(probs, labels)
        assert result["ece"] < 0.5

    def test_random_calibration(self):
        rng = np.random.RandomState(42)
        probs = rng.uniform(0, 1, 100)
        labels = rng.binomial(1, 0.5, 100)
        result = expected_calibration_error(probs, labels)
        assert result["ece"] >= 0.0


class TestHypothesisLedger:
    def test_add_hypothesis(self):
        ledger = HypothesisLedger()
        result = ledger.add_hypothesis(
            glyph_or_sequence=["200"],
            candidate_interpretations=[{"target": "king", "score": 0.8}],
            posterior_score=0.8,
            claim_level="C1_STRUCTUREAL",
            anchor_power=0.1,
            bootstrap_stability=0.5,
            negative_control_gap=1.5,
        )
        assert result["hypothesis_id"] == "HYP_000001"
        assert not result["blocked"]

    def test_block_overclaim(self):
        ledger = HypothesisLedger()
        result = ledger.add_hypothesis(
            glyph_or_sequence=["200"],
            candidate_interpretations=[{"target": "king", "score": 0.3}],
            posterior_score=0.3,
            claim_level="C5_TRANSLATION_STRONG",
            anchor_power=0.0,
            bootstrap_stability=0.1,
            negative_control_gap=0.5,
        )
        assert result["blocked"]

    def test_save_and_summary(self, tmp_path):
        ledger = HypothesisLedger()
        ledger.add_hypothesis(
            glyph_or_sequence=["200"],
            candidate_interpretations=[{"target": "moon", "score": 0.6}],
            posterior_score=0.6,
            claim_level="C2_FUNCTIONAL",
            anchor_power=0.1,
            bootstrap_stability=0.5,
            negative_control_gap=2.0,
        )
        ledger.save(tmp_path / "test_ledger.jsonl")
        assert (tmp_path / "test_ledger.jsonl").exists()

        summary = ledger.summary()
        assert summary["total_hypotheses"] == 1
