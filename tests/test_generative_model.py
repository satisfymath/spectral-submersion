"""Tests for generative model: claim blocking, hypothesis cards."""
import pytest
import numpy as np
from spectral_submersion.generative_model import (
    GenerativeConfig,
    RongorongoGenerativeModel,
)
from spectral_submersion.claims import ClaimLevel


class TestGenerativeConfig:
    def test_default_blocks_c5(self):
        config = GenerativeConfig()
        assert config.block_c5_without_external
        assert config.max_claim_level == ClaimLevel.C2_FUNCTIONAL


class TestRongorongoGenerativeModel:
    def test_basic_hypothesis_generation(self):
        model = RongorongoGenerativeModel()
        n_src, n_tgt = 5, 5
        rng = np.random.RandomState(42)
        coupling = rng.dirichlet(np.ones(n_tgt), size=n_src)
        for i in range(n_src):
            coupling[i] /= coupling[i].sum()

        result = model.process_transport_hypotheses(
            coupling_matrix=coupling,
            source_tokens=["A", "B", "C", "D", "E"],
            target_tokens=["moon", "king", "fish", "bird", "sun"],
            anchor_power=0.1,
            bootstrap_stability=0.5,
            negative_control_gap=2.0,
        )
        assert len(result) == 5
        assert all("max_claim_level" in h for h in result)

    def test_claims_blocked_without_anchors(self):
        model = RongorongoGenerativeModel()
        n_src, n_tgt = 3, 3
        coupling = np.eye(n_tgt) * 0.8 + 0.1 / n_tgt

        result = model.process_transport_hypotheses(
            coupling_matrix=coupling,
            source_tokens=["200", "076", "300"],
            target_tokens=["ra", "ki", "ma"],
            anchor_power=0.0,
            bootstrap_stability=0.1,
            negative_control_gap=0.3,
        )
        for h in result:
            assert h["claim_blocked"] or h["max_claim_level"] in (
                "C0_PALEOGRAPHIC", "C1_STRUCTUREAL"
            )

    def test_format_hypothesis_card(self):
        model = RongorongoGenerativeModel()
        coupling = np.eye(2) * 0.8 + 0.1
        coupling /= coupling.sum(axis=1, keepdims=True)

        result = model.process_transport_hypotheses(
            coupling_matrix=coupling,
            source_tokens=["200", "076"],
            target_tokens=["moon", "king"],
            anchor_power=0.5,
            bootstrap_stability=0.7,
            negative_control_gap=2.5,
        )
        card = model.format_hypothesis_card(result[0])
        assert "HYPOTHESIS" in card
        assert "Max claim level" in card