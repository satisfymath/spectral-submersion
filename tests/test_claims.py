"""Tests for claims module: claim levels, admissibility, overclaim risk."""

from spectral_submersion.claims import (
    ClaimLevel,
    CLAIM_LABELS,
    CLAIM_DESCRIPTIONS,
    FORBIDDEN_PER_LEVEL,
    ANCHOR_LEVEL_REQUIREMENTS,
    STABILITY_REQUIREMENTS,
    NEGATIVE_CONTROL_GAP_REQUIREMENTS,
    admissible,
    overclaim_risk,
    check_forbidden,
)


class TestClaimLevelEnum:
    def test_claim_levels_ordered(self):
        assert ClaimLevel.C0_PALEOGRAPHIC < ClaimLevel.C1_STRUCTUREAL
        assert ClaimLevel.C1_STRUCTUREAL < ClaimLevel.C2_FUNCTIONAL
        assert ClaimLevel.C2_FUNCTIONAL < ClaimLevel.C3_SEMANTIC_WEAK
        assert ClaimLevel.C3_SEMANTIC_WEAK < ClaimLevel.C4_PHONETIC_PARTIAL
        assert ClaimLevel.C4_PHONETIC_PARTIAL < ClaimLevel.C5_TRANSLATION_STRONG

    def test_all_levels_have_labels(self):
        for level in ClaimLevel:
            assert level in CLAIM_LABELS
            assert level in CLAIM_DESCRIPTIONS
            assert level in FORBIDDEN_PER_LEVEL

    def test_requirements_monotone(self):
        anchor = [ANCHOR_LEVEL_REQUIREMENTS[label] for label in ClaimLevel]
        stability = [STABILITY_REQUIREMENTS[label] for label in ClaimLevel]
        gap = [NEGATIVE_CONTROL_GAP_REQUIREMENTS[label] for label in ClaimLevel]
        assert anchor == sorted(anchor)
        assert stability == sorted(stability)
        assert gap == sorted(gap)


class TestAdmissible:
    def test_no_evidence_yields_c0(self):
        level = admissible(anchor_power=0.0, stability=0.0, neg_ctrl_gap=0.0)
        assert level == ClaimLevel.C0_PALEOGRAPHIC

    def test_moderate_evidence_yields_c1(self):
        level = admissible(anchor_power=0.0, stability=0.5, neg_ctrl_gap=1.5)
        assert level == ClaimLevel.C1_STRUCTUREAL

    def test_strong_evidence_without_external_caps_at_c4(self):
        level = admissible(
            anchor_power=0.8,
            stability=0.9,
            neg_ctrl_gap=4.5,
            external_evidence=False,
        )
        assert level == ClaimLevel.C4_PHONETIC_PARTIAL

    def test_c5_requires_external_evidence(self):
        level = admissible(
            anchor_power=0.9,
            stability=0.98,
            neg_ctrl_gap=6.0,
            external_evidence=True,
        )
        assert level == ClaimLevel.C5_TRANSLATION_STRONG

    def test_c5_blocked_without_external(self):
        level = admissible(
            anchor_power=0.9,
            stability=0.98,
            neg_ctrl_gap=6.0,
            external_evidence=False,
        )
        assert level == ClaimLevel.C4_PHONETIC_PARTIAL

    def test_monotonicity(self):
        level_low = admissible(anchor_power=0.0, stability=0.1, neg_ctrl_gap=0.5)
        level_high = admissible(anchor_power=0.5, stability=0.8, neg_ctrl_gap=3.0)
        assert level_high > level_low

    def test_max_level_override(self):
        level = admissible(
            anchor_power=0.9,
            stability=0.98,
            neg_ctrl_gap=6.0,
            external_evidence=True,
            max_level=ClaimLevel.C2_FUNCTIONAL,
        )
        assert level <= ClaimLevel.C2_FUNCTIONAL


class TestOverclaimRisk:
    def test_high_evidence_low_risk(self):
        risk = overclaim_risk(ClaimLevel.C1_STRUCTUREAL, 3.0)
        assert risk < 1.0

    def test_low_evidence_high_risk(self):
        risk = overclaim_risk(ClaimLevel.C5_TRANSLATION_STRONG, 0.1)
        assert risk > 1.0

    def test_zero_evidence(self):
        risk = overclaim_risk(ClaimLevel.C2_FUNCTIONAL, 0.0)
        assert risk > 1.0


class TestCheckForbidden:
    def test_forbidden_in_statement(self):
        found = check_forbidden(ClaimLevel.C0_PALEOGRAPHIC, "This glyph has meaning")
        assert len(found) >= 1

    def test_no_forbidden(self):
        found = check_forbidden(ClaimLevel.C5_TRANSLATION_STRONG, "verified reading")
        assert len(found) == 0
