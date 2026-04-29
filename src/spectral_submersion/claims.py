"""Claim levels and admissibility for auditable hypothesis generation.

Implements the C0-C5 claim hierarchy (Proposition 2.1, Theorem 2.2) from the
PhD upgrade guide. Every hypothesis output must carry a maximum admissible
claim level determined by evidence, stability, anchor power, and controls.
"""
from __future__ import annotations

import enum


class ClaimLevel(enum.IntEnum):
    C0_PALEOGRAPHIC = 0
    C1_STRUCTUREAL = 1
    C2_FUNCTIONAL = 2
    C3_SEMANTIC_WEAK = 3
    C4_PHONETIC_PARTIAL = 4
    C5_TRANSLATION_STRONG = 5


CLAIM_LABELS = {
    ClaimLevel.C0_PALEOGRAPHIC: "paleographic",
    ClaimLevel.C1_STRUCTUREAL: "structural",
    ClaimLevel.C2_FUNCTIONAL: "functional",
    ClaimLevel.C3_SEMANTIC_WEAK: "semantic_weak",
    ClaimLevel.C4_PHONETIC_PARTIAL: "phonetic_partial",
    ClaimLevel.C5_TRANSLATION_STRONG: "translation_strong",
}

CLAIM_DESCRIPTIONS = {
    ClaimLevel.C0_PALEOGRAPHIC: "This stroke/glyph exists with probability p",
    ClaimLevel.C1_STRUCTUREAL: "Frequency, position, repetition, co-occurrence",
    ClaimLevel.C2_FUNCTIONAL: "Possible marker, classifier, numeral, determinative",
    ClaimLevel.C3_SEMANTIC_WEAK: "Compatible with lunar/genealogical/ritual domain",
    ClaimLevel.C4_PHONETIC_PARTIAL: "Compatible with syllable/sound under external anchor",
    ClaimLevel.C5_TRANSLATION_STRONG: "Verifiable reading",
}

FORBIDDEN_PER_LEVEL = {
    ClaimLevel.C0_PALEOGRAPHIC: ["meaning", "semantics"],
    ClaimLevel.C1_STRUCTUREAL: ["semantic function", "translation"],
    ClaimLevel.C2_FUNCTIONAL: ["lexical translation"],
    ClaimLevel.C3_SEMANTIC_WEAK: ["literal reading"],
    ClaimLevel.C4_PHONETIC_PARTIAL: ["complete decipherment"],
    ClaimLevel.C5_TRANSLATION_STRONG: [],
}

ANCHOR_LEVEL_REQUIREMENTS = {
    ClaimLevel.C0_PALEOGRAPHIC: 0.0,
    ClaimLevel.C1_STRUCTUREAL: 0.0,
    ClaimLevel.C2_FUNCTIONAL: 0.1,
    ClaimLevel.C3_SEMANTIC_WEAK: 0.3,
    ClaimLevel.C4_PHONETIC_PARTIAL: 0.5,
    ClaimLevel.C5_TRANSLATION_STRONG: 0.8,
}

STABILITY_REQUIREMENTS = {
    ClaimLevel.C0_PALEOGRAPHIC: 0.0,
    ClaimLevel.C1_STRUCTUREAL: 0.3,
    ClaimLevel.C2_FUNCTIONAL: 0.5,
    ClaimLevel.C3_SEMANTIC_WEAK: 0.7,
    ClaimLevel.C4_PHONETIC_PARTIAL: 0.8,
    ClaimLevel.C5_TRANSLATION_STRONG: 0.95,
}

NEGATIVE_CONTROL_GAP_REQUIREMENTS = {
    ClaimLevel.C0_PALEOGRAPHIC: 0.0,
    ClaimLevel.C1_STRUCTUREAL: 1.0,
    ClaimLevel.C2_FUNCTIONAL: 2.0,
    ClaimLevel.C3_SEMANTIC_WEAK: 3.0,
    ClaimLevel.C4_PHONETIC_PARTIAL: 4.0,
    ClaimLevel.C5_TRANSLATION_STRONG: 5.0,
}


def admissible(
    anchor_power: float,
    stability: float,
    neg_ctrl_gap: float,
    external_evidence: bool = False,
    max_level: ClaimLevel = ClaimLevel.C5_TRANSLATION_STRONG,
) -> ClaimLevel:
    """Determine maximum admissible claim level given evidence metrics.

    Admissibility is monotone: if level j is admissible, so is every level i < j.

    Args:
        anchor_power: AnchorPower metric in [0, 1]. 0 = no symmetry breaking.
        stability: Bootstrap stability in [0, 1].
        neg_ctrl_gap: Negative control gap in sigma units.
        external_evidence: Whether strong external evidence is present
            (required for C5).
        max_level: Upper bound on claim level (e.g., config may cap at C2).

    Returns:
        Maximum admissible ClaimLevel.
    """
    admissible_level = ClaimLevel.C0_PALEOGRAPHIC
    for level in ClaimLevel:
        if level.value > max_level.value:
            break
        if anchor_power < ANCHOR_LEVEL_REQUIREMENTS[level]:
            break
        if stability < STABILITY_REQUIREMENTS[level]:
            break
        if neg_ctrl_gap < NEGATIVE_CONTROL_GAP_REQUIREMENTS[level]:
            break
        if level == ClaimLevel.C5_TRANSLATION_STRONG and not external_evidence:
            break
        admissible_level = level
    return admissible_level


def overclaim_risk(claim_level: ClaimLevel, evidence_level: float) -> float:
    """Compute OverclaimRisk index.

    OverclaimRisk = ClaimLevel / (1 + EvidenceLevel).

    If > 1, the claim should be blocked.
    EvidenceLevel combines anchor power, stability, and control gap.

    Args:
        claim_level: The claim level being made.
        evidence_level: Combined evidence level (anchor + stability + gap).

    Returns:
        Overclaim risk ratio. Values > 1 indicate overclaim.
    """
    return claim_level.value / (1.0 + evidence_level)


def check_forbidden(claim_level: ClaimLevel, statement: str) -> list[str]:
    """Check if a statement contains forbidden terms for a given claim level.

    Args:
        claim_level: The claim level of the hypothesis.
        statement: The natural language statement.

    Returns:
        List of forbidden terms found in the statement.
    """
    found = []
    for forbidden in FORBIDDEN_PER_LEVEL[claim_level]:
        if forbidden.lower() in statement.lower():
            found.append(forbidden)
    return found