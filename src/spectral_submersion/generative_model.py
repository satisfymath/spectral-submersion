"""Generative model for Rongorongo hypothesis generation.

Implements the probabilistic model from Section 10 of the guide:
p(X,Z,A,C,D,U,R | Y,K) = p(R|K) * p(Z,A|Y,R,K)
  * prod_t p(x_t|y_{A_t}, C_t, D_t, U_t, R, K)
           * p(C_t|C_{<t}, R, K)
           * p(D_t|t, artifact, K)
           * p(U_t|image, K)

The output is not a verified translation, but a ranked ledger of
falsifiable hypotheses with evidence, counterevidence, and uncertainty.
Section 11's impossibility theorem is enforced: claims C5 are blocked
unless strong external evidence is explicitly provided.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .claims import ClaimLevel, admissible
from .audit_metrics import HypothesisLedger


@dataclass
class GenerativeConfig:
    n_hypotheses: int = 10
    max_claim_level: ClaimLevel = ClaimLevel.C2_FUNCTIONAL
    require_negative_controls: bool = True
    require_anchor_for_c3: bool = True
    block_c5_without_external: bool = True
    min_bootstrap_stability: float = 0.3
    min_negative_control_gap: float = 2.0
    min_spectral_reliability: float = 0.3


class RongorongoGenerativeModel:
    """Generative model wrapper that processes transport hypotheses through
    the claim-admissibility filter.

    This model does NOT generate sequences directly. Instead, it takes
    the output of alignment/transport methods and:
    1. Assigns posterior scores from coupling matrices
    2. Evaluates evidence and counterevidence
    3. Computes anchor power, stability, and control gaps
    4. Determines maximum admissible claim level
    5. Blocks overclaiming automatically

    The impossibility theorem (Theorem 11.1) is enforced:
    without external anchors, only orbits are identifiable,
    not absolute semantic assignments.
    """

    def __init__(self, config: GenerativeConfig | None = None):
        self.config = config or GenerativeConfig()
        self.ledger = HypothesisLedger()

    def process_transport_hypotheses(
        self,
        coupling_matrix: np.ndarray,
        source_tokens: list[str],
        target_tokens: list[str],
        anchor_power: float = 0.0,
        bootstrap_stability: float = 0.0,
        negative_control_gap: float = 0.0,
        spectral_reliability: float = 0.0,
        external_evidence: bool = False,
        top_k: int = 5,
        config_hash: str = "",
    ) -> list[dict]:
        """Process transport coupling matrix into auditable hypotheses.

        For each source token, produce a hypothesis entry with:
        - Top-k candidate interpretations
        - Posterior score from coupling
        - Entropy (uncertainty)
        - Maximum admissible claim level
        - Evidence and counterevidence

        Args:
            coupling_matrix: Transport/alignment matrix (n_src x n_tgt).
            source_tokens: Source token labels.
            target_tokens: Target token labels.
            anchor_power: AnchorPower metric [0, 1].
            bootstrap_stability: Bootstrap stability [0, 1].
            negative_control_gap: NegCtrlGap in sigma units.
            spectral_reliability: SpectralReliability [0, 1].
            external_evidence: Whether strong external evidence is present.
            top_k: Number of top candidates per source token.
            config_hash: Hash of the configuration for reproducibility.

        Returns:
            List of hypothesis dicts with audited claims.
        """
        max_admissible = admissible(
            anchor_power=anchor_power,
            stability=bootstrap_stability,
            neg_ctrl_gap=negative_control_gap,
            external_evidence=external_evidence,
            max_level=self.config.max_claim_level,
        )

        raw_hypotheses = []
        for i, src in enumerate(source_tokens):
            probs = coupling_matrix[i]
            prob_sum = probs.sum()
            if prob_sum > 0:
                probs_norm = probs / prob_sum
            else:
                probs_norm = np.ones_like(probs) / len(probs)

            top_idx = np.argsort(-probs_norm)[:top_k]
            candidates = [
                {
                    "target": target_tokens[j],
                    "score": float(probs_norm[j]),
                    "raw_coupling": float(probs[j]),
                }
                for j in top_idx
            ]

            entropy = float(
                -(probs_norm[probs_norm > 0] * np.log(probs_norm[probs_norm > 0])).sum()
            )

            evidence = []
            counterevidence = []

            if anchor_power > 0.1:
                evidence.append(
                    {
                        "type": "anchor_power",
                        "description": f"AnchorPower={anchor_power:.3f} breaks some symmetries",
                        "score": anchor_power,
                    }
                )
            else:
                counterevidence.append(
                    {
                        "type": "no_anchors",
                        "description": "No anchors: only orbit-identifiable claims",
                    }
                )

            if bootstrap_stability > 0.5:
                evidence.append(
                    {
                        "type": "stability",
                        "description": f"Bootstrap stability={bootstrap_stability:.3f}",
                        "score": bootstrap_stability,
                    }
                )
            elif bootstrap_stability < 0.3:
                counterevidence.append(
                    {
                        "type": "unstable",
                        "description": f"Low stability={bootstrap_stability:.3f}: result is noise",
                    }
                )

            if negative_control_gap > 2.0:
                evidence.append(
                    {
                        "type": "negative_control",
                        "description": f"Controls exceeded by {negative_control_gap:.1f} sigma",
                        "score": negative_control_gap,
                    }
                )
            elif negative_control_gap < 1.0:
                counterevidence.append(
                    {
                        "type": "weak_controls",
                        "description": f"NegCtrlGap={negative_control_gap:.1f} sigma: insufficient evidence",
                    }
                )

            if entropy > 3.0:
                counterevidence.append(
                    {
                        "type": "high_entropy",
                        "description": f"Entropy={entropy:.2f}: near-uniform distribution",
                    }
                )

            result = self.ledger.add_hypothesis(
                glyph_or_sequence=[src],
                candidate_interpretations=candidates,
                posterior_score=float(np.max(probs_norm)),
                claim_level=max_admissible.name,
                evidence=evidence,
                counterevidence=counterevidence,
                anchor_power=anchor_power,
                bootstrap_stability=bootstrap_stability,
                negative_control_gap=negative_control_gap,
                spectral_reliability=spectral_reliability,
                config_hash=config_hash,
            )

            raw_hypotheses.append(
                {
                    "source_token": src,
                    "candidates": candidates,
                    "entropy": entropy,
                    "posterior_top1": float(np.max(probs_norm)),
                    "max_claim_level": max_admissible.name,
                    "claim_blocked": result["blocked"],
                    "overclaim_risk": result["overclaim_risk"],
                    "forbidden_claims": result["forbidden_claims"],
                }
            )

        return raw_hypotheses

    def format_hypothesis_card(
        self,
        hypothesis: dict,
    ) -> str:
        """Format a hypothesis as a human-readable card.

        Every card explicitly states:
        1. What is claimed
        2. Maximum admissible claim level
        3. Evidence and counterevidence
        4. Forbidden interpretations
        5. Overclaim risk

        Args:
            hypothesis: Dict from process_transport_hypotheses.

        Returns:
            Formatted string.
        """
        lines = []
        lines.append(f"= HYPOTHESIS: {hypothesis['source_token']} =")
        lines.append(f"Max claim level: {hypothesis['max_claim_level']}")
        lines.append(f"Blocked: {hypothesis['claim_blocked']}")
        lines.append(f"Overclaim risk: {hypothesis['overclaim_risk']:.3f}")
        lines.append(f"Forbidden claims: {hypothesis['forbidden_claims']}")
        lines.append("")
        lines.append("Top candidates:")
        for c in hypothesis["candidates"]:
            lines.append(f"  {c['target']}: {c['score']:.4f}")
        lines.append("")
        lines.append(f"Entropy: {hypothesis['entropy']:.3f}")
        lines.append(f"Posterior (top-1): {hypothesis['posterior_top1']:.4f}")
        return "\n".join(lines)

    def save_ledger(self, path: str) -> None:
        self.ledger.save(path)

    def ledger_summary(self) -> dict:
        return self.ledger.summary()
