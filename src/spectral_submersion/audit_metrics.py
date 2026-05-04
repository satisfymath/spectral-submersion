"""Audit metrics: negative control gap, calibration, overclaim risk, hypothesis ledger.

Implements the auditability framework from Part V of the PhD upgrade guide:
- Section 23: NegCtrlGap (gap against negative controls)
- Section 24: Bootstrap stability
- Section 25: ECE calibration
- Section 26: Overclaim Risk Index
- Hypothesis ledger with full provenance
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict

import numpy as np


def negative_control_gap(
    score_real: float,
    scores_negative: np.ndarray,
) -> dict:
    """Compute NegCtrlGap(S) from Section 23.

    NegCtrlGap(S) = (S(X) - E_{H0}[S]) / sd_{H0}[S]

    Interpretation:
    - < 1: no evidence
    - 1-2: weak
    - 2-3: moderate
    - 3-5: strong
    - > 5: very strong (but check for leakage)

    Args:
        score_real: Score on the real corpus.
        scores_negative: Array of scores on negative control corpora.

    Returns:
        Dict with gap, mean, std, and interpretation.
    """
    mu_neg = float(np.mean(scores_negative))
    std_neg = float(np.std(scores_negative))

    if std_neg < 1e-12:
        gap = float("inf") if score_real > mu_neg else 0.0
    else:
        gap = float((score_real - mu_neg) / std_neg)

    if gap < 1:
        interpretation = "no_evidence"
    elif gap < 2:
        interpretation = "weak"
    elif gap < 3:
        interpretation = "moderate"
    elif gap < 5:
        interpretation = "strong"
    else:
        interpretation = "very_strong_check_leakage"

    return {
        "gap": gap,
        "negative_mean": mu_neg,
        "negative_std": std_neg,
        "real_score": score_real,
        "interpretation": interpretation,
    }


def bootstrap_stability(
    scores: np.ndarray,
) -> float:
    """Compute bootstrap stability from Section 24.

    Stability(h) = E_{b,b'}[ sim(h^(b), h^(b')) ]

    For distributions: sim(Pi, Pi') = 1 - 0.5 * ||Pi - Pi'||_1

    Args:
        scores: Array of scores from bootstrap resamples.

    Returns:
        Mean pairwise similarity.
    """
    n = len(scores)
    if n < 2:
        return float("nan")

    scores = np.asarray(scores)
    # mean_val = float(np.mean(scores))
    std_val = float(np.std(scores))
    if std_val < 1e-12:
        return 1.0

    coefs = []
    for i in range(min(n, 100)):
        for j in range(i + 1, min(n, 100)):
            mean_pair = 0.5 * (scores[i] + scores[j])
            sim = 1.0 - 0.5 * abs(scores[i] - scores[j]) / (mean_pair + 1e-128)
            coefs.append(sim)
    return float(np.mean(coefs))


def bootstrap_coupling_stability(
    couplings: list[np.ndarray],
) -> dict:
    """Compute coupling stability from bootstrap samples.

    OTStability = E_{b,b'}[ ||Pi^(b) - Pi^(b')||_1 ]

    Args:
        couplings: List of transport plan matrices from bootstrap.

    Returns:
        Dict with mean_l1_distance and pairwise_stability.
    """
    n = len(couplings)
    if n < 2:
        return {"mean_l1_distance": float("nan"), "pairwise_stability": float("nan")}

    l1_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            l1_dists.append(float(np.linalg.norm(couplings[i] - couplings[j], 1)))

    mean_l1 = float(np.mean(l1_dists))

    total_mass = sum(c.sum() for c in couplings) / n
    stability = max(0.0, 1.0 - mean_l1 / (2 * total_mass + 1e-128))

    return {
        "mean_l1_distance": mean_l1,
        "pairwise_stability": float(stability),
    }


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute ECE (Expected Calibration Error) from Section 25.

    ECE = sum_m |B_m|/N * |acc(B_m) - conf(B_m)|

    On real Rongorongo there's no ground truth, so calibrate on
    synthetic data and report as "simulation-calibrated scores".

    Args:
        probabilities: Predicted probabilities (N,).
        labels: Binary labels (N,), 1 if correct correspondence.
        n_bins: Number of calibration bins.

    Returns:
        Dict with ECE, per-bin accuracy, per-bin confidence.
    """
    probs = np.asarray(probabilities)
    labs = np.asarray(labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_stats = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs > lo) & (probs <= hi)
        n_bin = mask.sum()
        if n_bin == 0:
            bin_stats.append({"bin": i, "n": 0, "accuracy": 0, "confidence": 0})
            continue

        bin_acc = float(labs[mask].mean())
        bin_conf = float(probs[mask].mean())
        ece += (n_bin / len(probs)) * abs(bin_acc - bin_conf)
        bin_stats.append(
            {
                "bin": i,
                "n": int(n_bin),
                "accuracy": bin_acc,
                "confidence": bin_conf,
            }
        )

    return {
        "ece": float(ece),
        "n_bins": n_bins,
        "bin_stats": bin_stats,
    }


@dataclass
class HypothesisEntry:
    glyph_or_sequence: list[str]
    candidate_interpretations: list[dict]
    posterior_score: float
    claim_level: str
    evidence: list[dict] = field(default_factory=list)
    counterevidence: list[dict] = field(default_factory=list)
    anchor_power: float = 0.0
    bootstrap_stability: float = 0.0
    negative_control_gap: float = 0.0
    spectral_reliability: float = 0.0
    forbidden_claims: list[str] = field(default_factory=list)
    overclaim_risk: float = 0.0
    config_hash: str = ""
    run_id: str = ""


class HypothesisLedger:
    """Auditable hypothesis ledger from Section 34 (Principle).

    Every hypothesis must be reported as a tuple:
    h = (x, Y_h, p_h, E_h, B_h, l_h, R_h)

    The ledger enforces that no hypothesis exceeds its admissible
    claim level based on evidence, stability, anchors, and controls.
    """

    def __init__(self, config: dict | None = None):
        self.hypotheses: list[HypothesisEntry] = []
        self.config = config or {}
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def add_hypothesis(
        self,
        glyph_or_sequence: list[str],
        candidate_interpretations: list[dict],
        posterior_score: float,
        claim_level: str,
        evidence: list[dict] | None = None,
        counterevidence: list[dict] | None = None,
        anchor_power: float = 0.0,
        bootstrap_stability: float = 0.0,
        negative_control_gap: float = 0.0,
        spectral_reliability: float = 0.0,
        config_hash: str = "",
    ) -> dict:
        from .claims import ClaimLevel, admissible, overclaim_risk, FORBIDDEN_PER_LEVEL

        try:
            level_enum = ClaimLevel[claim_level]
        except (KeyError, ValueError):
            level_enum = ClaimLevel.C0_PALEOGRAPHIC

        max_admissible = admissible(
            anchor_power=anchor_power,
            stability=bootstrap_stability,
            neg_ctrl_gap=negative_control_gap,
            external_evidence=False,
        )

        evidence_level = (
            anchor_power
            + bootstrap_stability
            + min(negative_control_gap / 5.0, 1.0)
            + spectral_reliability
        )
        ocr = overclaim_risk(level_enum, evidence_level)

        blocked = level_enum.value > max_admissible.value
        forbidden = FORBIDDEN_PER_LEVEL.get(level_enum, [])

        entry = HypothesisEntry(
            glyph_or_sequence=glyph_or_sequence,
            candidate_interpretations=candidate_interpretations,
            posterior_score=posterior_score,
            claim_level=claim_level,
            evidence=evidence or [],
            counterevidence=counterevidence or [],
            anchor_power=anchor_power,
            bootstrap_stability=bootstrap_stability,
            negative_control_gap=negative_control_gap,
            spectral_reliability=spectral_reliability,
            forbidden_claims=forbidden,
            overclaim_risk=ocr,
            config_hash=config_hash,
            run_id=self.run_id,
        )

        self.hypotheses.append(entry)

        return {
            "hypothesis_id": f"HYP_{len(self.hypotheses):06d}",
            "claim_level_requested": claim_level,
            "claim_level_admissible": max_admissible.name,
            "blocked": blocked,
            "overclaim_risk": ocr,
            "forbidden_claims": forbidden,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        entries = []
        for h in self.hypotheses:
            d = asdict(h)
            entries.append(d)
        with open(path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def summary(self) -> dict:
        from .claims import ClaimLevel

        level_counts = {}
        blocked_counts = 0
        for h in self.hypotheses:
            try:
                level_enum = ClaimLevel[h.claim_level]
            except (KeyError, ValueError):
                level_enum = ClaimLevel.C0_PALEOGRAPHIC
            key = level_enum.name
            level_counts[key] = level_counts.get(key, 0) + 1
            from .claims import admissible

            max_adm = admissible(
                anchor_power=h.anchor_power,
                stability=h.bootstrap_stability,
                neg_ctrl_gap=h.negative_control_gap,
            )
            if level_enum.value > max_adm.value:
                blocked_counts += 1

        return {
            "total_hypotheses": len(self.hypotheses),
            "level_counts": level_counts,
            "blocked_count": blocked_counts,
            "mean_overclaim_risk": (
                float(np.mean([h.overclaim_risk for h in self.hypotheses]))
                if self.hypotheses
                else 0.0
            ),
        }
