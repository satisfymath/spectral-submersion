"""PMI / PPMI matrix computation with context distribution smoothing."""

import numpy as np


def ppmi_matrix(
    C: np.ndarray, epsilon: float = 1e-9, alpha: float = 0.75
) -> np.ndarray:
    """Compute Positive Pointwise Mutual Information (PPMI) matrix.

    Applies context distribution smoothing (alpha parameter) as recommended
    by Levy & Goldberg (2015): P(j)^alpha is used instead of P(j) in the
    PMI denominator, down-weighting the effect of frequent context words.

    Args:
        C: Co-occurrence matrix (non-negative).
        epsilon: Smoothing constant to avoid division by zero.
        alpha: Context distribution smoothing parameter. alpha < 1 down-weights
               frequent context words. alpha=1.0 gives standard PMI.
               Recommended: 0.75 (Levy & Goldberg 2015).

    Returns:
        PPMI matrix of same shape as C.
    """
    C = C.astype(float)
    total = C.sum() + epsilon

    Pij = (C + epsilon) / total
    Pi = Pij.sum(axis=1, keepdims=True)
    Pj = Pij.sum(axis=0, keepdims=True)

    # Apply context distribution smoothing
    Pj_smooth = Pj**alpha
    Pj_smooth = Pj_smooth / Pj_smooth.sum()

    PMI = np.log(Pij / (Pi * Pj_smooth + epsilon))
    PPMI = np.maximum(PMI, 0.0)

    return PPMI
