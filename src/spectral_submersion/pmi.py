"""PMI / PPMI matrix computation."""
import numpy as np


def ppmi_matrix(C: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    """Compute Positive Pointwise Mutual Information (PPMI) matrix.

    Args:
        C: Co-occurrence matrix (non-negative).
        epsilon: Smoothing constant to avoid division by zero.

    Returns:
        PPMI matrix of same shape as C.
    """
    C = C.astype(float)
    total = C.sum() + epsilon

    Pij = (C + epsilon) / total
    Pi = Pij.sum(axis=1, keepdims=True)
    Pj = Pij.sum(axis=0, keepdims=True)

    PMI = np.log(Pij / (Pi * Pj + epsilon))
    PPMI = np.maximum(PMI, 0.0)

    return PPMI
