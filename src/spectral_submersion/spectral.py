"""Spectral embedding and SVD utilities."""
import numpy as np
from sklearn.utils.extmath import randomized_svd


def spectral_embedding(
    M: np.ndarray,
    k: int = 16,
    alpha: float = 0.5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute truncated SVD and construct spectral embedding.

    Args:
        M: Input matrix (e.g., PPMI).
        k: Number of latent dimensions.
        alpha: Exponent for singular values (0=directions only, 1=full weight).
        random_state: Random seed for randomized solver.

    Returns:
        Tuple of (embedding E, singular values S, right singular vectors Vt).
    """
    U, S, Vt = randomized_svd(M, n_components=k, random_state=random_state)
    E = U @ np.diag(S ** alpha)
    return E, S, Vt


def effective_rank(singular_values: np.ndarray) -> float:
    """Compute effective (entropic) rank from singular values."""
    s = np.asarray(singular_values, dtype=float)
    s = s[s > 0]
    if s.sum() == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))
