"""Alignment utilities: Procrustes, Moore-Penrose, and soft dictionary."""
import numpy as np


def orthogonal_procrustes(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Solve min_Q ||XQ - Y||_F subject to Q.T Q = I.

    Args:
        X: Source embeddings (n x d_x).
        Y: Target embeddings (n x d_y), assumed d_x == d_y.

    Returns:
        Orthogonal matrix Q of shape (d_x, d_y).
    """
    M = X.T @ Y
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    Q = U @ Vt
    return Q


def pairwise_squared_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute pairwise squared Euclidean distances between rows of X and Y."""
    X_norm = (X ** 2).sum(axis=1, keepdims=True)
    Y_norm = (Y ** 2).sum(axis=1, keepdims=True).T
    return X_norm + Y_norm - 2 * X @ Y.T


def soft_dictionary(D: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Convert distance matrix to soft nearest-neighbor probabilities.

    Args:
        D: Pairwise distance matrix (n_X x n_Y).
        temperature: Softmax temperature.

    Returns:
        Probability matrix Pi where Pi[i,j] ~ exp(-D[i,j] / tau).
    """
    logits = -D / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)
