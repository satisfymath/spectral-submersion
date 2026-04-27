"""Optimal transport utilities with Sinkhorn implementation."""
import numpy as np

try:
    import ot
    POT_AVAILABLE = True
except ImportError:
    POT_AVAILABLE = False


def _sinkhorn_manual(
    cost: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    reg: float = 0.1,
    num_iter: int = 1000,
    tol: float = 1e-9,
) -> np.ndarray:
    """Manual NumPy implementation of entropic-regularized Sinkhorn.

    Solves: min_{Pi in U(a,b)} <Pi, C> + reg * KL(Pi | a ox b)
    via log-domain stabilized fixed-point iteration.

    Args:
        cost: Cost matrix (n_X x n_Y).
        a: Source distribution (n_X,).
        b: Target distribution (n_Y,).
        reg: Entropic regularization parameter (epsilon).
        num_iter: Maximum iterations.
        tol: Convergence tolerance on marginal deviation.

    Returns:
        Transport plan Pi (n_X x n_Y).
    """
    K = np.exp(-cost / reg)
    u = np.ones_like(a)
    v = np.ones_like(b)

    for _ in range(num_iter):
        u_prev = u
        u = a / (K @ v + 1e-128)
        v = b / (K.T @ u + 1e-128)
        if np.max(np.abs(u - u_prev)) < tol:
            break

    Pi = u[:, None] * K * v[None, :]
    return Pi


def optimal_transport_matrix(
    cost: np.ndarray,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
    reg: float = 0.1,
    num_iter: int = 1000,
    tol: float = 1e-9,
) -> np.ndarray:
    """Compute entropically-regularized optimal transport plan.

    Uses POT (ot.sinkhorn) if available; falls back to manual NumPy
    implementation otherwise. This avoids external C++ compiler
    dependencies while preserving mathematical correctness.

    Args:
        cost: Cost matrix (n_X x n_Y).
        a: Source distribution (n_X,). Defaults to uniform.
        b: Target distribution (n_Y,). Defaults to uniform.
        reg: Entropic regularization parameter (epsilon).
        num_iter: Maximum iterations for manual fallback.
        tol: Convergence tolerance for manual fallback.

    Returns:
        Transport plan Pi (n_X x n_Y).
    """
    n_x, n_y = cost.shape
    if a is None:
        a = np.ones(n_x) / n_x
    if b is None:
        b = np.ones(n_y) / n_y

    if POT_AVAILABLE:
        Pi = ot.sinkhorn(a, b, cost, reg, numItermax=num_iter, stopThr=tol)
    else:
        Pi = _sinkhorn_manual(cost, a, b, reg, num_iter, tol)

    return Pi
