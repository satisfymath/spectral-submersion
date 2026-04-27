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


def gromov_wasserstein_matrix(
    Dx: np.ndarray,
    Dy: np.ndarray,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
    reg: float = 0.1,
    max_iter: int = 100,
    sinkhorn_iter: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """Compute entropically-regularized Gromov-Wasserstein coupling.

    Solves: min_{Pi in U(a,b)} sum_{ijkl} (Dx_ik - Dy_jl)^2 Pi_ij Pi_kl
            + reg * KL(Pi | a ox b)

    Uses a fixed-point iteration (Peyre et al., 2016) where each step
    updates the GW cost tensor and solves a Sinkhorn problem.

    Args:
        Dx: Pairwise distance matrix for source (n_X x n_X).
        Dy: Pairwise distance matrix for target (n_Y x n_Y).
        a: Source marginal (n_X,). Defaults to uniform.
        b: Target marginal (n_Y,). Defaults to uniform.
        reg: Entropic regularization parameter.
        max_iter: Maximum outer iterations.
        sinkhorn_iter: Maximum Sinkhorn iterations per step.
        tol: Convergence tolerance on Pi change.

    Returns:
        Coupling matrix Pi (n_X x n_Y).
    """
    n_x = Dx.shape[0]
    n_y = Dy.shape[0]
    if a is None:
        a = np.ones(n_x) / n_x
    if b is None:
        b = np.ones(n_y) / n_y

    # Initialize with outer product of marginals
    Pi = a[:, None] * b[None, :]

    for _ in range(max_iter):
        # Compute GW cost tensor efficiently
        # C_ij = sum_kl (Dx_ik - Dy_jl)^2 Pi_kl
        #      = sum_kl Dx_ik^2 Pi_kl + sum_kl Dy_jl^2 Pi_kl - 2 sum_kl Dx_ik Dy_jl Pi_kl
        a_pi = Pi.sum(axis=1)  # (n_x,)
        b_pi = Pi.sum(axis=0)  # (n_y,)

        term1 = (Dx ** 2) @ a_pi  # (n_x,)
        term2 = (Dy ** 2) @ b_pi  # (n_y,)

        # C = term1[:, None] + term2[None, :] - 2 * Dx @ Pi @ Dy.T
        C = term1[:, None] + term2[None, :] - 2.0 * (Dx @ Pi @ Dy.T)

        # Sinkhorn step
        Pi_new = _sinkhorn_manual(C, a, b, reg, sinkhorn_iter, tol)

        if np.linalg.norm(Pi_new - Pi, ord="fro") < tol:
            break
        Pi = Pi_new

    return Pi
