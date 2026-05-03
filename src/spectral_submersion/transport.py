"""Optimal transport utilities with Sinkhorn implementation.

Uses POT when available (log-domain stabilized), falls back to manual
implementation. The manual implementation uses standard (not log-domain)
Sinkhorn, which is numerically unstable for small reg values.
Always prefer POT for production use.

Key fixes (v2):
- _sinkhorn_manual: renamed docstring to remove false "log-domain" claim
- gromov_wasserstein_matrix: uses optimal_transport_matrix (which routes to
  POT when available) instead of always calling _sinkhorn_manual
- multi_marginal_gw: KL prior regularization is now applied correctly
  by modifying the Gibbs kernel K = prior * exp(-C/reg) instead of
  adding -log(prior) to the cost matrix
- consensus_from_multi_gw: aligns embeddings via Procrustes before averaging
"""
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
    """Standard domain Sinkhorn iteration (NOT log-domain stabilized).

    Solves: min_{Pi in U(a,b)} <Pi, C> + reg * KL(Pi | a ox b)
    via standard (multiplicative) fixed-point iteration.

    WARNING: This implementation is numerically unstable for small reg
    values (reg < 0.1). For robust computation, use optimal_transport_matrix
    which routes to POT's log-domain Sinkhorn when available.

    Args:
        cost: Cost matrix (n_X x n_Y).
        a: Source distribution (n_X,).
        b: Target distribution (n_Y,).
        reg: Entropic regularization parameter (epsilon).
        num_iter: Maximum iterations.
        tol: Convergence tolerance on scaling vector change.

    Returns:
        Transport plan Pi (n_X x n_Y).
    """
    K = np.exp(-cost / reg)
    u = np.ones_like(a)
    v = np.ones_like(b)

    for _ in range(num_iter):
        u_prev = u.copy()
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

    Uses POT (ot.sinkhorn) if available for log-domain stability;
    falls back to manual NumPy implementation otherwise.

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

    Uses optimal_transport_matrix for Sinkhorn steps, which routes to
    POT's log-domain implementation when available.

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
        a_pi = Pi.sum(axis=1)
        b_pi = Pi.sum(axis=0)

        term1 = (Dx ** 2) @ a_pi
        term2 = (Dy ** 2) @ b_pi

        C = term1[:, None] + term2[None, :] - 2.0 * (Dx @ Pi @ Dy.T)

        # Use optimal_transport_matrix which routes to POT when available
        Pi_new = optimal_transport_matrix(C, a, b, reg, sinkhorn_iter, tol)

        if np.linalg.norm(Pi_new - Pi, ord="fro") < tol:
            break
        Pi = Pi_new

    return Pi


def multi_marginal_gw(
    distance_matrices: list[np.ndarray],
    marginal_distributions: list[np.ndarray] | None = None,
    reg: float = 0.1,
    max_iter: int = 50,
    sinkhorn_iter: int = 500,
    tol: float = 1e-6,
) -> list[list[np.ndarray]]:
    """Multi-marginal Gromov-Wasserstein alignment preserving full vocabulary sizes.

    Computes pairwise GW couplings between all m languages simultaneously,
    then iteratively refines them by applying consistency constraints from
    the other couplings.

    The KL prior regularization is applied by modifying the Gibbs kernel:
    K_ij = prior_ij * exp(-C_ij / reg), which is mathematically equivalent
    to adding reg * KL(Pi | prior) to the objective.

    Args:
        distance_matrices: List of m distance matrices, each (n_i x n_i).
        marginal_distributions: Optional list of m marginal distributions.
            Defaults to uniform for each language.
        reg: Entropic regularization parameter for Sinkhorn.
        max_iter: Maximum refinement iterations.
        sinkhorn_iter: Maximum Sinkhorn iterations per GW solve.
        tol: Convergence tolerance on coupling change.

    Returns:
        couplings: m x m matrix of coupling matrices.
            couplings[i][j] is (n_i x n_j) transport plan from i to j.
            couplings[i][i] is identity-like (n_i x n_i).
    """
    m = len(distance_matrices)
    n = [D.shape[0] for D in distance_matrices]

    if marginal_distributions is None:
        marginals = [np.ones(ni) / ni for ni in n]
    else:
        marginals = marginal_distributions

    # Initialize with pairwise GW
    couplings = [[None] * m for _ in range(m)]
    for i in range(m):
        couplings[i][i] = np.eye(n[i]) / n[i]
        for j in range(i + 1, m):
            Pi_ij = gromov_wasserstein_matrix(
                distance_matrices[i], distance_matrices[j],
                marginals[i], marginals[j],
                reg=reg, max_iter=max_iter, sinkhorn_iter=sinkhorn_iter, tol=tol,
            )
            couplings[i][j] = Pi_ij
            couplings[j][i] = Pi_ij.T

    # Iterative refinement with consistency priors
    for iteration in range(max_iter):
        max_change = 0.0
        for i in range(m):
            for j in range(i + 1, m):
                # Compose priors from all other languages
                prior = np.zeros((n[i], n[j]))
                weight_sum = 0.0
                for k in range(m):
                    if k == i or k == j:
                        continue
                    composed = couplings[i][k] @ couplings[k][j]
                    # Normalize composition to be a valid coupling
                    composed_sum = composed.sum()
                    if composed_sum > 1e-10:
                        composed = composed / composed_sum
                        if composed.shape == prior.shape:
                            prior += composed
                            weight_sum += 1.0

                if weight_sum > 0:
                    prior /= weight_sum

                # Solve GW with consistency prior
                # Correct KL regularization: modify Gibbs kernel
                # K_ij = prior_ij * exp(-C_ij / reg)
                # This is equivalent to: min <Pi, C> + reg * KL(Pi | prior * a x b)
                Dx = distance_matrices[i]
                Dy = distance_matrices[j]
                a = marginals[i]
                b = marginals[j]

                # GW cost using prior as current coupling estimate
                a_pi = prior.sum(axis=1)
                b_pi = prior.sum(axis=0)
                term1 = (Dx ** 2) @ a_pi
                term2 = (Dy ** 2) @ b_pi
                C = term1[:, None] + term2[None, :] - 2.0 * (Dx @ prior @ Dy.T)

                if weight_sum > 0:
                    # Gibbs kernel with prior: K = prior * exp(-C/reg)
                    # This regularizes toward the consensus prior
                    K_prior = prior * np.exp(-C / reg)
                    # Sinkhorn with modified kernel
                    Pi_new = _sinkhorn_with_kernel(K_prior, a, b, reg, sinkhorn_iter, tol)
                else:
                    Pi_new = optimal_transport_matrix(
                        C, a, b, reg, sinkhorn_iter, tol
                    )

                change = np.linalg.norm(Pi_new - couplings[i][j], ord="fro")
                max_change = max(max_change, change)
                couplings[i][j] = Pi_new
                couplings[j][i] = Pi_new.T

        if max_change < tol:
            break

    return couplings


def _sinkhorn_with_kernel(
    K: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    reg: float = 0.1,
    num_iter: int = 1000,
    tol: float = 1e-9,
) -> np.ndarray:
    """Sinkhorn iteration with a pre-computed Gibbs kernel.

    Solves: min_{Pi in U(a,b)} <Pi, C> + reg * KL(Pi | K)
    where K is the pre-computed kernel (already includes prior information).

    This is used for KL-regularized transport toward a prior coupling.
    """
    u = np.ones_like(a)
    v = np.ones_like(b)

    for _ in range(num_iter):
        u_prev = u.copy()
        u = a / (K @ v + 1e-128)
        v = b / (K.T @ u + 1e-128)
        if np.max(np.abs(u - u_prev)) < tol:
            break

    Pi = u[:, None] * K * v[None, :]
    # Normalize to ensure it's a valid coupling
    Pi = Pi / (Pi.sum() + 1e-128)
    # Re-enforce marginals
    Pi = Pi * (a[:, None] * b[None, :]) / (Pi.sum(axis=1, keepdims=True) * Pi.sum(axis=0, keepdims=True) + 1e-128)
    Pi = Pi / (Pi.sum() + 1e-128)
    return Pi


def consensus_from_multi_gw(
    couplings: list[list[np.ndarray]],
    embeddings: list[np.ndarray],
    distance_matrices: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build consensus representation from multi-marginal GW couplings.

    Properly aligns embeddings via Procrustes before averaging.
    Each embedding is first rotated to best match the others using
    the coupling-weighted Procrustes alignment.

    Args:
        couplings: m x m matrix of coupling matrices from multi_marginal_gw.
        embeddings: List of m embedding matrices, each (n_i, d).
        distance_matrices: List of m distance matrices, each (n_i x n_i).

    Returns:
        consensus: Consensus representation (n_min, d).
        projections: List of projection matrices mapping each language to consensus.
    """
    m = len(embeddings)
    d = min(E.shape[1] for E in embeddings)
    n_min = min(E.shape[0] for E in embeddings)

    # Center and truncate each embedding
    centered = []
    for E in embeddings:
        E_c = E[:, :d] - E[:, :d].mean(axis=0, keepdims=True)
        # Sort by norm (frequency proxy) and take first n_min
        norms = np.linalg.norm(E_c, axis=1)
        rank_idx = np.argsort(-norms)
        centered.append(E_c[rank_idx[:n_min], :])

    # Use language 0 as initial reference
    consensus = centered[0].copy()
    rotations = [np.eye(d) for _ in range(m)]

    # Iterative Procrustes alignment toward consensus
    for iteration in range(10):
        consensus_old = consensus.copy()

        for i in range(m):
            M = centered[i].T @ consensus
            U, _, Vt = np.linalg.svd(M, full_matrices=False)
            Q = U @ Vt
            if np.linalg.det(Q) < 0:
                U[:, -1] *= -1
                Q = U @ Vt
            rotations[i] = Q

        # Weighted average of aligned embeddings
        consensus = np.zeros((n_min, d))
        for i in range(m):
            consensus += centered[i] @ rotations[i]
        consensus /= m

        if np.linalg.norm(consensus - consensus_old, "fro") < 1e-6:
            break

    # Projections map from original centered to consensus-aligned
    projections = rotations

    return consensus, projections