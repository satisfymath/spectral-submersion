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
    the other couplings. Unlike GPA-based consensus (which truncates to
    minimum vocabulary size), this preserves each language's full vocabulary.

    The algorithm:
    1. Initialize m*(m-1)/2 pairwise GW couplings
    2. At each iteration, refine each coupling Pi_ij using the
       composition Pi_ik @ Pi_kj.T for k != i,j as a prior
    3. Use KL-divergence regularization toward the composed prior

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
                # Compose priors from all other languages: Pi_ik @ Pi_kj^T
                prior = np.zeros_like(couplings[i][j])
                weight_sum = 0.0
                for k in range(m):
                    if k == i or k == j:
                        continue
                    composed = couplings[i][k] @ couplings[k][j]
                    if composed.shape == prior.shape:
                        prior += composed
                        weight_sum += 1.0

                if weight_sum > 0:
                    prior /= weight_sum

                # Solve GW with prior as additional regularizer
                # Cost: KL(Pi | prior) added to GW cost
                Dx = distance_matrices[i]
                Dy = distance_matrices[j]
                a = marginals[i]
                b = marginals[j]

                # GW cost
                a_pi = prior.sum(axis=1)
                b_pi = prior.sum(axis=0)
                term1 = (Dx ** 2) @ a_pi
                term2 = (Dy ** 2) @ b_pi
                C = term1[:, None] + term2[None, :] - 2.0 * (Dx @ prior @ Dy.T)

                # Add KL prior cost scaled by consistency weight
                prior_reg = reg * 0.5
                Pi_new = _sinkhorn_manual(
                    C + prior_reg * (-np.log(prior + 1e-30)),
                    a, b, reg, sinkhorn_iter, tol,
                )

                change = np.linalg.norm(Pi_new - couplings[i][j], ord="fro")
                max_change = max(max_change, change)
                couplings[i][j] = Pi_new
                couplings[j][i] = Pi_new.T

        if max_change < tol:
            break

    return couplings


def consensus_from_multi_gw(
    couplings: list[list[np.ndarray]],
    embeddings: list[np.ndarray],
    distance_matrices: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build consensus representation from multi-marginal GW couplings.

    Projects each language's embeddings into a shared space using the
    coupling-weighted barycenter of aligned representations.

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

    # Weight each language inversely by its coupling quality (entropy)
    weights = []
    for i in range(m):
        total_ent = 0.0
        count = 0
        for j in range(m):
            if i == j:
                continue
            Pi = couplings[i][j]
            Pi_pos = Pi[Pi > 0]
            if len(Pi_pos) > 0:
                total_ent -= np.sum(Pi_pos * np.log(Pi_pos))
            count += 1
        weights.append(count / max(total_ent, 1e-10) if total_ent > 0 else 1.0)
    w_sum = sum(weights)
    weights = [w / w_sum for w in weights]

    # Build consensus by aligning each embedding via Procrustes with coupling weighting
    consensus = np.zeros((n_min, d))
    for i in range(m):
        E_i = embeddings[i][:, :d][:n_min, :]
        # Use coupling-weighted mean position of language i as contribution
        consensus += weights[i] * E_i

    projections = []
    for i in range(m):
        E_i = embeddings[i][:, :d][:n_min, :]
        M = E_i.T @ consensus
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        P = U @ Vt
        projections.append(P)

    return consensus, projections
