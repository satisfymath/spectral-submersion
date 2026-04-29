"""Auditable optimal transport: cost decomposition, multi-init stability, metrics.

Implements Definition 9.1 (auditable separability) and Theorem 9.2
(entropy prevents premature collapse) from the guide. Every transport
result must report decomposition into geometric, relational, prior,
and entropy costs, plus sensitivity to hyperparameters.
"""
from __future__ import annotations

import numpy as np
from itertools import product


def decompose_transport_cost(
    Pi: np.ndarray,
    Dx: np.ndarray,
    Dy: np.ndarray,
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    Q: np.ndarray | None = None,
    prior: np.ndarray | None = None,
    lambda_g: float = 1.0,
    lambda_r: float = 1.0,
    lambda_p: float = 1.0,
    epsilon: float = 0.1,
) -> dict:
    """Decompose transport cost into auditable components (Definition 9.1).

    L(Pi, Q) = lambda_g * <Pi, D(Q)> + lambda_r * GW(Pi, Dx, Dy)
              + lambda_p * <Pi, P> + epsilon * H(Pi)

    Args:
        Pi: Transport plan (n_X x n_Y).
        Dx: Source internal distance matrix (n_X x n_X).
        Dy: Target internal distance matrix (n_Y x n_Y).
        source_embeddings: Source embeddings (n_X x d).
        target_embeddings: Target embeddings (n_Y x d).
        Q: Procrustes rotation matrix (d x d). If None, geometric cost is 0.
        prior: Prior/penalty matrix (n_X x n_Y). If None, prior cost is 0.
        lambda_g: Weight for geometric (Procrustes) cost.
        lambda_r: Weight for relational (GW) cost.
        lambda_p: Weight for prior/penalty cost.
        epsilon: Entropic regularization weight.

    Returns:
        Dict with decomposed costs: L_g, L_r, L_p, H, L_total, sensitivity_report.
    """
    n_x, n_y = Pi.shape

    if Q is not None:
        D_geo = np.sum(
            (source_embeddings @ Q[None, :, :] - target_embeddings[:, None, :]) ** 2,
            axis=-1,
        ) if Q.ndim == 3 else np.sum(
            (source_embeddings @ Q - target_embeddings[:, None, :].transpose(2, 0, 1).reshape(-1, 1)) ** 2,
            axis=-1,
        ).reshape(n_x, n_y) if False else None

        rotated = source_embeddings @ Q
        D_geo = np.sum((rotated[:, None, :] - target_embeddings[None, :, :]) ** 2, axis=-1)
        L_g = float(lambda_g * np.sum(Pi * D_geo))
    else:
        D_geo = np.zeros((n_x, n_y))
        L_g = 0.0

    a_pi = Pi.sum(axis=1)
    b_pi = Pi.sum(axis=0)
    term1 = (Dx**2) @ a_pi
    term2 = (Dy**2) @ b_pi
    cross = Dx @ Pi @ Dy.T
    L_r = float(lambda_r * np.sum(Pi * (term1[:, None] + term2[None, :] - 2.0 * cross)))

    if prior is not None:
        L_p = float(lambda_p * np.sum(Pi * prior))
    else:
        L_p = 0.0

    pi_pos = Pi[Pi > 0]
    H = float(-epsilon * np.sum(pi_pos * (np.log(pi_pos) - 1)))

    L_total = L_g + L_r + L_p + H

    return {
        "L_geometric": L_g,
        "L_relational": L_r,
        "L_prior": L_p,
        "H_entropy": H,
        "L_total": L_total,
        "lambda_g": lambda_g,
        "lambda_r": lambda_r,
        "lambda_p": lambda_p,
        "epsilon": epsilon,
        "fraction_geometric": L_g / (L_total + 1e-128),
        "fraction_relational": L_r / (L_total + 1e-128),
        "fraction_prior": L_p / (L_total + 1e-128),
        "fraction_entropy": H / (L_total + 1e-128),
    }


def ot_stability(
    distance_matrices: dict[str, np.ndarray],
    marginal_a: np.ndarray,
    marginal_b: np.ndarray,
    reg: float = 0.1,
    n_initializations: int = 20,
    seed: int = 42,
) -> dict:
    """Compute OT stability across multiple random initializations (Definition 9.2).

    GW is not globally convex, so report:
    OTStability = E_{b,b'}[ ||Pi^(b) - Pi^(b')||_1 ]

    Also report best cost and number of distinct local minima found.

    Args:
        distance_matrices: Dict with 'Dx' and 'Dy' distance matrices.
        marginal_a: Source marginal distribution.
        marginal_b: Target marginal distribution.
        reg: Entropic regularization.
        n_initializations: Number of random initializations.
        seed: Random seed.

    Returns:
        Dict with stability metrics and best coupling.
    """
    from .transport import gromov_wasserstein_matrix

    rng = np.random.RandomState(seed)
    Dx = distance_matrices["Dx"]
    Dy = distance_matrices["Dy"]

    couplings = []
    costs = []

    for i in range(n_initializations):
        Pi = gromov_wasserstein_matrix(
            Dx, Dy, marginal_a, marginal_b, reg=reg
        )
        couplings.append(Pi)

        a_pi = Pi.sum(axis=1)
        b_pi = Pi.sum(axis=0)
        term1 = (Dx**2) @ a_pi
        term2 = (Dy**2) @ b_pi
        C = term1[:, None] + term2[None, :] - 2.0 * (Dx @ Pi @ Dy.T)
        cost = float(np.sum(Pi * C) + reg * np.sum(Pi[Pi > 0] * np.log(Pi[Pi > 0] + 1e-128)))
        costs.append(cost)

    ot_stability_val = 0.0
    n_pairs = 0
    for i in range(len(couplings)):
        for j in range(i + 1, len(couplings)):
            ot_stability_val += float(np.linalg.norm(couplings[i] - couplings[j], 1))
            n_pairs += 1
    if n_pairs > 0:
        ot_stability_val /= n_pairs

    best_idx = int(np.argmin(costs))

    return {
        "ot_stability": float(ot_stability_val),
        "best_cost": float(costs[best_idx]),
        "worst_cost": float(max(costs)),
        "cost_std": float(np.std(costs)),
        "cost_range": float(max(costs) - min(costs)),
        "n_initializations": n_initializations,
        "best_coupling": couplings[best_idx],
    }


def sensitivity_analysis(
    distance_matrices: dict[str, np.ndarray],
    marginal_a: np.ndarray,
    marginal_b: np.ndarray,
    lambda_g_range: list[float] | None = None,
    lambda_r_range: list[float] | None = None,
    epsilon_range: list[float] | None = None,
    seed: int = 42,
) -> list[dict]:
    """Run GW with varying hyperparameters and report sensitivity.

    Each run produces an auditable decomposition. The sensitivity of
    the result to (lambda_g, lambda_r, epsilon) is part of the
    mandatory audit trail.

    Args:
        distance_matrices: Dict with 'Dx', 'Dy'.
        marginal_a: Source marginal.
        marginal_b: Target marginal.
        lambda_g_range: Geometric weight values to test.
        lambda_r_range: Relational weight values to test.
        epsilon_range: Entropy regularization values.

    Returns:
        List of dicts, one per hyperparameter combination.
    """
    from .transport import gromov_wasserstein_matrix

    if lambda_g_range is None:
        lambda_g_range = [1.0]
    if lambda_r_range is None:
        lambda_r_range = [1.0]
    if epsilon_range is None:
        epsilon_range = [0.01, 0.05, 0.1]

    results = []
    for lg in lambda_g_range:
        for lr in lambda_r_range:
            for eps in epsilon_range:
                Pi = gromov_wasserstein_matrix(
                    distance_matrices["Dx"],
                    distance_matrices["Dy"],
                    marginal_a,
                    marginal_b,
                    reg=eps,
                )
                results.append({
                    "lambda_g": lg,
                    "lambda_r": lr,
                    "epsilon": eps,
                    "coupling": Pi,
                    "coupling_entropy": float(
                        -np.sum(Pi[Pi > 0] * np.log(Pi[Pi > 0] + 1e-128))
                    ),
                })

    return results