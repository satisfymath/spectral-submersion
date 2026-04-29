"""Spectral stability, co-occurrence concentration, and regularized PPMI.

Implements Theorem 5.2 (spectral stability under perturbation), Corollary 5.3
(spectral rejection rule), Proposition 6.1 (co-occurrence concentration),
and Proposition 7.1 / Corollary 7.2 (SPPMI with structural priors).

These are essential for making spectral claims auditable: without stability
analysis, singular value gaps and embeddings cannot support claims C2+.
"""
from __future__ import annotations

import numpy as np
from typing import Sequence


def spectral_gap(singular_values: np.ndarray, k: int) -> float:
    """Compute the k-th singular gap delta_k = sigma_k - sigma_{k+1}.

    Args:
        singular_values: Array of singular values sorted descending.
        k: Index (1-based dimensionality). Gap between k-th and (k+1)-th.

    Returns:
        Singular gap delta_k.
    """
    sv = np.asarray(singular_values, dtype=float)
    if k >= len(sv):
        return 0.0
    return float(sv[k - 1] - sv[k])


def spectral_reliability(
    singular_values: np.ndarray,
    k: int,
    bootstrap_error: float,
) -> float:
    """Compute SpectralReliability_k (Corollary 5.3).

    SpectralReliability_k = max(0, 1 - epsilon_hat / delta_hat_k)

    If reliability <= 0, the k-dimensional embedding is unstable and
    cannot support claims C2 or higher.

    Args:
        singular_values: Empirical singular values sorted descending.
        k: Embedding dimensionality.
        bootstrap_error: Estimated spectral perturbation epsilon_hat
            from bootstrap resampling.

    Returns:
        Spectral reliability in [0, 1]. >0.5 is reasonable, <0.3 is concerning.
    """
    delta_k = spectral_gap(singular_values, k)
    if delta_k <= 0:
        return 0.0
    if bootstrap_error <= 0:
        return 1.0
    return max(0.0, 1.0 - bootstrap_error / delta_k)


def spectral_stability_bootstrap(
    sequences: Sequence[Sequence[int]],
    vocab_size: int,
    k: int,
    window_size: int = 3,
    n_bootstrap: int = 200,
    alpha: float = 0.75,
    random_state: int = 42,
) -> dict:
    """Compute bootstrap estimate of spectral perturbation error.

    Resamples the corpus with replacement, recomputes PPMI + SVD each time,
    and estimates the perturbation of singular values and subspaces.

    Args:
        sequences: List of token-ID sequences.
        vocab_size: Vocabulary size.
        k: Embedding dimensionality to test.
        window_size: Co-occurrence window.
        n_bootstrap: Number of bootstrap iterations.
        alpha: Context distribution smoothing for PPMI.
        random_state: Random seed.

    Returns:
        Dict with keys:
        - 'singular_values_mean': Mean singular values across bootstrap.
        - 'singular_values_std': Std of singular values across bootstrap.
        - 'delta_k_mean': Mean singular gap at dimension k.
        - 'epsilon_hat': Estimated perturbation (mean spectral std).
        - 'spectral_reliability': Reliability at dimension k.
        - 'reliable': Whether reliability > 0.3.
        - 'n_bootstrap': Number of bootstrap samples.
    """
    from .cooccurrence import cooccurrence_matrix_from_sequences
    from .pmi import ppmi_matrix
    from .spectral import spectral_embedding

    rng = np.random.RandomState(random_state)

    all_sv = []
    representations = []
    n_seqs = len(sequences)

    k_svd = k + 1

    for _ in range(n_bootstrap):
        idx = rng.choice(n_seqs, size=n_seqs, replace=True)
        boot_seqs = [sequences[i] for i in idx]

        C = cooccurrence_matrix_from_sequences(
            boot_seqs, vocab_size, window_size=window_size
        )
        M = ppmi_matrix(C, alpha=alpha)
        E, sv, Vt = spectral_embedding(M, k=k_svd)
        all_sv.append(sv)
        representations.append(E)

    sv_array = np.array(all_sv)
    sv_mean = sv_array.mean(axis=0)
    sv_std = sv_array.std(axis=0)

    delta_k_mean = spectral_gap(sv_mean, k)
    epsilon_hat = float(sv_std[:k].mean())

    reliability = spectral_reliability(sv_mean, k, epsilon_hat)

    return {
        "singular_values_mean": sv_mean.tolist(),
        "singular_values_std": sv_std.tolist(),
        "delta_k_mean": float(delta_k_mean),
        "epsilon_hat": epsilon_hat,
        "spectral_reliability": float(reliability),
        "reliable": reliability > 0.3,
        "n_bootstrap": n_bootstrap,
    }


def cooccurrence_coverage(cooccurrence_matrix: np.ndarray) -> float:
    """Compute CoocCoverage(h): fraction of observed pairs.

    CoocCoverage = |{(i,j): C_ij > 0}| / n^2

    If this is low, the co-occurrence matrix is statistically weak.

    Args:
        cooccurrence_matrix: Non-negative co-occurrence matrix (n x n).

    Returns:
        Fraction of non-zero entries.
    """
    n = cooccurrence_matrix.shape[0]
    if n == 0:
        return 0.0
    observed = np.sum(cooccurrence_matrix > 0)
    return float(observed) / (n * n)


def expected_pair_count(
    total_tokens: int,
    window_size: int,
    vocab_size: int,
) -> float:
    """Compute ExpectedPairCount = 2*h*T / n^2.

    If this is << 1, the co-occurrence matrix is statistically weak.
    Each cell has only O(T_h / n^2) expected observations.

    Args:
        total_tokens: Total number of tokens T in the corpus.
        window_size: Half-window h.
        vocab_size: Vocabulary size n.

    Returns:
        Expected number of observations per cell.
    """
    if vocab_size == 0:
        return 0.0
    t_h = 2 * window_size * total_tokens
    return t_h / (vocab_size**2)


def min_tokens_for_coverage(
    vocab_size: int,
    window_size: int,
    target_coverage: float = 0.5,
    eta: float = 0.1,
    alpha: float = 0.05,
    c_mixing: float = 1.0,
) -> float:
    """Estimate minimum T needed for co-occurrence concentration.

    From Proposition 6.1: to control all n^2 pairs to error eta
    with probability 1-alpha, we need T_h ~ (1/c*eta^2) * log(2n^2/alpha).

    Args:
        vocab_size: Vocabulary size n.
        window_size: Half-window h.
        target_coverage: Fraction of pairs we want to observe.
        eta: Desired error bound per pair.
        alpha: Probability of exceeding the bound.
        c_mixing: Mixing constant (lower = slower mixing).

    Returns:
        Estimated minimum total tokens needed.
    """
    if eta <= 0 or alpha <= 0 or c_mixing <= 0:
        return float("inf")
    t_h_per_pair = (1.0 / (c_mixing * eta**2)) * np.log(2 * vocab_size**2 / alpha)
    t_h_total = t_h_per_pair * (vocab_size**2) * target_coverage
    return float(t_h_total / (2 * window_size))


def sceptmi_matrix(
    C: np.ndarray,
    epsilon: float = 0.1,
    prior_type: str = "uniform",
    k_neg: float = 1.0,
    alpha: float = 0.75,
) -> np.ndarray:
    """Compute Structurally-regularized Positive PMI (SPPMI).

    Implements Corollary 7.2: replaces raw PMI with regularized estimates
    using a structural prior q_{ij} to handle sparse data.

    SPPMI_{ij} = max{ log(p_ij^eps / (p_i^eps * p_j^eps)) - log(k_neg), 0 }

    where:
    p_ij^eps = (C_ij + eps * q_ij) / (N + eps)
    p_i^eps = row marginal of p_ij^eps
    p_j^eps = column marginal of p_ij^eps

    Args:
        C: Co-occurrence matrix (non-negative).
        epsilon: Smoothing weight for the prior. Higher = more smoothing.
        prior_type: One of 'uniform', 'marginal_product', 'diagonal'.
            - 'uniform': q_ij = 1/n^2 (maximum smoothing, Laplace-like)
            - 'marginal_product': q_ij = p_i * p_j (preserves marginals)
            - 'diagonal': q_ij = delta_ij/n (favor self-co-occurrence)
        k_neg: Negative sampling shift (like word2vec's k parameter).
        alpha: Context distribution smoothing (Levy & Goldberg).

    Returns:
        SPPMI matrix of same shape as C.
    """
    C = C.astype(float)
    n = C.shape[0]
    N = C.sum()
    if N <= 0:
        return np.zeros_like(C)

    if prior_type == "uniform":
        q = np.ones((n, n)) / (n * n)
    elif prior_type == "marginal_product":
        row_marg = C.sum(axis=1, keepdims=True)
        col_marg = C.sum(axis=0, keepdims=True)
        q = (row_marg @ col_marg) / (N**2 + 1e-128)
        q = q / (q.sum() + 1e-128)
    elif prior_type == "diagonal":
        q = np.zeros((n, n))
        np.fill_diagonal(q, 1.0 / n)
    else:
        raise ValueError(f"Unknown prior_type: {prior_type}")

    C_smooth = C + epsilon * q
    N_smooth = N + epsilon

    P_ij = C_smooth / N_smooth
    P_i = P_ij.sum(axis=1, keepdims=True)
    P_j = P_ij.sum(axis=0, keepdims=True)

    P_j_smooth = P_j**alpha
    P_j_smooth = P_j_smooth / (P_j_smooth.sum() + 1e-128)

    PMI = np.log(P_ij / (P_i * P_j_smooth + 1e-128) + 1e-128)
    SPPMI = np.maximum(PMI - np.log(k_neg), 0.0)

    return SPPMI


def pmi_sensitivity(
    p_ij: np.ndarray,
    p_i: np.ndarray,
    p_j: np.ndarray,
) -> dict:
    """Compute local sensitivity of PMI as per Proposition 7.1.

    |df| <= |dp_ij|/p_ij + |dp_i|/p_i + |dp_j|/p_j

    Args:
        p_ij: Joint probability matrix.
        p_i: Row marginals.
        p_j: Column marginals.

    Returns:
        Dict with 'max_sensitivity', 'mean_sensitivity', 'pairs_at_risk'
        (fraction of pairs where sensitivity > 10).
    """
    eps = 1e-15
    sens = 1.0 / (p_ij + eps) + 1.0 / (p_i[:, None] + eps) + 1.0 / (p_j[None, :] + eps)
    return {
        "max_sensitivity": float(sens.max()),
        "mean_sensitivity": float(sens.mean()),
        "pairs_at_risk": float(np.mean(sens > 10.0)),
    }


def spectral_rejection_rule(
    singular_values: np.ndarray,
    bootstrap_error: np.ndarray,
    k_values: list[int] | None = None,
) -> list[dict]:
    """Apply Corollary 5.3 spectral rejection rule for multiple k values.

    If delta_hat_k <= 2 * epsilon_hat, the k-dimensional embedding
    should be marked as unstable and not support C2+ claims.

    Args:
        singular_values: Empirical singular values.
        bootstrap_error: Bootstrap-estimated errors per singular value.
        k_values: List of dimensions to test. Defaults to [4, 8, 16, 32].

    Returns:
        List of dicts per dimension k with stability assessment.
    """
    sv = np.asarray(singular_values, dtype=float)
    eps = np.asarray(bootstrap_error, dtype=float)

    if k_values is None:
        k_values = [4, 8, 16, 32]

    results = []
    for k in k_values:
        if k >= len(sv):
            results.append({
                "k": k,
                "delta_k": 0.0,
                "epsilon": 0.0,
                "reliability": 0.0,
                "stable": False,
                "claim_limit": "C0",
            })
            continue

        delta_k = spectral_gap(sv, k)
        eps_k = float(eps[:k].mean()) if k <= len(eps) else float(eps.mean())
        reliability = spectral_reliability(sv, k, eps_k)
        rejects = delta_k <= 2 * eps_k

        if reliability < 0.1:
            claim_limit = "C0"
        elif reliability < 0.3:
            claim_limit = "C1"
        elif reliability < 0.5:
            claim_limit = "C2"
        elif reliability < 0.7:
            claim_limit = "C2"
        else:
            claim_limit = "C2+"

        results.append({
            "k": k,
            "delta_k": float(delta_k),
            "epsilon": float(eps_k),
            "reliability": float(reliability),
            "stable": not rejects,
            "claim_limit": claim_limit,
        })

    return results