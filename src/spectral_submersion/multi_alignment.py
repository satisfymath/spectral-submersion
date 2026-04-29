"""Multi-language alignment and consensus space construction.

Implements Generalized Procrustes Analysis (GPA) for multiple languages
and builds a consensus latent space R from which lost languages are projected.

Key design decisions (v2):
- GPA alignment uses frequency-ranked correspondences (row i in each
  embedding corresponds to the i-th most frequent token), NOT arbitrary
  row ordering. This ensures Procrustes aligns semantically meaningful
  pairs.
- All embeddings are rotated in GPA (including reference), per standard GPA.
- Zero-padding is eliminated: we truncate to min_n using frequency-ranked rows.
- The "intersection" method is renamed to "union" (it computes subspace union).
- project_lost_to_consensus uses truncation only (no zero-padding).
- consensus_from_multi_gw aligns embeddings before averaging.
"""
import numpy as np
from typing import Sequence


def _frequency_ranks(embedding: np.ndarray) -> np.ndarray:
    """Return indices that sort rows by descending norm (frequency proxy).

    For embeddings sorted alphabetically, this provides a more meaningful
    correspondence than row index: high-norm rows correspond to frequent tokens.
    """
    norms = np.linalg.norm(embedding, axis=1)
    return np.argsort(-norms)


def generalized_procrustes(
    embeddings: list[np.ndarray],
    freq_rank: bool = True,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generalized Procrustes Analysis for multiple embedding spaces.

    Iteratively aligns all embeddings to a mean consensus configuration
    using Procrustes rotations. All embeddings are rotated (no fixed
    reference), and correspondences are established by frequency rank.

    Args:
        embeddings: List of embedding matrices [E_1, ..., E_m],
                    each of shape (n_i, d) with same d.
        freq_rank: If True, sort rows by norm before alignment to establish
                   meaningful cross-lingual correspondences.
        max_iter: Maximum iterations.
        tol: Convergence tolerance on mean configuration change.

    Returns:
        aligned: List of aligned embeddings, each (n_i, d).
        rotations: List of orthogonal rotation matrices Q_i (d, d).
    """
    m = len(embeddings)
    d = embeddings[0].shape[1]

    # Center each embedding
    centered = []
    for E in embeddings:
        E_c = E - E.mean(axis=0, keepdims=True)
        centered.append(E_c)

    # Consensus size: minimum vocabulary across languages
    min_n = min(E.shape[0] for E in centered)

    # Sort rows by norm (frequency proxy) to establish cross-lingual correspondence
    if freq_rank:
        sorted_centered = []
        for E_c in centered:
            rank_idx = np.argsort(-np.linalg.norm(E_c, axis=1))
            sorted_centered.append(E_c[rank_idx][:min_n, :])
        consensus_bases = sorted_centered
    else:
        consensus_bases = [E_c[:min_n, :] for E_c in centered]

    # Initialize consensus as mean
    consensus = np.mean(consensus_bases, axis=0)

    rotations = [np.eye(d) for _ in range(m)]
    aligned = [E.copy() for E in centered]

    for iteration in range(max_iter):
        consensus_old = consensus.copy()

        # Align ALL embeddings (including reference) to consensus
        for i in range(m):
            E_c = centered[i]
            n_i = E_c.shape[0]

            # Establish frequency-ranked correspondence
            if freq_rank:
                rank_idx = np.argsort(-np.linalg.norm(E_c, axis=1))
                E_ranked = E_c[rank_idx]
            else:
                E_ranked = E_c

            # Use first min_n rows (highest frequency) for Procrustes
            E_sub = E_ranked[:min_n, :]

            # Solve Procrustes: min_Q ||E_sub Q - consensus||_F, Q.T Q = I
            M = E_sub.T @ consensus
            U, _, Vt = np.linalg.svd(M, full_matrices=False)
            Q = U @ Vt

            # Check for reflection (det < 0) and correct
            if np.linalg.det(Q) < 0:
                U[:, -1] *= -1
                Q = U @ Vt

            rotations[i] = Q
            aligned[i] = E_c @ Q

        # Update consensus from aligned, frequency-ranked embeddings
        aligned_sub = []
        for i in range(m):
            if freq_rank:
                rank_idx = np.argsort(-np.linalg.norm(aligned[i], axis=1))
                aligned_sub.append(aligned[i][rank_idx][:min_n, :])
            else:
                aligned_sub.append(aligned[i][:min_n, :])

        consensus = np.mean(aligned_sub, axis=0)

        change = np.linalg.norm(consensus - consensus_old, "fro")
        if change < tol:
            break

    return aligned, rotations


def build_consensus_space(
    embeddings: list[np.ndarray],
    method: str = "gpa",
    target_dim: int | None = None,
    **kwargs,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Build consensus latent space R from multiple language embeddings.

    Args:
        embeddings: List of embedding matrices, each (n_i, d_i).
        method: 'gpa' for Generalized Procrustes, 'union' for SVD union.
        target_dim: Target dimension for R. If None, use min dimension.
        **kwargs: Extra args for GPA.

    Returns:
        R: Consensus embedding (min_n, target_dim).
        aligned: List of aligned embeddings.
        projections: List of projection matrices P_i mapping each language to R.
    """
    if target_dim is None:
        target_dim = min(E.shape[1] for E in embeddings)

    # Trim to common dimension
    trimmed = [E[:, :target_dim] for E in embeddings]

    if method == "gpa":
        aligned, rotations = generalized_procrustes(trimmed, **kwargs)
        min_n = min(a.shape[0] for a in aligned)
        # Build consensus from frequency-ranked aligned embeddings
        consensus_bases = []
        for a in aligned:
            rank_idx = np.argsort(-np.linalg.norm(a, axis=1))
            consensus_bases.append(a[rank_idx][:min_n, :])
        R = np.mean(consensus_bases, axis=0)
        projections = rotations
    elif method in ("union", "intersection"):
        # SVD of stacked embeddings extracts the shared principal subspace.
        # This is a UNION of subspaces (span-addition), not intersection.
        # Kept as 'intersection' for backward compatibility but documented correctly.
        stacked = np.vstack(trimmed)
        U, S, Vt = np.linalg.svd(stacked, full_matrices=False)
        R_basis = Vt[:target_dim, :].T
        aligned = []
        projections = []
        for E in trimmed:
            E_proj = E @ R_basis
            aligned.append(E_proj)
            projections.append(R_basis)
        min_n = min(a.shape[0] for a in aligned)
        R = np.mean([a[:min_n, :] for a in aligned], axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")

    return R, aligned, projections


def project_lost_to_consensus(
    E_lost: np.ndarray,
    R: np.ndarray,
    regularization: float = 1e-3,
) -> np.ndarray:
    """Project lost language embeddings into consensus space R.

    Solves min_W ||E_lost W - R_sub||_F with Tikhonov regularization,
    where R_sub is the first n_x rows of R (no zero-padding).

    Args:
        E_lost: Lost language embeddings (n_x, d_x).
        R: Consensus space (n_r, d_r), with d_r == d_x (after trimming).
        regularization: Tikhonov lambda.

    Returns:
        W: Projection matrix (d_x, d_r).
    """
    n_x, d_x = E_lost.shape
    n_r, d_r = R.shape

    # Use first min(n_x, n_r) rows only -- NO zero-padding
    n = min(n_x, n_r)
    d = min(d_x, d_r)

    E_lost_d = E_lost[:n, :d]
    R_sub_d = R[:n, :d]

    # Tikhonov solution: W = (A^T A + lambda I)^{-1} A^T B
    A = E_lost_d
    B = R_sub_d
    W = np.linalg.solve(A.T @ A + regularization * np.eye(d), A.T @ B)
    return W


def consensus_distance_matrix(
    E_lost: np.ndarray,
    E_candidates: list[np.ndarray],
    candidate_names: list[str],
    method: str = "gpa",
    target_dim: int | None = None,
    reg: float = 1e-3,
) -> dict:
    """Full pipeline: build consensus space from candidates, project lost language,
    and compute distances.

    Returns:
        Dictionary with consensus R, projections, W, and distance matrices.
    """
    R, aligned_cands, projections = build_consensus_space(
        E_candidates, method=method, target_dim=target_dim
    )

    d_consensus = R.shape[1]
    E_lost_trim = E_lost[:, :d_consensus]

    W = project_lost_to_consensus(E_lost_trim, R, regularization=reg)
    E_lost_proj = E_lost_trim @ W

    from spectral_submersion.alignment import pairwise_squared_distances

    distances = {}
    for name, E_cand in zip(candidate_names, aligned_cands):
        d = min(E_lost_proj.shape[1], E_cand.shape[1])
        D = pairwise_squared_distances(E_lost_proj[:, :d], E_cand[:, :d])
        distances[name] = D

    return {
        "R": R,
        "aligned_candidates": aligned_cands,
        "projections": projections,
        "W": W,
        "E_lost_proj": E_lost_proj,
        "distances": distances,
    }