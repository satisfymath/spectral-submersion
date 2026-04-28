"""Multi-language alignment and consensus space construction.

Implements Generalized Procrustes Analysis (GPA) for multiple languages
and builds a consensus latent space R from which lost languages are projected.
"""
import numpy as np
from typing import Sequence


def generalized_procrustes(
    embeddings: list[np.ndarray],
    ref_idx: int = 0,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generalized Procrustes Analysis for multiple embedding spaces.

    Iteratively aligns all embeddings to a mean consensus configuration.
    This constructs a shared latent space where isometric structures overlap.

    Args:
        embeddings: List of embedding matrices [E_1, ..., E_m],
                    each of shape (n_i, d) with same d.
        ref_idx: Initial reference index (default 0).
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

    # Fix consensus size to minimum n across all embeddings
    min_n = min(E.shape[0] for E in centered)

    # Initialize consensus as mean of first min_n rows of centered embeddings
    consensus = np.mean([E[:min_n, :] for E in centered], axis=0)
    rotations = [np.eye(d) for _ in range(m)]
    aligned = [E.copy() for E in centered]

    for _ in range(max_iter):
        consensus_old = consensus.copy()

        # Align each embedding to current consensus via Procrustes
        for i in range(m):
            if i == ref_idx:
                rotations[i] = np.eye(d)
                aligned[i] = centered[i]
                continue

            # Procrustes: min_Q ||E_i Q - consensus||_F, Q.T Q = I
            # For different sizes, use cross-covariance with consensus-sized slice
            n_i = centered[i].shape[0]
            if n_i >= min_n:
                M = centered[i][:min_n, :].T @ consensus
            else:
                # Pad centered[i] to min_n with zeros for covariance computation
                pad = np.zeros((min_n - n_i, d))
                E_pad = np.vstack([centered[i], pad])
                M = E_pad.T @ consensus
            U, _, Vt = np.linalg.svd(M, full_matrices=False)
            Q = U @ Vt
            rotations[i] = Q
            aligned[i] = centered[i] @ Q

        # Update consensus as mean over aligned embeddings (truncate to min_n)
        consensus = np.mean([a[:min_n, :] for a in aligned], axis=0)

        if np.linalg.norm(consensus - consensus_old, "fro") < tol:
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
        method: 'gpa' for Generalized Procrustes, 'intersection' for SVD intersection.
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
        R = np.mean([a[:min_n, :] for a in aligned], axis=0)
        projections = rotations
    elif method == "intersection":
        # Intersection method: take SVD of stacked embeddings and extract shared subspace
        # This is more algebraic but less geometrically intuitive.
        # We stack all embeddings vertically and take top-k SVD components.
        stacked = np.vstack(trimmed)
        U, S, Vt = np.linalg.svd(stacked, full_matrices=False)
        R_basis = Vt[:target_dim, :].T  # (target_dim, target_dim)
        # Project each language onto shared basis
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

    Solves min_W ||E_lost W - R_subset||_F with Tikhonov regularization,
    where R_subset is a subset of R matched by size.

    Args:
        E_lost: Lost language embeddings (n_x, d_x).
        R: Consensus space (n_r, d_r), with d_r == d_x (after trimming).
        regularization: Tikhonov lambda.

    Returns:
        W: Projection matrix (d_x, d_r).
    """
    n_x, d_x = E_lost.shape
    n_r, d_r = R.shape

    # Match sizes by truncating or padding R
    if n_r >= n_x:
        R_sub = R[:n_x, :]
    else:
        # Pad R with zeros to match n_x
        pad = np.zeros((n_x - n_r, d_r))
        R_sub = np.vstack([R, pad])

    # Match dimensions
    d = min(d_x, d_r)
    E_lost_d = E_lost[:, :d]
    R_sub_d = R_sub[:, :d]

    # Tikhonov solution
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

    # Determine consensus dimension
    d_consensus = R.shape[1]
    E_lost_trim = E_lost[:, :d_consensus]

    W = project_lost_to_consensus(E_lost_trim, R, regularization=reg)
    E_lost_proj = E_lost_trim @ W

    # Compute distances from projected lost to each aligned candidate
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
