"""Identifiability theory: no-free-decipherment, anchor power, orbit analysis.

Implements Proposition 1.1, Theorem 3.2, Corollary 3.3, and Definitions 4.1-4.3
from the PhD upgrade guide. Formalizes why unanchored decipherment is
ill-posed and how anchors break the permutation symmetry.
"""
from __future__ import annotations

import numpy as np
from itertools import permutations
from collections.abc import Callable


def verify_non_identifiability(
    vocab_size: int,
    structural_statistic: Callable[[np.ndarray], float],
    corpus: np.ndarray,
    n_permutations: int = 100,
    seed: int = 42,
) -> dict:
    """Empirically verify that a structural statistic is invariant under permutations.

    Generates random permutations of the glyph vocabulary, applies them to
    the corpus, and checks that the structural statistic is unchanged.
    This confirms Theorem 3.2: structural statistics are invariant under
    Sym(V_X) and cannot distinguish between semantically different assignments.

    Args:
        vocab_size: Size of the vocabulary V_X.
        structural_statistic: A function taking a co-occurrence or other
            matrix and returning a scalar (or array) statistic.
        corpus: Array of token IDs representing the corpus.
        n_permutations: Number of random permutations to test.
        seed: Random seed.

    Returns:
        Dict with keys: 'original_stat', 'max_deviation', 'is_invariant',
        'n_permutations_tested'.
    """
    rng = np.random.RandomState(seed)
    original_stat = structural_statistic(corpus)
    max_dev = 0.0

    for _ in range(n_permutations):
        perm = rng.permutation(vocab_size)
        permuted_corpus = perm[corpus]
        permuted_stat = structural_statistic(permuted_corpus)
        dev = np.max(np.abs(np.asarray(permuted_stat) - np.asarray(original_stat)))
        max_dev = max(max_dev, float(dev))

    return {
        "original_stat": original_stat,
        "max_deviation": max_dev,
        "is_invariant": max_dev < 1e-10,
        "n_permutations_tested": n_permutations,
    }


def orbit_size(vocab_size: int, anchored_glyphs: int) -> int:
    """Compute the size of the orbit under Sym(V_X) after fixing anchors.

    The remaining symmetry is Sym(V_X \\ anchored), so the orbit size is
    (vocab_size - anchored_glyphs)!.

    Args:
        vocab_size: |V_X|.
        anchored_glyphs: Number of glyphs with anchors that fix them.

    Returns:
        Size of the orbit (equivalence class).
    """
    remaining = vocab_size - anchored_glyphs
    if remaining < 0:
        raise ValueError(
            f"anchored_glyphs ({anchored_glyphs}) > vocab_size ({vocab_size})"
        )
    import math as _math
    return int(_math.factorial(remaining))


def anchor_power(
    automorphism_group_size: int,
    anchored_automorphism_size: int,
) -> float:
    """Compute AnchorPower(A) as defined in Definition 4.3.

    AnchorPower(A) = 1 - log(|Aut(G_X;A)| + 1) / log(|Aut(G_X)| + 1)

    - ~0: anchors do not break symmetry
    - ~1: anchors eliminate almost all structural ambiguity

    Args:
        automorphism_group_size: |Aut(G_X)| without anchors.
        anchored_automorphism_size: |Aut(G_X;A)| with anchors.

    Returns:
        Anchor power in [0, 1].
    """
    if automorphism_group_size <= 0:
        return 1.0
    return 1.0 - np.log(anchored_automorphism_size + 1) / np.log(
        automorphism_group_size + 1
    )


def compute_automorphism_size_upper_bound(
    weight_matrix: np.ndarray,
) -> int:
    """Compute an upper bound on |Aut(G_X)| by counting approximately
    equivalence classes of glyphs under structural similarity.

    Two glyphs i, j are in the same orbit if their row and column
    patterns in W are identical (necessary condition for automorphism).

    The true |Aut(G_X)| >= product of (orbit_sizes!) for each orbit.
    This computes an upper bound based on degree patterns.

    Args:
        weight_matrix: Weighted adjacency matrix W (n x n).

    Returns:
        Upper bound on automorphism group size.
    """
    n = weight_matrix.shape[0]
    row_sums = weight_matrix.sum(axis=1)
    col_sums = weight_matrix.sum(axis=0)
    diag = np.diag(weight_matrix)

    signatures = []
    for i in range(n):
        sig = tuple(sorted([
            round(row_sums[i], 8),
            round(col_sums[i], 8),
            round(diag[i], 8),
        ]))
        signatures.append(sig)

    from collections import Counter

    orbit_sizes = Counter(signatures)
    aut_size = 1
    for size in orbit_sizes.values():
        import math as _math
    aut_size *= int(_math.factorial(size))
    return aut_size


def anchor_condition_number(
    X_anchors: np.ndarray,
    Y_anchors: np.ndarray,
) -> float:
    """Compute AnchorCondition(A) = sigma_d(X_A^T Y_A) as in Theorem 8.1.

    A low value indicates geometrically degenerate anchors, meaning
    the Procrustes solution is sensitive to perturbation.

    Args:
        X_anchors: Source anchor embeddings (m x d).
        Y_anchors: Target anchor embeddings (m x d).

    Returns:
        Minimum singular value of X_anchors^T @ Y_anchors, which
        determines the stability of the Procrustes rotation.
    """
    C = X_anchors.T @ Y_anchors
    singular_values = np.linalg.svd(C, compute_uv=False)
    if len(singular_values) == 0:
        return 0.0
    d = min(X_anchors.shape[1], Y_anchors.shape[1])
    if d > len(singular_values):
        return 0.0
    return float(singular_values[d - 1])


def leave_one_anchor_out_stability(
    X_anchors: np.ndarray,
    Y_anchors: np.ndarray,
    n_bootstrap: int = 100,
    seed: int = 42,
) -> dict:
    """Compute Procrustes stability under leave-one-anchor-out.

    For each anchor removed, compute Q and measure deviation from
    the full-anchor Q. Also compute bootstrap stability QStability.

    Args:
        X_anchors: Source anchor embeddings (m x d).
        Y_anchors: Target anchor embeddings (m x d).
        n_bootstrap: Bootstrap iterations.
        seed: Random seed.

    Returns:
        Dict with keys: 'loo_deviations', 'q_stability',
        'anchor_condition', 'loo_mean_deviation'.
    """
    from .alignment import orthogonal_procrustes

    m = X_anchors.shape[0]
    rng = np.random.RandomState(seed)

    if m < 3:
        return {
            "loo_deviations": [],
            "q_stability": float("nan"),
            "anchor_condition": anchor_condition_number(X_anchors, Y_anchors),
            "loo_mean_deviation": float("nan"),
        }

    Q_full = orthogonal_procrustes(X_anchors, Y_anchors)

    loo_deviations = []
    for i in range(m):
        idx = [j for j in range(m) if j != i]
        X_loo = X_anchors[idx]
        Y_loo = Y_anchors[idx]
        Q_loo = orthogonal_procrustes(X_loo, Y_loo)
        dev = np.linalg.norm(Q_loo - Q_full, "fro")
        loo_deviations.append(float(dev))

    q_stability = 0.0
    q_samples = []
    for _ in range(n_bootstrap):
        idx = rng.choice(m, size=m, replace=True)
        X_b = X_anchors[idx]
        Y_b = Y_anchors[idx]
        Q_b = orthogonal_procrustes(X_b, Y_b)
        q_samples.append(Q_b)

    for i in range(len(q_samples)):
        for j in range(i + 1, len(q_samples)):
            q_stability += np.linalg.norm(q_samples[i] - q_samples[j], "fro")

    n_pairs = n_bootstrap * (n_bootstrap - 1) / 2
    if n_pairs > 0:
        q_stability /= n_pairs

    return {
        "loo_deviations": loo_deviations,
        "q_stability": float(q_stability),
        "anchor_condition": anchor_condition_number(X_anchors, Y_anchors),
        "loo_mean_deviation": float(np.mean(loo_deviations)),
    }