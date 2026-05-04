"""Evaluation utilities: controls, bootstrap, and relational metrics."""

import numpy as np


def permute_corpus(sequences: list[list[str]]) -> list[list[str]]:
    """Shuffle tokens globally while preserving sequence lengths."""
    all_tokens = [tok for seq in sequences for tok in seq]
    np.random.shuffle(all_tokens)
    permuted = []
    idx = 0
    for seq in sequences:
        n = len(seq)
        permuted.append(all_tokens[idx : idx + n])
        idx += n
    return permuted


def random_corpus_same_frequency(sequences: list[list[str]]) -> list[list[str]]:
    """Generate random corpus with same token frequency distribution."""
    all_tokens = [tok for seq in sequences for tok in seq]
    random_tokens = np.random.choice(
        all_tokens, size=len(all_tokens), replace=True
    ).tolist()
    random_seqs = []
    idx = 0
    for seq in sequences:
        n = len(seq)
        random_seqs.append(random_tokens[idx : idx + n])
        idx += n
    return random_seqs


def random_corpus_uniform(
    sequences: list[list[str]], vocab: list[str]
) -> list[list[str]]:
    """Generate random corpus from uniform token distribution."""
    total = sum(len(seq) for seq in sequences)
    random_tokens = np.random.choice(vocab, size=total, replace=True).tolist()
    random_seqs = []
    idx = 0
    for seq in sequences:
        n = len(seq)
        random_seqs.append(random_tokens[idx : idx + n])
        idx += n
    return random_seqs


def relational_distortion(
    Pi: np.ndarray,
    Dx: np.ndarray,
    Dy: np.ndarray,
) -> float:
    """Compute Gromov-Wasserstein-style relational distortion for a coupling Pi.

    L(Pi) = sum_{i,i',j,j'} |Dx(i,i') - Dy(j,j')|^2 * Pi_{ij} * Pi_{i'j'}

    Lower is better (preserves relational structure).

    Args:
        Pi: Coupling matrix (n_X x n_Y), rows sum to 1.
        Dx: Pairwise distance matrix for source (n_X x n_X).
        Dy: Pairwise distance matrix for target (n_Y x n_Y).

    Returns:
        Scalar distortion value.
    """
    # Efficient O(n_x^2 n_y + n_x n_y^2) computation using matrix products.
    # Expand (Dx_{ii'} - Dy_{jj'})^2 = Dx^2 + Dy^2 - 2 Dx Dy.
    # L = sum_{ij} Pi_{ij} * [sum_{i'} Dx_{ii'}^2 + sum_{j'} Dy_{jj'}^2
    #                         - 2 * sum_{i'j'} Dx_{ii'} Dy_{jj'} Pi_{i'j'}]
    # The cross term simplifies to: C = Dx @ Pi @ Dy.T
    A = (Dx**2).sum(axis=1, keepdims=True)  # (n_x, 1)
    B = (Dy**2).sum(axis=1, keepdims=True)  # (n_y, 1)
    C = Dx @ Pi @ Dy.T  # (n_x, n_y)
    L = np.sum(Pi * (A + B.T - 2.0 * C))
    return float(L)


def geometric_distortion(
    Pi: np.ndarray,
    D: np.ndarray,
) -> float:
    """Compute expected geometric distortion: <Pi, D>.

    Args:
        Pi: Coupling matrix (n_X x n_Y).
        D: Pairwise distance matrix (n_X x n_Y).

    Returns:
        Scalar <Pi, D>.
    """
    return float(np.sum(Pi * D))
