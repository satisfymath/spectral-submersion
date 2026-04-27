"""Co-occurrence matrix construction with document/line boundary awareness."""
from collections.abc import Sequence

import numpy as np


def build_vocab(tokens: list[str]) -> dict[str, int]:
    """Build vocabulary mapping token -> index, sorted alphabetically."""
    return {tok: i for i, tok in enumerate(sorted(set(tokens)))}


def cooccurrence_matrix_from_sequences(
    sequences: Sequence[Sequence[int]],
    vocab_size: int,
    window_size: int = 3,
    inverse_distance: bool = True,
) -> np.ndarray:
    """Build co-occurrence matrix from token-ID sequences, respecting boundaries.

    Args:
        sequences: List of token-ID sequences (e.g., per line/document).
        vocab_size: Size of vocabulary.
        window_size: Context window radius.
        inverse_distance: If True, weight by 1/distance; otherwise uniform.

    Returns:
        Co-occurrence matrix of shape (vocab_size, vocab_size).
    """
    C = np.zeros((vocab_size, vocab_size), dtype=float)

    for token_ids in sequences:
        n = len(token_ids)
        for t, center in enumerate(token_ids):
            left = max(0, t - window_size)
            right = min(n, t + window_size + 1)

            for u in range(left, right):
                if u == t:
                    continue
                context = token_ids[u]
                distance = abs(u - t)
                weight = 1.0 / distance if inverse_distance else 1.0
                C[center, context] += weight

    return C


def cooccurrence_matrix(
    token_ids: list[int],
    vocab_size: int,
    window_size: int = 3,
    inverse_distance: bool = True,
) -> np.ndarray:
    """Legacy single-sequence co-occurrence (no boundary awareness)."""
    return cooccurrence_matrix_from_sequences(
        [token_ids], vocab_size, window_size, inverse_distance
    )
