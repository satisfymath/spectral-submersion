"""Tests for co-occurrence module."""


from spectral_submersion.cooccurrence import (
    build_vocab,
    cooccurrence_matrix,
    cooccurrence_matrix_from_sequences,
)


def test_build_vocab():
    tokens = ["x", "y", "z"]
    vocab = build_vocab(tokens)
    assert vocab == {"x": 0, "y": 1, "z": 2}


def test_cooccurrence_basic():
    token_ids = [0, 1, 0, 2]
    C = cooccurrence_matrix(
        token_ids, vocab_size=3, window_size=1, inverse_distance=False
    )
    # window=1, uniform weight
    # center 0 at t=0: context 1 -> C[0,1] += 1
    # center 1 at t=1: context 0,0 -> C[1,0] += 2
    # center 0 at t=2: context 1,2 -> C[0,1] += 1, C[0,2] += 1
    # center 2 at t=3: context 0 -> C[2,0] += 1
    assert C[0, 1] == 2.0
    assert C[0, 2] == 1.0
    assert C[1, 0] == 2.0
    assert C[2, 0] == 1.0


def test_cooccurrence_from_sequences_respects_boundaries():
    seqs = [[0, 1], [1, 0]]
    C = cooccurrence_matrix_from_sequences(
        seqs, vocab_size=2, window_size=2, inverse_distance=False
    )
    # Sequence [0,1]: 0 sees 1, 1 sees 0
    # Sequence [1,0]: 1 sees 0, 0 sees 1
    # No cross-sequence contamination
    assert C[0, 1] == 2.0
    assert C[1, 0] == 2.0
    assert C[0, 0] == 0.0
    assert C[1, 1] == 0.0
