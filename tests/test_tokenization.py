"""Tests for tokenization module."""
import pandas as pd
import pytest

from spectral_submersion.tokenization import (
    read_corpus,
    normalize_tokens,
    build_vocab,
    tokens_to_ids,
    get_sequences_by_line,
)


def test_build_vocab():
    tokens = ["a", "b", "a", "c"]
    vocab = build_vocab(tokens)
    assert vocab == {"a": 0, "b": 1, "c": 2}


def test_tokens_to_ids():
    vocab = {"a": 0, "b": 1}
    ids = tokens_to_ids(["a", "b", "a"], vocab)
    assert ids == [0, 1, 0]


def test_get_sequences_by_line():
    df = pd.DataFrame({
        "doc_id": ["d1", "d1", "d1", "d2", "d2"],
        "line_id": [1, 1, 2, 1, 1],
        "position": [1, 2, 1, 1, 2],
        "token": ["a", "b", "c", "d", "e"],
    })
    sequences = get_sequences_by_line(df)
    assert sequences == [["a", "b"], ["c"], ["d", "e"]]


def test_normalize_tokens_lowercase():
    df = pd.DataFrame({
        "doc_id": ["d1"],
        "line_id": [1],
        "position": [1],
        "token": ["Hello "]
    })
    norm = normalize_tokens(df, lowercase=True, strip=True)
    assert norm["token"].iloc[0] == "hello"
