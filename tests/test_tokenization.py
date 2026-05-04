"""Tests for tokenization module."""

import pandas as pd

from spectral_submersion.tokenization import (
    build_vocab,
    collapse_repetitions,
    get_abab_aware_sequences,
    get_repetition_aware_sequences,
    get_sequences_by_line,
    normalize_tokens,
    tokens_to_ids,
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
    df = pd.DataFrame(
        {
            "doc_id": ["d1", "d1", "d1", "d2", "d2"],
            "line_id": [1, 1, 2, 1, 1],
            "position": [1, 2, 1, 1, 2],
            "token": ["a", "b", "c", "d", "e"],
        }
    )
    sequences = get_sequences_by_line(df)
    assert sequences == [["a", "b"], ["c"], ["d", "e"]]


def test_normalize_tokens_lowercase():
    df = pd.DataFrame(
        {"doc_id": ["d1"], "line_id": [1], "position": [1], "token": ["Hello "]}
    )
    norm = normalize_tokens(df, lowercase=True, strip=True)
    assert norm["token"].iloc[0] == "hello"


def test_collapse_repetitions_no_repeats():
    assert collapse_repetitions(["a", "b", "c"]) == ["a", "b", "c"]


def test_collapse_repetitions_double():
    assert collapse_repetitions(["a", "a", "b"]) == ["a_REP2", "b"]


def test_collapse_repetitions_triple():
    assert collapse_repetitions(["a", "a", "a", "b"]) == ["a_REP3", "b"]


def test_collapse_repetitions_quad():
    assert collapse_repetitions(["a", "a", "a", "a", "b"]) == ["a_REP4", "b"]


def test_collapse_repetitions_exceeds_max():
    assert collapse_repetitions(["a", "a", "a", "a", "a", "b"], max_repeat=4) == [
        "a_REP4",
        "a",
        "b",
    ]


def test_collapse_repetitions_multiple_runs():
    assert collapse_repetitions(["a", "a", "b", "b", "b", "c"]) == [
        "a_REP2",
        "b_REP3",
        "c",
    ]


def test_collapse_repetitions_empty():
    assert collapse_repetitions([]) == []


def test_collapse_repetitions_single():
    assert collapse_repetitions(["x"]) == ["x"]


def test_get_repetition_aware_sequences():
    df = pd.DataFrame(
        {
            "doc_id": ["d1", "d1", "d1", "d1"],
            "line_id": [1, 1, 1, 1],
            "position": [1, 2, 3, 4],
            "token": ["a", "a", "a", "b"],
        }
    )
    collapsed, pure = get_repetition_aware_sequences(df)
    assert collapsed == [["a_REP3", "b"]]
    assert pure == [["a", "a", "a", "b"]]


def test_get_abab_aware_sequences():
    df = pd.DataFrame(
        {
            "doc_id": ["d1", "d1", "d1", "d1"],
            "line_id": [1, 1, 1, 1],
            "position": [1, 2, 3, 4],
            "token": ["a", "b", "a", "b"],
        }
    )
    result = get_abab_aware_sequences(df)
    assert result == [["a_b_ABAB"]]
