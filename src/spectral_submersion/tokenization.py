"""Tokenization utilities with document/line boundary handling."""
from collections.abc import Sequence

import pandas as pd


def read_corpus(path: str) -> pd.DataFrame:
    """Read corpus CSV and validate required columns."""
    df = pd.read_csv(path)
    required = {"doc_id", "line_id", "position", "token"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


def normalize_tokens(
    df: pd.DataFrame,
    lowercase: bool = True,
    strip: bool = True,
) -> pd.DataFrame:
    """Normalize token strings in a DataFrame."""
    df = df.copy()
    df["token"] = df["token"].astype(str)
    if strip:
        df["token"] = df["token"].str.strip()
    if lowercase:
        df["token"] = df["token"].str.lower()
    return df


def build_vocab(
    tokens: Sequence[str],
    min_frequency: int = 1,
) -> dict[str, int]:
    """Build vocabulary mapping token -> index, sorted alphabetically."""
    counts = pd.Series(tokens).value_counts()
    if min_frequency > 1:
        counts = counts[counts >= min_frequency]
    vocab = {tok: i for i, tok in enumerate(sorted(counts.index))}
    return vocab


def tokens_to_ids(
    tokens: Sequence[str],
    vocab: dict[str, int],
    unk_token: str | None = None,
) -> list[int]:
    """Map token sequence to integer IDs using vocabulary."""
    unk_id = vocab.get(unk_token, -1) if unk_token else -1
    return [vocab.get(tok, unk_id) for tok in tokens]


def get_sequences_by_line(
    df: pd.DataFrame,
) -> list[list[str]]:
    """Extract token sequences grouped by (doc_id, line_id), respecting boundaries."""
    sequences = []
    grouped = df.sort_values(["doc_id", "line_id", "position"]).groupby(
        ["doc_id", "line_id"]
    )
    for _, group in grouped:
        sequences.append(group["token"].tolist())
    return sequences
