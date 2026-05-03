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


def collapse_repetitions(
    sequence: list[str],
    max_repeat: int = 4,
) -> list[str]:
    """Collapse consecutive identical tokens into repetition-aware tokens.

    For each run of k identical tokens, emit:
    - One token suffixed with _REPk (if k >= 2 and k <= max_repeat)
    - If k > max_repeat, emit one _REP{max_repeat} followed by (k - max_repeat) bare tokens
    - Single occurrences pass through unchanged

    Example: ['440', '440', '440', '300'] -> ['440_REP3', '300']
    Example: ['440', '300'] -> ['440', '300']
    """
    if not sequence:
        return []
    result = []
    i = 0
    while i < len(sequence):
        tok = sequence[i]
        j = i + 1
        while j < len(sequence) and sequence[j] == tok:
            j += 1
        run_len = j - i
        if run_len == 1:
            result.append(tok)
        elif run_len <= max_repeat:
            result.append(f"{tok}_REP{run_len}")
        else:
            result.append(f"{tok}_REP{max_repeat}")
            for _ in range(run_len - max_repeat):
                result.append(tok)
        i = j
    return result


def get_repetition_aware_sequences(
    df: pd.DataFrame,
    max_repeat: int = 4,
) -> list[list[str]]:
    """Extract sequences with consecutive repetitions collapsed into pattern-aware tokens.

    Returns (sequences, pure_sequences) where:
    - sequences: repetition-collapsed token lists
    - pure_sequences: original (uncollapsed) token lists
    """
    pure = get_sequences_by_line(df)
    collapsed = [collapse_repetitions(seq, max_repeat=max_repeat) for seq in pure]
    return collapsed, pure


def get_abab_aware_sequences(
    df: pd.DataFrame,
) -> list[list[str]]:
    """Extract sequences with ABAB patterns marked as composite tokens.

    Detects ABAB patterns and replaces them with A_BAB composite tokens.
    Consecutive repetitions are also collapsed.
    """
    from spectral_submersion.tokenization import collapse_repetitions
    sequences = get_sequences_by_line(df)
    result = []
    for seq in sequences:
        seq = collapse_repetitions(seq)
        out = []
        i = 0
        while i < len(seq):
            if (
                i + 3 < len(seq)
                and seq[i] == seq[i + 2]
                and seq[i + 1] == seq[i + 3]
                and seq[i] != seq[i + 1]
            ):
                out.append(f"{seq[i]}_{seq[i+1]}_ABAB")
                i += 4
            else:
                out.append(seq[i])
                i += 1
        result.append(out)
    return result
