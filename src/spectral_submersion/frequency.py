"""Frequency analysis utilities."""
import numpy as np
import pandas as pd


def token_frequencies(tokens: list[str]) -> pd.DataFrame:
    """Compute absolute and relative token frequencies with Zipf-style ranking."""
    counts = pd.Series(tokens).value_counts().rename("count").reset_index()
    # pandas >= 2.0 compatibility: rename columns explicitly
    counts.columns = ["token", "count"]
    total = counts["count"].sum()
    counts["probability"] = counts["count"] / total
    counts["rank"] = np.arange(1, len(counts) + 1)
    return counts


def entropy(probabilities: np.ndarray) -> float:
    """Shannon entropy in nats."""
    p = np.asarray(probabilities, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def zipf_mandelbrot_prediction(
    ranks: np.ndarray,
    N: int,
    s: float = 1.0,
    q: float = 0.0,
) -> np.ndarray:
    """Predict frequencies under Zipf-Mandelbrot law: f ~ 1 / (r + q)^s."""
    denom = np.sum(1.0 / (np.arange(1, N + 1) + q) ** s)
    return 1.0 / ((ranks + q) ** s * denom)
