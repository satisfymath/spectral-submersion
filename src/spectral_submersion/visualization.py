"""Visualization utilities."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_token_frequencies(
    ranks: np.ndarray,
    counts: np.ndarray,
    title: str = "Token Frequency Distribution",
    save_path: str | Path | None = None,
) -> None:
    """Log-log plot of token rank vs frequency (Zipf plot)."""
    plt.figure(figsize=(10, 5))
    plt.plot(ranks, counts, marker="o", linestyle="none", alpha=0.7)
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_singular_values(
    singular_values: np.ndarray,
    title: str = "Singular Value Spectrum",
    save_path: str | Path | None = None,
) -> None:
    """Plot singular value decay."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(singular_values) + 1), singular_values, marker="o")
    plt.xlabel("Index")
    plt.ylabel("Singular Value")
    plt.title(title)
    plt.yscale("log")
    plt.grid(True, ls="--", alpha=0.5)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
