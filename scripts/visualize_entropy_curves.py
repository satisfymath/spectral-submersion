"""Visualize conditional entropy curves (Rao et al. 2009 style).

Plots H_n = H(token | previous n-1 tokens) for n=1,2,3 (unigram, bigram, trigram)
across real corpus and controls. Linguistic systems typically show a smooth
decay curve; non-linguistic systems may show flat or irregular curves.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_entropy_curves(csv_path: str, output_path: str):
    """Plot conditional entropy curves from entropy analysis CSV."""
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "real": "#2E86AB",
        "permuted": "#A23B72",
        "random_same_freq": "#F18F01",
        "random_uniform": "#C73E1D",
    }
    labels = {
        "real": "Real corpus",
        "permuted": "Permuted",
        "random_same_freq": "Random (same freq)",
        "random_uniform": "Random (uniform)",
    }

    x = [1, 2, 3]
    x_labels = ["Unigram\nH(X)", "Bigram\nH(X|prev)", "Trigram\nH(X|prev2,prev)"]

    for _, row in df.iterrows():
        variant = row["variant"]
        y = [
            row["h_unconditional"],
            row["h_bigram_conditional"],
            row["h_trigram_conditional"],
        ]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.5,
            markersize=10,
            color=colors.get(variant, "gray"),
            label=labels.get(variant, variant),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Conditional Entropy (bits)", fontsize=12)
    ax.set_title("Conditional Entropy Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Saved entropy curve plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot conditional entropy curves")
    parser.add_argument(
        "--input", default="reports/tables/entropy_analysis_indus_real.csv"
    )
    parser.add_argument("--output", default="reports/figures/entropy_curves.png")
    args = parser.parse_args()
    plot_entropy_curves(args.input, args.output)


if __name__ == "__main__":
    main()
