"""Generate comparative visualizations across all corpora.

Creates multi-panel figures comparing:
- Frequency distributions
- Singular value spectra
- Effective rank vs vocabulary size
- Conditional entropy curves
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_corpus_comparison(output_path: str):
    """Create comprehensive comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Corpus Comparison: Synthetic vs Rongorongo-like vs Indus Real", fontsize=16, fontweight="bold")

    # Panel 1: Frequency distribution (Zipf plot)
    ax = axes[0, 0]
    for corpus_name, csv_path, color in [
        ("Synthetic PCFG", "data/raw/lost_language/corpus_synthetic.csv", "#2E86AB"),
        ("Rongorongo-like", "data/raw/lost_language/corpus_rongorongo_v2.csv", "#A23B72"),
        ("Indus Real", "data/raw/lost_language/corpus_indus_real.csv", "#F18F01"),
    ]:
        if not Path(csv_path).exists():
            continue
        df = pd.read_csv(csv_path)
        freq = df["token"].value_counts().reset_index()
        freq.columns = ["token", "count"]
        freq["rank"] = np.arange(1, len(freq) + 1)
        ax.loglog(freq["rank"], freq["count"], marker="o", markersize=3,
                  label=corpus_name, color=color, alpha=0.8)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Frequency")
    ax.set_title("Token Frequency (Zipf plot)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Singular value spectra
    ax = axes[0, 1]
    for corpus_name, sv_path, color in [
        ("Synthetic PCFG", "data/processed/singular_values_synthetic_v2.npy", "#2E86AB"),
        ("Rongorongo-like", "data/processed/singular_values_rongorongo_v2.npy", "#A23B72"),
        ("Indus Real", "data/processed/sv_indus_real.npy", "#F18F01"),
    ]:
        if not Path(sv_path).exists():
            continue
        S = np.load(sv_path)
        ax.plot(range(1, len(S) + 1), S, marker="o", markersize=3,
                label=corpus_name, color=color, alpha=0.8)
    ax.set_xlabel("Index")
    ax.set_ylabel("Singular Value")
    ax.set_title("Singular Value Spectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Panel 3: Effective rank comparison bar chart
    ax = axes[1, 0]
    controls_data = []
    for corpus_label, csv_path in [
        ("Synthetic PCFG", "reports/tables/control_comparison_v2.csv"),
        ("Rongorongo-like", "reports/tables/control_comparison_rongorongo_v2.csv"),
        ("Indus Real", "reports/tables/control_comparison_indus_real.csv"),
    ]:
        if not Path(csv_path).exists():
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            controls_data.append({
                "corpus": corpus_label,
                "variant": row["variant"],
                "effective_rank": row["effective_rank"],
            })

    if controls_data:
        cdf = pd.DataFrame(controls_data)
        variants = ["real", "permuted", "random_same_freq", "random_uniform"]
        x = np.arange(len(variants))
        width = 0.25
        colors = {"Synthetic PCFG": "#2E86AB", "Rongorongo-like": "#A23B72", "Indus Real": "#F18F01"}
        for i, corpus in enumerate(["Synthetic PCFG", "Rongorongo-like", "Indus Real"]):
            sub = cdf[cdf["corpus"] == corpus]
            vals = [sub[sub["variant"] == v]["effective_rank"].values[0] if len(sub[sub["variant"] == v]) > 0 else 0 for v in variants]
            ax.bar(x + i * width, vals, width, label=corpus, color=colors.get(corpus), alpha=0.8)
        ax.set_xticks(x + width)
        ax.set_xticklabels(["Real", "Permuted", "Rand Freq", "Rand Uniform"])
        ax.set_ylabel("Effective Rank")
        ax.set_title("Effective Rank by Corpus and Control")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    # Panel 4: Sequence length distribution
    ax = axes[1, 1]
    for corpus_name, csv_path, color in [
        ("Synthetic PCFG", "data/raw/lost_language/corpus_synthetic.csv", "#2E86AB"),
        ("Rongorongo-like", "data/raw/lost_language/corpus_rongorongo_v2.csv", "#A23B72"),
        ("Indus Real", "data/raw/lost_language/corpus_indus_real.csv", "#F18F01"),
    ]:
        if not Path(csv_path).exists():
            continue
        df = pd.read_csv(csv_path)
        lengths = df.groupby("line_id").size()
        bins = np.arange(1, lengths.max() + 2)
        ax.hist(lengths, bins=bins, alpha=0.5, label=corpus_name, color=color, edgecolor="black")
    ax.set_xlabel("Sequence Length (tokens)")
    ax.set_ylabel("Count")
    ax.set_title("Sequence Length Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Saved comparison figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate corpus comparison visualizations")
    parser.add_argument("--output", default="reports/figures/corpus_comparison.png")
    args = parser.parse_args()
    plot_corpus_comparison(args.output)


if __name__ == "__main__":
    main()
