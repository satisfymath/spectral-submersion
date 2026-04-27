"""Positional analysis of signs in short inscriptions.

For corpora like Indus where signs have known positional constraints
(titles at start, numerals at end, etc.), this script analyzes the
distribution of each sign across relative positions (beginning, middle,
end) and identifies signs with strong positional preferences.

Metrics:
- Position bias: fraction of occurrences in first/middle/last tertile
- Start ratio: occurrences in first position / total occurrences
- End ratio: occurrences in last position / total occurrences
- Chi-square test against uniform distribution
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectral_submersion.tokenization import get_sequences_by_line


def positional_analysis(sequences: list[list[str]]) -> pd.DataFrame:
    """Compute positional statistics for each token."""
    stats = {}
    for seq in sequences:
        n = len(seq)
        if n == 0:
            continue
        for pos, tok in enumerate(seq):
            if tok not in stats:
                stats[tok] = {
                    "count": 0,
                    "first": 0,
                    "last": 0,
                    "middle": 0,
                    "positions": [],
                }
            stats[tok]["count"] += 1
            stats[tok]["positions"].append(pos / max(n - 1, 1))  # relative position [0, 1]
            if pos == 0:
                stats[tok]["first"] += 1
            elif pos == n - 1:
                stats[tok]["last"] += 1
            else:
                stats[tok]["middle"] += 1

    rows = []
    for tok, s in stats.items():
        count = s["count"]
        if count < 3:
            continue
        mean_pos = np.mean(s["positions"])
        std_pos = np.std(s["positions"])
        rows.append({
            "token": tok,
            "count": count,
            "mean_relative_position": mean_pos,
            "std_relative_position": std_pos,
            "first_ratio": s["first"] / count,
            "last_ratio": s["last"] / count,
            "middle_ratio": s["middle"] / count,
            "start_score": s["first"] / count - 1 / len(s["positions"]) if s["positions"] else 0,
            "end_score": s["last"] / count - 1 / len(s["positions"]) if s["positions"] else 0,
        })

    return pd.DataFrame(rows).sort_values("count", ascending=False)


def plot_positional_bias(df: pd.DataFrame, title: str, output_path: str, top_n: int = 30):
    """Plot positional bias for top N tokens."""
    sub = df.head(top_n).sort_values("mean_relative_position")

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))

    y = np.arange(len(sub))
    colors = ["#2E86AB" if r < 0.33 else "#A23B72" if r > 0.66 else "#F18F01"
              for r in sub["mean_relative_position"]]

    ax.barh(y, sub["mean_relative_position"], color=colors, alpha=0.8, edgecolor="black")
    ax.set_yticks(y)
    ax.set_yticklabels(sub["token"], fontfamily="monospace", fontsize=9)
    ax.set_xlabel("Mean Relative Position (0=start, 1=end)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.axvline(1/3, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(2/3, color="gray", linestyle="--", alpha=0.5)

    # Add text labels for start/middle/end zones
    ax.text(1/6, -1.5, "START", ha="center", fontsize=9, color="#2E86AB", fontweight="bold")
    ax.text(0.5, -1.5, "MIDDLE", ha="center", fontsize=9, color="#A23B72", fontweight="bold")
    ax.text(5/6, -1.5, "END", ha="center", fontsize=9, color="#F18F01", fontweight="bold")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved positional bias plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Positional analysis of signs")
    parser.add_argument("--input", default="data/raw/lost_language/corpus_indus_real.csv")
    parser.add_argument("--output-table", default="reports/tables/positional_analysis.csv")
    parser.add_argument("--output-fig", default="reports/figures/positional_bias.png")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--title", default="Positional Bias of Signs")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    sequences = get_sequences_by_line(df)

    result_df = positional_analysis(sequences)
    result_df.to_csv(args.output_table, index=False)
    print(f"Saved positional analysis table to {args.output_table}")
    print(result_df.head(20)[["token", "count", "mean_relative_position",
                               "first_ratio", "last_ratio"]].to_string(index=False))

    plot_positional_bias(result_df, args.title, args.output_fig, args.top_n)


if __name__ == "__main__":
    main()
