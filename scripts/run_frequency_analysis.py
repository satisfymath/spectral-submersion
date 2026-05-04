"""Run frequency analysis on processed corpus."""

from pathlib import Path

import pandas as pd

from spectral_submersion.frequency import token_frequencies, entropy
from spectral_submersion.visualization import plot_token_frequencies


def main():
    in_path = Path("data/processed/lost_tokens.csv")
    out_table = Path("reports/tables/frequency_table.csv")
    out_fig = Path("reports/figures/token_frequency.png")

    df = pd.read_csv(in_path)
    tokens = df["token"].tolist()

    freq = token_frequencies(tokens)
    H = entropy(freq["probability"].values)

    out_table.parent.mkdir(parents=True, exist_ok=True)
    freq.to_csv(out_table, index=False)

    plot_token_frequencies(
        ranks=freq["rank"].values,
        counts=freq["count"].values,
        title="Token Frequency Distribution",
        save_path=out_fig,
    )

    print(f"Frequency table saved to {out_table}")
    print(f"Figure saved to {out_fig}")
    print(f"Entropy: {H:.4f} nats")
    print(f"Top 5 tokens:")
    print(freq.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
