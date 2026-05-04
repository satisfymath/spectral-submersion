"""Generate final hypothesis report."""

from pathlib import Path

import numpy as np
import pandas as pd

from spectral_submersion.frequency import token_frequencies, entropy
from spectral_submersion.spectral import effective_rank
from spectral_submersion.reporting import generate_markdown_report


def main():
    in_path = Path("data/processed/lost_tokens.csv")
    sv_path = Path("data/processed/singular_values_lost.npy")
    out_report = Path("reports/final/first_hypothesis_report.md")

    df = pd.read_csv(in_path)
    tokens = df["token"].tolist()
    freq = token_frequencies(tokens)
    H = entropy(freq["probability"].values)

    S = np.load(sv_path) if sv_path.exists() else np.array([])
    r_eff = effective_rank(S) if S.size > 0 else 0.0

    metadata = {
        "num_docs": df["doc_id"].nunique(),
        "num_lines": df.groupby(["doc_id", "line_id"]).ngroups,
        "num_tokens": len(df),
        "vocab_size": df["token"].nunique(),
    }
    freq_stats = {
        "entropy": H,
        "top_tokens": freq.head(10)["token"].tolist(),
    }
    spectral_stats = {
        "effective_rank": r_eff,
        "embedding_dim": len(S) if S.size > 0 else 0,
    }

    generate_markdown_report(
        metadata=metadata,
        frequency_stats=freq_stats,
        spectral_stats=spectral_stats,
        output_path=out_report,
    )

    print(f"Report saved to {out_report}")


if __name__ == "__main__":
    main()
