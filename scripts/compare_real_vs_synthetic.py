"""Compare Indus real corpus against Indus-like synthetic benchmark.

Generates side-by-side statistics: vocabulary overlap, length distribution,
frequency distribution (Zipf fit), positional bias patterns, and spectral
properties. Quantifies how well the synthetic model approximates the real data.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from spectral_submersion.tokenization import get_sequences_by_line


def zipf_fit(ranks: np.ndarray, counts: np.ndarray) -> tuple[float, float, float]:
    """Fit Zipf law: log(count) = log(C) - s * log(rank). Returns (s, C, r2)."""
    log_ranks = np.log(ranks)
    log_counts = np.log(counts)
    slope, intercept, r_value, _, _ = stats.linregress(log_ranks, log_counts)
    return -slope, np.exp(intercept), r_value ** 2


def compare_corpora(real_path: str, synth_path: str) -> dict:
    """Compare real and synthetic corpora across multiple dimensions."""
    real_df = pd.read_csv(real_path)
    synth_df = pd.read_csv(synth_path)

    real_seqs = get_sequences_by_line(real_df)
    synth_seqs = get_sequences_by_line(synth_df)

    # Basic stats
    real_tokens = sum(len(s) for s in real_seqs)
    synth_tokens = sum(len(s) for s in synth_seqs)

    real_vocab = set()
    for s in real_seqs:
        real_vocab.update(s)
    synth_vocab = set()
    for s in synth_seqs:
        synth_vocab.update(s)

    real_lengths = [len(s) for s in real_seqs]
    synth_lengths = [len(s) for s in synth_seqs]

    real_freq = pd.Series([t for s in real_seqs for t in s]).value_counts()
    synth_freq = pd.Series([t for s in synth_seqs for t in s]).value_counts()

    # Zipf fit
    real_ranks = np.arange(1, len(real_freq) + 1)
    real_s, real_C, real_r2 = zipf_fit(real_ranks, real_freq.values)
    synth_ranks = np.arange(1, len(synth_freq) + 1)
    synth_s, synth_C, synth_r2 = zipf_fit(synth_ranks, synth_freq.values)

    results = {
        "real_n_inscriptions": len(real_seqs),
        "synth_n_inscriptions": len(synth_seqs),
        "real_n_tokens": real_tokens,
        "synth_n_tokens": synth_tokens,
        "real_vocab_size": len(real_vocab),
        "synth_vocab_size": len(synth_vocab),
        "real_mean_length": np.mean(real_lengths),
        "synth_mean_length": np.mean(synth_lengths),
        "real_std_length": np.std(real_lengths),
        "synth_std_length": np.std(synth_lengths),
        "real_zipf_slope": real_s,
        "synth_zipf_slope": synth_s,
        "real_zipf_r2": real_r2,
        "synth_zipf_r2": synth_r2,
        "vocab_overlap": len(real_vocab & synth_vocab),
    }

    return results, real_freq, synth_freq, real_lengths, synth_lengths


def main():
    parser = argparse.ArgumentParser(description="Compare real and synthetic Indus corpora")
    parser.add_argument("--real", default="data/raw/lost_language/corpus_indus_real.csv")
    parser.add_argument("--synthetic", default="data/raw/lost_language/corpus_indus_like.csv")
    parser.add_argument("--output-table", default="reports/tables/real_vs_synthetic_comparison.csv")
    parser.add_argument("--output-freq", default="reports/tables/frequency_comparison.csv")
    args = parser.parse_args()

    results, real_freq, synth_freq, real_lengths, synth_lengths = compare_corpora(args.real, args.synthetic)

    print("=" * 60)
    print("REAL vs SYNTHETIC COMPARISON")
    print("=" * 60)
    for key, val in results.items():
        print(f"{key}: {val:.4f}" if isinstance(val, float) else f"{key}: {val}")

    # Length distributions
    print("\nLength distribution (real):")
    print(pd.Series(real_lengths).value_counts().sort_index().head(10).to_string())
    print("\nLength distribution (synthetic):")
    print(pd.Series(synth_lengths).value_counts().sort_index().head(10).to_string())

    # Save comparison table
    summary_df = pd.DataFrame([results])
    Path(args.output_table).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output_table, index=False)
    print(f"\nSaved comparison to {args.output_table}")

    # Frequency comparison (top 20)
    top_n = 20
    real_top = real_freq.head(top_n).reset_index()
    real_top.columns = ["token", "real_count"]
    synth_top = synth_freq.head(top_n).reset_index()
    synth_top.columns = ["token", "synth_count"]
    merged = pd.merge(real_top, synth_top, on="token", how="outer").fillna(0)
    merged.to_csv(args.output_freq, index=False)
    print(f"Saved frequency comparison to {args.output_freq}")
    print(merged.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
