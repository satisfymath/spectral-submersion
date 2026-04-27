"""Run negative controls and compare spectral metrics.

Generates three negative controls from the synthetic corpus:
1. Permuted: shuffle tokens globally, preserve sequence lengths.
2. Random-same-freq: sample tokens with replacement from empirical distribution.
3. Random-uniform: sample tokens uniformly from vocabulary.

Then computes co-occurrence, PPMI, SVD, and effective rank for each.
Results are saved as CSV for comparison.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from spectral_submersion.cooccurrence import (
    build_vocab,
    cooccurrence_matrix_from_sequences,
)
from spectral_submersion.evaluation import (
    permute_corpus,
    random_corpus_same_frequency,
    random_corpus_uniform,
)
from spectral_submersion.pmi import ppmi_matrix
from spectral_submersion.spectral import spectral_embedding, effective_rank
from spectral_submersion.tokenization import get_sequences_by_line, tokens_to_ids


def build_svd_pipeline(sequences: list[list[str]], k: int = 16, seed: int = 42):
    vocab = build_vocab([tok for seq in sequences for tok in seq])
    seq_ids = [tokens_to_ids(seq, vocab) for seq in sequences]
    C = cooccurrence_matrix_from_sequences(seq_ids, len(vocab), window_size=3)
    M = ppmi_matrix(C)
    E, S, Vt = spectral_embedding(M, k=k, alpha=0.5, random_state=seed)
    r_eff = effective_rank(S)
    return {
        "vocab_size": len(vocab),
        "effective_rank": r_eff,
        "top_sv": float(S[0]) if len(S) > 0 else 0.0,
        "sum_sv": float(S.sum()),
        "embedding_shape": str(E.shape),
    }


def main():
    parser = argparse.ArgumentParser(description="Run negative controls")
    parser.add_argument("--input", default="data/raw/lost_language/corpus_synthetic.csv")
    parser.add_argument("--output", default="reports/tables/control_comparison.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    sequences = get_sequences_by_line(df)

    results = []

    # Real corpus
    real_stats = build_svd_pipeline(sequences)
    real_stats["variant"] = "real"
    results.append(real_stats)

    # Permuted
    perm_seqs = permute_corpus(sequences)
    perm_stats = build_svd_pipeline(perm_seqs)
    perm_stats["variant"] = "permuted"
    results.append(perm_stats)

    # Random same frequency
    rand_freq_seqs = random_corpus_same_frequency(sequences)
    rand_freq_stats = build_svd_pipeline(rand_freq_seqs)
    rand_freq_stats["variant"] = "random_same_freq"
    results.append(rand_freq_stats)

    # Random uniform
    vocab_list = sorted({tok for seq in sequences for tok in seq})
    rand_unif_seqs = random_corpus_uniform(sequences, vocab_list)
    rand_unif_stats = build_svd_pipeline(rand_unif_seqs)
    rand_unif_stats["variant"] = "random_uniform"
    results.append(rand_unif_stats)

    out_df = pd.DataFrame(results)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    print(out_df.to_string(index=False))
    print(f"\nSaved comparison to {args.output}")

    # Basic sanity check: real should have lower effective rank than uniform
    real_r = out_df[out_df["variant"] == "real"]["effective_rank"].values[0]
    unif_r = out_df[out_df["variant"] == "random_uniform"]["effective_rank"].values[0]
    if real_r < unif_r:
        print(f"\n[SANITY CHECK PASSED] Real effective rank ({real_r:.2f}) < Uniform ({unif_r:.2f})")
    else:
        print(f"\n[SANITY CHECK WARNING] Real effective rank ({real_r:.2f}) >= Uniform ({unif_r:.2f})")


if __name__ == "__main__":
    main()
