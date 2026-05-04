"""Conditional entropy analysis for short inscription corpora.

For corpora with very short sequences (e.g., Indus script, ~5 signs per
inscription), window-based co-occurrence is noisy because there are few
context pairs per sequence. This script computes conditional entropy of
bigrams and trigrams directly, which works even for short sequences.

Metrics:
- H(token) : unconditional entropy
- H(token|prev) : conditional entropy given previous token
- H(token|prev,prev2) : conditional entropy given previous two tokens
- Perplexity : 2^H for each level
- Comparison with permuted and random controls

Reference: Rao et al. (2009) "Entropic Evidence for Linguistic Structure
in the Indus Script", Science 324(5931):1165.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spectral_submersion.evaluation import (
    permute_corpus,
    random_corpus_same_frequency,
    random_corpus_uniform,
)
from spectral_submersion.tokenization import get_sequences_by_line


def entropy(probs: np.ndarray) -> float:
    """Compute Shannon entropy in bits."""
    p = np.asarray(probs, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def conditional_entropy_bigram(sequences: list[list[str]]) -> dict:
    """Compute H(Y|X) from bigram counts."""
    unigram_counts = {}
    bigram_counts = {}

    for seq in sequences:
        for i, tok in enumerate(seq):
            unigram_counts[tok] = unigram_counts.get(tok, 0) + 1
            if i > 0:
                prev = seq[i - 1]
                bigram_counts[(prev, tok)] = bigram_counts.get((prev, tok), 0) + 1

    total_unigrams = sum(unigram_counts.values())
    total_bigrams = sum(bigram_counts.values())

    # H(Y|X) = sum_x p(x) * H(Y|X=x)
    h_cond = 0.0
    for prev, count_prev in unigram_counts.items():
        p_prev = count_prev / total_unigrams
        # Get all bigrams starting with prev
        sub_counts = {k: v for k, v in bigram_counts.items() if k[0] == prev}
        if not sub_counts:
            continue
        sub_total = sum(sub_counts.values())
        sub_probs = np.array([v / sub_total for v in sub_counts.values()])
        h_sub = entropy(sub_probs)
        h_cond += p_prev * h_sub

    # H(Y) unconditional
    unigram_probs = np.array([v / total_unigrams for v in unigram_counts.values()])
    h_uncond = entropy(unigram_probs)

    return {
        "h_unconditional": h_uncond,
        "h_bigram_conditional": h_cond,
        "perplexity_unconditional": 2**h_uncond,
        "perplexity_bigram": 2**h_cond,
        "n_unigrams": len(unigram_counts),
        "n_bigrams": len(bigram_counts),
        "total_tokens": total_unigrams,
    }


def conditional_entropy_trigram(sequences: list[list[str]]) -> dict:
    """Compute H(Z|X,Y) from trigram counts."""
    bigram_counts = {}
    trigram_counts = {}

    for seq in sequences:
        for i, tok in enumerate(seq):
            if i > 0:
                prev = seq[i - 1]
                bigram_counts[(prev, tok)] = bigram_counts.get((prev, tok), 0) + 1
            if i > 1:
                prev2 = seq[i - 2]
                prev = seq[i - 1]
                trigram_counts[(prev2, prev, tok)] = (
                    trigram_counts.get((prev2, prev, tok), 0) + 1
                )

    total_bigrams = sum(bigram_counts.values())
    total_trigrams = sum(trigram_counts.values())

    h_cond = 0.0
    for (prev2, prev), count_bp in bigram_counts.items():
        p_bp = count_bp / total_bigrams
        sub_counts = {
            k: v for k, v in trigram_counts.items() if k[0] == prev2 and k[1] == prev
        }
        if not sub_counts:
            continue
        sub_total = sum(sub_counts.values())
        sub_probs = np.array([v / sub_total for v in sub_counts.values()])
        h_sub = entropy(sub_probs)
        h_cond += p_bp * h_sub

    return {
        "h_trigram_conditional": h_cond,
        "perplexity_trigram": 2**h_cond if h_cond > 0 else 0.0,
        "n_trigrams": len(trigram_counts),
        "total_bigrams": total_bigrams,
    }


def analyze_sequences(sequences: list[list[str]], label: str) -> dict:
    """Run full entropy analysis on a set of sequences."""
    result = {"variant": label}
    result.update(conditional_entropy_bigram(sequences))
    result.update(conditional_entropy_trigram(sequences))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Conditional entropy analysis for short inscriptions"
    )
    parser.add_argument(
        "--input", default="data/raw/lost_language/corpus_indus_real.csv"
    )
    parser.add_argument("--output", default="reports/tables/entropy_analysis.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    sequences = get_sequences_by_line(df)

    # Build vocab for uniform random
    vocab = list({tok for seq in sequences for tok in seq})

    results = []

    # Real corpus
    results.append(analyze_sequences(sequences, "real"))

    # Permuted
    perm_seqs = permute_corpus(sequences)
    results.append(analyze_sequences(perm_seqs, "permuted"))

    # Random same frequency
    rand_freq_seqs = random_corpus_same_frequency(sequences)
    results.append(analyze_sequences(rand_freq_seqs, "random_same_freq"))

    # Random uniform
    rand_uniform_seqs = random_corpus_uniform(sequences, vocab)
    results.append(analyze_sequences(rand_uniform_seqs, "random_uniform"))

    out_df = pd.DataFrame(results)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print("=" * 70)
    print("Conditional Entropy Analysis")
    print("=" * 70)
    print(
        out_df[
            [
                "variant",
                "h_unconditional",
                "h_bigram_conditional",
                "h_trigram_conditional",
                "perplexity_unconditional",
                "perplexity_bigram",
                "perplexity_trigram",
                "n_unigrams",
                "n_bigrams",
                "n_trigrams",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved to {out_path}")

    # Sanity check: real should have lower conditional entropy than uniform
    real_h = out_df[out_df["variant"] == "real"]["h_bigram_conditional"].values[0]
    uniform_h = out_df[out_df["variant"] == "random_uniform"][
        "h_bigram_conditional"
    ].values[0]
    print(
        f"\n[SANITY CHECK] Real bigram H={real_h:.4f} vs Uniform bigram H={uniform_h:.4f}"
    )
    if real_h < uniform_h:
        print("  PASS: Real corpus has lower conditional entropy than uniform random.")
    else:
        print(
            "  FAIL: Real corpus does NOT have lower conditional entropy than uniform random."
        )
        print(
            "  This suggests the corpus lacks sequential predictability at the bigram level."
        )


if __name__ == "__main__":
    main()
