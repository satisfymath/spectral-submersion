"""Train n-gram models and compute perplexity.

Implements character/sign-level n-gram language models with Laplace
(add-1) smoothing. Tests n=1 (unigram), n=2 (bigram), n=3 (trigram).
Perplexity is computed on held-out test sequences and compared against
permuted/random baselines. Lower perplexity = better model fit = more
predictable structure.

Also reports cross-entropy in bits per sign.
"""

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from spectral_submersion.tokenization import get_sequences_by_line


def train_ngram(sequences: list[list[str]], n: int) -> tuple[dict, dict]:
    """Train n-gram model with Laplace smoothing."""
    vocab = set()
    for seq in sequences:
        vocab.update(seq)
    vocab = sorted(vocab)
    vocab_size = len(vocab)
    token_to_id = {t: i for i, t in enumerate(vocab)}

    # Count n-grams and (n-1)-grams
    ngram_counts = Counter()
    context_counts = Counter()

    for seq in sequences:
        padded = ["<s>"] * (n - 1) + seq + ["</s>"]
        for i in range(n - 1, len(padded)):
            ngram = tuple(padded[i - n + 1 : i + 1])
            context = ngram[:-1]
            ngram_counts[ngram] += 1
            context_counts[context] += 1

    return ngram_counts, context_counts, vocab_size, token_to_id, vocab


def compute_perplexity(
    sequences: list[list[str]],
    ngram_counts: Counter,
    context_counts: Counter,
    vocab_size: int,
    n: int,
) -> float:
    """Compute perplexity on test sequences with Laplace smoothing."""
    log_prob = 0.0
    token_count = 0

    for seq in sequences:
        padded = ["<s>"] * (n - 1) + seq + ["</s>"]
        for i in range(n - 1, len(padded)):
            ngram = tuple(padded[i - n + 1 : i + 1])
            context = ngram[:-1]
            count_ngram = ngram_counts.get(ngram, 0)
            count_context = context_counts.get(context, 0)
            # Laplace smoothing
            prob = (count_ngram + 1) / (count_context + vocab_size)
            log_prob += math.log2(prob)
            token_count += 1

    if token_count == 0:
        return float("inf")
    cross_entropy = -log_prob / token_count
    perplexity = 2**cross_entropy
    return perplexity, cross_entropy


def split_train_test(
    sequences: list[list[str]], test_ratio: float = 0.2, seed: int = 42
) -> tuple:
    """Split sequences into train and test."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(sequences))
    rng.shuffle(indices)
    split = int(len(sequences) * (1 - test_ratio))
    train_idx = indices[:split]
    test_idx = indices[split:]
    train = [sequences[i] for i in train_idx]
    test = [sequences[i] for i in test_idx]
    return train, test


def main():
    parser = argparse.ArgumentParser(
        description="Train n-gram models and compute perplexity"
    )
    parser.add_argument(
        "--input", default="data/raw/lost_language/corpus_indus_real.csv"
    )
    parser.add_argument("--output", default="reports/tables/ngram_perplexity.csv")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--max-n", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    sequences = get_sequences_by_line(df)

    # Generate controls
    rng = np.random.default_rng(args.seed)
    all_tokens = [t for s in sequences for t in s]
    permuted = []
    for s in sequences:
        perm = list(s)
        rng.shuffle(perm)
        permuted.append(perm)

    # Random with same frequency
    freq_dist = pd.Series(all_tokens).value_counts(normalize=True)
    random_seqs = []
    for s in sequences:
        random_seqs.append(
            rng.choice(freq_dist.index, size=len(s), p=freq_dist.values).tolist()
        )

    results = []

    for corpus_name, corpus_seqs in [
        ("real", sequences),
        ("permuted", permuted),
        ("random_freq", random_seqs),
    ]:
        train, test = split_train_test(
            corpus_seqs, test_ratio=args.test_ratio, seed=args.seed
        )

        for n in range(1, args.max_n + 1):
            ngram_counts, context_counts, vocab_size, token_to_id, vocab = train_ngram(
                train, n
            )
            ppl, ce = compute_perplexity(
                test, ngram_counts, context_counts, vocab_size, n
            )
            results.append(
                {
                    "corpus": corpus_name,
                    "n": n,
                    "vocab_size": vocab_size,
                    "train_tokens": sum(len(s) for s in train),
                    "test_tokens": sum(len(s) for s in test),
                    "perplexity": ppl,
                    "cross_entropy_bits": ce,
                }
            )
            print(
                f"{corpus_name:12s} n={n} | vocab={vocab_size} | perplexity={ppl:.2f} | cross-entropy={ce:.3f} bits/sign"
            )

    results_df = pd.DataFrame(results)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False)
    print(f"\nSaved results to {args.output}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Real vs Controls")
    print("=" * 60)
    pivot = results_df.pivot_table(index="n", columns="corpus", values="perplexity")
    print(pivot.to_string())


if __name__ == "__main__":
    main()
