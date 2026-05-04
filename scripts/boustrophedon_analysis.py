"""Boustrophedon-aware sequence modeling for Rongorongo.

Rongorongo is believed to be written in boustrophedon (alternating direction):
- Odd lines read left-to-right
- Even lines read right-to-left (and possibly upside-down)

This script:
1. Identifies line pairs and their orientation
2. Builds boustrophedon-aware bigrams (crossing line boundaries)
3. Computes co-occurrence matrices with boustrophedon direction
4. Compares spectral properties of boustrophedon vs linear co-occurrence
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
from scipy.linalg import svd

from spectral_submersion.tokenization import read_corpus, get_sequences_by_line


def build_boustrophedon_sequences(sequences, reverse_even=True):
    """Build boustrophedon-aware sequences.

    For even-indexed lines (0-based), reverse the token order to simulate
    reading right-to-left as in boustrophedon writing.

    Returns list of sequences where even lines are reversed.
    """
    boustrophedon = []
    for i, seq in enumerate(sequences):
        if reverse_even and i % 2 == 1:
            boustrophedon.append(list(reversed(seq)))
        else:
            boustrophedon.append(list(seq))
    return boustrophedon


def build_cross_line_bigrams(sequences, reverse_even=True):
    """Build bigrams that cross line boundaries in boustrophedon reading.

    In boustrophedon, the last token of line i connects to the last token
    of line i+1 (because line i+1 reads right-to-left and starts where line i ends).
    """
    cross_bigrams = Counter()
    for i in range(len(sequences) - 1):
        seq_i = sequences[i]
        seq_next = sequences[i + 1]
        if not seq_i or not seq_next:
            continue

        if i % 2 == 0:
            # Odd line ends at position len-1
            # Next line (even, reversed) starts at position len-1
            token_end = seq_i[-1]
            if reverse_even and (i + 1) % 2 == 1:
                token_start = seq_next[-1]  # reversed, so "start" is last position
            else:
                token_start = seq_next[0]
            cross_bigrams[(token_end, token_start)] += 1
        else:
            # Even line (reversed) ends at position 0
            # Next line (odd, normal) starts at position 0
            if reverse_even:
                token_end = seq_i[0]  # reversed, so "end" is first position
            else:
                token_end = seq_i[-1]
            token_start = seq_next[0]
            cross_bigrams[(token_end, token_start)] += 1

    return cross_bigrams


def build_cooccurrence_matrix(sequences, vocab, window=2):
    """Build window-based co-occurrence matrix from sequences."""
    token_to_idx = {t: i for i, t in enumerate(vocab)}
    n = len(vocab)
    cooc = np.zeros((n, n), dtype=np.float64)

    for seq in sequences:
        for i, tok in enumerate(seq):
            if tok not in token_to_idx:
                continue
            idx_i = token_to_idx[tok]
            for j in range(max(0, i - window), min(len(seq), i + window + 1)):
                if i == j:
                    continue
                tok_j = seq[j]
                if tok_j not in token_to_idx:
                    continue
                idx_j = token_to_idx[tok_j]
                weight = 1.0 / abs(i - j)
                cooc[idx_i, idx_j] += weight

    return cooc


def build_pmi_matrix(cooc):
    """Build PPMI matrix from co-occurrence matrix."""
    total = cooc.sum()
    if total == 0:
        return cooc

    row_sums = cooc.sum(axis=1)
    col_sums = cooc.sum(axis=0)

    pmi = np.zeros_like(cooc)
    for i in range(cooc.shape[0]):
        for j in range(cooc.shape[1]):
            if cooc[i, j] > 0 and row_sums[i] > 0 and col_sums[j] > 0:
                pmi[i, j] = np.log2(
                    cooc[i, j] * total / (row_sums[i] * col_sums[j] + 1e-10)
                )

    pmi = np.maximum(pmi, 0)
    return pmi


def compute_effective_rank(E, d=16):
    """Compute effective rank (r_eff) from SVD singular values."""
    if E.shape[1] < 2:
        return 1.0
    _, s, _ = svd(E[:, :d], full_matrices=False)
    s = s[s > 1e-10]
    if len(s) == 0:
        return 0.0
    r_eff = np.sum(s**2) / s[0] ** 2
    return r_eff


def main():
    out_dir = Path("reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    rr_path = "data/raw/lost_language/corpus_rongorongo_real.xml.csv"
    rr_df = read_corpus(rr_path)
    rr_seqs = get_sequences_by_line(rr_df)

    print(
        f"Rongorongo: {len(rr_seqs)} lines, "
        f"{sum(len(s) for s in rr_seqs)} total tokens"
    )
    print(
        f"Line lengths: min={min(len(s) for s in rr_seqs if s)}, "
        f"max={max(len(s) for s in rr_seqs)}, "
        f"mean={np.mean([len(s) for s in rr_seqs]):.1f}"
    )

    # Build vocabulary
    all_tokens = [t for seq in rr_seqs for t in seq]
    freq = Counter(all_tokens)
    vocab = sorted(freq.keys())
    print(f"Vocabulary size: {len(vocab)}")

    # ====== Experiment 1: Co-occurrence matrices ======
    print("\n=== Experiment 1: Co-occurrence Matrices ===")

    # Linear (standard) sequences
    cooc_linear = build_cooccurrence_matrix(rr_seqs, vocab, window=2)
    pmi_linear = build_pmi_matrix(cooc_linear)

    # Boustrophedon sequences
    rr_boust = build_boustrophedon_sequences(rr_seqs, reverse_even=True)
    cooc_boust = build_cooccurrence_matrix(rr_boust, vocab, window=2)
    pmi_boust = build_pmi_matrix(cooc_boust)

    # Random direction baseline: randomly reverse half the lines
    rng = np.random.default_rng(42)
    rr_random = []
    for i, seq in enumerate(rr_seqs):
        if rng.random() < 0.5:
            rr_random.append(list(reversed(seq)))
        else:
            rr_random.append(list(seq))
    cooc_random = build_cooccurrence_matrix(rr_random, vocab, window=2)
    pmi_random = build_pmi_matrix(cooc_random)

    # SVD and effective rank
    d = 16
    r_eff_linear = compute_effective_rank(pmi_linear, d)
    r_eff_boust = compute_effective_rank(pmi_boust, d)
    r_eff_random = compute_effective_rank(pmi_random, d)

    print(f"  Linear (standard) r_eff = {r_eff_linear:.2f}")
    print(f"  Boustrophedon r_eff = {r_eff_boust:.2f}")
    print(f"  Random direction r_eff = {r_eff_random:.2f}")

    # Difference in structure
    delta = r_eff_boust - r_eff_linear
    print(f"  Delta (boust - linear) = {delta:.4f}")
    print(
        f"  {'Boustrophedon IMPROVES structure' if delta < 0 else 'Boustrophedon does NOT improve structure'}"
    )

    # ====== Experiment 2: Cross-line bigram analysis ======
    print("\n=== Experiment 2: Cross-line Bigram Analysis ===")

    cross_boust = build_cross_line_bigrams(rr_seqs, reverse_even=True)
    cross_linear = build_cross_line_bigrams(rr_seqs, reverse_even=False)

    # Compare bigram distributions
    total_boust = sum(cross_boust.values())
    total_linear = sum(cross_linear.values())

    print(f"  Total cross-line bigrams (boustrophedon): {total_boust}")
    print(f"  Total cross-line bigrams (linear): {total_linear}")

    # Top cross-line bigrams in boustrophedon
    print("\n  Top-10 boustrophedon cross-line bigrams:")
    for (t1, t2), cnt in cross_boust.most_common(10):
        print(f"    {t1:10s} -> {t2:10s}: {cnt}")

    print("\n  Top-10 linear cross-line bigrams:")
    for (t1, t2), cnt in cross_linear.most_common(10):
        print(f"    {t1:10s} -> {t2:10s}: {cnt}")

    # Entropy of cross-line bigram distributions
    def bigram_entropy(bigrams):
        counts = np.array(list(bigrams.values()), dtype=np.float64)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-30))

    h_boust = bigram_entropy(cross_boust)
    h_linear = bigram_entropy(cross_linear)
    print(f"\n  Cross-line bigram entropy (boustrophedon): {h_boust:.2f} bits")
    print(f"  Cross-line bigram entropy (linear): {h_linear:.2f} bits")
    print(f"  Lower entropy = more structured cross-line patterns")
    print(
        f"  {'Boustrophedon has more structured cross-line patterns' if h_boust < h_linear else 'Linear reading has more structured cross-line patterns'}"
    )

    # ====== Experiment 3: Leave-one-line-out predictability ======
    print("\n=== Experiment 3: Line Boundary Predictability ===")

    # For each line boundary, compute how predictable the start of line i+1
    # is given the end of line i (in both reading orders)

    success_boust = 0
    success_linear = 0
    n_boundaries = 0

    for i in range(len(rr_seqs) - 1):
        seq_i = rr_seqs[i]
        seq_next = rr_seqs[i + 1]
        if not seq_i or not seq_next:
            continue

        n_boundaries += 1
        end_token = seq_i[-1]

        # Linear: next line starts at position 0
        start_linear = seq_next[0]

        # Boustrophedon: odd lines reversed, so end->start
        if i % 2 == 0:
            start_boust = seq_next[
                -1
            ]  # even line ends, next (odd, reversed) starts from end
        else:
            start_boust = seq_next[
                0
            ]  # odd line (reversed) ends, next (even) starts from start

    # ====== Experiment 4: Plausibility of boustrophedon reading ======
    print("\n=== Experiment 4: Boustrophedon Plausibility ===")

    # Compare within-line structure preservation
    # In boustrophedon, even lines should read as coherent sequences when reversed

    within_line_entropies_boust = []
    within_line_entropies_linear = []

    for i, seq in enumerate(rr_seqs):
        if len(seq) < 3:
            continue

        # Compute bigram entropy for this line
        bigrams_fwd = Counter()
        for j in range(len(seq) - 1):
            bigrams_fwd[(seq[j], seq[j + 1])] += 1
        h_fwd = bigram_entropy(bigrams_fwd)
        within_line_entropies_linear.append(h_fwd)

        # If boustrophedon, reversed even lines should also have structure
        if i % 2 == 1:
            seq_rev = list(reversed(seq))
        else:
            seq_rev = seq

        bigrams_rev = Counter()
        for j in range(len(seq_rev) - 1):
            bigrams_rev[(seq_rev[j], seq_rev[j + 1])] += 1
        h_rev = bigram_entropy(bigrams_rev)
        within_line_entropies_boust.append(h_rev)

    print(
        f"  Mean within-line entropy (linear reading): {np.mean(within_line_entropies_linear):.3f}"
    )
    print(
        f"  Mean within-line entropy (boustrophedon reading): {np.mean(within_line_entropies_boust):.3f}"
    )

    # Per-line entropy comparison
    # In boustrophedon, even lines when reversed should maintain similar structure
    even_entropies_fwd = []
    even_entropies_rev = []
    odd_entropies = []

    for i, seq in enumerate(rr_seqs):
        if len(seq) < 3:
            continue
        # Forward bigram entropy
        bigrams = Counter()
        for j in range(len(seq) - 1):
            bigrams[(seq[j], seq[j + 1])] += 1
        h = bigram_entropy(bigrams)

        if i % 2 == 1:  # Even line (boustrophedon: reversed)
            bigrams_rev = Counter()
            seq_rev = list(reversed(seq))
            for j in range(len(seq_rev) - 1):
                bigrams_rev[(seq_rev[j], seq_rev[j + 1])] += 1
            h_rev = bigram_entropy(bigrams_rev)
            even_entropies_fwd.append(h)
            even_entropies_rev.append(h_rev)
        else:
            odd_entropies.append(h)

    if even_entropies_fwd:
        print(f"\n  Even lines (potentially reversed in boustrophedon):")
        print(f"    Forward mean entropy: {np.mean(even_entropies_fwd):.3f}")
        print(f"    Reversed mean entropy: {np.mean(even_entropies_rev):.3f}")
        print(
            f"    {'Reversal REDUCES entropy (boustrophedon plausible)' if np.mean(even_entropies_rev) < np.mean(even_entropies_fwd) else 'Reversal INCREASES entropy (boustrophedon less plausible)'}"
        )

    print(f"\n  Odd lines (forward reading):")
    print(f"    Mean entropy: {np.mean(odd_entropies):.3f}")

    # ====== Experiment 5: Tablet-by-tablet analysis ======
    print("\n=== Experiment 5: Tablet-by-Tablet Analysis ===")

    for doc_id in sorted(rr_df["doc_id"].unique()):
        doc_seqs = [
            seq
            for seq, df_seq in [(s, rr_df[rr_df["doc_id"] == doc_id]) for s in rr_seqs]
            if len(seq) > 0
        ][
            :1
        ]  # just check if doc exists

        # Filter sequences by doc_id
        doc_groups = rr_df[rr_df["doc_id"] == doc_id].groupby("line_id")
        doc_sequences = []
        for _, group in doc_groups:
            seq = group.sort_values("position")["token"].tolist()
            doc_sequences.append(seq)

        if not doc_sequences:
            continue

        # Compute boustrophedon co-occurrence for this tablet
        doc_boust = build_boustrophedon_sequences(doc_sequences, reverse_even=True)
        doc_linear_seqs = doc_sequences

        total_tokens = sum(len(s) for s in doc_sequences)
        n_lines = len(doc_sequences)

        # Compute entropy difference
        bigrams_l = Counter()
        bigrams_b = Counter()
        for seq in doc_linear_seqs:
            for j in range(len(seq) - 1):
                bigrams_l[(seq[j], seq[j + 1])] += 1
        for seq in doc_boust:
            for j in range(len(seq) - 1):
                bigrams_b[(seq[j], seq[j + 1])] += 1

        h_l = bigram_entropy(bigrams_l)
        h_b = bigram_entropy(bigrams_b)

        print(
            f"  Tablet {doc_id}: {n_lines} lines, {total_tokens} tokens, "
            f"H_linear={h_l:.2f}, H_boust={h_b:.2f}, "
            f"delta={h_b - h_l:.3f}"
        )

    # Save results
    results = pd.DataFrame(
        [
            {
                "experiment": "cooccurrence",
                "r_eff_linear": r_eff_linear,
                "r_eff_boustrophedon": r_eff_boust,
                "r_eff_random": r_eff_random,
                "delta": delta,
            }
        ]
    )
    results.to_csv(out_dir / "boustrophedon_results.csv", index=False)
    print(f"\nResults saved to {out_dir}/boustrophedon_results.csv")


if __name__ == "__main__":
    main()
