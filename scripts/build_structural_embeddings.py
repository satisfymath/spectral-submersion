"""Build structural feature embeddings for Rongorongo tokens.

Instead of window-based co-occurrence (which fails for high TTR, short sequences),
this builds per-token feature vectors directly from:
1. Positional features: P(first), P(last), normalized position statistics
2. Repetition features: P(AA|token), P(AAA|token), mean repeat run length
3. Distributional features: Shannon entropy of context distribution, frequency rank
4. Bigram features: conditional entropy H(next|token), most likely successor entropy

These features encode the structural properties that our analysis has shown to be
the strongest signals in Rongorongo: positional bias and repetition patterns.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy

from spectral_submersion.tokenization import get_sequences_by_line
from spectral_submersion.spectral import effective_rank


def compute_structural_features(df, sequences):
    tokens_all = [t for seq in sequences for t in seq]
    freq = Counter(tokens_all)
    vocab = sorted(freq.keys())
    token_to_idx = {t: i for i, t in enumerate(vocab)}
    n_tokens = len(vocab)

    positions = defaultdict(list)
    first_count = Counter()
    last_count = Counter()
    repeat_run_lengths = defaultdict(list)
    successor_counts = defaultdict(Counter)
    line_lengths = []

    for seq in sequences:
        line_lengths.append(len(seq))
        if not seq:
            continue
        first_count[seq[0]] += 1
        last_count[seq[-1]] += 1
        for pos, tok in enumerate(seq):
            positions[tok].append(pos / max(len(seq) - 1, 1))

        i = 0
        while i < len(seq):
            j = i + 1
            while j < len(seq) and seq[j] == seq[i]:
                j += 1
            run_len = j - i
            if run_len >= 2:
                repeat_run_lengths[seq[i]].append(run_len)
            i = j if run_len > 1 else i + 1

        for i in range(len(seq) - 1):
            successor_counts[seq[i]][seq[i + 1]] += 1

    total_first = sum(first_count.values()) or 1
    total_last = sum(last_count.values()) or 1

    features = np.zeros((n_tokens, 10), dtype=np.float64)
    feature_names = [
        "log_freq",
        "type_token_indicator",
        "first_ratio",
        "last_ratio",
        "pos_mean",
        "pos_std",
        "rep_rate",
        "mean_run_len",
        "successor_entropy",
        "freq_rank_norm",
    ]

    total = sum(freq.values())
    sorted_tokens = sorted(freq.keys(), key=lambda t: freq[t], reverse=True)
    rank_map = {t: i for i, t in enumerate(sorted_tokens)}

    for tok in vocab:
        idx = token_to_idx[tok]
        f = freq[tok]
        features[idx, 0] = np.log1p(f)
        features[idx, 1] = 1.0 if f == 1 else 0.0
        features[idx, 2] = first_count.get(tok, 0) / f
        features[idx, 3] = last_count.get(tok, 0) / f
        pos_list = positions.get(tok, [0.5])
        features[idx, 4] = np.mean(pos_list)
        features[idx, 5] = np.std(pos_list) if len(pos_list) > 1 else 0.0
        repeat_runs = repeat_run_lengths.get(tok, [])
        features[idx, 6] = len(repeat_runs) / f if f > 0 else 0.0
        features[idx, 7] = np.mean(repeat_runs) if repeat_runs else 0.0
        succ = successor_counts.get(tok, Counter())
        if succ:
            probs = np.array(list(succ.values()), dtype=np.float64)
            probs = probs / probs.sum()
            features[idx, 8] = entropy(probs)
        else:
            features[idx, 8] = 0.0
        features[idx, 9] = rank_map.get(tok, n_tokens) / n_tokens

    return vocab, features, feature_names


def compute_structural_features_collapsed(sequences, pure_sequences):
    collapsed_all = [t for seq in sequences for t in seq]
    freq = Counter(collapsed_all)
    vocab = sorted(freq.keys())
    token_to_idx = {t: i for i, t in enumerate(vocab)}
    n_tokens = len(vocab)

    positions = defaultdict(list)
    first_count = Counter()
    last_count = Counter()
    successor_counts = defaultdict(Counter)

    for seq in sequences:
        if not seq:
            continue
        first_count[seq[0]] += 1
        last_count[seq[-1]] += 1
        for pos, tok in enumerate(seq):
            positions[tok].append(pos / max(len(seq) - 1, 1))
        for i in range(len(seq) - 1):
            successor_counts[seq[i]][seq[i + 1]] += 1

    total = sum(freq.values())
    sorted_tokens = sorted(freq.keys(), key=lambda t: freq[t], reverse=True)
    rank_map = {t: i for i, t in enumerate(sorted_tokens)}

    features = np.zeros((n_tokens, 10), dtype=np.float64)
    feature_names = [
        "log_freq",
        "type_token_indicator",
        "first_ratio",
        "last_ratio",
        "pos_mean",
        "pos_std",
        "is_repeat_token",
        "rep_suffix",
        "successor_entropy",
        "freq_rank_norm",
    ]

    for tok in vocab:
        idx = token_to_idx[tok]
        f = freq[tok]
        features[idx, 0] = np.log1p(f)
        features[idx, 1] = 1.0 if f == 1 else 0.0
        features[idx, 2] = first_count.get(tok, 0) / f
        features[idx, 3] = last_count.get(tok, 0) / f
        pos_list = positions.get(tok, [0.5])
        features[idx, 4] = np.mean(pos_list)
        features[idx, 5] = np.std(pos_list) if len(pos_list) > 1 else 0.0
        features[idx, 6] = 1.0 if "_REP" in tok else 0.0
        features[idx, 7] = int(tok.split("_REP")[-1]) if "_REP" in tok else 0.0
        succ = successor_counts.get(tok, Counter())
        if succ:
            probs = np.array(list(succ.values()), dtype=np.float64)
            probs = probs / probs.sum()
            features[idx, 8] = entropy(probs)
        else:
            features[idx, 8] = 0.0
        features[idx, 9] = rank_map.get(tok, n_tokens) / n_tokens

    return vocab, features, feature_names


def permuted_sequences(sequences, seed=42):
    rng = np.random.default_rng(seed)
    permuted = []
    for seq in sequences:
        arr = list(seq)
        rng.shuffle(arr)
        permuted.append(arr)
    return permuted


def uniform_sequences(sequences, seed=42):
    rng = np.random.default_rng(seed)
    all_tokens = sorted(set(t for seq in sequences for t in seq))
    uniform = []
    for seq in sequences:
        uniform.append([rng.choice(all_tokens) for _ in range(len(seq))])
    return uniform


def main():
    parser = argparse.ArgumentParser(description="Build structural feature embeddings")
    parser.add_argument("--input", required=True, help="Input CSV")
    parser.add_argument(
        "--output-dir", default="data/processed", help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from spectral_submersion.tokenization import get_repetition_aware_sequences

    pure_seqs = get_sequences_by_line(df)
    collapsed_seqs, _ = get_repetition_aware_sequences(df)
    perm_seqs = permuted_sequences(pure_seqs, args.seed)
    unif_seqs = uniform_sequences(pure_seqs, args.seed)

    print("=== Structural Feature Embeddings (Pure Tokenization) ===\n")
    results = {}

    for label, seqs in [
        ("real", pure_seqs),
        ("permuted", perm_seqs),
        ("random_uniform", unif_seqs),
    ]:
        vocab, features, fnames = compute_structural_features(df, seqs)
        U, S, Vt = np.linalg.svd(features, full_matrices=False)
        r_eff = float(effective_rank(S))
        results[label] = {
            "vocab_size": len(vocab),
            "n_features": features.shape[1],
            "r_eff": r_eff,
            "singular_values": S.tolist()[:10],
        }
        np.save(output_dir / f"structural_features_{label}.npy", features)
        with open(output_dir / f"structural_features_{label}.vocab.json", "w") as f:
            json.dump({t: i for i, t in enumerate(vocab)}, f)
        print(
            f"  {label:>15}: vocab={len(vocab)}, r_eff={r_eff:.4f}, features={features.shape[1]}"
        )

    print("\n=== Structural Feature Embeddings (Collapsed Tokenization) ===\n")
    from spectral_submersion.tokenization import get_repetition_aware_sequences

    collapsed_seqs, pure_seqs_ref = get_repetition_aware_sequences(df)
    perm_collapsed = permuted_sequences(collapsed_seqs, args.seed)
    unif_collapsed = uniform_sequences(collapsed_seqs, args.seed)

    for label, seqs_col, seqs_pure in [
        ("real_collapsed", collapsed_seqs, pure_seqs_ref),
        ("permuted_collapsed", perm_collapsed, perm_seqs),
        ("random_uniform_collapsed", unif_collapsed, unif_seqs),
    ]:
        vocab, features, fnames = compute_structural_features_collapsed(
            seqs_col, seqs_pure
        )
        U, S, Vt = np.linalg.svd(features, full_matrices=False)
        r_eff = float(effective_rank(S))
        results[label] = {
            "vocab_size": len(vocab),
            "n_features": features.shape[1],
            "r_eff": r_eff,
            "singular_values": S.tolist()[:10],
        }
        np.save(output_dir / f"structural_features_{label}.npy", features)
        with open(output_dir / f"structural_features_{label}.vocab.json", "w") as f:
            json.dump({t: i for i, t in enumerate(vocab)}, f)
        print(
            f"  {label:>25}: vocab={len(vocab)}, r_eff={r_eff:.4f}, features={features.shape[1]}"
        )

    print("\n=== Sanity Check: Structural Embeddings ===\n")
    r_real = results["real"]["r_eff"]
    r_perm = results["permuted"]["r_eff"]
    r_unif = results["random_uniform"]["r_eff"]
    r_real_c = results["real_collapsed"]["r_eff"]
    r_perm_c = results["permuted_collapsed"]["r_eff"]
    r_unif_c = results["random_uniform_collapsed"]["r_eff"]

    print(
        f"  Pure tokenization:  r_eff(real)={r_real:.4f}  r_eff(perm)={r_perm:.4f}  r_eff(unif)={r_unif:.4f}"
    )
    print(
        f"  Collapsed tokeniz.: r_eff(real)={r_real_c:.4f}  r_eff(perm)={r_perm_c:.4f}  r_eff(unif)={r_unif_c:.4f}"
    )

    if r_real > r_perm > r_unif:
        print(f"  ✓ Pure: Inverted sanity check PASSES (real > permuted > uniform)")
    elif r_real > r_perm:
        print(
            f"  ✓ Pure: Partial inverted check (real > permuted, but permuted <= uniform)"
        )
    else:
        print(f"  ✗ Pure: Inverted sanity check FAILS (real <= permuted)")

    if r_real_c > r_perm_c > r_unif_c:
        print(
            f"  ✓ Collapsed: Inverted sanity check PASSES (real > permuted > uniform)"
        )
    elif r_real_c > r_perm_c:
        print(
            f"  ✓ Collapsed: Partial inverted check (real > permuted, but permuted <= uniform)"
        )
    else:
        print(f"  ✗ Collapsed: Inverted sanity check FAILS (real <= permuted)")

    out_path = output_dir / "structural_feature_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
