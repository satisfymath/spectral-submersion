"""Bootstrap confidence intervals for effective rank and key metrics.

Resamples tokens within each corpus to generate bootstrap distributions
of r_eff for structural features and co-occurrence embeddings.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from scipy.linalg import svd
from scipy.stats import entropy

from spectral_submersion.tokenization import read_corpus, get_sequences_by_line
from spectral_submersion.spectral import effective_rank


def bootstrap_r_eff_cooccurrence(
    sequences, vocab, n_bootstrap=200, window=3, k=16, seed=42
):
    """Bootstrap r_eff for co-occurrence embeddings by resampling sequences."""
    rng = np.random.default_rng(seed)
    r_effs = []

    for b in range(n_bootstrap):
        # Resample sequences with replacement
        idx = rng.choice(len(sequences), size=len(sequences), replace=True)
        boot_seqs = [sequences[i] for i in idx]

        # Build cooccurrence matrix
        token_to_idx = {t: i for i, t in enumerate(vocab)}
        n = len(vocab)
        C = np.zeros((n, n), dtype=np.float64)

        for seq in boot_seqs:
            ids = [token_to_idx[t] for t in seq if t in token_to_idx]
            for i_pos, i in enumerate(ids):
                for j_pos in range(
                    max(0, i_pos - window), min(len(ids), i_pos + window + 1)
                ):
                    if i_pos != j_pos:
                        weight = 1.0 / abs(i_pos - j_pos)
                        C[i, ids[j_pos]] += weight

        # PPMI
        total = C.sum() + 1e-9
        Pij = (C + 1e-9) / total
        Pi = Pij.sum(axis=1, keepdims=True)
        Pj = Pij.sum(axis=0, keepdims=True)
        Pj_smooth = Pj**0.75
        Pj_smooth = Pj_smooth / Pj_smooth.sum()
        PMI = np.log(Pij / (Pi * Pj_smooth + 1e-9))
        PPMI = np.maximum(PMI, 0.0)

        _, S, _ = svd(PPMI, full_matrices=False)
        r = float(effective_rank(S))
        r_effs.append(r)

    return np.array(r_effs)


def bootstrap_r_eff_structural(df, sequences, n_bootstrap=200, seed=42):
    """Bootstrap r_eff for structural features by resampling sequences."""
    rng = np.random.default_rng(seed)
    r_effs = []

    for b in range(n_bootstrap):
        idx = rng.choice(len(sequences), size=len(sequences), replace=True)
        boot_seqs = [sequences[i] for i in idx]

        # Compute structural features
        all_tokens = [t for seq in boot_seqs for t in seq]
        freq = Counter(all_tokens)
        vocab = sorted(freq.keys())
        n_tokens = len(vocab)
        token_to_idx = {t: i for i, t in enumerate(vocab)}

        pos_data = defaultdict(list)
        first_count = Counter()
        last_count = Counter()
        succ = defaultdict(Counter)

        for seq in boot_seqs:
            if not seq:
                continue
            first_count[seq[0]] += 1
            last_count[seq[-1]] += 1
            for p, tok in enumerate(seq):
                pos_data[tok].append(p / max(len(seq) - 1, 1))
            for i in range(len(seq) - 1):
                succ[seq[i]][seq[i + 1]] += 1

        total = sum(freq.values())
        sorted_toks = sorted(freq.keys(), key=lambda t: freq[t], reverse=True)
        rank_map = {t: i for i, t in enumerate(sorted_toks)}

        F = np.zeros((n_tokens, 10))
        for tok in vocab:
            i = token_to_idx[tok]
            f = freq[tok]
            F[i, 0] = np.log1p(f)
            F[i, 1] = 1.0 if f == 1 else 0.0
            F[i, 2] = first_count.get(tok, 0) / max(f, 1)
            F[i, 3] = last_count.get(tok, 0) / max(f, 1)
            p_list = pos_data.get(tok, [0.5])
            F[i, 4] = np.mean(p_list)
            F[i, 5] = np.std(p_list) if len(p_list) > 1 else 0.0
            repeat_runs = []
            for seq in boot_seqs:
                j = 0
                while j < len(seq):
                    k = j + 1
                    while k < len(seq) and seq[k] == seq[j]:
                        k += 1
                    if k - j >= 2:
                        repeat_runs.append(k - j)
                    j = k
            tok_runs = [
                r
                for r in repeat_runs
                if any(seq[j : j + r].count(tok) > 0 for seq in boot_seqs)
            ]
            F[i, 6] = len(tok_runs) / max(f, 1)
            F[i, 7] = np.mean(tok_runs) if tok_runs else 0.0
            s = succ[tok]
            if s:
                probs = np.array(list(s.values()), dtype=np.float64)
                probs = probs / probs.sum()
                F[i, 8] = float(entropy(probs))
            F[i, 9] = rank_map.get(tok, n_tokens) / n_tokens

        _, S, _ = svd(F, full_matrices=False)
        r = float(effective_rank(S))
        r_effs.append(r)

    return np.array(r_effs)


def main():
    out_dir = Path("reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 60)

    # Rongorongo real
    rr_path = "data/raw/lost_language/corpus_rongorongo_real.xml.csv"
    rr_df = read_corpus(rr_path)
    rr_seqs = get_sequences_by_line(rr_df)
    all_tokens = [t for seq in rr_seqs for t in seq]
    vocab = sorted(set(all_tokens))

    print(
        f"\nRongorongo real: {len(rr_seqs)} lines, {len(all_tokens)} tokens, {len(vocab)} types"
    )

    # Structural features bootstrap
    print("\n--- Structural Features Bootstrap (n=200) ---")
    r_effs_struct = bootstrap_r_eff_structural(rr_df, rr_seqs, n_bootstrap=200, seed=42)
    print(f"  r_eff mean = {np.mean(r_effs_struct):.4f}")
    print(f"  r_eff std  = {np.std(r_effs_struct):.4f}")
    print(
        f"  95% CI = [{np.percentile(r_effs_struct, 2.5):.4f}, {np.percentile(r_effs_struct, 97.5):.4f}]"
    )

    # Cooccurrence bootstrap
    print("\n--- Cooccurrence Bootstrap (n=200) ---")
    r_effs_cooc = bootstrap_r_eff_cooccurrence(rr_seqs, vocab, n_bootstrap=200, seed=42)
    print(f"  r_eff mean = {np.mean(r_effs_cooc):.4f}")
    print(f"  r_eff std  = {np.std(r_effs_cooc):.4f}")
    print(
        f"  95% CI = [{np.percentile(r_effs_cooc, 2.5):.4f}, {np.percentile(r_effs_cooc, 97.5):.4f}]"
    )

    # Compare with point estimates
    print("\n--- Point Estimates vs Bootstrap ---")
    sv_rr = np.load("data/processed/sv_rongorongo_real.npy")
    r_point_cooc = float(effective_rank(sv_rr))
    print(f"  Cooccurrence r_eff (point): {r_point_cooc:.4f}")
    print(f"  Cooccurrence r_eff (bootstrap mean): {np.mean(r_effs_cooc):.4f}")
    print(f"  Structural r_eff (point, corrected): 5.0821")
    print(f"  Structural r_eff (bootstrap mean): {np.mean(r_effs_struct):.4f}")

    # Save
    results = pd.DataFrame(
        {
            "method": ["structural"] * len(r_effs_struct)
            + ["cooccurrence"] * len(r_effs_cooc),
            "r_eff": np.concatenate([r_effs_struct, r_effs_cooc]),
        }
    )
    results.to_csv(out_dir / "bootstrap_ci.csv", index=False)
    print(f"\nSaved to {out_dir / 'bootstrap_ci.csv'}")


if __name__ == "__main__":
    main()
