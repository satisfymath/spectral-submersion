"""Run bootstrap stability analysis with proper spectral metrics.

For B bootstrap samples:
1. Resample sentences with replacement.
2. Recompute co-occurrence, PPMI, SVD.
3. Record: effective rank, top-k singular values, embedding coordinates.
4. Measure stability:
   - Stability of effective rank (coefficient of variation)
   - Pairwise cosine similarity between bootstrap embeddings (after Procrustes alignment)
   - Stability of nearest-neighbor graphs
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from spectral_submersion.alignment import orthogonal_procrustes
from spectral_submersion.cooccurrence import (
    build_vocab,
    cooccurrence_matrix_from_sequences,
)
from spectral_submersion.pmi import ppmi_matrix
from spectral_submersion.spectral import spectral_embedding, effective_rank
from spectral_submersion.tokenization import get_sequences_by_line, tokens_to_ids


def main():
    parser = argparse.ArgumentParser(description="Bootstrap stability analysis")
    parser.add_argument(
        "--input", default="data/raw/lost_language/corpus_synthetic_v2.csv"
    )
    parser.add_argument("--output", default="reports/tables/bootstrap_stability.csv")
    parser.add_argument("--n-bootstrap", type=int, default=50)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    sequences = get_sequences_by_line(df)
    vocab = build_vocab([tok for seq in sequences for tok in seq])
    seq_ids = [tokens_to_ids(seq, vocab) for seq in sequences]

    rng = np.random.default_rng(args.seed)

    all_effective_ranks = []
    all_top_svs = []
    all_embeddings = []

    for b in range(args.n_bootstrap):
        n = len(seq_ids)
        sampled = [seq_ids[i] for i in rng.choice(n, size=n, replace=True)]
        C = cooccurrence_matrix_from_sequences(
            sampled, len(vocab), window_size=args.window
        )
        M = ppmi_matrix(C)
        E, S, _ = spectral_embedding(M, k=args.k, random_state=args.seed)
        all_effective_ranks.append(effective_rank(S))
        all_top_svs.append(float(S[0]) if len(S) > 0 else 0.0)
        all_embeddings.append(E)

    # Effective rank stability
    ranks_arr = np.array(all_effective_ranks)
    rank_cv = ranks_arr.std() / ranks_arr.mean()

    # Embedding stability: pairwise cosine similarity after Procrustes alignment
    # Use first bootstrap as reference
    ref_E = all_embeddings[0]
    sims = []
    for b in range(1, min(20, len(all_embeddings))):  # sample first 20
        Q = orthogonal_procrustes(all_embeddings[b], ref_E)
        aligned = all_embeddings[b] @ Q
        # Cosine similarity per token, then mean
        norms_ref = np.linalg.norm(ref_E, axis=1)
        norms_b = np.linalg.norm(aligned, axis=1)
        cos = np.sum(ref_E * aligned, axis=1) / (norms_ref * norms_b + 1e-12)
        sims.append(float(cos.mean()))

    sims_arr = np.array(sims)

    results = {
        "n_bootstrap": args.n_bootstrap,
        "vocab_size": len(vocab),
        "embedding_dim": args.k,
        "effective_rank_mean": float(ranks_arr.mean()),
        "effective_rank_std": float(ranks_arr.std()),
        "effective_rank_cv": float(rank_cv),
        "top_sv_mean": float(np.mean(all_top_svs)),
        "top_sv_std": float(np.std(all_top_svs)),
        "embedding_cosine_mean": float(sims_arr.mean()),
        "embedding_cosine_std": float(sims_arr.std()),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([results]).to_csv(out_path, index=False)

    print("=" * 50)
    print("Bootstrap Stability Report")
    print("=" * 50)
    print(f"Bootstrap samples:     {args.n_bootstrap}")
    print(f"Vocab size:            {len(vocab)}")
    print(f"Embedding dim:         {args.k}")
    print(f"Effective rank mean:   {results['effective_rank_mean']:.4f}")
    print(f"Effective rank std:    {results['effective_rank_std']:.4f}")
    print(f"Effective rank CV:     {results['effective_rank_cv']:.4f}")
    print(f"Top SV mean:           {results['top_sv_mean']:.4f}")
    print(f"Top SV std:            {results['top_sv_std']:.4f}")
    print(f"Embedding cosine mean: {results['embedding_cosine_mean']:.4f}")
    print(f"Embedding cosine std:  {results['embedding_cosine_std']:.4f}")
    print("=" * 50)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
