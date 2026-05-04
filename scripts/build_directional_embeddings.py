"""Build directional spectral embeddings (left/right context separation).

This creates asymmetric embeddings where each token has:
- An 'input' embedding from left-context co-occurrence
- An 'output' embedding from right-context co-occurrence

The combined representation captures positional/directional structure,
which is crucial for scripts with strong left-to-right or right-to-left
regularities (e.g., boustrophedon, agglutinative morphology).

The matrices C_left and C_right are converted to PPMI separately,
then concatenated before SVD.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from spectral_submersion.cooccurrence import (
    build_vocab,
    directional_cooccurrence_matrix_from_sequences,
)
from spectral_submersion.pmi import ppmi_matrix
from spectral_submersion.spectral import spectral_embedding, effective_rank
from spectral_submersion.tokenization import get_sequences_by_line, tokens_to_ids
from spectral_submersion.visualization import plot_singular_values


def main():
    parser = argparse.ArgumentParser(
        description="Build directional spectral embeddings"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sv-output", required=True)
    parser.add_argument("--fig", default=None)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-vocab", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    tokens = df["token"].tolist()

    if args.max_vocab:
        from collections import Counter

        counts = Counter(tokens)
        top_n = counts.most_common(args.max_vocab)
        allowed = {tok for tok, _ in top_n}
        df = df[df["token"].isin(allowed)].copy()
        tokens = df["token"].tolist()
        print(f"Filtered to top {args.max_vocab} tokens ({len(df)} rows remaining)")

    vocab = build_vocab(tokens)
    sequences = get_sequences_by_line(df)
    seq_ids = [tokens_to_ids(seq, vocab) for seq in sequences]

    C_left, C_right = directional_cooccurrence_matrix_from_sequences(
        seq_ids, vocab_size=len(vocab), window_size=args.window, inverse_distance=True
    )

    M_left = ppmi_matrix(C_left, epsilon=1e-9)
    M_right = ppmi_matrix(C_right, epsilon=1e-9)

    # Concatenate left and right PPMI matrices row-wise
    # Each token gets a vector [PPMI_left(token, :) | PPMI_right(token, :)]
    M_combined = np.hstack([M_left, M_right])

    E, S, Vt = spectral_embedding(
        M_combined, k=args.k, alpha=args.alpha, random_state=args.seed
    )
    r_eff = effective_rank(S)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.sv_output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, E)
    np.save(args.sv_output, S)

    vocab_path = Path(args.output).with_suffix(".vocab.json")
    import json

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    if args.fig:
        plot_singular_values(
            S, title=f"Directional SVD ({Path(args.input).stem})", save_path=args.fig
        )

    print(f"Input: {args.input}")
    print(f"Vocab size: {len(vocab)}")
    print(f"Combined matrix shape: {M_combined.shape}")
    print(f"Embeddings shape: {E.shape}")
    print(f"Effective rank: {r_eff:.4f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
