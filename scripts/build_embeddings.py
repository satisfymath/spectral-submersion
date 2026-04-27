"""Build spectral embeddings (PPMI + SVD) from any corpus CSV.

Usage:
    PYTHONPATH=src python scripts/build_embeddings.py \
        --input data/processed/lost_tokens.csv \
        --output data/processed/embeddings_lost.npy \
        --sv-output data/processed/singular_values_lost.npy \
        --fig reports/figures/singular_values_lost.png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from spectral_submersion.cooccurrence import (
    build_vocab,
    cooccurrence_matrix_from_sequences,
)
from spectral_submersion.pmi import ppmi_matrix
from spectral_submersion.spectral import spectral_embedding, effective_rank
from spectral_submersion.tokenization import get_sequences_by_line, tokens_to_ids
from spectral_submersion.visualization import plot_singular_values


def main():
    parser = argparse.ArgumentParser(description="Build spectral embeddings from corpus CSV")
    parser.add_argument("--input", required=True, help="Input CSV with doc_id, line_id, position, token")
    parser.add_argument("--output", required=True, help="Output .npy path for embeddings")
    parser.add_argument("--sv-output", required=True, help="Output .npy path for singular values")
    parser.add_argument("--fig", default=None, help="Optional path to save singular value plot")
    parser.add_argument("--k", type=int, default=16, help="Embedding dimension")
    parser.add_argument("--alpha", type=float, default=0.5, help="Singular value exponent")
    parser.add_argument("--window", type=int, default=3, help="Co-occurrence window size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-vocab", type=int, default=None, help="Keep only top N most frequent tokens")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    tokens = df["token"].tolist()

    # If max_vocab specified, filter to top N frequent tokens
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

    C = cooccurrence_matrix_from_sequences(
        seq_ids, vocab_size=len(vocab), window_size=args.window, inverse_distance=True
    )
    M = ppmi_matrix(C, epsilon=1e-9)

    E, S, Vt = spectral_embedding(M, k=args.k, alpha=args.alpha, random_state=args.seed)
    r_eff = effective_rank(S)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.sv_output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, E)
    np.save(args.sv_output, S)

    # Save vocab mapping for later decoding
    vocab_path = Path(args.output).with_suffix(".vocab.json")
    import json
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    if args.fig:
        plot_singular_values(S, title=f"Singular Value Spectrum ({Path(args.input).stem})", save_path=args.fig)

    print(f"Input: {args.input}")
    print(f"Vocab size: {len(vocab)}")
    print(f"Embeddings shape: {E.shape}")
    print(f"Effective rank: {r_eff:.4f}")
    print(f"Saved embeddings to {args.output}")
    print(f"Saved singular values to {args.sv_output}")


if __name__ == "__main__":
    main()
