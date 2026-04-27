"""Run SVD and build spectral embeddings."""
from pathlib import Path

import numpy as np
import pandas as pd

from spectral_submersion.tokenization import (
    read_corpus,
    normalize_tokens,
    build_vocab,
    tokens_to_ids,
    get_sequences_by_line,
)
from spectral_submersion.cooccurrence import (
    cooccurrence_matrix_from_sequences,
    build_vocab as build_vocab_cooc,
)
from spectral_submersion.pmi import ppmi_matrix
from spectral_submersion.spectral import spectral_embedding, effective_rank
from spectral_submersion.visualization import plot_singular_values


def main():
    in_path = Path("data/processed/lost_tokens.csv")
    out_embed = Path("data/processed/embeddings_lost.npy")
    out_sv = Path("data/processed/singular_values_lost.npy")
    out_fig = Path("reports/figures/singular_values.png")

    df = pd.read_csv(in_path)
    tokens = df["token"].tolist()
    vocab = build_vocab_cooc(tokens)
    sequences = get_sequences_by_line(df)
    seq_ids = [tokens_to_ids(seq, vocab) for seq in sequences]

    C = cooccurrence_matrix_from_sequences(
        seq_ids, vocab_size=len(vocab), window_size=3, inverse_distance=True
    )
    M = ppmi_matrix(C, epsilon=1e-9)

    E, S, Vt = spectral_embedding(M, k=16, alpha=0.5, random_state=42)
    r_eff = effective_rank(S)

    out_embed.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_embed, E)
    np.save(out_sv, S)

    plot_singular_values(S, save_path=out_fig)

    print(f"Embeddings shape: {E.shape}")
    print(f"Effective rank: {r_eff:.4f}")
    print(f"Saved embeddings to {out_embed}")
    print(f"Saved singular values to {out_sv}")
    print(f"Saved plot to {out_fig}")


if __name__ == "__main__":
    main()
