"""Run alignment between lost language and candidate language embeddings.

Handles both matched-size (Procrustes) and unmatched-size (direct distance) regimes.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from spectral_submersion.alignment import (
    orthogonal_procrustes,
    pairwise_squared_distances,
)
from spectral_submersion.transport import optimal_transport_matrix
from spectral_submersion.decoding import generate_symbol_hypotheses
from spectral_submersion.reporting import save_hypotheses


def main():
    parser = argparse.ArgumentParser(description="Align lost language with candidate")
    parser.add_argument("--lost-embed", default="data/processed/embeddings_lost.npy")
    parser.add_argument("--lost-vocab", default="data/processed/embeddings_lost.vocab.json")
    parser.add_argument("--candidate-embed", required=True)
    parser.add_argument("--candidate-vocab", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output", default="reports/hypotheses/candidate_dictionary.yaml")
    parser.add_argument("--reg", type=float, default=0.1, help="OT entropic regularization")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature for NN fallback")
    args = parser.parse_args()

    lost_embed_path = Path(args.lost_embed)
    lost_vocab_path = Path(args.lost_vocab)
    cand_embed_path = Path(args.candidate_embed)
    cand_vocab_path = Path(args.candidate_vocab)

    if not cand_embed_path.exists():
        print(f"Candidate embeddings not found at {cand_embed_path}")
        return

    E_lost = np.load(lost_embed_path)
    E_cand = np.load(cand_embed_path)

    with open(lost_vocab_path, "r", encoding="utf-8") as f:
        lost_vocab = json.load(f)
    with open(cand_vocab_path, "r", encoding="utf-8") as f:
        cand_vocab = json.load(f)

    lost_tokens = [tok for tok, _ in sorted(lost_vocab.items(), key=lambda x: x[1])]
    cand_tokens = [tok for tok, _ in sorted(cand_vocab.items(), key=lambda x: x[1])]

    # Ensure same latent dimension
    d = min(E_lost.shape[1], E_cand.shape[1])
    E_lost = E_lost[:, :d]
    E_cand = E_cand[:, :d]

    n_lost, n_cand = E_lost.shape[0], E_cand.shape[0]
    use_procrustes = n_lost == n_cand

    if use_procrustes:
        Q = orthogonal_procrustes(E_lost, E_cand)
        D = pairwise_squared_distances(E_lost @ Q, E_cand)
        method_note = "procrustes"
    else:
        # Without anchors or matched vocabularies, Procrustes is undefined.
        # Fall back to direct pairwise distance in the common latent space.
        Q = None
        D = pairwise_squared_distances(E_lost, E_cand)
        method_note = "direct_distance_no_procrustes"
        print(f"[WARNING] Vocabulary sizes differ ({n_lost} vs {n_cand}). "
              f"Skipping Procrustes; using direct distance. Results are exploratory only.")

    # Optimal transport (Sinkhorn)
    Pi_ot = optimal_transport_matrix(D, reg=args.reg)

    # Simple softmax baseline
    logits = -D / args.temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    Pi_nn = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    hypotheses_ot = generate_symbol_hypotheses(Pi_ot, lost_tokens, cand_tokens, top_k=5)
    hypotheses_nn = generate_symbol_hypotheses(Pi_nn, lost_tokens, cand_tokens, top_k=5)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_hypotheses(
        hypotheses_ot,
        out_path.with_name(f"{args.candidate_name}_dictionary_ot.yaml"),
        format="yaml",
    )
    save_hypotheses(
        hypotheses_nn,
        out_path.with_name(f"{args.candidate_name}_dictionary_nn.yaml"),
        format="yaml",
    )

    if use_procrustes:
        distortion = float(np.linalg.norm(E_lost @ Q - E_cand, "fro"))
    else:
        distortion = float(np.mean(np.min(D, axis=1)))

    print(f"Alignment with {args.candidate_name}")
    print(f"  Lost vocab: {len(lost_tokens)}, Candidate vocab: {len(cand_tokens)}")
    print(f"  Embedding dim used: {d}")
    print(f"  Method: {method_note}")
    print(f"  Distortion metric: {distortion:.4f}")
    print(f"  OT plan entropy: {float(-(Pi_ot[Pi_ot>0]*np.log(Pi_ot[Pi_ot>0])).sum()):.4f}")
    print(f"  Saved OT hypotheses to {out_path.with_name(f'{args.candidate_name}_dictionary_ot.yaml')}")
    print(f"  Saved NN hypotheses to {out_path.with_name(f'{args.candidate_name}_dictionary_nn.yaml')}")


if __name__ == "__main__":
    main()
