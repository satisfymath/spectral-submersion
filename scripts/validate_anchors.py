"""Validate anchor-based Procrustes alignment on synthetic ground-truth data.

Procedure:
1. Load lost and candidate embeddings (same vocab size, known permutation).
2. Load ground-truth anchors.
3. Split anchors into train (for Procrustes) and test (for evaluation).
4. Solve Procrustes on train anchors.
5. Apply Q to all lost embeddings.
6. For each test lost token, find nearest candidate in aligned space.
7. Report accuracy@1, accuracy@5, mean reciprocal rank.

This validates the mathematical pipeline under ideal conditions.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from spectral_submersion.alignment import orthogonal_procrustes


def accuracy_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks <= k))


def mean_reciprocal_rank(ranks: np.ndarray) -> float:
    return float(np.mean(1.0 / ranks))


def main():
    parser = argparse.ArgumentParser(description="Validate synthetic anchor recovery")
    parser.add_argument(
        "--lost-embed", default="data/processed/embeddings_synthetic_v2.npy"
    )
    parser.add_argument(
        "--lost-vocab", default="data/processed/embeddings_synthetic_v2.vocab.json"
    )
    parser.add_argument(
        "--candidate-embed", default="data/processed/embeddings_synthetic_candidate.npy"
    )
    parser.add_argument(
        "--candidate-vocab",
        default="data/processed/embeddings_synthetic_candidate.vocab.json",
    )
    parser.add_argument(
        "--anchors", default="data/raw/candidate_languages/synthetic_anchors.json"
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.20,
        help="Fraction of anchors used for training Procrustes",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load embeddings
    E_lost = np.load(args.lost_embed)
    E_cand = np.load(args.candidate_embed)
    with open(args.lost_vocab, "r", encoding="utf-8") as f:
        lost_vocab = json.load(f)
    with open(args.candidate_vocab, "r", encoding="utf-8") as f:
        cand_vocab = json.load(f)

    # Build index -> token lookups
    lost_idx_to_tok = {i: t for t, i in lost_vocab.items()}
    cand_idx_to_tok = {i: t for t, i in cand_vocab.items()}

    # Load anchors
    with open(args.anchors, "r", encoding="utf-8") as f:
        anchors = json.load(f)

    # Split anchors
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(anchors))
    n_train = max(1, int(len(anchors) * args.train_fraction))
    train_anchors = [anchors[i] for i in perm[:n_train]]
    test_anchors = [anchors[i] for i in perm[n_train:]]

    # Build train matrices
    X_train = np.array([E_lost[lost_vocab[a["lost_token"]]] for a in train_anchors])
    Y_train = np.array(
        [E_cand[cand_vocab[a["candidate_token"]]] for a in train_anchors]
    )

    # Solve Procrustes on train anchors
    Q = orthogonal_procrustes(X_train, Y_train)

    # Align all lost embeddings
    E_lost_aligned = E_lost @ Q

    # Evaluate on test anchors
    test_ranks = []
    for a in test_anchors:
        lost_idx = lost_vocab[a["lost_token"]]
        true_cand_idx = cand_vocab[a["candidate_token"]]
        # Nearest neighbor in aligned space
        dists = np.linalg.norm(E_lost_aligned[lost_idx] - E_cand, axis=1)
        rank = 1 + np.argsort(dists).tolist().index(true_cand_idx)
        test_ranks.append(rank)

    test_ranks = np.array(test_ranks)

    print("=" * 60)
    print("Synthetic Anchor Recovery Benchmark")
    print("=" * 60)
    print(f"Total anchors:      {len(anchors)}")
    print(f"Train anchors:      {len(train_anchors)} ({args.train_fraction:.0%})")
    print(f"Test anchors:       {len(test_anchors)}")
    print(f"Embedding dim:      {E_lost.shape[1]}")
    print(f"Train distortion:   {np.linalg.norm(X_train @ Q - Y_train, 'fro'):.6f}")
    print("-" * 60)
    print(f"Accuracy@1:         {accuracy_at_k(test_ranks, 1):.4f}")
    print(f"Accuracy@5:         {accuracy_at_k(test_ranks, 5):.4f}")
    print(f"Accuracy@10:        {accuracy_at_k(test_ranks, 10):.4f}")
    print(f"MRR:                {mean_reciprocal_rank(test_ranks):.4f}")
    print(f"Median rank:        {int(np.median(test_ranks))}")
    print(f"Mean rank:          {test_ranks.mean():.2f}")
    print("=" * 60)

    # Save results
    out = {
        "n_train": len(train_anchors),
        "n_test": len(test_anchors),
        "train_distortion": float(np.linalg.norm(X_train @ Q - Y_train, "fro")),
        "accuracy_at_1": accuracy_at_k(test_ranks, 1),
        "accuracy_at_5": accuracy_at_k(test_ranks, 5),
        "accuracy_at_10": accuracy_at_k(test_ranks, 10),
        "mrr": mean_reciprocal_rank(test_ranks),
        "median_rank": int(np.median(test_ranks)),
        "mean_rank": float(test_ranks.mean()),
    }
    out_path = Path("reports/tables/anchor_recovery_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved benchmark to {out_path}")


if __name__ == "__main__":
    main()
