"""Validate anchor recovery using multi-language consensus space.

This compares single-candidate Procrustes vs. consensus-space projection
for recovering ground-truth synthetic anchors.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from spectral_submersion.alignment import orthogonal_procrustes, pairwise_squared_distances
from spectral_submersion.multi_alignment import build_consensus_space, project_lost_to_consensus


def accuracy_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks <= k))


def mean_reciprocal_rank(ranks: np.ndarray) -> float:
    return float(np.mean(1.0 / ranks))


def evaluate_recovery(E_lost_aligned, E_cand, lost_vocab, cand_vocab, anchors, train_anchors):
    """Evaluate anchor recovery given aligned lost and candidate embeddings."""
    train_set = {(a["lost_token"], a["candidate_token"]) for a in train_anchors}
    test_anchors = [a for a in anchors if (a["lost_token"], a["candidate_token"]) not in train_set]

    test_ranks = []
    for a in test_anchors:
        lost_idx = lost_vocab[a["lost_token"]]
        true_cand_idx = cand_vocab[a["candidate_token"]]
        dists = np.linalg.norm(E_lost_aligned[lost_idx] - E_cand, axis=1)
        rank = 1 + np.argsort(dists).tolist().index(true_cand_idx)
        test_ranks.append(rank)

    test_ranks = np.array(test_ranks)
    return {
        "n_test": len(test_anchors),
        "accuracy_at_1": accuracy_at_k(test_ranks, 1),
        "accuracy_at_5": accuracy_at_k(test_ranks, 5),
        "accuracy_at_10": accuracy_at_k(test_ranks, 10),
        "mrr": mean_reciprocal_rank(test_ranks),
        "median_rank": int(np.median(test_ranks)),
        "mean_rank": float(test_ranks.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-language anchor recovery validation")
    parser.add_argument("--lost-embed", default="data/processed/embeddings_synthetic_v2.npy")
    parser.add_argument("--lost-vocab", default="data/processed/embeddings_synthetic_v2.vocab.json")
    parser.add_argument("--candidate-embed", default="data/processed/embeddings_synthetic_transformed.npy")
    parser.add_argument("--candidate-vocab", default="data/processed/embeddings_synthetic_transformed.vocab.json")
    parser.add_argument("--anchors", default="data/raw/candidate_languages/synthetic_anchors.json")
    parser.add_argument("--auxiliary-embeds", nargs="+", default=[
        "data/processed/embeddings_rap.npy",
        "data/processed/embeddings_fj.npy",
        "data/processed/embeddings_ty.npy",
        "data/processed/embeddings_to.npy",
    ], help="Additional candidate embeddings to build consensus")
    parser.add_argument("--train-fraction", type=float, default=0.20)
    parser.add_argument("--method", default="gpa", choices=["gpa", "intersection"])
    parser.add_argument("--target-dim", type=int, default=16)
    parser.add_argument("--tikhonov", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    E_lost = np.load(args.lost_embed)
    E_cand = np.load(args.candidate_embed)
    with open(args.lost_vocab, "r", encoding="utf-8") as f:
        lost_vocab = json.load(f)
    with open(args.candidate_vocab, "r", encoding="utf-8") as f:
        cand_vocab = json.load(f)

    with open(args.anchors, "r", encoding="utf-8") as f:
        anchors = json.load(f)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(anchors))
    n_train = max(1, int(len(anchors) * args.train_fraction))
    train_anchors = [anchors[i] for i in perm[:n_train]]

    print("=" * 70)
    print("Multi-Language Consensus Anchor Recovery Benchmark")
    print("=" * 70)
    print(f"Total anchors:      {len(anchors)}")
    print(f"Train anchors:      {n_train} ({args.train_fraction:.0%})")

    # Baseline: single-candidate Procrustes on train anchors
    X_train = np.array([E_lost[lost_vocab[a["lost_token"]]] for a in train_anchors])
    Y_train = np.array([E_cand[cand_vocab[a["candidate_token"]]] for a in train_anchors])
    Q_single = orthogonal_procrustes(X_train, Y_train)
    E_lost_single = E_lost @ Q_single
    baseline = evaluate_recovery(E_lost_single, E_cand, lost_vocab, cand_vocab, anchors, train_anchors)

    print("\n--- Baseline: Single-Candidate Procrustes ---")
    print(f"Accuracy@1:         {baseline['accuracy_at_1']:.4f}")
    print(f"Accuracy@5:         {baseline['accuracy_at_5']:.4f}")
    print(f"MRR:                {baseline['mrr']:.4f}")
    print(f"Median rank:        {baseline['median_rank']}")

    # Multi-language consensus: include candidate + auxiliary languages
    aux_embeds = []
    for path in args.auxiliary_embeds:
        p = Path(path)
        if p.exists():
            aux_embeds.append(np.load(p))
            print(f"Loaded auxiliary: {p.name}")

    all_embeds = [E_cand] + aux_embeds
    print(f"\nBuilding consensus from {len(all_embeds)} languages (method={args.method}) ...")
    R, aligned, projections = build_consensus_space(
        all_embeds, method=args.method, target_dim=args.target_dim
    )
    print(f"Consensus R shape: {R.shape}")

    # Project lost into consensus using ONLY train anchors (no cheating)
    # We solve a Procrustes from lost_train to R_train (where R_train is the
    # aligned candidate portion of consensus corresponding to train anchors).
    # Since the candidate is the first in all_embeds, aligned[0] is the candidate in R.
    E_cand_consensus = aligned[0]

    # But we want a fair comparison: project lost using train anchors to consensus.
    # Build train matrices in lost space
    X_train_lost = np.array([E_lost[lost_vocab[a["lost_token"]]] for a in train_anchors])
    Y_train_consensus = np.array([E_cand_consensus[cand_vocab[a["candidate_token"]]] for a in train_anchors])

    Q_cons = orthogonal_procrustes(X_train_lost, Y_train_consensus)
    E_lost_consensus = E_lost @ Q_cons

    multi = evaluate_recovery(E_lost_consensus, E_cand_consensus, lost_vocab, cand_vocab, anchors, train_anchors)

    print("\n--- Multi-Language Consensus Procrustes ---")
    print(f"Accuracy@1:         {multi['accuracy_at_1']:.4f}")
    print(f"Accuracy@5:         {multi['accuracy_at_5']:.4f}")
    print(f"MRR:                {multi['mrr']:.4f}")
    print(f"Median rank:        {multi['median_rank']}")

    print("\n--- Improvement over baseline ---")
    print(f"Acc@1 delta:        {multi['accuracy_at_1'] - baseline['accuracy_at_1']:+.4f}")
    print(f"Acc@5 delta:        {multi['accuracy_at_5'] - baseline['accuracy_at_5']:+.4f}")
    print(f"MRR delta:          {multi['mrr'] - baseline['mrr']:+.4f}")
    print("=" * 70)

    out = {
        "baseline": baseline,
        "multi": multi,
        "deltas": {
            "accuracy_at_1": multi["accuracy_at_1"] - baseline["accuracy_at_1"],
            "accuracy_at_5": multi["accuracy_at_5"] - baseline["accuracy_at_5"],
            "mrr": multi["mrr"] - baseline["mrr"],
        },
        "config": {
            "n_train": n_train,
            "method": args.method,
            "target_dim": args.target_dim,
            "n_auxiliary": len(aux_embeds),
        },
    }
    out_path = Path("reports/tables/multi_language_anchor_recovery.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
