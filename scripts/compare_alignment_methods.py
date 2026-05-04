"""Compare alignment methods using relational (GW-style) and geometric distortion.

Evaluates three coupling strategies:
1. Optimal Transport (Sinkhorn)
2. Soft nearest-neighbor (softmax over distances)
3. Random uniform coupling (baseline)

For same-size vocabularies, also evaluates Procrustes alignment.
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
from spectral_submersion.evaluation import relational_distortion, geometric_distortion


def build_random_coupling(n_x: int, n_y: int) -> np.ndarray:
    """Build a row-stochastic random coupling matrix."""
    Pi = np.random.rand(n_x, n_y)
    Pi = Pi / Pi.sum(axis=1, keepdims=True)
    return Pi


def main():
    parser = argparse.ArgumentParser(description="Compare alignment distortions")
    parser.add_argument("--lost-embed", required=True)
    parser.add_argument("--candidate-embed", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--reg", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", default="reports/tables/alignment_comparison.json")
    args = parser.parse_args()

    E_lost = np.load(args.lost_embed)
    E_cand = np.load(args.candidate_embed)

    d = min(E_lost.shape[1], E_cand.shape[1])
    E_lost = E_lost[:, :d]
    E_cand = E_cand[:, :d]

    n_lost, n_cand = E_lost.shape[0], E_cand.shape[0]
    same_size = n_lost == n_cand

    # Compute distance matrices
    if same_size:
        Q = orthogonal_procrustes(E_lost, E_cand)
        D_geo = pairwise_squared_distances(E_lost @ Q, E_cand)
        # Relational distances in aligned space
        Dx = pairwise_squared_distances(E_lost @ Q, E_lost @ Q)
        Dy = pairwise_squared_distances(E_cand, E_cand)
        method_note = "procrustes"
    else:
        D_geo = pairwise_squared_distances(E_lost, E_cand)
        Dx = pairwise_squared_distances(E_lost, E_lost)
        Dy = pairwise_squared_distances(E_cand, E_cand)
        method_note = "direct_distance"

    # Couplings
    Pi_ot = optimal_transport_matrix(D_geo, reg=args.reg)
    logits = -D_geo / args.temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    Pi_nn = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    Pi_rand = build_random_coupling(n_lost, n_cand)

    results = {
        "candidate": args.candidate_name,
        "n_lost": n_lost,
        "n_cand": n_cand,
        "embedding_dim": d,
        "method": method_note,
        "same_size": same_size,
        "comparisons": {},
    }

    for name, Pi in [("ot", Pi_ot), ("nn", Pi_nn), ("random", Pi_rand)]:
        results["comparisons"][name] = {
            "geometric_distortion": geometric_distortion(Pi, D_geo),
            "relational_distortion": relational_distortion(Pi, Dx, Dy),
            "coupling_entropy": float(-(Pi[Pi > 0] * np.log(Pi[Pi > 0])).sum()),
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Alignment comparison: {args.candidate_name}")
    print(f"  Size: {n_lost} x {n_cand}, dim={d}, method={method_note}")
    print("-" * 50)
    for name, metrics in results["comparisons"].items():
        print(
            f"  {name:8s} | Geo={metrics['geometric_distortion']:.4f} | "
            f"Rel={metrics['relational_distortion']:.4f} | "
            f"H={metrics['coupling_entropy']:.2f}"
        )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
