"""Run Gromov-Wasserstein alignment and compare against direct OT.

This script evaluates whether GW (relational) alignment produces
meaningfully different couplings than direct geometric OT when
vocabularies have different sizes and no anchors exist.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from spectral_submersion.alignment import pairwise_squared_distances
from spectral_submersion.transport import optimal_transport_matrix, gromov_wasserstein_matrix
from spectral_submersion.evaluation import relational_distortion, geometric_distortion


def main():
    parser = argparse.ArgumentParser(description="Run GW alignment comparison")
    parser.add_argument("--lost-embed", required=True)
    parser.add_argument("--candidate-embed", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--reg", type=float, default=0.5)
    parser.add_argument("--gw-max-iter", type=int, default=50)
    parser.add_argument("--output", default="reports/tables/gw_comparison.json")
    args = parser.parse_args()

    E_lost = np.load(args.lost_embed)
    E_cand = np.load(args.candidate_embed)

    d = min(E_lost.shape[1], E_cand.shape[1])
    E_lost = E_lost[:, :d]
    E_cand = E_cand[:, :d]

    # Distance matrices
    Dx = pairwise_squared_distances(E_lost, E_lost)
    Dy = pairwise_squared_distances(E_cand, E_cand)
    D_geo = pairwise_squared_distances(E_lost, E_cand)

    # Direct OT
    Pi_ot = optimal_transport_matrix(D_geo, reg=args.reg)

    # GW alignment
    Pi_gw = gromov_wasserstein_matrix(Dx, Dy, reg=args.reg, max_iter=args.gw_max_iter)

    # Random baseline
    n_x, n_y = E_lost.shape[0], E_cand.shape[0]
    Pi_rand = np.random.rand(n_x, n_y)
    Pi_rand = Pi_rand / Pi_rand.sum(axis=1, keepdims=True)

    results = {
        "candidate": args.candidate_name,
        "n_lost": n_x,
        "n_cand": n_y,
        "embedding_dim": d,
        "reg": args.reg,
        "methods": {},
    }

    for name, Pi in [("ot", Pi_ot), ("gw", Pi_gw), ("random", Pi_rand)]:
        results["methods"][name] = {
            "geometric_distortion": geometric_distortion(Pi, D_geo),
            "relational_distortion": relational_distortion(Pi, Dx, Dy),
            "coupling_entropy": float(-(Pi[Pi > 0] * np.log(Pi[Pi > 0])).sum()),
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"GW alignment comparison: {args.candidate_name}")
    print(f"  Size: {n_x} x {n_y}, dim={d}, reg={args.reg}")
    print("-" * 50)
    for name, metrics in results["methods"].items():
        print(f"  {name:8s} | Geo={metrics['geometric_distortion']:.4f} | "
              f"Rel={metrics['relational_distortion']:.4f} | "
              f"H={metrics['coupling_entropy']:.2f}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
