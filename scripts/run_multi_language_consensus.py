"""Run multi-language consensus pipeline.

Builds consensus space R from multiple candidate languages,
projects lost language into R, and generates comparative metrics.

Usage:
    .venv/bin/python scripts/run_multi_language_consensus.py \
        --lost-embed data/processed/embeddings_synthetic_v2.npy \
        --config configs/candidate_languages.yaml \
        --output reports/tables/multi_language_consensus.csv \
        --method gpa \
        --target-dim 16
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from spectral_submersion.multi_alignment import (
    build_consensus_space,
    project_lost_to_consensus,
)
from spectral_submersion.alignment import pairwise_squared_distances
from spectral_submersion.transport import optimal_transport_matrix
from spectral_submersion.evaluation import relational_distortion, geometric_distortion


def main():
    parser = argparse.ArgumentParser(description="Multi-language consensus alignment")
    parser.add_argument("--lost-embed", required=True, help="Path to lost language embeddings .npy")
    parser.add_argument("--config", default="configs/candidate_languages.yaml")
    parser.add_argument("--output", default="reports/tables/multi_language_consensus.csv")
    parser.add_argument("--method", default="gpa", choices=["gpa", "intersection"])
    parser.add_argument("--target-dim", type=int, default=None)
    parser.add_argument("--reg", type=float, default=0.5, help="OT regularization")
    parser.add_argument("--tikhonov", type=float, default=1e-3, help="Tikhonov regularization for projection")
    args = parser.parse_args()

    E_lost = np.load(args.lost_embed)
    print(f"Lost embeddings: {E_lost.shape}")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    candidates = config.get("candidates", [])
    candidate_embeds = []
    candidate_names = []

    for cand in candidates:
        name = cand["name"]
        code = cand.get("code", name)
        embed_path = Path(f"data/processed/embeddings_{code}.npy")
        if not embed_path.exists():
            print(f"  Skipping {name}: {embed_path} not found")
            continue
        E_cand = np.load(embed_path)
        candidate_embeds.append(E_cand)
        candidate_names.append(name)
        print(f"  Loaded {name}: {E_cand.shape}")

    if len(candidate_embeds) < 2:
        raise ValueError("Need at least 2 candidate embeddings for consensus")

    print(f"\nBuilding consensus space via {args.method} from {len(candidate_embeds)} languages ...")
    R, aligned_cands, projections = build_consensus_space(
        candidate_embeds,
        method=args.method,
        target_dim=args.target_dim,
    )
    print(f"Consensus R shape: {R.shape}")

    print("Projecting lost language into consensus space ...")
    d_consensus = R.shape[1]
    E_lost_trim = E_lost[:, :d_consensus]
    W = project_lost_to_consensus(E_lost_trim, R, regularization=args.tikhonov)
    E_lost_proj = E_lost_trim @ W
    print(f"Projected lost embeddings: {E_lost_proj.shape}")

    print("\nEvaluating against each candidate in consensus space ...")
    results = []
    for name, E_cand in zip(candidate_names, aligned_cands):
        d = min(E_lost_proj.shape[1], E_cand.shape[1])
        E_lost_d = E_lost_proj[:, :d]
        E_cand_d = E_cand[:, :d]

        D_geo = pairwise_squared_distances(E_lost_d, E_cand_d)
        Dx = pairwise_squared_distances(E_lost_d, E_lost_d)
        Dy = pairwise_squared_distances(E_cand_d, E_cand_d)

        Pi_ot = optimal_transport_matrix(D_geo, reg=args.reg)

        geo = geometric_distortion(Pi_ot, D_geo)
        rel = relational_distortion(Pi_ot, Dx, Dy)
        ent = float(-(Pi_ot[Pi_ot > 0] * np.log(Pi_ot[Pi_ot > 0])).sum())

        results.append({
            "candidate": name,
            "family": next(c.get("family", "unknown") for c in candidates if c["name"] == name),
            "n_cand": E_cand.shape[0],
            "dim": d,
            "geo_dist": geo,
            "rel_dist": rel,
            "entropy": ent,
        })
        print(f"  {name:15s} | Geo={geo:.2f} | Rel={rel:.2f} | Ent={ent:.2f}")

    # Also compute a baseline: average distance to all candidates
    avg_rel_dist = np.mean([r["rel_dist"] for r in results])
    print(f"\nAverage relational distortion across candidates: {avg_rel_dist:.2f}")

    df = pd.DataFrame(results)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # Save full result bundle
    bundle_path = out_path.with_suffix(".json")
    with open(bundle_path, "w", encoding="utf-8") as f:
        json.dump({
            "lost_shape": list(E_lost.shape),
            "consensus_shape": list(R.shape),
            "method": args.method,
            "target_dim": args.target_dim,
            "candidates": results,
            "average_rel_dist": float(avg_rel_dist),
        }, f, indent=2, ensure_ascii=False)

    print(f"\nSaved CSV to {out_path}")
    print(f"Saved JSON to {bundle_path}")


if __name__ == "__main__":
    main()
