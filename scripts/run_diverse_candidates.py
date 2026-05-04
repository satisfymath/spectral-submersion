"""Batch comparison with diverse candidate pool.

Compares a reference corpus against multiple candidate languages
including non-Polynesian controls (English, Spanish, Japanese, etc.).
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from spectral_submersion.alignment import pairwise_squared_distances
from spectral_submersion.transport import optimal_transport_matrix
from spectral_submersion.evaluation import relational_distortion, geometric_distortion


def load_embed(path: str):
    import numpy as np

    E = np.load(path)
    # Trim to consistent dimension
    return E


def compare_pair(lost_path: str, cand_path: str, cand_name: str, reg: float = 0.5):
    import numpy as np

    E_lost = np.load(lost_path)
    E_cand = np.load(cand_path)

    d = min(E_lost.shape[1], E_cand.shape[1])
    E_lost = E_lost[:, :d]
    E_cand = E_cand[:, :d]

    D_geo = pairwise_squared_distances(E_lost, E_cand)
    Dx = pairwise_squared_distances(E_lost, E_lost)
    Dy = pairwise_squared_distances(E_cand, E_cand)

    Pi_ot = optimal_transport_matrix(D_geo, reg=reg)

    n_x, n_y = E_lost.shape[0], E_cand.shape[0]
    Pi_rand = np.random.rand(n_x, n_y)
    Pi_rand = Pi_rand / Pi_rand.sum(axis=1, keepdims=True)

    return {
        "candidate": cand_name,
        "n_lost": n_x,
        "n_cand": n_y,
        "dim": d,
        "ot_geo": geometric_distortion(Pi_ot, D_geo),
        "ot_rel": relational_distortion(Pi_ot, Dx, Dy),
        "ot_entropy": float(-(Pi_ot[Pi_ot > 0] * np.log(Pi_ot[Pi_ot > 0])).sum()),
        "rand_geo": geometric_distortion(Pi_rand, D_geo),
        "rand_rel": relational_distortion(Pi_rand, Dx, Dy),
    }


def main():
    parser = argparse.ArgumentParser(description="Batch diverse candidate comparison")
    parser.add_argument(
        "--lost-embed", default="data/processed/embeddings_rongorongo_v2.npy"
    )
    parser.add_argument(
        "--output", default="reports/tables/diverse_candidate_comparison.csv"
    )
    parser.add_argument("--reg", type=float, default=0.5)
    args = parser.parse_args()

    # Diverse candidate pool
    candidates = [
        ("maori", "data/processed/embeddings_mi.npy"),
        ("tahitian", "data/processed/embeddings_ty.npy"),
        ("hawaiian", "data/processed/embeddings_haw.npy"),
        ("samoan", "data/processed/embeddings_sm.npy"),
        ("tongan", "data/processed/embeddings_to.npy"),
        ("fijian", "data/processed/embeddings_fj.npy"),
        ("rapa_nui", "data/processed/embeddings_rap.npy"),
        ("english", "data/processed/embeddings_en.npy"),
        ("spanish", "data/processed/embeddings_es.npy"),
        ("japanese", "data/processed/embeddings_ja.npy"),
    ]

    results = []
    for name, path in candidates:
        if not Path(path).exists():
            print(f"Skipping {name}: {path} not found")
            continue
        print(f">>> Comparing against {name} ...")
        res = compare_pair(args.lost_embed, path, name, args.reg)
        results.append(res)

    df = pd.DataFrame(results)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\n{'='*70}")
    print("Diverse Candidate Comparison (all vs reference)")
    print(f"{'='*70}")
    print(df.to_string(index=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
