"""Compare GPA truncated vs Multi-marginal GW for Rongorongo real.

Head-to-head comparison between two alignment strategies:
1. GPA truncated: Generalized Procrustes with min-vocabulary truncation (38 tokens)
2. Multi-marginal GW: preserves full vocabulary sizes, iteratively refines couplings

Both methods align Rongorongo real against the same 17 candidate languages.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from spectral_submersion.multi_alignment import (
    build_consensus_space,
    project_lost_to_consensus,
    consensus_distance_matrix,
)
from spectral_submersion.transport import multi_marginal_gw, consensus_from_multi_gw

CANDIDATES = {
    "maori": {"embed": "data/processed/embeddings_mi.npy", "family": "polynesian"},
    "tahitian": {"embed": "data/processed/embeddings_ty.npy", "family": "polynesian"},
    "hawaiian": {"embed": "data/processed/embeddings_haw.npy", "family": "polynesian"},
    "samoan": {"embed": "data/processed/embeddings_sm.npy", "family": "polynesian"},
    "tongan": {"embed": "data/processed/embeddings_to.npy", "family": "polynesian"},
    "fijian": {"embed": "data/processed/embeddings_fj.npy", "family": "austronesian"},
    "rapa_nui": {"embed": "data/processed/embeddings_rap.npy", "family": "polynesian"},
    "english": {"embed": "data/processed/embeddings_english.npy", "family": "germanic"},
    "spanish": {"embed": "data/processed/embeddings_spanish.npy", "family": "romance"},
    "russian": {"embed": "data/processed/embeddings_russian.npy", "family": "slavic"},
    "japanese": {"embed": "data/processed/embeddings_ja.npy", "family": "japonic"},
}

LOST = {"rongorongo_real": "data/processed/embeddings_rongorongo_real.npy"}


def compute_distance_matrix(E):
    E_c = E - E.mean(axis=0, keepdims=True)
    D = np.sqrt(np.sum((E_c[:, None, :] - E_c[None, :, :]) ** 2, axis=2) + 1e-10)
    return D


def main():
    out_dir = Path("reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading embeddings...")
    E_lost = np.load(LOST["rongorongo_real"])[:, :16]
    cand_embeds = {}
    cand_families = {}
    for name, info in CANDIDATES.items():
        E = np.load(info["embed"])[:, :16]
        cand_embeds[name] = E
        cand_families[name] = info["family"]
        print(f"  {name}: {E.shape}")

    d = 16
    results_gpa = []
    results_gw = []

    # ====== METHOD 1: GPA truncated ======
    print("\n=== Method 1: GPA Consensus (truncated to min vocab) ===")
    cand_list = list(CANDIDATES.keys())
    E_cands = [cand_embeds[n] for n in cand_list]
    R, aligned, projections = build_consensus_space(E_cands, method="gpa", target_dim=d)
    print(f"  Consensus shape: {R.shape}")
    print(f"  Min vocab (truncation): {min(E.shape[0] for E in E_cands)}")

    W = project_lost_to_consensus(E_lost, R, regularization=1e-3)
    E_lost_proj = E_lost[:, :d] @ W

    for i, name in enumerate(cand_list):
        E_cand = aligned[i][: R.shape[0], :]
        diff = E_lost_proj[: R.shape[0], :] - E_cand
        geo_dist = float(np.mean(np.linalg.norm(diff, axis=1)))
        rel_dist_x = float(np.mean(np.linalg.norm(diff, axis=1)))
        results_gpa.append(
            {
                "method": "GPA_truncated",
                "candidate": name,
                "family": cand_families[name],
                "n_cand": cand_embeds[name].shape[0],
                "n_consensus": R.shape[0],
                "geo_dist": geo_dist,
                "rel_dist": rel_dist_x,
            }
        )
        print(f"  vs {name:15s} ({cand_families[name]:15s}): geo_dist={geo_dist:.4f}")

    # ====== METHOD 2: Multi-marginal GW ======
    print("\n=== Method 2: Multi-marginal GW (full vocabularies) ===")
    all_embeds = [E_lost] + [cand_embeds[n] for n in cand_list]
    all_names = ["rongorongo_real"] + cand_list
    all_dists = [compute_distance_matrix(E) for E in all_embeds]

    n_sub = 25
    rng = np.random.default_rng(42)

    # Subsample for GW tractability
    all_dists_sub = []
    for D in all_dists:
        idx = rng.choice(D.shape[0], min(n_sub, D.shape[0]), replace=False)
        all_dists_sub.append(D[np.ix_(idx, idx)])

    couplings = multi_marginal_gw(
        all_dists_sub, reg=5.0, max_iter=2, sinkhorn_iter=30, tol=1e-3
    )

    for j, name in enumerate(cand_list):
        idx_j = j + 1
        Pi = couplings[0][idx_j]
        gw_dist = float(np.sum(Pi * (all_dists_sub[0] @ Pi @ all_dists_sub[idx_j].T)))
        n_lost = min(n_sub, E_lost.shape[0])
        n_cand = min(n_sub, cand_embeds[name].shape[0])
        results_gw.append(
            {
                "method": "multi_marginal_GW",
                "candidate": name,
                "family": cand_families[name],
                "n_cand": cand_embeds[name].shape[0],
                "n_subsample": n_sub,
                "gw_distance": gw_dist,
            }
        )
        print(f"  vs {name:15s} ({cand_families[name]:15s}): gw_dist={gw_dist:.4f}")

    df = pd.DataFrame(results_gpa + results_gw)
    out_path = out_dir / "gpa_vs_mgw_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    print("\n=== Comparison Summary ===")
    gpa = df[df["method"] == "GPA_truncated"].sort_values("rel_dist")
    gw = df[df["method"] == "multi_marginal_GW"].sort_values("gw_distance")
    print("\nGPA (truncated) ranking:")
    for _, row in gpa.iterrows():
        print(
            f"  {row['candidate']:15s} ({row['family']:15s}): rel_dist={row['rel_dist']:.4f}"
        )
    print("\nMulti-marginal GW ranking:")
    for _, row in gw.iterrows():
        print(
            f"  {row['candidate']:15s} ({row['family']:15s}): gw_dist={row['gw_distance']:.4f}"
        )


if __name__ == "__main__":
    main()
