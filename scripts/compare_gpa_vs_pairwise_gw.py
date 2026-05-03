"""Compare GPA truncated vs Pairwise GW for Rongorongo real.

Runs both methods sequentially, prints results as they come.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from spectral_submersion.multi_alignment import (
    build_consensus_space,
    project_lost_to_consensus,
)
from spectral_submersion.transport import gromov_wasserstein_matrix

CANDIDATES = {
    "maori":      ("data/processed/embeddings_mi.npy",        "polynesian"),
    "tahitian":    ("data/processed/embeddings_ty.npy",        "polynesian"),
    "hawaiian":    ("data/processed/embeddings_haw.npy",       "polynesian"),
    "samoan":      ("data/processed/embeddings_sm.npy",        "polynesian"),
    "tongan":      ("data/processed/embeddings_to.npy",        "polynesian"),
    "fijian":      ("data/processed/embeddings_fj.npy",        "austronesian"),
    "rapa_nui":   ("data/processed/embeddings_rap.npy",       "polynesian"),
    "english":     ("data/processed/embeddings_english.npy",   "germanic"),
    "spanish":     ("data/processed/embeddings_spanish.npy",   "romance"),
    "german":      ("data/processed/embeddings_german.npy",    "germanic"),
    "russian":     ("data/processed/embeddings_russian.npy",   "slavic"),
    "french":      ("data/processed/embeddings_french.npy",   "romance"),
    "italian":     ("data/processed/embeddings_italian.npy",  "romance"),
    "portuguese":  ("data/processed/embeddings_portuguese.npy","romance"),
    "japanese":    ("data/processed/embeddings_ja.npy",        "japonic"),
    "arabic":      ("data/processed/embeddings_ar.npy",        "semitic"),
    "korean":      ("data/processed/embeddings_ko.npy",       "koreanic"),
}

LOST_PATH = "data/processed/embeddings_rongorongo_real.npy"
N_SUB = 20


def dist_matrix(E):
    E_c = E - E.mean(axis=0, keepdims=True)
    D = np.sqrt(np.sum((E_c[:, None, :] - E_c[None, :, :]) ** 2, axis=2) + 1e-10)
    return D


def main():
    out_dir = Path("reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    d = 16

    E_lost = np.load(LOST_PATH)[:, :d]
    print(f"Lost: {E_lost.shape}", flush=True)

    cand_list = list(CANDIDATES.keys())
    cand_embeds = {}
    cand_families = {}
    for name, (path, family) in CANDIDATES.items():
        E = np.load(path)[:, :d]
        cand_embeds[name] = E
        cand_families[name] = family
        print(f"  {name}: {E.shape}", flush=True)

    # ====== METHOD 1: GPA truncated ======
    print("\n=== GPA Consensus (truncated) ===", flush=True)
    E_cands = [cand_embeds[n] for n in cand_list]
    n_min = min(E.shape[0] for E in E_cands)
    print(f"  n_min={n_min}", flush=True)
    E_trunc = [E[:n_min, :d] for E in E_cands]

    R, aligned, projections = build_consensus_space(E_trunc, method="gpa", target_dim=d)
    W = project_lost_to_consensus(E_lost[:n_min, :d], R, regularization=1e-3)
    E_lost_proj = E_lost[:n_min, :d] @ W

    gpa_results = []
    for i, name in enumerate(cand_list):
        E_cand = aligned[i][:R.shape[0], :]
        diff = E_lost_proj[:R.shape[0], :] - E_cand
        geo_dist = float(np.mean(np.linalg.norm(diff, axis=1)))
        reli = np.linalg.norm(diff, axis=1) / (np.linalg.norm(E_cand, axis=1) + 1e-10)
        rel_dist = float(np.mean(reli))
        gpa_results.append({
            "method": "GPA_truncated", "candidate": name,
            "family": cand_families[name], "n_cand": cand_embeds[name].shape[0],
            "n_consensus": R.shape[0], "geo_dist": geo_dist, "rel_dist": rel_dist,
        })
        print(f"  {name:15s}: geo={geo_dist:.4f} rel={rel_dist:.4f}", flush=True)

    # ====== METHOD 2: Pairwise GW ======
    print("\n=== Pairwise GW ===", flush=True)
    rng = np.random.default_rng(42)
    D_lost = dist_matrix(E_lost)
    idx_lost = rng.choice(E_lost.shape[0], min(N_SUB, E_lost.shape[0]), replace=False)
    D_lost_sub = D_lost[np.ix_(idx_lost, idx_lost)]

    gw_results = []
    for name in cand_list:
        E_cand = cand_embeds[name][:, :d]
        D_cand = dist_matrix(E_cand)
        idx_cand = rng.choice(E_cand.shape[0], min(N_SUB, E_cand.shape[0]), replace=False)
        D_cand_sub = D_cand[np.ix_(idx_cand, idx_cand)]

        Pi = gromov_wasserstein_matrix(D_lost_sub, D_cand_sub, reg=2.0,
                                        max_iter=10, sinkhorn_iter=50, tol=1e-3)
        gw_dist = float(np.sum(Pi * (D_lost_sub @ Pi @ D_cand_sub)))
        entropy = float(-np.sum(Pi * np.log(Pi + 1e-30)))
        gw_results.append({
            "method": "pairwise_GW", "candidate": name,
            "family": cand_families[name], "n_cand": E_cand.shape[0],
            "n_sub": N_SUB, "gw_distance": gw_dist, "coupling_entropy": entropy,
        })
        print(f"  {name:15s}: gw_dist={gw_dist:.4f} entropy={entropy:.4f}", flush=True)

    # Save
    df = pd.DataFrame(gpa_results + gw_results)
    out_path = out_dir / "gpa_vs_pairwise_gw_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}", flush=True)

    # Compare rankings
    from scipy.stats import spearmanr
    gpa_df = df[df["method"] == "GPA_truncated"].sort_values("rel_dist")
    gw_df = df[df["method"] == "pairwise_GW"].sort_values("gw_distance")

    print("\n" + "=" * 70)
    print("GPA ranking:", flush=True)
    for rank, (_, r) in enumerate(gpa_df.iterrows(), 1):
        m = " *" if r["family"] == "polynesian" else ""
        print(f"  {rank:2d}. {r['candidate']:15s} ({r['family']:15s}): rel_dist={r['rel_dist']:.4f}{m}")

    print("\nGW ranking:", flush=True)
    for rank, (_, r) in enumerate(gw_df.iterrows(), 1):
        m = " *" if r["family"] == "polynesian" else ""
        print(f"  {rank:2d}. {r['candidate']:15s} ({r['family']:15s}): gw_dist={r['gw_distance']:.4f}{m}")

    common = gpa_df["candidate"].tolist()
    gpa_r = list(range(len(common)))
    gw_r = [list(gw_df["candidate"]).index(n) for n in common]
    rho, p_val = spearmanr(gpa_r, gw_r)
    print(f"\nSpearman rho={rho:.4f}, p={p_val:.6f}")


if __name__ == "__main__":
    main()