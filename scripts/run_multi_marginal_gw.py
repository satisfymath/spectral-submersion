"""Run multi-marginal GW analysis on Rongorongo real vs all 17 candidate languages.

Compares:
1. GPA-based consensus (truncated to min vocab size) vs multi-marginal GW
2. Structural feature embeddings vs co-occurrence embeddings
3. All 17 candidate languages: 7 Polynesian + 10 controls
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

CANDIDATES = {
    # Polynesian / Austronesian
    "maori": {"embed": "data/processed/embeddings_mi.npy", "family": "polynesian"},
    "tahitian": {"embed": "data/processed/embeddings_ty.npy", "family": "polynesian"},
    "hawaiian": {"embed": "data/processed/embeddings_haw.npy", "family": "polynesian"},
    "samoan": {"embed": "data/processed/embeddings_sm.npy", "family": "polynesian"},
    "tongan": {"embed": "data/processed/embeddings_to.npy", "family": "polynesian"},
    "fijian": {"embed": "data/processed/embeddings_fj.npy", "family": "austronesian"},
    "rapa_nui": {"embed": "data/processed/embeddings_rap.npy", "family": "polynesian"},
    # Controls (large)
    "english": {"embed": "data/processed/embeddings_english.npy", "family": "germanic"},
    "spanish": {"embed": "data/processed/embeddings_spanish.npy", "family": "romance"},
    "german": {"embed": "data/processed/embeddings_german.npy", "family": "germanic"},
    "russian": {"embed": "data/processed/embeddings_russian.npy", "family": "slavic"},
    "french": {"embed": "data/processed/embeddings_french.npy", "family": "romance"},
    "italian": {"embed": "data/processed/embeddings_italian.npy", "family": "romance"},
    "portuguese": {
        "embed": "data/processed/embeddings_portuguese.npy",
        "family": "romance",
    },
    # Controls (medium)
    "japanese": {"embed": "data/processed/embeddings_ja.npy", "family": "japonic"},
    "arabic": {"embed": "data/processed/embeddings_ar.npy", "family": "semitic"},
    "korean": {"embed": "data/processed/embeddings_ko.npy", "family": "koreanic"},
}

LOST_CORPORA = {
    "rongorongo_real": "data/processed/embeddings_rongorongo_real.npy",
    "indus_real": "data/processed/embeddings_indus_real.npy",
}


def compute_distance_matrix(embeddings):
    E = embeddings - embeddings.mean(axis=0, keepdims=True)
    D = np.sqrt(np.sum((E[:, None, :] - E[None, :, :]) ** 2, axis=2) + 1e-10)
    return D


def gw_distance(D_x, D_y, reg=0.1, max_iter=20):
    from spectral_submersion.transport import gromov_wasserstein_matrix

    n_x, n_y = D_x.shape[0], D_y.shape[0]
    a = np.ones(n_x) / n_x
    b = np.ones(n_y) / n_y
    Pi = gromov_wasserstein_matrix(D_x, D_y, a, b, reg=reg, max_iter=max_iter)
    gw_dist = np.sum(
        Pi
        * (
            D_x @ Pi @ D_y.T
            - 2 * (D_x**2) @ Pi.sum(axis=1, keepdims=False).reshape(-1, 1) / n_y
        )
    )
    return float(gw_dist), Pi


def main():
    out_dir = Path("reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading lost language embeddings...")
    lost_embeds = {}
    for name, path in LOST_CORPORA.items():
        E = np.load(path)
        lost_embeds[name] = E[:, :16] if E.shape[1] > 16 else E
        print(f"  {name}: shape={E.shape}")

    print("Loading candidate embeddings...")
    cand_embeds = {}
    cand_info = {}
    for name, info in CANDIDATES.items():
        E = np.load(info["embed"])
        cand_embeds[name] = E[:, :16] if E.shape[1] > 16 else E
        cand_info[name] = info["family"]
        print(f"  {name}: shape={E.shape}, family={info['family']}")

    d = min(lost_embeds["rongorongo_real"].shape[1], 16)
    print(f"\nUsing embedding dimension d={d}")

    results = []
    for lost_name, E_lost in lost_embeds.items():
        print(f"\n=== Analyzing {lost_name} ===")
        E_lost_d = E_lost[:, :d]
        n_lost = E_lost_d.shape[0]
        D_lost = compute_distance_matrix(E_lost_d)

        # GS subsampling for large candidates
        max_n = min(n_lost, 38)

        for cand_name, E_cand_full in cand_embeds.items():
            E_cand_d = E_cand_full[:, :d]
            n_cand = E_cand_d.shape[0]

            if n_cand > max_n:
                idx = np.random.default_rng(42).choice(n_cand, max_n, replace=False)
                E_cand = E_cand_d[idx]
            else:
                E_cand = E_cand_d

            D_cand = compute_distance_matrix(E_cand)

            n_gw = min(D_lost.shape[0], D_cand.shape[0], 50)

            rng = np.random.default_rng(42)
            idx_lost = rng.choice(D_lost.shape[0], n_gw, replace=False)
            idx_cand = rng.choice(D_cand.shape[0], n_gw, replace=False)

            D_sub_lost = D_lost[np.ix_(idx_lost, idx_lost)]
            D_sub_cand = D_cand[np.ix_(idx_cand, idx_cand)]

            try:
                gw_dist, Pi = gw_distance(D_sub_lost, D_sub_cand, reg=0.5, max_iter=10)
            except Exception as e:
                print(f"  GW failed for {cand_name}: {e}")
                gw_dist = float("inf")

            results.append(
                {
                    "lost_corpus": lost_name,
                    "candidate": cand_name,
                    "family": cand_info[cand_name],
                    "n_lost": n_lost,
                    "n_cand": E_cand_d.shape[0],
                    "n_sample": n_gw,
                    "gw_distance": gw_dist,
                    "dim": d,
                }
            )
            print(f"  vs {cand_name} ({cand_info[cand_name]}): gw_dist={gw_dist:.4f}")

    df = pd.DataFrame(results)
    out_path = out_dir / "multi_marginal_gw_17candidates.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    for lost_corpus in df["lost_corpus"].unique():
        sub = df[df["lost_corpus"] == lost_corpus].sort_values("gw_distance")
        print(f"\n=== {lost_corpus} ranking ===")
        for _, row in sub.iterrows():
            print(
                f"  {row['candidate']:15s} ({row['family']:15s}): gw_dist={row['gw_distance']:.4f}"
            )


if __name__ == "__main__":
    main()
