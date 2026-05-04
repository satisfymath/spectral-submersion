"""Run pairwise GW for remaining candidates (European languages with large vocab)."""

import sys
import numpy as np
from pathlib import Path
from spectral_submersion.transport import gromov_wasserstein_matrix

LOST_PATH = "data/processed/embeddings_rongorongo_real.npy"
N_SUB = 15

REMAINING = {
    "english": ("data/processed/embeddings_english.npy", "germanic"),
    "spanish": ("data/processed/embeddings_spanish.npy", "romance"),
    "german": ("data/processed/embeddings_german.npy", "germanic"),
    "russian": ("data/processed/embeddings_russian.npy", "slavic"),
    "french": ("data/processed/embeddings_french.npy", "romance"),
    "italian": ("data/processed/embeddings_italian.npy", "romance"),
    "portuguese": ("data/processed/embeddings_portuguese.npy", "romance"),
    "japanese": ("data/processed/embeddings_ja.npy", "japonic"),
    "arabic": ("data/processed/embeddings_ar.npy", "semitic"),
    "korean": ("data/processed/embeddings_ko.npy", "koreanic"),
}


def dist_matrix(E):
    E_c = E - E.mean(axis=0, keepdims=True)
    return np.sqrt(np.sum((E_c[:, None, :] - E_c[None, :, :]) ** 2, axis=2) + 1e-10)


def main():
    d = 16
    E_lost = np.load(LOST_PATH)[:, :d]
    rng = np.random.default_rng(42)
    D_lost = dist_matrix(E_lost)
    idx_lost = rng.choice(E_lost.shape[0], min(N_SUB, E_lost.shape[0]), replace=False)
    D_lost_sub = D_lost[np.ix_(idx_lost, idx_lost)]

    for name, (path, family) in REMAINING.items():
        E_cand = np.load(path)[:, :d]
        D_cand = dist_matrix(E_cand)
        idx_cand = rng.choice(
            E_cand.shape[0], min(N_SUB, E_cand.shape[0]), replace=False
        )
        D_cand_sub = D_cand[np.ix_(idx_cand, idx_cand)]

        Pi = gromov_wasserstein_matrix(
            D_lost_sub, D_cand_sub, reg=2.0, max_iter=8, sinkhorn_iter=30, tol=1e-3
        )
        gw_dist = float(np.sum(Pi * (D_lost_sub @ Pi @ D_cand_sub)))
        entropy = float(-np.sum(Pi * np.log(Pi + 1e-30)))
        print(
            f"{name},{family},{E_cand.shape[0]},{N_SUB},{gw_dist:.6f},{entropy:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
