"""Run pairwise GW - subsample first, then compute distances."""

import sys
import numpy as np
from pathlib import Path
from spectral_submersion.transport import gromov_wasserstein_matrix

LOST_PATH = "data/processed/embeddings_rongorongo_real.npy"
N_SUB = 15

ALL_CANDIDATES = {
    "maori": ("data/processed/embeddings_mi.npy", "polynesian"),
    "tahitian": ("data/processed/embeddings_ty.npy", "polynesian"),
    "hawaiian": ("data/processed/embeddings_haw.npy", "polynesian"),
    "samoan": ("data/processed/embeddings_sm.npy", "polynesian"),
    "tongan": ("data/processed/embeddings_to.npy", "polynesian"),
    "fijian": ("data/processed/embeddings_fj.npy", "austronesian"),
    "rapa_nui": ("data/processed/embeddings_rap.npy", "polynesian"),
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


def dist_matrix_sub(E, n_sub, rng):
    """Subsample first, then compute distance matrix."""
    idx = rng.choice(E.shape[0], min(n_sub, E.shape[0]), replace=False)
    E_sub = E[idx]
    E_c = E_sub - E_sub.mean(axis=0, keepdims=True)
    D = np.sqrt(np.sum((E_c[:, None, :] - E_c[None, :, :]) ** 2, axis=2) + 1e-10)
    return D


def main():
    d = 16
    rng = np.random.default_rng(42)
    E_lost = np.load(LOST_PATH)[:, :d]
    D_lost = dist_matrix_sub(E_lost, N_SUB, rng)

    print("name,family,n_vocab,n_sub,gw_distance,coupling_entropy", flush=True)
    for name, (path, family) in ALL_CANDIDATES.items():
        E_cand = np.load(path)[:, :d]
        D_cand = dist_matrix_sub(E_cand, N_SUB, rng)
        Pi = gromov_wasserstein_matrix(
            D_lost, D_cand, reg=2.0, max_iter=8, sinkhorn_iter=30, tol=1e-3
        )
        gw_dist = float(np.sum(Pi * (D_lost @ Pi @ D_cand)))
        entropy = float(-np.sum(Pi * np.log(Pi + 1e-30)))
        print(
            f"{name},{family},{E_cand.shape[0]},{N_SUB},{gw_dist:.6f},{entropy:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
