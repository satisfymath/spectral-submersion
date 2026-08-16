"""F3: singular-value spectra with bootstrap bands: real vs three negative
controls (permuted, frequency-matched random, uniform random). Log scale.
50 bootstrap resamples of lines. Seed 42.
Anti-conclusion: curve separation is the spectral signature of local
co-occurrence structure (C1); it carries no semantic content.
"""
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from spectral_submersion.tokenization import get_sequences_by_line
from spectral_submersion.cooccurrence import cooccurrence_matrix_from_sequences
from spectral_submersion.pmi import ppmi_matrix

SEED = 42
K = 40
B = 50
OUT = Path("paper_v3/figures")
COLORS = {"real": "#0072B2", "permuted": "#D55E00",
          "random (matched freq)": "#009E73", "random (uniform)": "#CC79A7"}


def spectrum(seqs, vocab):
    idx = {t: i for i, t in enumerate(vocab)}
    ids = [[idx[t] for t in s if t in idx] for s in seqs]
    C = cooccurrence_matrix_from_sequences(ids, len(vocab), window_size=2)
    M = ppmi_matrix(C)
    s = np.linalg.svd(M, compute_uv=False)
    return s[:K]


def variants(seqs, rng, freqs, vocab):
    flat = [t for s in seqs for t in s]
    rng.shuffle(flat)
    it = iter(flat)
    perm = [[next(it) for _ in s] for s in seqs]
    probs = np.array([freqs[t] for t in vocab], dtype=float)
    probs /= probs.sum()
    rmatch = [[vocab[i] for i in rng.choice(len(vocab), size=len(s), p=probs)] for s in seqs]
    runif = [[vocab[i] for i in rng.integers(0, len(vocab), size=len(s))] for s in seqs]
    return {"permuted": perm, "random (matched freq)": rmatch, "random (uniform)": runif}


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv("data/raw/lost_language/corpus_rongorongo_real.xml.csv")
    seqs = [[t for t in s if t != "_"] for s in get_sequences_by_line(df)]
    freqs = Counter(t for s in seqs for t in s)
    vocab = sorted(freqs)

    curves = {name: [] for name in COLORS}
    for b in range(B):
        bidx = rng.integers(0, len(seqs), len(seqs))
        bs = [seqs[i] for i in bidx]
        curves["real"].append(spectrum(bs, vocab))
        for name, vs in variants(bs, rng, freqs, vocab).items():
            curves[name].append(spectrum(vs, vocab))

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = np.arange(1, K + 1)
    for name, c in COLORS.items():
        arr = np.stack(curves[name])
        med = np.median(arr, 0)
        lo, hi = np.percentile(arr, [2.5, 97.5], axis=0)
        ax.plot(x, med, color=c, lw=1.8, label=name)
        ax.fill_between(x, lo, hi, color=c, alpha=0.18, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("singular value index")
    ax.set_ylabel("singular value (log)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_title(f"PPMI spectra with bootstrap 95% bands (B={B}), real RR corpus vs controls",
                 fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig3_spectrum_bootstrap.{ext}", dpi=300, bbox_inches="tight")
    print("saved fig3")


if __name__ == "__main__":
    main()
