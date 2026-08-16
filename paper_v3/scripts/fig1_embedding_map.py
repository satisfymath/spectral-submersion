"""F1: PCA map of spectral embeddings of the 941 real RR types, colored by
Barthel series, with a permuted-control panel. Seed 42.

Anti-conclusion (goes in caption): clusters aligned with Barthel series are
C1 structural signal about co-occurrence regularity; they do NOT establish
iconic or semantic identity of any glyph (C2 ceiling, Prop. P7).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from spectral_submersion.tokenization import get_sequences_by_line
from spectral_submersion.cooccurrence import cooccurrence_matrix_from_sequences
from spectral_submersion.pmi import ppmi_matrix

SEED = 42
CORPUS = "data/raw/lost_language/corpus_rongorongo_real.xml.csv"
OUT = Path("paper_v3/figures")

# Okabe-Ito
SERIES = [
    ("001-099 geometric", 1, 99, "#0072B2"),
    ("100-199", 100, 199, "#56B4E9"),
    ("200-399 anthropomorph", 200, 399, "#D55E00"),
    ("400-599", 400, 599, "#E69F00"),
    ("600-699 bird", 600, 699, "#009E73"),
    ("700-799 fish/marine", 700, 799, "#CC79A7"),
]


def base(tok):
    d = ""
    for ch in tok:
        if ch.isdigit():
            d += ch
        else:
            break
    return int(d[:3]) if len(d) >= 3 else (int(d) if d else None)


def series_color(tok):
    b = base(tok)
    if b is None:
        return "gray", "other"
    for name, lo, hi, c in SERIES:
        if lo <= b <= hi:
            return c, name
    return "gray", "other"


def embed(seqs, vocab):
    idx = {t: i for i, t in enumerate(vocab)}
    ids = [[idx[t] for t in s] for s in seqs]
    C = cooccurrence_matrix_from_sequences(ids, len(vocab), window_size=2)
    M = ppmi_matrix(C)
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    E = U[:, :16] * S[:16]
    # PCA to 2D
    E = E - E.mean(0)
    _, _, Vt = np.linalg.svd(E, full_matrices=False)
    return E @ Vt[:2].T


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(CORPUS)
    seqs = [[t for t in s if t != "_"] for s in get_sequences_by_line(df)]
    from collections import Counter
    freqs = Counter(t for s in seqs for t in s)
    vocab = sorted(freqs)

    X_real = embed(seqs, vocab)
    # permuted control: shuffle tokens across the corpus preserving line lengths
    flat = [t for s in seqs for t in s]
    rng.shuffle(flat)
    it = iter(flat)
    seqs_perm = [[next(it) for _ in s] for s in seqs]
    X_perm = embed(seqs_perm, vocab)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharex=False)
    for ax, X, title in [(axes[0], X_real, "Real corpus"),
                         (axes[1], X_perm, "Permuted control")]:
        for name, lo, hi, c in SERIES + [("other", -1, -1, "gray")]:
            xs, ys, ss = [], [], []
            for i, t in enumerate(vocab):
                col, sname = series_color(t)
                if sname == name or (name == "other" and sname == "other"):
                    xs.append(X[i, 0]); ys.append(X[i, 1])
                    ss.append(8 + 3 * np.sqrt(freqs[t]))
            ax.scatter(xs, ys, s=ss, c=(c if name != "other" else "lightgray"),
                       alpha=0.75, edgecolors="none", label=name if ax is axes[0] else None)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#cccccc")
    axes[0].legend(fontsize=7, loc="best", framealpha=0.9)
    fig.suptitle("Spectral embeddings of 941 real Rongorongo types (PPMI+SVD, w=2, k=16), "
                 "colored by Barthel numeric series", fontsize=10)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_embedding_map.{ext}", dpi=300, bbox_inches="tight")
    print("saved fig1")


if __name__ == "__main__":
    main()
