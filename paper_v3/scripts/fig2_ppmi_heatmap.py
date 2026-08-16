"""F2: PPMI heatmap reordered by Louvain communities, real vs permuted.
Top-120 most frequent types for legibility. Seed 42.
Anti-conclusion: block structure evidences co-occurrence communities (C1);
community membership does not identify meaning (C2 ceiling).
"""
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from spectral_submersion.tokenization import get_sequences_by_line
from spectral_submersion.cooccurrence import cooccurrence_matrix_from_sequences
from spectral_submersion.pmi import ppmi_matrix

SEED = 42
TOP = 120
OUT = Path("paper_v3/figures")


def ppmi_for(seqs, vocab):
    idx = {t: i for i, t in enumerate(vocab)}
    ids = [[idx[t] for t in s if t in idx] for s in seqs]
    C = cooccurrence_matrix_from_sequences(ids, len(vocab), window_size=2)
    return ppmi_matrix(C)


def louvain_order(M, vocab):
    G = nx.Graph()
    G.add_nodes_from(range(len(vocab)))
    n = len(vocab)
    for i in range(n):
        for j in range(i + 1, n):
            if M[i, j] > 0.2:
                G.add_edge(i, j, weight=M[i, j])
    comms = nx.community.louvain_communities(G, weight="weight", seed=SEED)
    comms = sorted(comms, key=len, reverse=True)
    order = [i for c in comms for i in sorted(c)]
    sizes = [len(c) for c in comms]
    return order, sizes


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv("data/raw/lost_language/corpus_rongorongo_real.xml.csv")
    seqs = [[t for t in s if t != "_"] for s in get_sequences_by_line(df)]
    freqs = Counter(t for s in seqs for t in s)
    vocab = [t for t, _ in freqs.most_common(TOP)]

    M_real = ppmi_for(seqs, vocab)
    flat = [t for s in seqs for t in s]
    rng.shuffle(flat)
    it = iter(flat)
    seqs_perm = [[next(it) for _ in s] for s in seqs]
    M_perm = ppmi_for(seqs_perm, vocab)

    order, sizes = louvain_order(M_real, vocab)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0))
    vmax = np.percentile(M_real, 99)
    for ax, M, title in [(axes[0], M_real, f"Real (Louvain order, {len(sizes)} communities)"),
                         (axes[1], M_perm, "Permuted control (same order)")]:
        ax.imshow(M[np.ix_(order, order)], cmap="Blues", vmin=0, vmax=vmax,
                  interpolation="nearest")
        # community boundaries
        pos = 0
        for s in sizes[:-1]:
            pos += s
            ax.axhline(pos - 0.5, color="#D55E00", lw=0.7)
            ax.axvline(pos - 0.5, color="#D55E00", lw=0.7)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"PPMI matrix of the top-{TOP} real RR types, reordered by Louvain communities",
                 fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig2_ppmi_heatmap.{ext}", dpi=300, bbox_inches="tight")
    print("saved fig2; community sizes:", sizes)


if __name__ == "__main__":
    main()
