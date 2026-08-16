"""F9: bigram network of the real RR corpus with the actual vector glyph
drawings as node markers. Edges ∝ bigram frequency, node ring color =
Louvain community (Okabe-Ito). Top-28 glyph bases with available SVGs.
Seed 42. Anti-conclusion: communities are co-occurrence structure (C1),
not semantic groupings.
"""
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from spectral_submersion.tokenization import get_sequences_by_line

SEED = 42
TOP = 28
GLYPH_DIR = Path("data/rongorongo/glyph_svgs")
OUT = Path("paper_v3/figures")
OKABE = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#F0E442"]


def base(tok):
    m = re.match(r"(\d+)", tok)
    return m.group(1).zfill(3) if m else None


def main():
    df = pd.read_csv("data/raw/lost_language/corpus_rongorongo_real.xml.csv")
    seqs = [[base(t) for t in s if t != "_" and base(t)] for s in get_sequences_by_line(df)]
    freqs = Counter(t for s in seqs for t in s)
    avail = {p.stem for p in GLYPH_DIR.glob("*.png")}
    nodes = [t for t, _ in freqs.most_common(60) if t in avail][:TOP]
    nodeset = set(nodes)

    bi = Counter()
    for s in seqs:
        for a, b in zip(s, s[1:]):
            if a in nodeset and b in nodeset and a != b:
                bi[tuple(sorted((a, b)))] += 1

    G = nx.Graph()
    G.add_nodes_from(nodes)
    for (a, b), w in bi.items():
        if w >= 2:
            G.add_edge(a, b, weight=w)
    comms = sorted(nx.community.louvain_communities(G, weight="weight", seed=SEED),
                   key=len, reverse=True)
    comm_of = {n: i for i, c in enumerate(comms) for n in c}

    pos = nx.spring_layout(G, weight="weight", seed=SEED, k=0.35)

    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    wmax = max(d["weight"] for _, _, d in G.edges(data=True))
    for a, b, d in G.edges(data=True):
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color="#999999", alpha=0.25 + 0.55 * d["weight"] / wmax,
                lw=0.5 + 3.5 * d["weight"] / wmax, zorder=1)
    for n in G.nodes:
        c = OKABE[comm_of.get(n, 0) % len(OKABE)]
        ax.scatter(*pos[n], s=1400, facecolor="white", edgecolor=c, linewidths=2.5,
                   zorder=2)
        img = mpimg.imread(str(GLYPH_DIR / f"{n}.png"))
        ab = AnnotationBbox(OffsetImage(img, zoom=0.11), pos[n], frameon=False, zorder=3)
        ax.add_artist(ab)
        ax.annotate(n, pos[n], textcoords="offset points", xytext=(0, -26),
                    ha="center", fontsize=6.5, color="#555555", zorder=4)
    ax.axis("off")
    ax.set_title(f"Bigram network of the top-{TOP} real RR glyph bases "
                 "(edges ∝ adjacency counts; ring color = Louvain community)",
                 fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig9_glyph_network.{ext}", dpi=300, bbox_inches="tight")
    print("saved fig9; communities:", [len(c) for c in comms])


if __name__ == "__main__":
    main()
