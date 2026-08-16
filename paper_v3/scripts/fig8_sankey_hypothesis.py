"""F8: flow diagram grammatical category -> Barthel series, counted from
the v4 parallel corpus. Title carries the claim level: WORKING HYPOTHESIS
(C2). Matplotlib-only Sankey-style ribbon plot. Seed 42.
"""
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import generate_massive_parallel_v4 as v4  # noqa: E402

OUT = Path("paper_v3/figures")
SERIES = [("001-099", 1, 99, "#0072B2"), ("100-199", 100, 199, "#56B4E9"),
          ("200-399", 200, 399, "#D55E00"), ("400-599", 400, 599, "#E69F00"),
          ("600-699", 600, 699, "#009E73"), ("700-799", 700, 799, "#CC79A7")]
CATS = ["det", "part", "num", "noun", "name", "verb"]
CAT_COLOR = {"det": "#999999", "part": "#bbbbbb", "num": "#777777",
             "noun": "#0072B2", "name": "#D55E00", "verb": "#E69F00"}


def series_of(tok):
    d = ""
    for ch in tok:
        if ch.isdigit():
            d += ch
        else:
            break
    if len(d) < 2:
        return None
    b = int(d[:3]) if len(d) >= 3 else int(d)
    for name, lo, hi, _ in SERIES:
        if lo <= b <= hi:
            return name
    return None


def main():
    df = pd.read_csv("data/raw/lost_language/parallel_rongorongo_real_v4.csv")
    flows = Counter()
    for _, row in df.head(20000).iterrows():
        cats = row["categories"].split()
        glyphs = row["target_glyphs"].split()
        # align pre-reduplication: walk glyphs, skip immediate repeats beyond source len
        gi = 0
        for cat in cats:
            if gi >= len(glyphs):
                break
            s = series_of(glyphs[gi])
            if s:
                flows[(cat, s)] += 1
            gi += 1
    cat_tot = Counter()
    ser_tot = Counter()
    for (c, s), n in flows.items():
        cat_tot[c] += n
        ser_tot[s] += n

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    total = sum(cat_tot.values())
    gap = 0.015
    # left stacks (categories), right stacks (series)
    ly, lpos = 1.0, {}
    for c in CATS:
        h = cat_tot[c] / total * (1 - gap * len(CATS))
        lpos[c] = (ly - h, ly)
        ax.add_patch(mpatches.Rectangle((0.02, ly - h), 0.06, h,
                                        color=CAT_COLOR[c], ec="white"))
        ax.text(0.005, ly - h / 2, c, ha="right", va="center", fontsize=9)
        ly -= h + gap
    ry, rpos = 1.0, {}
    for name, _, _, col in SERIES:
        h = ser_tot[name] / total * (1 - gap * len(SERIES))
        rpos[name] = (ry - h, ry)
        ax.add_patch(mpatches.Rectangle((0.92, ry - h), 0.06, h, color=col, ec="white"))
        ax.text(0.995, ry - h / 2, name, ha="left", va="center", fontsize=9)
        ry -= h + gap

    lcur = {c: lpos[c][1] for c in CATS}
    rcur = {s: rpos[s][1] for s, *_ in [(n, 0) for n, _, _, _ in SERIES]}
    for c in CATS:
        for name, _, _, col in SERIES:
            n = flows.get((c, name), 0)
            if not n:
                continue
            h = n / total * (1 - gap * len(CATS))
            y0a, y0b = lcur[c], lcur[c] - h
            y1a, y1b = rcur[name], rcur[name] - h
            lcur[c] -= h
            rcur[name] -= h
            verts = [(0.08, y0a), (0.5, y0a), (0.92, y1a), (0.92, y1b),
                     (0.5, y0b), (0.08, y0b), (0.08, y0a)]
            codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4,
                     MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
            # matplotlib CURVE4 needs pairs of control points; build two beziers
            path = MplPath(
                [(0.08, y0a), (0.45, y0a), (0.55, y1a), (0.92, y1a),
                 (0.92, y1b), (0.55, y1b), (0.45, y0b), (0.08, y0b), (0.08, y0a)],
                [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                 MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                 MplPath.CLOSEPOLY])
            ax.add_patch(mpatches.PathPatch(path, color=col, alpha=0.35, lw=0))
    ax.set_xlim(-0.08, 1.10)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")
    ax.set_title("WORKING HYPOTHESIS (claim level C2):\n"
                 "grammatical category → Barthel series mapping in the v4 parallel corpus",
                 fontsize=10)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig8_sankey_hypothesis.{ext}", dpi=300, bbox_inches="tight")
    print("saved fig8")


if __name__ == "__main__":
    main()
