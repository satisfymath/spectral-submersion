"""F6: KDE/violin of bits-per-glyph: real held-out lines vs systems.
Real held-out uses the LM trained on {A,B,C,E} (no memorization); systems
use the full-corpus LM, per the evaluation convention (stated in caption).
Seed 42. Anti-conclusion: values below the real mean reflect the selection
bias formalized in Prop. P6, not superior authenticity.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from translate_to_rongorongo_v6 import RealGlyphLM  # noqa: E402
from evaluate_rongorongo_v6 import real_stats  # noqa: E402

SEED = 42
OUT = Path("paper_v3/figures")
COLORS = {"real held-out (D,F)": "#000000", "v6 beam+fusion": "#009E73",
          "v6 greedy": "#E69F00", "baseline bigram": "#D55E00",
          "baseline template": "#56B4E9"}


def main():
    suite = json.loads(Path("reports/v3_evaluation_suite.json").read_text())
    lm_full = RealGlyphLM("data/raw/lost_language/corpus_rongorongo_real.xml.csv")

    df = pd.read_csv("data/raw/lost_language/corpus_rongorongo_real.xml.csv")
    tmp = Path("reports/_lm_train_split_f6.csv")
    df[df.doc_id.isin({"A", "B", "C", "E"})].to_csv(tmp, index=False)
    lm_train = RealGlyphLM(str(tmp))
    tmp.unlink()
    lines, _, _ = real_stats("data/raw/lost_language/corpus_rongorongo_real.xml.csv")
    real_bits = [lm_train.bits_per_glyph(seq) for doc, seq in lines
                 if doc in ("D", "F") and seq]

    def sys_bits(name):
        outs = suite["systems"][name]["outputs"]
        return [lm_full.bits_per_glyph(o.split()) for o in outs if o.split()]

    data = {
        "real held-out (D,F)": real_bits,
        "v6 beam+fusion": sys_bits("v6_beam_fusion"),
        "v6 greedy": sys_bits("v6_greedy"),
        "baseline bigram": sys_bits("baseline_bigram"),
        "baseline template": sys_bits("baseline_template"),
    }

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    names = list(data)
    parts = ax.violinplot([data[n] for n in names], showmedians=True, widths=0.8)
    for body, n in zip(parts["bodies"], names):
        body.set_facecolor(COLORS[n])
        body.set_alpha(0.45)
    for k in ("cmedians", "cbars", "cmins", "cmaxes"):
        parts[k].set_color("#444444")
        parts[k].set_linewidth(1.0)
    rng = np.random.default_rng(SEED)
    for i, n in enumerate(names, 1):
        x = rng.normal(i, 0.045, len(data[n]))
        ax.scatter(x, data[n], s=9, color=COLORS[n], alpha=0.8, zorder=3)
    ax.axhline(8.257, color="#000000", lw=1.1, ls=(0, (4, 3)))
    ax.text(len(names) + 0.35, 8.257, "real held-out\nmean 8.26", fontsize=7.5,
            va="center")
    ax.set_xticks(range(1, len(names) + 1),
                  [n.replace(" ", "\n", 1) for n in names], fontsize=8)
    ax.set_ylabel("bits/glyph under real trigram LM")
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Per-line bits/glyph: real held-out vs generated outputs", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig6_violin_bits.{ext}", dpi=300, bbox_inches="tight")
    print("saved fig6")


if __name__ == "__main__":
    main()
