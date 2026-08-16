"""F7: radar of the 6 metrics for 4 systems, all normalized to [0,1]
where 1 = best. Normalizations stated in caption:
 VR, BA, D2 as-is; JS -> 1-JS; RM -> 1 at RM<=2 (real reduplication),
 linear down to 0 at RM>=50; b/g -> 1 - |b/g - 8.26|/4 clipped to [0,1]
 (closeness to the real held-out reference, per Prop. P6 the raw minimum
 is not the target). Seed 42.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path("paper_v3/figures")
SYSTEMS = [("baseline_bigram", "baseline bigram", "#D55E00"),
           ("v5_greedy", "v5 greedy", "#56B4E9"),
           ("v6_greedy", "v6 greedy", "#E69F00"),
           ("v6_beam_fusion", "v6 beam+fusion", "#009E73")]
METRICS = ["VR", "BA", "1-JS", "D2", "rep. sanity", "b/g proximity"]


def scores(p):
    rm = p["RM"]
    rep = 1.0 if rm <= 2 else max(0.0, 1 - (rm - 2) / 48)
    bg = p["bg"]
    bgs = 0.0 if bg is None or np.isnan(bg) else max(0.0, 1 - abs(bg - 8.257) / 4)
    return [p["VR"], p["BA"], 1 - p["JS"], p["D2"], rep, bgs]


def main():
    suite = json.loads(Path("reports/v3_evaluation_suite.json").read_text())
    N = len(METRICS)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    ang += ang[:1]

    fig, ax = plt.subplots(figsize=(5.6, 5.2), subplot_kw={"polar": True})
    for key, label, color in SYSTEMS:
        p = suite["systems"][key]["point"]
        v = scores(p)
        v += v[:1]
        ax.plot(ang, v, color=color, lw=1.8, label=label)
        ax.fill(ang, v, color=color, alpha=0.10)
    ax.set_xticks(ang[:-1], METRICS, fontsize=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.5", "0.75", "1"], fontsize=7)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", bbox_to_anchor=(1.18, -0.08), fontsize=8)
    ax.set_title("Six-metric profile (1 = best; normalizations in caption)",
                 fontsize=10, pad=18)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig7_radar.{ext}", dpi=300, bbox_inches="tight")
    print("saved fig7")


if __name__ == "__main__":
    main()
