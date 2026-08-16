"""F5: lambda-sweep Pareto (realism vs fidelity) with bootstrap CIs.
Empirical verification of Prop. P6(i): bits/glyph is monotone in lambda.
Operating point lambda=0.35 marked; mode-collapse zone (lambda>=0.85,
bits < 6, bigram-baseline-like) shaded. Seed 42.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path("paper_v3/figures")


def main():
    data = json.loads(Path("reports/lambda_sweep.json").read_text())
    lam = [d["lambda"] for d in data]
    bits = [d["bits"] for d in data]
    bits_lo = [d["bits_ci"][0] for d in data]
    bits_hi = [d["bits_ci"][1] for d in data]
    f1 = [d["f1"] for d in data]
    f1_lo = [d["f1_ci"][0] for d in data]
    f1_hi = [d["f1_ci"][1] for d in data]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))

    ax = axes[0]
    ax.plot(lam, bits, color="#0072B2", lw=1.8, marker="o", ms=3, label="bits/glyph (realism)")
    ax.fill_between(lam, bits_lo, bits_hi, color="#0072B2", alpha=0.18, lw=0)
    ax.axhline(8.257, color="#000000", lw=1.0, ls=(0, (4, 3)))
    ax.text(0.68, 8.32, "real held-out mean 8.26", fontsize=7.5)
    ax.axvspan(0.85, 1.0, color="#D55E00", alpha=0.10)
    ax.text(0.925, 7.7, "mode-\ncollapse\nzone", fontsize=7, ha="center", color="#D55E00")
    ax.axvline(0.35, color="#009E73", lw=1.4)
    ax.text(0.36, 8.0, "operating point\nλ=0.35", fontsize=7.5, color="#009E73")
    ax.set_xlabel("fusion weight λ")
    ax.set_ylabel("bits/glyph under real LM")
    ax.grid(alpha=0.25)
    ax.set_title("Monotone in λ, as predicted by Prop. P6(i)", fontsize=9)

    ax = axes[1]
    ax.plot(bits, f1, color="#555555", lw=1.0, alpha=0.6, zorder=1)
    sc = ax.scatter(bits, f1, c=lam, cmap="viridis", s=34, zorder=2)
    for d in data:
        if d["lambda"] in (0.0, 0.35, 1.0):
            ax.annotate(f"λ={d['lambda']:.2f}", (d["bits"], d["f1"]),
                        textcoords="offset points", xytext=(6, 5), fontsize=7.5)
    ax.errorbar(bits, f1,
                xerr=[np.array(bits) - np.array(bits_lo), np.array(bits_hi) - np.array(bits)],
                yerr=[np.array(f1) - np.array(f1_lo), np.array(f1_hi) - np.array(f1)],
                fmt="none", ecolor="#aaaaaa", elinewidth=0.7, zorder=1)
    fig.colorbar(sc, ax=ax, label="λ", shrink=0.85)
    ax.set_xlabel("bits/glyph under real LM (← more LM-mode-seeking)")
    ax.set_ylabel("token F1 vs corpus-generator reference")
    ax.grid(alpha=0.25)
    ax.set_title("Fidelity–realism trade-off (95% bootstrap CIs)", fontsize=9)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig5_lambda_pareto.{ext}", dpi=300, bbox_inches="tight")
    print("saved fig5")


if __name__ == "__main__":
    main()
