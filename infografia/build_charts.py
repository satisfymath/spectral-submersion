"""Charts for the vertical infographic. Every number comes from
reports/v3_evaluation_suite.json, reports/lambda_sweep.json, the real corpus
and the paper's own figure conventions (paper_v3/scripts/fig3, fig6).
"""
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from translate_to_rongorongo_v6 import RealGlyphLM  # noqa: E402
from evaluate_rongorongo_v6 import real_stats  # noqa: E402
from spectral_submersion.tokenization import get_sequences_by_line  # noqa: E402
from spectral_submersion.cooccurrence import cooccurrence_matrix_from_sequences  # noqa: E402
from spectral_submersion.pmi import ppmi_matrix  # noqa: E402

OUT = Path("infografia/charts"); OUT.mkdir(parents=True, exist_ok=True)
CORPUS = "data/raw/lost_language/corpus_rongorongo_real.xml.csv"
PROPOSAL = "002 002 004 004 280 280 450 063 063 004 004 002 002 004 004 430 022"

# validated palette (dataviz validator, light mode): burdeos, cobre, azul, oliva
BURDEOS, COBRE, AZUL, OLIVA = "#8a2f2f", "#c2702a", "#2a6f9e", "#8a8a2a"
INK, INK2, MUTED, GRID = "#2b2118", "#5a4a3a", "#8c7b68", "#e3d9c6"
SURFACE = "none"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 12.5,
    "savefig.transparent": True,
})


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("left", "bottom"): ax.spines[s].set_linewidth(0.8)


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", transparent=True)
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", transparent=True)
    plt.close(fig); print("saved", name)


suite = json.loads(Path("reports/v3_evaluation_suite.json").read_text())
sweep = json.loads(Path("reports/lambda_sweep.json").read_text())
lm_full = RealGlyphLM(CORPUS)
df = pd.read_csv(CORPUS)
tmp = Path("reports/_lm_train_split_info.csv")
df[df.doc_id.isin({"A", "B", "C", "E"})].to_csv(tmp, index=False)
lm_train = RealGlyphLM(str(tmp)); tmp.unlink()
lines, _, _ = real_stats(CORPUS)
real_bits = [lm_train.bits_per_glyph(seq) for doc, seq in lines if doc in ("D", "F") and seq]
REAL_MEAN = float(np.mean(real_bits))
print(f"real held-out (D,F): n={len(real_bits)} mean={REAL_MEAN:.3f} "
      f"p5={np.percentile(real_bits,5):.2f} p95={np.percentile(real_bits,95):.2f}")
prop_bits = lm_full.bits_per_glyph(PROPOSAL.split())
print(f"proposal line bits (full LM) = {prop_bits:.3f}")

# ---------- Chart 1: systems vs baselines/ablations (b/g with CI, D2, RM) ----------
S = suite["systems"]
rows = [  # label, key, group
    ("v6 beam + fusión LM (sistema)", "v6_beam_fusion", "sys"),
    ("v6 greedy (sin beam)", "v6_greedy", "abl"),
    ("ablación λ = 0 (sin fusión)", "abl_no_fusion", "abl"),
    ("ablación sin penalización rep.", "abl_no_reppenalty", "abl"),
    ("ablación beam = 1", "abl_beam1", "abl"),
    ("baseline LSTM", "baseline_lstm", "base"),
    ("baseline conteos léxico+bigrama", "baseline_bigram", "base"),
    ("baseline plantilla (sin aprendizaje)", "baseline_template", "base"),
]
keys = set(S)
rows = [r for r in rows if r[1] in keys]
print("systems in suite:", sorted(keys))
fig, ax = plt.subplots(figsize=(7.6, 4.6))
style(ax); ax.grid(axis="x", color=GRID, lw=0.8); ax.grid(axis="y", visible=False)
ys = np.arange(len(rows))[::-1]
col = {"sys": BURDEOS, "abl": COBRE, "base": OLIVA}
ax.axvspan(np.percentile(real_bits, 25), np.percentile(real_bits, 75), color=AZUL, alpha=0.12, lw=0)
ax.axvline(REAL_MEAN, color=AZUL, lw=1.4, ls=(0, (4, 3)))
ax.text(REAL_MEAN + 0.05, len(rows) - 0.55, f"real held-out\nmedia {REAL_MEAN:.2f}\n(banda = cuartiles)", color=AZUL, fontsize=9, va="top")
for y, (label, key, grp) in zip(ys, rows):
    p, ci = S[key]["point"], S[key]["ci95"]
    ax.plot(ci["bg"], [y, y], color=col[grp], lw=2, solid_capstyle="round")
    ax.plot(p["bg"], y, "o", ms=9 if grp == "sys" else 7, color=col[grp], mec="white", mew=1.5)
    ax.text(9.55, y, f"{p['D2']:.2f}", va="center", ha="right", fontsize=9.5, color=INK2)
    ax.text(10.25, y, f"{p['RM']:.0f}", va="center", ha="right", fontsize=9.5,
            color=BURDEOS if p["RM"] > 2 else INK2, fontweight="bold" if p["RM"] > 2 else "normal")
ax.text(9.55, len(rows) - 0.2, "distinct-2", ha="right", fontsize=9, color=MUTED)
ax.text(10.25, len(rows) - 0.2, "rep. máx.", ha="right", fontsize=9, color=MUTED)
ax.set_yticks(ys, [r[0] for r in rows], fontsize=10)
ax.set_xlim(5.4, 10.4); ax.set_ylim(-0.7, len(rows) - 0.1)
ax.set_xlabel("bits/glifo bajo el LM trigrama de las tablillas reales (IC 95 % bootstrap, B = 1000)")
ax.set_title("Seis métricas, 20 glosas de test: solo el sistema completo es a la vez\nno degenerado (rep. máx. 2), diverso (distinct-2 0.72) y dentro de la banda real")
for s in ("left",): ax.spines[s].set_visible(False)
save(fig, "c1_systems")

# ---------- Chart 2: lambda sweep ----------
lam = np.array([r["lambda"] for r in sweep]); bits = np.array([r["bits"] for r in sweep])
lo = np.array([r["bits_ci"][0] for r in sweep]); hi = np.array([r["bits_ci"][1] for r in sweep])
fig, ax = plt.subplots(figsize=(7.6, 3.9)); style(ax)
ax.axvspan(0.85, 1.0, color=COBRE, alpha=0.14, lw=0)
ax.text(0.925, 6.55, "colapso\nmodal", ha="center", va="top", fontsize=9.5, color=COBRE)
ax.fill_between(lam, lo, hi, color=BURDEOS, alpha=0.15, lw=0)
ax.plot(lam, bits, color=BURDEOS, lw=2.2)
ax.axhline(REAL_MEAN, color=AZUL, lw=1.4, ls=(0, (4, 3)))
ax.text(0.01, REAL_MEAN + 0.06, f"referencia real held-out {REAL_MEAN:.2f}", color=AZUL, fontsize=9.5)
i35 = int(np.argmin(np.abs(lam - 0.35)))
ax.plot(0.35, bits[i35], "o", ms=11, color=BURDEOS, mec="white", mew=2)
ax.annotate(f"punto de operación λ = 0.35\n{bits[i35]:.2f} bits/glifo", (0.35, bits[i35]),
            xytext=(0.43, bits[i35] + 0.55), fontsize=10, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set_xlabel("peso de fusión λ del LM real"); ax.set_ylabel("bits/glifo decodificados")
ax.set_xlim(0, 1); ax.set_title(f"Barrido de λ: monótono {bits[0]:.2f} → {bits[-1]:.2f} (P6). Un LM más pesado no vuelve\nla salida más «auténtica»: la vuelve más repetitiva")
save(fig, "c2_lambda")

# ---------- Chart 3: per-line bits, real vs systems, with the proposal line ----------
def sys_bits(name):
    return [lm_full.bits_per_glyph(o.split()) for o in S[name]["outputs"] if o.split()]
data = [("real held-out\n(D, F)", real_bits, AZUL), ("v6 beam+fusión", sys_bits("v6_beam_fusion"), BURDEOS),
        ("v6 greedy", sys_bits("v6_greedy"), COBRE), ("baseline\nbigrama", sys_bits("baseline_bigram"), OLIVA),
        ("baseline\nplantilla", sys_bits("baseline_template"), OLIVA)]
fig, ax = plt.subplots(figsize=(7.6, 4.3)); style(ax)
rng = np.random.default_rng(42)
for i, (name, vals, c) in enumerate(data, 1):
    parts = ax.violinplot([vals], positions=[i], showextrema=False, widths=0.75)
    for b in parts["bodies"]: b.set_facecolor(c); b.set_alpha(0.22); b.set_edgecolor("none")
    ax.scatter(rng.normal(i, 0.05, len(vals)), vals, s=14, color=c, alpha=0.85, zorder=3, edgecolors="white", linewidths=0.5)
    ax.plot([i - 0.22, i + 0.22], [np.median(vals)] * 2, color=INK, lw=1.6, zorder=4)
ax.axhline(REAL_MEAN, color=AZUL, lw=1.2, ls=(0, (4, 3)))
ax.plot(2, prop_bits, marker="*", ms=20, color=BURDEOS, mec="white", mew=1.5, zorder=6)
ax.annotate(f"línea de la petición\n{prop_bits:.2f} bits/glifo", (2, prop_bits), xytext=(2.55, prop_bits - 1.05),
            fontsize=10, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set_xticks(range(1, 6), [d[0] for d in data], fontsize=10)
ax.set_ylabel("bits/glifo por línea"); ax.set_ylim(3.6, 10.6)
ax.set_title("Distribución por línea: líneas reales (LM entrenado solo en A, B, C, E) frente a salidas\ngeneradas (LM completo). Un valor bajo no es «mejor»: es búsqueda de moda (P6)")
save(fig, "c3_lines")

import os
if os.environ.get("SKIP_C4") == "1":
    raise SystemExit(0)
# ---------- Chart 4: negative result — PPMI spectra real vs controls (F3 recomputed) ----------
K, B = 40, 50
def spectrum(seqs, vocab):
    idx = {t: i for i, t in enumerate(vocab)}
    ids = [[idx[t] for t in s if t in idx] for s in seqs]
    M = ppmi_matrix(cooccurrence_matrix_from_sequences(ids, len(vocab), window_size=2))
    return np.linalg.svd(M, compute_uv=False)[:K]
rng = np.random.default_rng(42)
seqs = [[t for t in s if t != "_"] for s in get_sequences_by_line(df)]
freqs = Counter(t for s in seqs for t in s); vocab = sorted(freqs)
probs = np.array([freqs[t] for t in vocab], float); probs /= probs.sum()
curves = {"real": [], "permutado": [], "aleatorio (frec. igualada)": [], "aleatorio (uniforme)": []}
cache = OUT / "c4_curves.json"
if cache.exists():
    curves = {k: [np.array(v) for v in vs] for k, vs in json.load(open(cache)).items()}
for _ in range(0 if cache.exists() else B):
    bs = [seqs[i] for i in rng.integers(0, len(seqs), len(seqs))]
    curves["real"].append(spectrum(bs, vocab))
    flat = [t for s in bs for t in s]; rng.shuffle(flat); it = iter(flat)
    curves["permutado"].append(spectrum([[next(it) for _ in s] for s in bs], vocab))
    curves["aleatorio (frec. igualada)"].append(spectrum([[vocab[i] for i in rng.choice(len(vocab), len(s), p=probs)] for s in bs], vocab))
    curves["aleatorio (uniforme)"].append(spectrum([[vocab[i] for i in rng.integers(0, len(vocab), len(s))] for s in bs], vocab))
if not cache.exists():
    json.dump({k: [list(map(float, v)) for v in vs] for k, vs in curves.items()}, open(cache, "w"))
cols = {"real": AZUL, "permutado": BURDEOS, "aleatorio (frec. igualada)": COBRE, "aleatorio (uniforme)": OLIVA}
fig, ax = plt.subplots(figsize=(7.6, 3.9)); style(ax)
x = np.arange(1, K + 1)
for name, c in cols.items():
    arr = np.stack(curves[name]); med = np.median(arr, 0); l, h = np.percentile(arr, [2.5, 97.5], axis=0)
    ax.plot(x, med, color=c, lw=2, label=name); ax.fill_between(x, l, h, color=c, alpha=0.15, lw=0)
ax.legend(frameon=False, fontsize=9.5, loc="upper right", labelcolor=INK2)
ax.set_yscale("log"); ax.set_xlabel("índice del valor singular"); ax.set_ylabel("valor singular (log)")
ax.set_xlim(1, K)
ax.set_title(f"Resultado negativo: el espectro PPMI real NO se separa de los controles permutado\ny de frecuencia igualada (bandas 95 %, B = {B}). Solo el uniforme se separa, trivialmente")
save(fig, "c4_spectrum")
json.dump({"real_heldout_mean": REAL_MEAN, "real_heldout_n": len(real_bits), "proposal_bits_fullLM": prop_bits,
           "lambda_035_bits": float(bits[i35]), "sweep_first_last": [float(bits[0]), float(bits[-1])]},
          open(OUT / "numbers.json", "w"), indent=1)
