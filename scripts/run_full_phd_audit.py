#!/usr/bin/env python3
"""Comprehensive PhD audit: stability, claims, negative controls, and metrics.

Produces figures and tables comparing the OLD pipeline (plain PPMI + SVD)
vs the NEW pipeline (SPPMI + stability diagnostics + claim levels).
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, "src")

from spectral_submersion.tokenization import (
    read_corpus,
    build_vocab,
    tokens_to_ids,
    get_sequences_by_line,
)
from spectral_submersion.cooccurrence import cooccurrence_matrix_from_sequences
from spectral_submersion.pmi import ppmi_matrix
from spectral_submersion.spectral import spectral_embedding, effective_rank
from spectral_submersion.stability import (
    spectral_gap,
    spectral_reliability,
    spectral_stability_bootstrap,
    cooccurrence_coverage,
    expected_pair_count,
    sceptmi_matrix,
    pmi_sensitivity,
    spectral_rejection_rule,
    min_tokens_for_coverage,
)
from spectral_submersion.evaluation import (
    permute_corpus,
    random_corpus_same_frequency,
)
from spectral_submersion.identifiability import (
    verify_non_identifiability,
    anchor_power,
    compute_automorphism_size_upper_bound,
)
from spectral_submersion.claims import (
    ClaimLevel,
    admissible,
    overclaim_risk,
    CLAIM_LABELS,
)
from spectral_submersion.audit_metrics import (
    negative_control_gap,
    bootstrap_stability,
    expected_calibration_error,
    HypothesisLedger,
)
from spectral_submersion.synthetic_experiments import (
    experiment_permutation_recovery,
    experiment_calendar_model,
    find_parallel_passages,
    generate_permuted_corpus,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects
import matplotlib.pyplot as plt

OUT = Path("runs/phd_audit")
OUT.mkdir(parents=True, exist_ok=True)
FIGS = OUT / "figures"
FIGS.mkdir(exist_ok=True)

N_BOOT = 50
N_NEG_CTRL = 30


def seqs_str_to_ids(seqs_str, vocab):
    return [tokens_to_ids(s, vocab) for s in seqs_str]


def filter_valid(seqs_ids):
    return [
        [t for t in s if t is not None and t >= 0]
        for s in seqs_ids
        if len([t for t in s if t is not None and t >= 0]) > 0
    ]


print("=" * 70)
print("PHD UPGRADE AUDIT: OLD vs NEW Pipeline Comparison")
print("=" * 70)

CORPORA = {}

for name, path in [
    ("PCFG_v2", "data/raw/lost_language/corpus_synthetic_v2.csv"),
    ("RR_like", "data/raw/lost_language/corpus_rongorongo_v2.csv"),
    ("RR_real", "data/raw/lost_language/corpus_rongorongo_real.xml.csv"),
    ("Indus", "data/raw/lost_language/corpus_indus_real.csv"),
    ("Positional", "data/raw/lost_language/corpus_positional_synthetic.csv"),
]:
    p = Path(path)
    if not p.exists():
        print(f"  {name}: NOT FOUND ({path})")
        continue
    df = read_corpus(str(p))
    tokens = df["token"].tolist()
    vocab = build_vocab(tokens)
    token_ids = tokens_to_ids(tokens, vocab)
    seqs_str = get_sequences_by_line(df)
    seqs_int = filter_valid(seqs_str_to_ids(seqs_str, vocab))
    CORPORA[name] = {
        "df": df,
        "tokens": tokens,
        "vocab": vocab,
        "token_ids": token_ids,
        "sequences": seqs_int,
        "seqs_str": seqs_str,
        "vocab_size": len(vocab),
        "total_tokens": len(token_ids),
    }
    print(
        f"  {name}: {len(vocab)} types, {len(token_ids)} tokens, {len(seqs_int)} lines"
    )

# ============================================================
# 1. SPECTRAL STABILITY COMPARISON (OLD: PPMI vs NEW: SPPMI)
# ============================================================
print("\n" + "=" * 70)
print("1. SPECTRAL STABILITY: PPMI vs SPPMI across window sizes")
print("=" * 70)

stability_results = {}

for cname, cdata in CORPORA.items():
    vs = cdata["vocab_size"]
    seqs = cdata["sequences"]
    k = min(16, vs - 1)
    for ws in [2, 3, 5]:
        for matrix_type in ["PPMI", "SPPMI_uniform", "SPPMI_marginal"]:
            key = f"{cname}_w{ws}_{matrix_type}"
            try:
                C = cooccurrence_matrix_from_sequences(seqs, vs, window_size=ws)
                if matrix_type == "PPMI":
                    M = ppmi_matrix(C, alpha=0.75)
                elif matrix_type == "SPPMI_uniform":
                    M = sceptmi_matrix(C, epsilon=0.1, prior_type="uniform")
                else:
                    M = sceptmi_matrix(C, epsilon=0.1, prior_type="marginal_product")

                E, sv, Vt = spectral_embedding(M, k=k)
                r_eff = effective_rank(sv)
                cov = cooccurrence_coverage(C)
                epc = expected_pair_count(cdata["total_tokens"], ws, vs)
                min_t = min_tokens_for_coverage(vs, ws)

                stability_results[key] = {
                    "corpus": cname,
                    "window": ws,
                    "matrix": matrix_type,
                    "r_eff": float(r_eff),
                    "sv_top5": sv[:5].tolist() if len(sv) >= 5 else sv.tolist(),
                    "coverage": float(cov),
                    "expected_pair_count": float(epc),
                    "min_tokens_needed": float(min_t),
                }
                print(f"  {key}: r_eff={r_eff:.2f}, cov={cov:.4f}, EPC={epc:.3f}")
            except Exception as e:
                print(f"  FAILED {key}: {e}")
                import traceback

                traceback.print_exc()

# Bootstrap stability (key configs only)
print("\n  Running bootstrap spectral stability...")
for cname in ["PCFG_v2", "RR_real", "Indus"]:
    if cname not in CORPORA:
        continue
    cdata = CORPORA[cname]
    vs = cdata["vocab_size"]
    seqs = cdata["sequences"]
    k = min(16, vs - 1)
    ws = 3
    print(f"    Bootstrapping {cname} (k={k}, w={ws})...")
    try:
        result = spectral_stability_bootstrap(
            seqs,
            vs,
            k=k,
            window_size=ws,
            n_bootstrap=N_BOOT,
            alpha=0.75,
            random_state=42,
        )
        boot_key = f"{cname}_w{ws}_bootstrap"
        stability_results[boot_key] = result
        print(
            f"      delta_k={result['delta_k_mean']:.4f}, eps={result['epsilon_hat']:.4f}, "
            f"rel={result['spectral_reliability']:.4f}, stable={result['reliable']}"
        )
    except Exception as e:
        print(f"      FAILED: {e}")
        import traceback

        traceback.print_exc()

# ============================================================
# 2. NEGATIVE CONTROL GAPS
# ============================================================
print("\n" + "=" * 70)
print("2. NEGATIVE CONTROL GAPS")
print("=" * 70)

neg_ctrl_results = {}

for cname, cdata in CORPORA.items():
    vs = cdata["vocab_size"]
    seqs = cdata["sequences"]
    vocab = cdata["vocab"]
    seqs_str = cdata["seqs_str"]
    k = min(16, vs - 1)
    ws = 3

    try:
        C_real = cooccurrence_matrix_from_sequences(seqs, vs, window_size=ws)
        M_real = ppmi_matrix(C_real, alpha=0.75)
        _, sv_real, _ = spectral_embedding(M_real, k=k)
        score_real = float(np.sum(sv_real[:4]))

        neg_scores = []
        for i in range(N_NEG_CTRL):
            try:
                perm_seqs = permute_corpus(seqs_str)
                perm_ids = filter_valid(seqs_str_to_ids(perm_seqs, vocab))
                if not perm_ids or all(len(s) == 0 for s in perm_ids):
                    continue
                C_perm = cooccurrence_matrix_from_sequences(
                    perm_ids, vs, window_size=ws
                )
                M_perm = ppmi_matrix(C_perm, alpha=0.75)
                _, sv_perm, _ = spectral_embedding(M_perm, k=k)
                neg_scores.append(float(np.sum(sv_perm[:4])))
            except Exception:
                continue

        for i in range(N_NEG_CTRL):
            try:
                rand_seqs = random_corpus_same_frequency(seqs_str)
                rand_ids = filter_valid(seqs_str_to_ids(rand_seqs, vocab))
                if not rand_ids or all(len(s) == 0 for s in rand_ids):
                    continue
                C_rand = cooccurrence_matrix_from_sequences(
                    rand_ids, vs, window_size=ws
                )
                M_rand = ppmi_matrix(C_rand, alpha=0.75)
                _, sv_rand, _ = spectral_embedding(M_rand, k=k)
                neg_scores.append(float(np.sum(sv_rand[:4])))
            except Exception:
                continue

        if len(neg_scores) < 10:
            print(f"  {cname}: Not enough negative controls ({len(neg_scores)})")
            continue

        gap_result = negative_control_gap(score_real, np.array(neg_scores))

        M_real_sppmi = sceptmi_matrix(
            C_real, epsilon=0.1, prior_type="marginal_product"
        )
        _, sv_sppmi, _ = spectral_embedding(M_real_sppmi, k=k)

        neg_ctrl_results[cname] = {
            "window": ws,
            "ppmi_score": score_real,
            "sppmi_score": float(np.sum(sv_sppmi[:4])),
            "neg_ctrl_gap": gap_result["gap"],
            "neg_mean": gap_result["negative_mean"],
            "neg_std": gap_result["negative_std"],
            "interpretation": gap_result["interpretation"],
            "n_neg_controls": len(neg_scores),
            "r_eff_ppmi": float(effective_rank(sv_real)),
            "r_eff_sppmi": float(effective_rank(sv_sppmi)),
            "coverage": float(cooccurrence_coverage(C_real)),
            "expected_pair_count": float(
                expected_pair_count(cdata["total_tokens"], ws, vs)
            ),
            "min_tokens_needed": float(min_tokens_for_coverage(vs, ws)),
        }
        print(
            f"  {cname}: gap={gap_result['gap']:.2f}σ ({gap_result['interpretation']}), "
            f"r_eff(PPMI)={neg_ctrl_results[cname]['r_eff_ppmi']:.2f}, "
            f"r_eff(SPPMI)={neg_ctrl_results[cname]['r_eff_sppmi']:.2f}"
        )
    except Exception as e:
        print(f"  {cname} FAILED: {e}")
        import traceback

        traceback.print_exc()

# ============================================================
# 3. RELIABILITY TABLE (Paper-mandatory)
# ============================================================
print("\n" + "=" * 70)
print("3. SPECTRAL RELIABILITY TABLE")
print("=" * 70)

reliability_table = []

for cname in ["PCFG_v2", "RR_real", "Indus"]:
    boot_key = f"{cname}_w3_bootstrap"
    if boot_key not in stability_results:
        continue
    result = stability_results[boot_key]
    sv_mean = np.array(result["singular_values_mean"])
    sv_std = np.array(result["singular_values_std"])

    for k in [4, 8, min(16, len(sv_mean))]:
        try:
            rejection = spectral_rejection_rule(sv_mean, sv_std, k_values=[k])
            if rejection:
                r = rejection[0]
                reliability_table.append(
                    {
                        "corpus": cname,
                        "window": 3,
                        "k": k,
                        "delta_k": r["delta_k"],
                        "epsilon": r["epsilon"],
                        "reliability": r["reliability"],
                        "stable": r["stable"],
                        "claim_limit": r["claim_limit"],
                    }
                )
                print(
                    f"  {cname} k={k}: delta={r['delta_k']:.4f}, "
                    f"eps={r['epsilon']:.4f}, rel={r['reliability']:.4f}, "
                    f"stable={r['stable']}, claim<{r['claim_limit']}"
                )
        except Exception as e:
            print(f"  {cname} k={k}: FAILED ({e})")

# ============================================================
# 4. IDENTIFIABILITY VERIFICATION
# ============================================================
print("\n" + "=" * 70)
print("4. IDENTIFIABILITY VERIFICATION (Theorem 3.2)")
print("=" * 70)

ident_results = {}

for cname, cdata in CORPORA.items():
    vs = cdata["vocab_size"]
    token_ids = np.array(cdata["token_ids"])

    def make_sv_stat(vocab_size):
        def sv_stat(c):
            C = cooccurrence_matrix_from_sequences(
                [c.tolist()], vocab_size, window_size=2
            )
            return float(np.linalg.svd(C, compute_uv=False)[0])

        return sv_stat

    try:
        result = verify_non_identifiability(
            vs, make_sv_stat(vs), token_ids, n_permutations=30, seed=42
        )
        ident_results[cname] = result
        print(
            f"  {cname}: invariant={result['is_invariant']}, max_dev={result['max_deviation']:.2e}"
        )
    except Exception as e:
        print(f"  {cname}: FAILED ({e})")
        import traceback

        traceback.print_exc()

# ============================================================
# 5. CLAIM LEVEL AUDIT
# ============================================================
print("\n" + "=" * 70)
print("5. CLAIM LEVEL AUDIT")
print("=" * 70)

ledger = HypothesisLedger()

scenarios = [
    ("no_anchors", 0.0, 0.3, 1.5, 0.2),
    ("weak_anchors", 0.15, 0.5, 2.5, 0.4),
    ("moderate_anchors", 0.4, 0.7, 3.5, 0.6),
]

for scenario_name, ap, stab, ncg, sr in scenarios:
    result = ledger.add_hypothesis(
        glyph_or_sequence=["RR_200"],
        candidate_interpretations=[{"target": "moon", "score": 0.6}],
        posterior_score=0.6,
        claim_level="C2_FUNCTIONAL",
        anchor_power=ap,
        bootstrap_stability=stab,
        negative_control_gap=ncg,
        spectral_reliability=sr,
    )
    adm_level = result["claim_level_admissible"]
    print(
        f"  {scenario_name}: admissible={adm_level}, "
        f"blocked={result['blocked']}, OCR={result['overclaim_risk']:.3f}"
    )

align_path = Path("reports/tables/multi_marginal_gw_17candidates.csv")
if align_path.exists():
    import csv

    with open(align_path) as f:
        reader = csv.DictReader(f)
        real_audit = [row for row in reader]
    print(f"\n  Loaded {len(real_audit)} alignment results")

# ============================================================
# 6. SYNTHETIC EXPERIMENTS
# ============================================================
print("\n" + "=" * 70)
print("6. SYNTHETIC EXPERIMENTS")
print("=" * 70)

synth_results = {}

for cname in ["PCFG_v2"]:
    if cname not in CORPORA:
        continue
    cdata = CORPORA[cname]
    vs = cdata["vocab_size"]
    seqs = cdata["sequences"]
    k = min(16, vs - 1)

    print("  Permutation recovery...")
    try:
        perm_seqs, perm_map = generate_permuted_corpus(seqs, vs, seed=42)
        result = experiment_permutation_recovery(
            perm_seqs, seqs, vs, vs, n_anchors=20, window_size=3, k=k, seed=42
        )
        synth_results["permutation_recovery"] = {
            "acc_at_1": result["acc_at_k"].get(1, None),
            "acc_at_5": result["acc_at_k"].get(5, None),
            "mrr": result["mrr"],
        }
        print(
            f"    Acc@1={result['acc_at_k'].get(1, 'N/A')}, "
            f"Acc@5={result['acc_at_k'].get(5, 'N/A')}, MRR={result['mrr']:.3f}"
        )
    except Exception as e:
        print(f"    FAILED: {e}")
        import traceback

        traceback.print_exc()

    print("  Parallel passages...")
    try:
        parallels = find_parallel_passages(seqs, edit_distance_threshold=0.3)
        synth_results["parallel_passages"] = {"n_parallels": len(parallels)}
        print(f"    Found {len(parallels)} parallel passages")
    except Exception as e:
        print(f"    FAILED: {e}")

    print("  Calendar model...")
    try:
        cal = experiment_calendar_model(seqs, vs, n_lunar_phases=30)
        synth_results["calendar_model"] = {
            "ngram_bic": cal["ngram_bic"],
            "calendar_bic": cal["calendar_bic"],
            "delta_bic": cal["delta_bic"],
            "calendar_preferred": cal["calendar_preferred"],
        }
        print(
            f"    n-gram BIC={cal['ngram_bic']:.1f}, calendar BIC={cal['calendar_bic']:.1f}, "
            f"delta={cal['delta_bic']:.1f}, preferred={'calendar' if cal['calendar_preferred'] else 'n-gram'}"
        )
    except Exception as e:
        print(f"    FAILED: {e}")

# ============================================================
# 7. FIGURES
# ============================================================
print("\n" + "=" * 70)
print("7. GENERATING FIGURES")
print("=" * 70)

# Figure 1: Singular value spectra comparison
fig, axes = plt.subplots(1, min(3, len(CORPORA)), figsize=(6 * min(3, len(CORPORA)), 5))
if len(CORPORA) == 1:
    axes = [axes]
corpus_list = list(CORPORA.keys())[:3]
colors = ["#2ecc71", "#e74c3c", "#3498db"]

for idx, cname in enumerate(corpus_list):
    cdata = CORPORA[cname]
    vs = cdata["vocab_size"]
    seqs = cdata["sequences"]
    k = min(32, vs - 1)
    ax = axes[idx] if len(corpus_list) > 1 else axes[0]

    for ws in [2, 3, 5]:
        try:
            C = cooccurrence_matrix_from_sequences(seqs, vs, window_size=ws)
            M_ppmi = ppmi_matrix(C, alpha=0.75)
            _, sv_ppmi, _ = spectral_embedding(M_ppmi, k=k)

            M_sppmi = sceptmi_matrix(C, epsilon=0.1, prior_type="marginal_product")
            _, sv_sppmi, _ = spectral_embedding(M_sppmi, k=k)

            ks = range(1, len(sv_ppmi) + 1)
            ax.plot(
                ks,
                sv_ppmi,
                "-",
                color=colors[idx],
                alpha=0.5 + 0.2 * (ws / 5),
                label=f"PPMI w={ws}",
            )
            ax.plot(
                ks,
                sv_sppmi,
                "--",
                color=colors[idx],
                alpha=0.5 + 0.2 * (ws / 5),
                label=f"SPPMI w={ws}",
            )
        except Exception as e:
            print(f"    Warning: couldn't plot {cname} w={ws}: {e}")

    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value")
    ax.set_title(f'{cname}\n(V={vs}, T={cdata["total_tokens"]})')
    ax.legend(fontsize=6)
    ax.set_yscale("log")

fig.suptitle("Singular Value Spectra: PPMI vs SPPMI", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGS / "fig1_spectral_comparison.png", dpi=150)
print("  Saved fig1_spectral_comparison.png")

# Figure 2: Negative Control Gap bar chart
if neg_ctrl_results:
    fig, ax = plt.subplots(figsize=(max(8, len(neg_ctrl_results) * 2), 6))
    corpora_names = list(neg_ctrl_results.keys())
    gaps = [neg_ctrl_results[c]["neg_ctrl_gap"] for c in corpora_names]
    r_effs = [neg_ctrl_results[c]["r_eff_ppmi"] for c in corpora_names]
    coverages = [neg_ctrl_results[c]["coverage"] for c in corpora_names]

    x = np.arange(len(corpora_names))
    width = 0.25

    ax.bar(x - width, gaps, width, label="NegCtrlGap (sigma)", color="#e74c3c")
    ax.bar(
        x, [c * 10 for c in coverages], width, label="Coverage (x10)", color="#3498db"
    )
    ax.bar(x + width, r_effs, width, label="r_eff (PPMI)", color="#2ecc71")

    ax.set_xlabel("Corpus")
    ax.set_ylabel("Value")
    ax.set_title("Negative Control Gap, Coverage & Effective Rank")
    ax.set_xticks(x)
    ax.set_xticklabels(corpora_names, rotation=15)
    ax.legend()
    ax.axhline(y=2, color="red", linestyle="--", alpha=0.5)
    ax.axhline(y=3, color="darkred", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(FIGS / "fig2_neg_ctrl_gaps.png", dpi=150)
    print("  Saved fig2_neg_ctrl_gaps.png")

# Figure 3: Claim Level Decision Boundaries
fig, ax = plt.subplots(figsize=(10, 7))
anchor_powers = np.linspace(0, 1, 100)
stabilities = np.linspace(0, 1, 100)
AP, ST = np.meshgrid(anchor_powers, stabilities)

claim_grid = np.zeros_like(AP)
for i in range(AP.shape[0]):
    for j in range(AP.shape[1]):
        claim_grid[i, j] = float(
            admissible(
                anchor_power=AP[i, j],
                stability=ST[i, j],
                neg_ctrl_gap=2.0,
                external_evidence=False,
            )
        )

im = ax.contourf(
    AP,
    ST,
    claim_grid,
    levels=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
    colors=["#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff", "#9b59b6"],
)
ax.contour(
    AP, ST, claim_grid, levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors="black", linewidths=0.5
)
cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4])
cbar.set_ticklabels(["C0", "C1", "C2", "C3", "C4"])
ax.set_xlabel("Anchor Power")
ax.set_ylabel("Bootstrap Stability")
ax.set_title(
    "Admissible Claim Level vs Anchor Power & Stability\n(NegCtrlGap = 2.0 sigma)"
)

for name, ap, st in [
    ("No anchors", 0.0, 0.3),
    ("Weak", 0.15, 0.5),
    ("Moderate", 0.4, 0.7),
]:
    ax.plot(ap, st, "w*", markersize=15, markeredgecolor="black")
    ax.annotate(
        name,
        (ap, st),
        textcoords="offset points",
        xytext=(10, 5),
        fontsize=9,
        color="white",
        fontweight="bold",
        path_effects=[
            matplotlib.patheffects.withStroke(linewidth=2, foreground="black")
        ],
    )

fig.tight_layout()
fig.savefig(FIGS / "fig3_claim_levels.png", dpi=150)
print("  Saved fig3_claim_levels.png")

# Figure 4: PMI Sensitivity heatmap
fig, axes = plt.subplots(1, min(2, len(CORPORA)), figsize=(14, 5))
if min(2, len(CORPORA)) == 1:
    axes = [axes]
plotted_corpora = [("PCFG_v2", "PCFG"), ("Indus", "Indus")]

for idx, (cname, title) in enumerate(plotted_corpora[: len(axes)]):
    if cname not in CORPORA:
        continue
    cdata = CORPORA[cname]
    vs = cdata["vocab_size"]
    seqs = cdata["sequences"]
    ax = axes[idx]

    C = cooccurrence_matrix_from_sequences(seqs, vs, window_size=3)
    N = C.sum() + 1e-15
    P = C / N
    p_i = P.sum(axis=1)
    p_j = P.sum(axis=0)

    sens_result = pmi_sensitivity(P, p_i, p_j)
    # Build sensitivity matrix from individual pair sensitivities
    E_PMI = np.log2((P + 1e-15) / (p_i[:, None] * p_j[None, :] + 1e-15) + 1e-15)
    # Sensitivity approximation: higher where PMI is unstable
    sens_matrix = (
        1.0 / (P + 1e-10) + 1.0 / (p_i[:, None] + 1e-10) + 1.0 / (p_j[None, :] + 1e-10)
    )
    sens_matrix = np.abs(E_PMI) * sens_matrix

    n_show = min(30, vs)
    im = ax.imshow(
        sens_matrix[:n_show, :n_show],
        cmap="YlOrRd",
        aspect="auto",
        vmin=0,
        vmax=np.percentile(sens_matrix[:n_show, :n_show], 95),
    )
    ax.set_xlabel("Context token j")
    ax.set_ylabel("Target token i")
    ax.set_title(
        f'{title} (V={vs}, T={cdata["total_tokens"]})\nPMI sensitivity ({n_show}x{n_show})'
    )
    fig.colorbar(im, ax=ax, label="Sensitivity")

fig.suptitle(
    "PMI Sensitivity: Where Co-occurrence Statistics Are Unreliable",
    fontsize=14,
    fontweight="bold",
)
fig.tight_layout()
fig.savefig(FIGS / "fig4_pmi_sensitivity.png", dpi=150)
print("  Saved fig4_pmi_sensitivity.png")

# Figure 5: Overclaim Risk vs Evidence
fig, ax = plt.subplots(figsize=(10, 6))
evidence_levels = np.linspace(0, 5, 200)
for level, color in [
    (ClaimLevel.C1_STRUCTUREAL, "#ffd93d"),
    (ClaimLevel.C2_FUNCTIONAL, "#6bcb77"),
    (ClaimLevel.C3_SEMANTIC_WEAK, "#4d96ff"),
    (ClaimLevel.C4_PHONETIC_PARTIAL, "#9b59b6"),
    (ClaimLevel.C5_TRANSLATION_STRONG, "#e74c3c"),
]:
    risks = [overclaim_risk(level, e) for e in evidence_levels]
    label = f"{CLAIM_LABELS[level]} (C{level.value})"
    ax.plot(evidence_levels, risks, label=label, color=color, linewidth=2)

ax.axhline(
    y=1.0,
    color="red",
    linestyle="--",
    linewidth=1,
    alpha=0.7,
    label="Overclaim threshold",
)
ax.fill_between(evidence_levels, 0, 1, alpha=0.1, color="green", label="Safe zone")
ax.set_xlabel("Evidence Level")
ax.set_ylabel("Overclaim Risk")
ax.set_title("Overclaim Risk by Claim Level vs Evidence Strength")
ax.legend(loc="upper right")
ax.set_ylim(0, 6)

fig.tight_layout()
fig.savefig(FIGS / "fig5_overclaim_risk.png", dpi=150)
print("  Saved fig5_overclaim_risk.png")

# ============================================================
# 8. SAVE RESULTS
# ============================================================
print("\n" + "=" * 70)
print("8. COMPILING AUDIT RESULTS")
print("=" * 70)

all_results = {
    "stability_comparison": {
        k: v for k, v in stability_results.items() if not k.endswith("_bootstrap")
    },
    "bootstrap_stability": {
        k: v for k, v in stability_results.items() if k.endswith("_bootstrap")
    },
    "negative_controls": neg_ctrl_results,
    "reliability_table": reliability_table,
    "identifiability": {
        k: {
            "is_invariant": v["is_invariant"],
            "max_deviation": float(v["max_deviation"]),
        }
        for k, v in ident_results.items()
    },
    "synthetic_experiments": {},
}


def convert(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert(v) for v in obj]
    if isinstance(obj, ClaimLevel):
        return int(obj)
    return obj


all_results = convert(all_results)
for k, v in synth_results.items():
    all_results["synthetic_experiments"][k] = convert(v)

with open(OUT / "phd_audit_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"  Saved phd_audit_results.json")

# ============================================================
# 9. TEXT SUMMARY
# ============================================================
summary_lines = [
    "=" * 70,
    "PHD UPGRADE AUDIT: DETAILED SUMMARY",
    "=" * 70,
    "",
    "1. SPECTRAL STABILITY (PPMI vs SPPMI)",
    "-" * 50,
]

for key, val in stability_results.items():
    if key.endswith("_bootstrap"):
        continue
    summary_lines.append(f"  {key}:")
    summary_lines.append(
        f"    r_eff={val['r_eff']:.2f}, coverage={val['coverage']:.4f}, "
        f"EPC={val['expected_pair_count']:.3f}, min_T={val['min_tokens_needed']:.0f}"
    )

summary_lines.extend(
    [
        "",
        "2. NEGATIVE CONTROL GAPS",
        "-" * 50,
    ]
)
for cname, data in neg_ctrl_results.items():
    summary_lines.append(f"  {cname}:")
    summary_lines.append(
        f"    NegCtrlGap = {data['neg_ctrl_gap']:.2f}sigma ({data['interpretation']})"
    )
    summary_lines.append(
        f"    r_eff(PPMI) = {data['r_eff_ppmi']:.2f}, r_eff(SPPMI) = {data['r_eff_sppmi']:.2f}"
    )
    summary_lines.append(
        f"    Coverage = {data['coverage']:.4f}, EPC = {data['expected_pair_count']:.3f}"
    )

summary_lines.extend(
    [
        "",
        "3. IDENTIFIABILITY",
        "-" * 50,
    ]
)
for cname, data in ident_results.items():
    summary_lines.append(
        f"  {cname}: invariant={data['is_invariant']}, max_dev={data['max_deviation']:.2e}"
    )

summary_lines.extend(
    [
        "",
        "4. BOOTSTRAP STABILITY",
        "-" * 50,
    ]
)
for key, val in stability_results.items():
    if not key.endswith("_bootstrap"):
        continue
    summary_lines.append(
        f"  {key}: delta_k={val['delta_k_mean']:.4f}, eps={val['epsilon_hat']:.4f}, "
        f"reliability={val['spectral_reliability']:.4f}, stable={val['reliable']}"
    )

summary_lines.extend(
    [
        "",
        "5. RELIABILITY TABLE",
        "-" * 50,
        f"  {'Corpus':<12} {'k':>3} {'delta_k':>8} {'epsilon':>8} {'reliability':>12} {'stable':>7} {'claim_limit':>12}",
    ]
)
for r in reliability_table:
    summary_lines.append(
        f"  {r['corpus']:<12} {r['k']:>3} "
        f"{r['delta_k']:>8.4f} {r['epsilon']:>8.4f} {r['reliability']:>12.4f} "
        f"{str(r['stable']):>7} {r['claim_limit']:>12}"
    )

summary_lines.extend(
    [
        "",
        "6. CLAIM LEVEL AUDIT",
        "-" * 50,
        "  Scenario          | Admissible | Blocked | Overclaim Risk",
        "  " + "-" * 60,
    ]
)
for scenario_name, ap, stab, ncg, sr in scenarios:
    ocr = overclaim_risk(ClaimLevel.C2_FUNCTIONAL, ap + stab + ncg / 5 + sr)
    summary_lines.append(
        f"  {scenario_name:<20} | C{int(admissible(anchor_power=ap, stability=stab, neg_ctrl_gap=ncg))} | "
        f"{ocr > 1.0!s:<7} | {ocr:.3f}"
    )

summary_lines.extend(
    [
        "",
        "7. MANDATORY AUDIT CHECKLIST",
        "-" * 50,
        "  1. Mathematical object: Co-occurrence -> PPMI/SPPMI -> SVD embedding",
        "  2. Hypothesis space: Permutation orbits under Sym(V_X)",
        "  3. Non-identifiability verified"
        + (
            " CHECK"
            if all(v["is_invariant"] for v in ident_results.values())
            else " FAIL"
        ),
        "  4. Anchor power metric: See claim level boundaries",
        "  5. Spectral reliability: See bootstrap table",
        "  6. Negative controls: See NegCtrlGap table",
        "  7. Max claim (no anchors): C1/C2 depending on stability",
        "  8. SPPMI sensitivity: epsilon=0.1, prior=marginal",
        "  9. Counterevidence: See overclaim risk figures",
        "  10. Results: runs/phd_audit/phd_audit_results.json",
        "",
        "=" * 70,
        "AUDIT COMPLETE",
        "=" * 70,
    ]
)

with open(OUT / "phd_audit_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))

print(f"\n  Saved phd_audit_summary.txt")
print(f"  Saved phd_audit_results.json")
print(f"  Saved 5 figures in {FIGS}/")
print("\n" + "=" * 70)
print("AUDIT COMPLETE. See runs/phd_audit/ for all results.")
print("=" * 70)
