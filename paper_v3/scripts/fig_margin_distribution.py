"""T4 <-> experiment cross-validation: empirical margin distribution of the
synthetic anchor benchmark with the theoretical correctness threshold of
Prop. A.margin overlaid. Predicted Acc@1 = fraction of test pairs whose
margin exceeds the threshold; compared against observed NN Acc@1.
Seed 42 (same split as validate_anchors.py).
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path("paper_v3/figures")
SEED = 42


def procrustes(X, Y):
    U, _, Vt = np.linalg.svd(X.T @ Y)
    return U @ Vt


def main():
    E_lost = np.load("data/processed/embeddings_synthetic_v2.npy")
    E_cand = np.load("data/processed/embeddings_synthetic_candidate.npy")
    lost_vocab = json.loads(Path("data/processed/embeddings_synthetic_v2.vocab.json").read_text())
    cand_vocab = json.loads(Path("data/processed/embeddings_synthetic_candidate.vocab.json").read_text())
    anchors = json.loads(Path("data/raw/candidate_languages/synthetic_anchors.json").read_text())

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(anchors))
    n_train = max(1, int(len(anchors) * 0.20))
    train = [anchors[i] for i in perm[:n_train]]
    test = [anchors[i] for i in perm[n_train:]]

    Xtr = np.array([E_lost[lost_vocab[a["lost_token"]]] for a in train])
    Ytr = np.array([E_cand[cand_vocab[a["candidate_token"]]] for a in train])
    Omega = procrustes(Xtr, Ytr)

    # Theoretical rotation error bound (Thm T4): 2||B^T E||_F / sigma_r(B)^2
    E_res = Xtr @ Omega - Ytr
    sigma_r = np.linalg.svd(Xtr, compute_uv=False)[-1]
    rot_err_bound = 2 * np.linalg.norm(Xtr.T @ E_res) / max(sigma_r**2, 1e-12)

    margins, thresholds, correct = [], [], []
    for a in test:
        x = E_lost[lost_vocab[a["lost_token"]]]
        y_true_idx = cand_vocab[a["candidate_token"]]
        proj = x @ Omega
        dists = np.linalg.norm(E_cand - proj, axis=1)
        order = np.argsort(dists)
        correct.append(order[0] == y_true_idx)
        d_true = np.linalg.norm(E_cand[y_true_idx] - proj)
        d_second = dists[order[1]] if order[0] == y_true_idx else dists[order[0]]
        # margin as defined in Prop: second-best minus true, from the true target's view
        y = E_cand[y_true_idx]
        d_all = np.linalg.norm(E_cand - y, axis=1)
        d_all[y_true_idx] = np.inf
        gamma = d_all.min()  # distance from true target to nearest competitor
        e_norm = np.linalg.norm(y - x @ Omega)
        margins.append(gamma)
        thresholds.append(2 * (np.linalg.norm(x) * rot_err_bound + e_norm))

    margins = np.array(margins)
    thresholds = np.array(thresholds)
    correct = np.array(correct)
    acc_obs = correct.mean()
    acc_pred = (margins > thresholds).mean()

    # margin as empirical predictor: AUC of margin for correctness
    order = np.argsort(margins)
    ranks = np.empty(len(margins))
    ranks[order] = np.arange(1, len(margins) + 1)
    n_pos, n_neg = correct.sum(), (~correct).sum()
    auc = (ranks[correct].sum() - n_pos * (n_pos + 1) / 2) / max(n_pos * n_neg, 1)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    bins = np.linspace(margins.min(), margins.max(), 24)
    ax.hist(margins[correct], bins=bins, color="#009E73", alpha=0.65,
            label=f"NN-correct pairs (n={n_pos})")
    ax.hist(margins[~correct], bins=bins, color="#D55E00", alpha=0.65,
            label=f"NN-incorrect pairs (n={n_neg})")
    ax.text(0.02, 0.95,
            f"observed Acc@1 = {acc_obs:.3f}\n"
            f"margin-as-predictor AUC = {auc:.3f}\n"
            f"T4 sufficient threshold: vacuous here\n"
            f"(guarantees {100*acc_pred:.0f}% — bound loose by design,\n"
            f"sufficient not necessary; reported as such)",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(fc="white", ec="#cccccc"))
    ax.set_xlabel("empirical margin γ (distance to nearest competitor)")
    ax.set_ylabel("test pairs")
    ax.legend(fontsize=8)
    ax.set_title("Margins vs. NN-matching correctness (synthetic anchor benchmark)",
                 fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_margin_distribution.{ext}", dpi=300, bbox_inches="tight")
    print(f"observed Acc@1={acc_obs:.3f}  predicted-by-margin={acc_pred:.3f}  "
          f"rot_err_bound={rot_err_bound:.3f}")


if __name__ == "__main__":
    main()
