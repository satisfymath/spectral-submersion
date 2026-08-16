"""Task 2.6: formal permutation tests for positional bias, with
Benjamini-Hochberg correction.

For each sign with frequency >= min_freq in the Indus real corpus:
  H0: the sign's first-position (resp. last-position) ratio is what one
      would observe if tokens were exchangeable within inscriptions.
  Test: permute tokens within each inscription (preserving lengths and
      the multiset of signs per inscription), N=10000 permutations,
      one-sided p-value for the observed ratio.
BH-corrected q-values at FDR 0.05. Seed 42.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_PERM = 10000
MIN_FREQ = 15
CORPUS = "data/raw/lost_language/corpus_indus_real.csv"


def bh(pvals):
    p = np.asarray(pvals)
    order = np.argsort(p)
    q = np.empty_like(p)
    m = len(p)
    prev = 1.0
    for rank_i, idx in enumerate(reversed(order), 1):
        i = m - rank_i  # 0-based rank from top
        val = p[idx] * m / (i + 1)
        prev = min(prev, val)
        q[idx] = prev
    return q


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(CORPUS)
    # column autodetect
    tok_col = "token" if "token" in df.columns else df.columns[-1]
    doc_col = "doc_id" if "doc_id" in df.columns else df.columns[0]
    pos_col = "position" if "position" in df.columns else None

    seqs = []
    for _, g in df.groupby(doc_col, sort=False):
        g = g.sort_values(pos_col) if pos_col else g
        seqs.append([str(t) for t in g[tok_col]])

    from collections import Counter
    freqs = Counter(t for s in seqs for t in s)
    signs = [t for t, c in freqs.items() if c >= MIN_FREQ]

    def ratios(seqs):
        first, last, tot = Counter(), Counter(), Counter()
        for s in seqs:
            if not s:
                continue
            first[s[0]] += 1
            last[s[-1]] += 1
            tot.update(s)
        return first, last, tot

    f_obs, l_obs, tot = ratios(seqs)

    # Permutation null: shuffle tokens within each inscription
    null_first = {t: 0 for t in signs}
    null_last = {t: 0 for t in signs}
    for _ in range(N_PERM):
        perm = [list(rng.permutation(s)) for s in seqs]
        f_p, l_p, _ = ratios(perm)
        for t in signs:
            if f_p.get(t, 0) >= f_obs.get(t, 0):
                null_first[t] += 1
            if l_p.get(t, 0) >= l_obs.get(t, 0):
                null_last[t] += 1

    rows = []
    for t in signs:
        p_first = (null_first[t] + 1) / (N_PERM + 1)
        p_last = (null_last[t] + 1) / (N_PERM + 1)
        rows.append({"sign": t, "freq": freqs[t],
                     "first_ratio": round(f_obs.get(t, 0) / max(1, sum(1 for s in seqs if t in s)), 3),
                     "p_first": p_first, "p_last": p_last})

    q_first = bh([r["p_first"] for r in rows])
    q_last = bh([r["p_last"] for r in rows])
    for r, qf, ql in zip(rows, q_first, q_last):
        r["q_first"] = float(qf)
        r["q_last"] = float(ql)
        r["sig_first_fdr05"] = bool(qf <= 0.05)
        r["sig_last_fdr05"] = bool(ql <= 0.05)

    rows.sort(key=lambda r: r["q_first"])
    out = Path("reports/tables/positional_permutation_tests.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seed": SEED, "n_perm": N_PERM, "min_freq": MIN_FREQ,
                               "n_signs_tested": len(rows), "results": rows}, indent=2))
    sig_f = [r["sign"] for r in rows if r["sig_first_fdr05"]]
    sig_l = [r["sign"] for r in rows if r["sig_last_fdr05"]]
    print(f"Tested {len(rows)} signs. FDR<=0.05: first-position {sig_f}; last-position {sig_l}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
