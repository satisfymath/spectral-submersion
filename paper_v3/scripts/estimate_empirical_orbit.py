"""T2 companion: estimate the size N(n) of the epsilon(n)-empirical orbit
for the real RR corpus.

Lower bound (exact): relabelings that permute signs within identical-
frequency classes preserve the unigram profile exactly, so
log N >= sum_c log(m_c!) over frequency classes of size m_c.
Refinement (Monte Carlo): among frequency-preserving permutations, the
fraction whose bigram-count deviation stays within twice the sampling
radius eps(n) (Thm T3, estimated by line bootstrap) is estimated by
sampling. Seed 42.
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from spectral_submersion.tokenization import get_sequences_by_line

SEED = 42
N_BOOT = 30
N_MC = 200


def bigram_counts(seqs):
    c = Counter()
    for s in seqs:
        for a, b in zip(s, s[1:]):
            c[(a, b)] += 1
    return c


def l2_dev(c1, c2):
    keys = set(c1) | set(c2)
    return math.sqrt(sum((c1.get(k, 0) - c2.get(k, 0)) ** 2 for k in keys))


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv("data/raw/lost_language/corpus_rongorongo_real.xml.csv")
    seqs = [[t for t in s if t != "_"] for s in get_sequences_by_line(df)]
    freqs = Counter(t for s in seqs for t in s)

    # exact lower bound from frequency classes
    classes = Counter(freqs.values())
    logN_freq = sum(math.lgamma(m + 1) for f, m in classes.items())
    n_classes_nontrivial = sum(1 for f, m in classes.items() if m > 1)

    # sampling radius: bootstrap deviation of bigram counts under line resampling
    base_bi = bigram_counts(seqs)
    devs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(seqs), len(seqs))
        devs.append(l2_dev(bigram_counts([seqs[i] for i in idx]), base_bi))
    eps = float(np.median(devs))

    # MC refinement: fraction of frequency-preserving relabelings within 2*eps
    by_freq = {}
    for t, f in freqs.items():
        by_freq.setdefault(f, []).append(t)
    accept = 0
    for _ in range(N_MC):
        mapping = {}
        for f, toks in by_freq.items():
            perm = rng.permutation(len(toks))
            for i, t in enumerate(toks):
                mapping[t] = toks[perm[i]]
        relab = [[mapping[t] for t in s] for s in seqs]
        if l2_dev(bigram_counts(relab), base_bi) <= 2 * eps:
            accept += 1
    frac = accept / N_MC
    logN_emp = logN_freq + (math.log(frac) if frac > 0 else -float("inf"))

    fano_floor = 1 - math.log(2) / logN_freq if logN_freq > math.log(2) else 0.0
    out = {
        "seed": SEED,
        "log_N_freq_preserving": logN_freq,
        "log10_N_freq_preserving": logN_freq / math.log(10),
        "nontrivial_freq_classes": n_classes_nontrivial,
        "eps_bigram_bootstrap_median": eps,
        "mc_fraction_within_2eps": frac,
        "log_N_empirical_orbit_estimate": logN_emp,
        "fano_error_floor_freq_orbit": fano_floor,
        "note": ("log N is a LOWER bound restricted to exact unigram-preserving "
                 "relabelings; the empirical orbit of Def. A.8 (population "
                 "statistics within sampling error) is at least this large."),
    }
    Path("reports/empirical_orbit_estimate.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
