"""Lambda sweep for fusion decoding (Task 2.3): fidelity vs realism Pareto.

For lambda in {0, 0.05, ..., 1.0}:
  - realism  : mean bits/glyph of decoded outputs under the real-corpus LM
  - fidelity : token-level F1 between the decoded output and the
               corpus-generator reference for the same sentence (the
               synthetic ground truth the translator was trained toward)
Bootstrap 95% CIs over the test sentences (B=500). Seed 42.
Empirically verifies the monotonicity predicted by Proposition P6(i).
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from translate_to_rongorongo_v6 import RealGlyphLM, beam_translate, load_model  # noqa: E402
from evaluate_rongorongo_v6 import TEST_SENTENCES  # noqa: E402
import generate_massive_parallel_v4 as v4  # noqa: E402

SEED = 42


def reference_for(sentence, sampler, rng):
    """Corpus-generator reference translation (synthetic ground truth)."""
    words = [w.strip(",") for w in sentence.lower().split()]
    word_cat = {}
    for cat, ws in v4.SOURCE_VOCAB.items():
        for w in ws:
            word_cat.setdefault(w.lower(), cat)
    out, prev = [], None
    for pos, w in enumerate(words):
        cat = word_cat.get(w, "part")
        g = sampler.sample(w, cat, pos, words[pos - 1] if pos else "<bos>", prev, rng)
        out.append(g)
        prev = g
    return out


def token_f1(hyp, ref):
    h, r = Counter(hyp), Counter(ref)
    overlap = sum((h & r).values())
    if not hyp or not ref:
        return 0.0
    p, rec = overlap / len(hyp), overlap / len(ref)
    return 2 * p * rec / (p + rec) if p + rec else 0.0


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    model, sv, tv = load_model("models/rongorongo_translator_v6")
    lm = RealGlyphLM("data/raw/lost_language/corpus_rongorongo_real.xml.csv")
    unigrams, bigrams = v4.load_real_stats("data/raw/lost_language/corpus_rongorongo_real.xml.csv")
    sampler = v4.GlyphSamplerV4(unigrams, bigrams, bigram_mix=0.45)
    rng = random.Random(SEED)
    refs = [reference_for(s, sampler, rng) for s in TEST_SENTENCES]

    lambdas = [round(0.05 * i, 2) for i in range(21)]
    results = []
    boot = np.random.default_rng(SEED)
    for lam in lambdas:
        outs = [beam_translate(model, sv, tv, lm, s, beam=5, lm_weight=lam)
                for s in TEST_SENTENCES]
        bits = [lm.bits_per_glyph(o.split()) for o in outs]
        f1s = [token_f1(o.split(), r) for o, r in zip(outs, refs)]
        n = len(bits)
        bb, bf = [], []
        for _ in range(500):
            idx = boot.integers(0, n, n)
            bb.append(np.mean([bits[i] for i in idx]))
            bf.append(np.mean([f1s[i] for i in idx]))
        results.append({
            "lambda": lam,
            "bits": float(np.mean(bits)),
            "bits_ci": [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))],
            "f1": float(np.mean(f1s)),
            "f1_ci": [float(np.percentile(bf, 2.5)), float(np.percentile(bf, 97.5))],
        })
        print(f"lambda={lam:.2f}  bits={results[-1]['bits']:.3f}  F1={results[-1]['f1']:.3f}")

    out = Path("reports/lambda_sweep.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
