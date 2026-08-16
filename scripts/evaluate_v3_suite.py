"""v3 evaluation suite: baselines + ablations + bootstrap CIs.

Systems evaluated with the same 6 metrics (VR, BA, JS, D2, RM, b/g):
  - baseline_template : category-restricted random template (no learning)
  - baseline_bigram   : lexical p(g|w) + target-bigram reranking (count-based)
  - baseline_lstm     : small LSTM seq2seq (if models/rongorongo_lstm exists)
  - v5_greedy         : previous synthetic-code transformer
  - v6_greedy         : v6 transformer, greedy
  - v6_beam_fusion    : v6 + beam 5 + LM fusion (lambda=0.35) [full system]
  - abl_no_fusion     : v6 beam 5, lambda=0
  - abl_no_reppenalty : v6 beam 5 + fusion, rep_penalty=0
  - abl_beam1         : v6 beam 1 + fusion
  - abl_noclass       : v6 retrained w/o class restriction (if model exists)
  - abl_noaug         : v6 retrained w/o masking augmentation (if model exists)

Bootstrap CIs (95%, B=1000, resampling the test sentences) for every metric.
Seed fixed: 42.
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from translate_to_rongorongo_v6 import RealGlyphLM, beam_translate, load_model  # noqa: E402
from translate_to_rongorongo import translate as greedy_translate  # noqa: E402
from translate_to_rongorongo import load_model as load_model_v5  # noqa: E402
from evaluate_rongorongo_v6 import TEST_SENTENCES, real_stats, js_divergence  # noqa: E402
import generate_massive_parallel_v4 as v4  # noqa: E402

SEED = 42


# ---------------- Baseline A: category-restricted random template ----------------
class TemplateBaseline:
    def __init__(self, real_corpus):
        unigrams, bigrams = v4.load_real_stats(real_corpus)
        self.sampler = v4.GlyphSamplerV4(unigrams, bigrams, bigram_mix=0.0)  # no bigram smoothing
        self.word_cat = {}
        for cat, words in v4.SOURCE_VOCAB.items():
            for w in words:
                self.word_cat.setdefault(w.lower(), cat)

    def translate(self, text, rng):
        out = []
        for w in text.lower().split():
            w = w.strip(",")
            cat = self.word_cat.get(w, "part")
            cls = self.sampler.word_class(w, cat)
            pool, probs = self.sampler.pools[cls]
            out.append(rng.choices(pool, weights=probs, k=1)[0])
        return " ".join(out)


# ---------------- Baseline B: lexical + bigram counts from parallel corpus ----------------
class BigramBaseline:
    def __init__(self, parallel_csv, top_k=5):
        df = pd.read_csv(parallel_csv)
        lex = defaultdict(Counter)
        bi = defaultdict(Counter)
        for _, row in df.iterrows():
            src = row["source_text"].split()
            tgt = row["target_glyphs"].split()
            for w in src:
                for g in tgt:
                    lex[w][g] += 1
            for a, b in zip(tgt, tgt[1:]):
                bi[a][b] += 1
        self.lex = lex
        self.bi = bi
        self.top_k = top_k

    def translate(self, text):
        prev = None
        out = []
        for w in text.lower().split():
            w = w.strip(",")
            cands = self.lex.get(w)
            if not cands:
                cands = Counter({"001": 1})
            top = cands.most_common(self.top_k)
            total_w = sum(c for _, c in top)
            best, best_score = None, -1e18
            for g, c in top:
                s = np.log(c / total_w)
                if prev is not None:
                    prev_c = self.bi.get(prev, Counter())
                    s += np.log((prev_c.get(g, 0) + 0.5) / (sum(prev_c.values()) + 0.5 * len(top)))
                if s > best_score:
                    best, best_score = g, s
            out.append(best)
            prev = best
        return " ".join(out)


# ---------------- Metrics with bootstrap ----------------
def per_sentence_metrics(outputs, real_uni, real_bi, lm):
    """Return per-sentence arrays so we can bootstrap over sentences."""
    rows = []
    for out in outputs:
        toks = out.split()
        bigr = list(zip(toks, toks[1:]))
        vr = sum(1 for t in toks if t in real_uni) / max(len(toks), 1)
        ba = (sum(1 for b in bigr if b in real_bi) / len(bigr)) if bigr else 0.0
        run, max_run = 1, 1
        for a, b in zip(toks, toks[1:]):
            run = run + 1 if a == b else 1
            max_run = max(max_run, run)
        bits = lm.bits_per_glyph(toks) if toks and vr > 0.5 else np.nan
        rows.append({"vr": vr, "ba": ba, "max_run": max_run, "bits": bits,
                     "len": len(toks), "toks": toks})
    return rows


def aggregate(rows, real_uni):
    toks = [t for r in rows for t in r["toks"]]
    out_uni = Counter(toks)
    bigrams = [b for r in rows for b in zip(r["toks"], r["toks"][1:])]
    return {
        "VR": float(np.mean([r["vr"] for r in rows])),
        "BA": float(np.mean([r["ba"] for r in rows])),
        "JS": js_divergence(out_uni, real_uni),
        "D2": len(set(bigrams)) / max(len(bigrams), 1),
        "RM": int(max(r["max_run"] for r in rows)),
        "bg": float(np.nanmean([r["bits"] for r in rows])),
    }


def bootstrap_ci(rows, real_uni, B=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(rows)
    stats = defaultdict(list)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        agg = aggregate([rows[i] for i in idx], real_uni)
        for k, v in agg.items():
            stats[k].append(v)
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
            for k, v in stats.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-corpus", default="data/raw/lost_language/corpus_rongorongo_real.xml.csv")
    parser.add_argument("--parallel", default="data/raw/lost_language/parallel_rongorongo_real_v4.csv")
    parser.add_argument("--output", default="reports/v3_evaluation_suite.json")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    lines, real_uni, real_bi = real_stats(args.real_corpus)
    lm = RealGlyphLM(args.real_corpus)

    systems = {}

    # Baseline A
    tb = TemplateBaseline(args.real_corpus)
    rng = random.Random(SEED)
    systems["baseline_template"] = [tb.translate(s, rng) for s in TEST_SENTENCES]

    # Baseline B
    bb = BigramBaseline(args.parallel)
    systems["baseline_bigram"] = [bb.translate(s) for s in TEST_SENTENCES]

    # Baseline C: LSTM (optional, if trained)
    lstm_dir = Path("models/rongorongo_lstm")
    if (lstm_dir / "model.pt").exists():
        from train_lstm_baseline import load_lstm, lstm_translate
        lmodel, lsv, ltv = load_lstm(str(lstm_dir))
        systems["baseline_lstm"] = [lstm_translate(lmodel, lsv, ltv, s) for s in TEST_SENTENCES]

    # v5
    m5, sv5, tv5 = load_model_v5("models/rongorongo_translator_v5")
    systems["v5_greedy"] = [greedy_translate(m5, sv5, tv5, s) for s in TEST_SENTENCES]

    # v6 variants
    m6, sv6, tv6 = load_model("models/rongorongo_translator_v6")
    systems["v6_greedy"] = [greedy_translate(m6, sv6, tv6, s) for s in TEST_SENTENCES]
    systems["v6_beam_fusion"] = [beam_translate(m6, sv6, tv6, lm, s, beam=5, lm_weight=0.35)
                                 for s in TEST_SENTENCES]
    systems["abl_no_fusion"] = [beam_translate(m6, sv6, tv6, lm, s, beam=5, lm_weight=0.0)
                                for s in TEST_SENTENCES]
    systems["abl_no_reppenalty"] = [beam_translate(m6, sv6, tv6, lm, s, beam=5, lm_weight=0.35,
                                                   rep_penalty=0.0)
                                    for s in TEST_SENTENCES]
    systems["abl_beam1"] = [beam_translate(m6, sv6, tv6, lm, s, beam=1, lm_weight=0.35)
                            for s in TEST_SENTENCES]

    # Retrained ablations (optional, if finished)
    for name, mdir in [("abl_noclass", "models/rongorongo_translator_v6_noclass"),
                       ("abl_noaug", "models/rongorongo_translator_v6_noaug")]:
        p = Path(mdir)
        if (p / "src_vocab.pkl").exists():
            ma, sva, tva = load_model(mdir)
            systems[name] = [beam_translate(ma, sva, tva, lm, s, beam=5, lm_weight=0.35)
                             for s in TEST_SENTENCES]

    results = {"seed": SEED, "n_test": len(TEST_SENTENCES), "systems": {}}
    for name, outs in systems.items():
        rows = per_sentence_metrics(outs, real_uni, real_bi, lm)
        agg = aggregate(rows, real_uni)
        ci = bootstrap_ci(rows, real_uni)
        results["systems"][name] = {"point": agg, "ci95": ci, "outputs": outs}
        print(f"{name:22s} VR={agg['VR']:.2f} BA={agg['BA']:.2f} JS={agg['JS']:.2f} "
              f"D2={agg['D2']:.2f} RM={agg['RM']:2d} b/g={agg['bg']:.2f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
