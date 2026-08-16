"""Evaluate translator v6 (real Barthel targets) vs v5 (synthetic codes)
against the REAL Rongorongo corpus statistics.

Metrics per system, over a fixed test set of Rapanui-like sentences:
  - vocab_realism : fraction of output tokens attested in the real corpus
  - bigram_realism: fraction of output bigrams attested in the real corpus
  - bits_per_glyph: cross-entropy under the real-corpus trigram LM
                    (only defined when tokens are real Barthel codes)
  - js_unigram    : Jensen-Shannon divergence between output unigram dist
                    and the real corpus unigram dist (lower = closer)
  - distinct_1/2  : output diversity
  - max_rep_run   : longest immediate-repetition run (degeneracy check)

Reference bound: bits_per_glyph of held-out real tablet lines under the
same LM (train on A,B,C,E; test on D,F) - the target to approach.
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from translate_to_rongorongo_v6 import RealGlyphLM, beam_translate, load_model  # noqa: E402
from translate_to_rongorongo import translate as greedy_translate  # noqa: E402
from translate_to_rongorongo import load_model as load_model_v5  # noqa: E402

TEST_SENTENCES = [
    "te tangata haere ki te moana",
    "he vahine noho i te hare",
    "ko Hotu Matua e kai ika",
    "te tamaiti moe i te po",
    "e haere au ki te maunga",
    "te manu rere i te raa",
    "he vai inu mo te tangata",
    "ko Maui tiki ake te ahi",
    "e aroha au e aroha au ki a kou",
    "te tamaiti toa o te manava tonu",
    "e haere au ma te rakau i raro",
    "ka ala te hana",
    "te vahine korero ki te ariki",
    "he ika nui i te tai",
    "ko Tangaroa moe i te moana",
    "te matua kai maa ma te tamaiti",
    "e rere te manu ki te maunga",
    "he tangata patu i te rakau",
    "te mahina whiti i te po",
    "ko Hina noho i te ana",
]


def real_stats(corpus_csv):
    df = pd.read_csv(corpus_csv)
    lines = []
    for _, line_df in df.groupby(["doc_id", "line_id"], sort=False):
        seq = [t for t in line_df.sort_values("position")["token"].astype(str) if t != "_"]
        lines.append((line_df["doc_id"].iloc[0], seq))
    uni = Counter(t for _, seq in lines for t in seq)
    bi = Counter(tuple(p) for _, seq in lines for p in zip(seq, seq[1:]))
    return lines, uni, bi


def js_divergence(p: Counter, q: Counter):
    keys = set(p) | set(q)
    pa = np.array([p.get(k, 0) for k in keys], dtype=float)
    qa = np.array([q.get(k, 0) for k in keys], dtype=float)
    pa /= pa.sum()
    qa /= qa.sum()
    m = 0.5 * (pa + qa)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(pa, m) + 0.5 * kl(qa, m)


def metrics_for(outputs, real_uni, real_bi, lm=None):
    toks = [t for out in outputs for t in out.split()]
    bigrams = [b for out in outputs for b in zip(out.split(), out.split()[1:])]
    out_uni = Counter(toks)
    vocab_realism = sum(c for t, c in out_uni.items() if t in real_uni) / max(len(toks), 1)
    bigram_realism = (sum(1 for b in bigrams if b in real_bi) / max(len(bigrams), 1))
    distinct1 = len(out_uni) / max(len(toks), 1)
    out_bi = Counter(bigrams)
    distinct2 = len(out_bi) / max(len(bigrams), 1)
    max_run = 0
    for out in outputs:
        seq = out.split()
        run = 1
        for a, b in zip(seq, seq[1:]):
            run = run + 1 if a == b else 1
            max_run = max(max_run, run)
    res = {
        "vocab_realism": round(vocab_realism, 4),
        "bigram_realism": round(bigram_realism, 4),
        "js_unigram_vs_real": round(js_divergence(out_uni, real_uni), 4),
        "distinct_1": round(distinct1, 4),
        "distinct_2": round(distinct2, 4),
        "max_rep_run": max_run,
        "mean_len": round(np.mean([len(o.split()) for o in outputs]), 2),
    }
    if lm is not None and vocab_realism > 0.5:
        res["bits_per_glyph_realLM"] = round(
            float(np.mean([lm.bits_per_glyph(o.split()) for o in outputs])), 3)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-corpus", default="data/raw/lost_language/corpus_rongorongo_real.xml.csv")
    parser.add_argument("--v5-dir", default="models/rongorongo_translator_v5")
    parser.add_argument("--v6-dir", default="models/rongorongo_translator_v6")
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--lm-weight", type=float, default=0.35)
    parser.add_argument("--output", default="reports/rongorongo_v6_evaluation.json")
    args = parser.parse_args()

    lines, real_uni, real_bi = real_stats(args.real_corpus)
    lm_full = RealGlyphLM(args.real_corpus)

    # Reference: held-out real lines (D, F) under LM trained on A, B, C, E
    train_docs = {"A", "B", "C", "E"}
    tmp = Path("reports") / "_lm_train_split.csv"
    df = pd.read_csv(args.real_corpus)
    df[df.doc_id.isin(train_docs)].to_csv(tmp, index=False)
    lm_train = RealGlyphLM(str(tmp))
    heldout_bits = float(np.mean([
        lm_train.bits_per_glyph(seq) for doc, seq in lines if doc in ("D", "F") and seq]))
    tmp.unlink()

    results = {"reference_heldout_real_bits_per_glyph": round(heldout_bits, 3)}

    # v5 (greedy, synthetic codes)
    m5, sv5, tv5 = load_model_v5(args.v5_dir)
    out_v5 = [greedy_translate(m5, sv5, tv5, s) for s in TEST_SENTENCES]
    results["v5_greedy"] = metrics_for(out_v5, real_uni, real_bi, lm_full)

    # v6 greedy
    m6, sv6, tv6 = load_model(args.v6_dir)
    out_v6g = [greedy_translate(m6, sv6, tv6, s) for s in TEST_SENTENCES]
    results["v6_greedy"] = metrics_for(out_v6g, real_uni, real_bi, lm_full)

    # v6 beam + shallow fusion
    out_v6b = [beam_translate(m6, sv6, tv6, lm_full, s,
                              beam=args.beam, lm_weight=args.lm_weight)
               for s in TEST_SENTENCES]
    results["v6_beam_fusion"] = metrics_for(out_v6b, real_uni, real_bi, lm_full)

    results["examples"] = [
        {"input": s, "v5": a, "v6_beam_fusion": b}
        for s, a, b in zip(TEST_SENTENCES[:8], out_v5[:8], out_v6b[:8])
    ]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps({k: v for k, v in results.items() if k != "examples"}, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
