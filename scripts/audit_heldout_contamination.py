"""Task 2.4: held-out contamination audit.

Reports overlap of long n-grams (n >= 5) between the held-out tablets
{D, F} and the LM-training tablets {A, B, C, E}. Known textual parallels
(H/P/Q Grand Tradition; Gv/K) lie OUTSIDE this corpus, but D and E share
scribal-school material per the literature, so any overlap must be
reported, not assumed away. Also reports the 2 real types not covered by
the v4 parallel corpus (Task 2.7).
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

CORPUS = "data/raw/lost_language/corpus_rongorongo_real.xml.csv"
PARALLEL = "data/raw/lost_language/parallel_rongorongo_real_v4.csv"


def line_seqs(df, docs):
    out = []
    for _, line_df in df[df.doc_id.isin(docs)].groupby(["doc_id", "line_id"], sort=False):
        seq = [t for t in line_df.sort_values("position")["token"].astype(str) if t != "_"]
        out.append(seq)
    return out


def ngrams(seqs, n):
    return Counter(tuple(s[i:i + n]) for s in seqs for i in range(len(s) - n + 1))


def main():
    df = pd.read_csv(CORPUS)
    train = line_seqs(df, {"A", "B", "C", "E"})
    held = line_seqs(df, {"D", "F"})

    report = {"train_tablets": ["A", "B", "C", "E"], "heldout_tablets": ["D", "F"],
              "train_tokens": sum(len(s) for s in train),
              "heldout_tokens": sum(len(s) for s in held),
              "ngram_overlap": {}}
    for n in range(5, 11):
        tr, he = ngrams(train, n), ngrams(held, n)
        shared = set(tr) & set(he)
        held_total = sum(he.values())
        held_covered = sum(c for g, c in he.items() if g in shared)
        report["ngram_overlap"][n] = {
            "heldout_ngrams": held_total,
            "shared_types": len(shared),
            "heldout_tokens_covered": held_covered,
            "coverage_pct": round(100 * held_covered / held_total, 2) if held_total else 0.0,
            "examples": [" ".join(g) for g in list(shared)[:3]],
        }
        print(f"n={n}: held-out n-grams {held_total}, shared types {len(shared)}, "
              f"coverage {report['ngram_overlap'][n]['coverage_pct']}%")

    # Task 2.7: the uncovered types
    real_types = set(t for t in df.token.astype(str) if t != "_")
    par = pd.read_csv(PARALLEL)
    covered = set(t for s in par["target_glyphs"] for t in s.split())
    missing = sorted(real_types - covered)
    freqs = Counter(t for t in df.token.astype(str) if t != "_")
    report["uncovered_types"] = [
        {"token": t, "freq": freqs[t],
         "reason": ("no leading digits -> no Barthel base parseable"
                    if not t[:1].isdigit() else "base outside every class range")}
        for t in missing
    ]
    print("Uncovered types:", report["uncovered_types"])

    out = Path("reports/heldout_contamination_audit.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
