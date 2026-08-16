"""Ablation corpus: v4 WITHOUT the category->Barthel-range restriction.

Every class pool is replaced by the full real vocabulary (still weighted by
real unigram frequency, still bigram-smoothed). Isolates the contribution of
the class-hypothesis mapping (claim level C2) to translator quality.
Seed-fixed; registered for the v3 ablation table.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

import generate_massive_parallel_v4 as v4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-corpus", default="data/raw/lost_language/corpus_rongorongo_real.xml.csv")
    parser.add_argument("--n-pairs", type=int, default=60000)
    parser.add_argument("--bigram-mix", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/raw/lost_language/parallel_rongorongo_real_v4_noclass.csv")
    args = parser.parse_args()

    # Remove the class hypothesis: every class covers the full Barthel range.
    for cls in v4.CLASS_RANGES:
        v4.CLASS_RANGES[cls] = [(1, 999)]

    unigrams, bigrams = v4.load_real_stats(args.real_corpus)
    sampler = v4.GlyphSamplerV4(unigrams, bigrams, bigram_mix=args.bigram_mix)
    rng = random.Random(args.seed)
    rows = []
    for i in range(1, args.n_pairs + 1):
        pair = v4.generate_sentence_pair(sampler, rng)
        pair["pair_id"] = i
        rows.append(pair)
    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Generated {len(df)} ablation pairs (no class restriction) to {args.output}")


if __name__ == "__main__":
    main()
