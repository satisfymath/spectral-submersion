"""Generate parallel corpus v4: Rapanui-like source -> REAL Barthel glyph codes.

Key upgrade over v3: targets are no longer synthetic d/n/v codes but real
Barthel numeric codes drawn from the real Rongorongo corpus (tablets A-F,
RR-corpus phspaelti). Glyph selection is:

  1. Constrained by a CLASS-HYPOTHESIS mapping (semantic category -> Barthel
     numeric range). This mapping is a documented HYPOTHESIS, not a claim
     (see guia_phd_upgrade: iconic anchors are weak; cross-script benchmark
     did not pass C2.5). It follows the traditional Barthel taxonomy:
       001-099  geometric / simple signs      -> determiners, particles, numerals
       200-399  anthropomorphs                -> person nouns, proper names
       400-599  anthropomorph variants/limbs  -> verbs (action figures)
       600-699  bird signs                    -> bird/sky domain nouns
       700-799  fish/marine signs             -> fish/sea domain nouns
  2. Weighted by REAL unigram frequency within the class pool.
  3. Smoothed by REAL bigram transitions: with prob --bigram-mix, the glyph
     is re-sampled from P(g | prev_g) restricted to the class pool, so the
     output sequences inherit the real corpus' transition statistics.

Same (word, context) -> same glyph via deterministic cache (consistency),
same repetition process as v3 (a real Rongorongo feature).
"""
import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from generate_massive_parallel_v3 import (
    SOURCE_VOCAB,
    sample_word,
    zipf_probs,
)

# --- Class hypothesis: semantic category -> Barthel numeric range(s) ---
CLASS_RANGES = {
    "det": [(1, 59)],
    "part": [(1, 99)],
    "num": [(1, 99)],
    "noun_person": [(200, 399)],
    "name": [(200, 399)],
    "verb": [(400, 599)],
    "noun_bird": [(600, 699)],
    "noun_fish": [(700, 799)],
    "noun_other": [(60, 199), (600, 799)],
}

# Domain-specific word -> class overrides (hypothesis level, not claims)
WORD_CLASS = {
    "manu": "noun_bird", "moka": "noun_bird", "pepe": "noun_bird",
    "ika": "noun_fish", "ura": "noun_fish", "moko": "noun_fish", "roe": "noun_fish",
    "tangata": "noun_person", "vahine": "noun_person", "tamaiti": "noun_person",
    "tama": "noun_person", "tane": "noun_person", "wahine": "noun_person",
    "matua": "noun_person", "tamahine": "noun_person", "mokopuna": "noun_person",
    "tuahine": "noun_person", "tungane": "noun_person", "ariki": "noun_person",
}

CATEGORY_CLASS = {
    "det": "det", "part": "part", "num": "num",
    "name": "name", "verb": "verb", "noun": "noun_other",
}


def barthel_base(token: str) -> int | None:
    """Extract the numeric base of a Barthel token like '430!' or '022bfy'."""
    digits = ""
    for ch in token:
        if ch.isdigit():
            digits += ch
        else:
            break
    if len(digits) < 2:
        return None
    return int(digits[:3]) if len(digits) >= 3 else int(digits)


def load_real_stats(corpus_csv: str):
    """Unigram and bigram statistics from the real corpus, per line."""
    df = pd.read_csv(corpus_csv)
    unigrams = Counter()
    bigrams = defaultdict(Counter)
    for _, line_df in df.groupby(["doc_id", "line_id"], sort=False):
        seq = [t for t in line_df.sort_values("position")["token"].astype(str) if t != "_"]
        unigrams.update(seq)
        for a, b in zip(seq, seq[1:]):
            bigrams[a][b] += 1
    return unigrams, bigrams


def build_class_pools(unigrams: Counter):
    """Assign every real glyph to the class pools whose range contains it."""
    pools = {}
    for cls, ranges in CLASS_RANGES.items():
        pool, weights = [], []
        for tok, freq in unigrams.items():
            base = barthel_base(tok)
            if base is None:
                continue
            if any(lo <= base <= hi for lo, hi in ranges):
                pool.append(tok)
                weights.append(freq)
        w = np.asarray(weights, dtype=float)
        pools[cls] = (pool, w / w.sum())
    return pools


class GlyphSamplerV4:
    def __init__(self, unigrams, bigrams, bigram_mix=0.45):
        self.pools = build_class_pools(unigrams)
        self.bigrams = bigrams
        self.bigram_mix = bigram_mix
        self.cache = {}

    def word_class(self, word, category):
        return WORD_CLASS.get(word, CATEGORY_CLASS[category])

    def sample(self, word, category, position, prev_word, prev_glyph, rng):
        cls = self.word_class(word, category)
        pool, probs = self.pools[cls]

        ctx = f"{word}:{prev_word}:{position % 4}"
        if ctx in self.cache:
            base_glyph = self.cache[ctx]
        else:
            local_rng = random.Random(hash(ctx) % 10_000_000 + position * 31)
            base_glyph = local_rng.choices(pool, weights=probs, k=1)[0]
            self.cache[ctx] = base_glyph

        # Bigram smoothing: prefer real transitions from prev_glyph within class
        if prev_glyph is not None and rng.random() < self.bigram_mix:
            nexts = self.bigrams.get(prev_glyph)
            if nexts:
                pool_set = set(pool)
                cands = [(g, c) for g, c in nexts.items() if g in pool_set]
                if cands:
                    glyphs, counts = zip(*cands)
                    return rng.choices(glyphs, weights=counts, k=1)[0]
        return base_glyph


PATTERNS = [
    (3.0, ["det", "noun", "verb", "part", "det", "noun"]),
    (3.0, ["det", "noun", "verb", "part", "det", "noun", "part"]),
    (2.0, ["name", "verb", "det", "noun", "part"]),
    (2.0, ["det", "noun", "verb", "num", "noun", "part"]),
    (1.5, ["part", "det", "noun", "verb", "part", "det", "noun"]),
    (1.5, ["det", "noun", "part", "det", "noun", "verb", "part"]),
    (1.0, ["name", "name", "verb", "det", "noun", "part", "part"]),
    (1.0, ["det", "noun", "verb", "part", "name", "part"]),
    (0.8, ["num", "det", "noun", "verb", "part", "det", "noun"]),
    (0.5, ["det", "noun", "verb", "det", "noun", "verb", "part", "det", "noun"]),
    (0.5, ["part", "name", "verb", "num", "noun", "part", "det", "noun"]),
    (0.3, ["det", "noun", "verb", "part", "det", "noun", "verb", "part", "det", "noun"]),
]


def generate_sentence_pair(sampler, rng, max_len=16):
    line_len = rng.randint(3, max_len)
    weights = [w for w, _ in PATTERNS]
    cats = list(rng.choices([p for _, p in PATTERNS], weights=weights, k=1)[0])
    if len(cats) > line_len:
        cats = cats[:line_len]
    while len(cats) < line_len:
        cats.append(rng.choice(["noun", "verb", "part", "det"]))

    source, glyphs = [], []
    prev_glyph = None
    for pos, cat in enumerate(cats):
        word = sample_word(cat, rng)
        source.append(word)
        prev_word = source[-2] if pos > 0 else "<bos>"
        g = sampler.sample(word, cat, pos, prev_word, prev_glyph, rng)
        glyphs.append(g)
        prev_glyph = g

    # Repetition process (real Rongorongo feature)
    final_glyphs = []
    for g in glyphs:
        final_glyphs.append(g)
        if rng.random() < 0.12:
            final_glyphs.append(g)
        elif rng.random() < 0.03:
            final_glyphs.extend([g, g])

    return {
        "source_text": " ".join(source),
        "target_glyphs": " ".join(final_glyphs),
        "source_len": len(source),
        "target_len": len(final_glyphs),
        "categories": " ".join(cats),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-corpus", default="data/raw/lost_language/corpus_rongorongo_real.xml.csv")
    parser.add_argument("--n-pairs", type=int, default=60000)
    parser.add_argument("--bigram-mix", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/raw/lost_language/parallel_rongorongo_real_v4.csv")
    args = parser.parse_args()

    unigrams, bigrams = load_real_stats(args.real_corpus)
    print(f"Real corpus: {sum(unigrams.values())} tokens, {len(unigrams)} types")
    sampler = GlyphSamplerV4(unigrams, bigrams, bigram_mix=args.bigram_mix)
    for cls, (pool, _) in sampler.pools.items():
        print(f"  class {cls:12s}: {len(pool)} real glyphs")

    rng = random.Random(args.seed)
    rows = []
    for i in range(1, args.n_pairs + 1):
        pair = generate_sentence_pair(sampler, rng)
        pair["pair_id"] = i
        rows.append(pair)
    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    tgt_vocab = set(t for s in df["target_glyphs"] for t in s.split())
    print(f"Generated {len(df)} pairs to {args.output}")
    print(f"  Source mean len: {df['source_len'].mean():.2f}, target mean len: {df['target_len'].mean():.2f}")
    print(f"  Unique target glyphs: {len(tgt_vocab)} (all real Barthel codes)")


if __name__ == "__main__":
    main()
