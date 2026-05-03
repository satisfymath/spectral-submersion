"""Build parallel corpus: Rapa Nui / Polynesian → Rongorongo.

Maps sentences from candidate languages to Rongorongo glyph sequences
using structural POS-to-glyph-category alignment. This creates
synthetic parallel data for training the seq2seq translator.

Mapping hypothesis (based on Polynesian typology):
- DET/ART → det glyphs
- NOUN → noun glyphs
- VERB → verb glyphs
- NUM → num glyphs
- PROPN/NAME → name glyphs
- ADP/PART → part glyphs
- Other → rare glyphs
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse glyph inventory from generator v3
# Replicated from generate_rongorongo_v3 to avoid import path issues
GLYPHS = {
    "det": [f"g{i:03d}" for i in range(1, 16)],
    "noun": [f"g{i:03d}" for i in range(16, 51)],
    "verb": [f"g{i:03d}" for i in range(51, 71)],
    "num": [f"g{i:03d}" for i in range(71, 81)],
    "name": [f"g{i:03d}" for i in range(81, 96)],
    "part": [f"g{i:03d}" for i in range(96, 106)],
    "rare": [f"g{i:03d}" for i in range(106, 121)],
}
ALL_GLYPHS = [g for cat in GLYPHS.values() for g in cat]


def _zipf_probs(items, alpha=1.1):
    import numpy as np
    ranks = np.arange(1, len(items) + 1)
    probs = 1.0 / (ranks ** alpha)
    probs = probs / probs.sum()
    return dict(zip(items, probs))


CATEGORY_PROBS = {cat: _zipf_probs(glyphs) for cat, glyphs in GLYPHS.items()}


def sample_glyph(category, rng):
    glyphs = list(CATEGORY_PROBS[category].keys())
    probs = list(CATEGORY_PROBS[category].values())
    return rng.choices(glyphs, weights=probs, k=1)[0]


# Simple heuristic POS mapping for Rapa Nui / Polynesian
def heuristic_pos(token: str, position: int, sentence_len: int) -> str:
    """Assign pseudo-POS based on token properties and position."""
    t = token.lower().strip()
    # Articles/determiners: 'te', 'he', 'tau', 'nga' (Polynesian)
    if t in {"te", "he", "tau", "nga", "na", "a", "o", "e", "ko", "ka"}:
        return "det"
    # Numbers
    if t.isdigit() or t in {"tahi", "rua", "toru", "ha", "rima", "ono", "hitu", "va", "iva", "hongahuru"}:
        return "num"
    # Common verbs (Polynesian)
    if t in {"haere", "noho", "kai", "inu", "moe", "ora", "mate", "tangi", "kite", "korero",
             "hula", "ala", "ai", "a\u02bbo", "hana", "makemake", "hele", "hana\u02bb", "ho\u02bbomaika\u02bbi"}:
        return "verb"
    # Proper names / titles (capitalized in source)
    if token[0].isupper() and position > 0:
        return "name"
    # Pronouns
    if t in {"au", "koe", "ia", "matou", "tatou", "ratou", "tana", "taku", "ona", "aku"}:
        return "noun"  # Treat pronouns as nouns
    # Prepositions / particles
    if t in {"i", "ki", "mai", "atu", "i\u02bb", "ma", "mo", "na", "no", "pe\u02bbi"}:
        return "part"
    # Default: noun for short words, verb for longer, mixed
    if len(t) <= 3:
        return "noun"
    elif len(t) <= 5:
        return random.choice(["noun", "verb"])
    else:
        return random.choice(["noun", "verb", "rare"])


def map_sentence_to_glyphs(tokens: list[str], rng: random.Random) -> list[str]:
    """Map a token sequence to Rongorongo glyph sequence."""
    glyphs = []
    for pos, tok in enumerate(tokens):
        cat = heuristic_pos(tok, pos, len(tokens))
        g = sample_glyph(cat, rng)
        glyphs.append(g)
    return glyphs


def build_parallel_corpus(
    candidate_path: str,
    output_path: str,
    seed: int = 42,
    max_pairs: int | None = None,
) -> pd.DataFrame:
    """Build parallel sentence-glyph pairs from candidate language tokens."""
    rng = random.Random(seed)
    np.random.seed(seed)

    df = pd.read_csv(candidate_path)
    required = {"doc_id", "line_id", "position", "token"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Group tokens into sentences by line_id
    sentences = []
    for _, group in df.groupby(["doc_id", "line_id"]):
        group = group.sort_values("position")
        tokens = group["token"].tolist()
        if len(tokens) >= 2:
            sentences.append(tokens)

    if max_pairs:
        rng.shuffle(sentences)
        sentences = sentences[:max_pairs]

    rows = []
    for sent_id, tokens in enumerate(sentences, start=1):
        glyphs = map_sentence_to_glyphs(tokens, rng)
        # Apply Rongorongo-specific post-processing
        # 1. Double repetition with 15% probability per position
        final_glyphs = []
        for g in glyphs:
            final_glyphs.append(g)
            if rng.random() < 0.15:
                final_glyphs.append(g)

        rows.append({
            "sent_id": sent_id,
            "source_lang": Path(candidate_path).stem.replace("_tokens", ""),
            "source_text": " ".join(tokens),
            "target_glyphs": " ".join(final_glyphs),
            "source_len": len(tokens),
            "target_len": len(final_glyphs),
        })

    out_df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"Built parallel corpus: {len(out_df)} pairs")
    print(f"  Source tokens mean: {out_df['source_len'].mean():.2f}")
    print(f"  Target glyphs mean: {out_df['target_len'].mean():.2f}")
    print(f"Saved to {output_path}")

    return out_df


def main():
    parser = argparse.ArgumentParser(description="Build parallel Rongorongo corpus")
    parser.add_argument("--candidate", default="data/raw/candidate_languages/rap_tokens.csv")
    parser.add_argument("--output", default="data/raw/lost_language/parallel_rongorongo_rap.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()
    build_parallel_corpus(args.candidate, args.output, args.seed, args.max_pairs)


if __name__ == "__main__":
    main()
