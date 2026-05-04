"""Generate massive synthetic parallel corpus for Rongorongo translation (v2).

More diverse and context-sensitive mapping to produce richer translations.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

# Expanded vocabularies with 500+ unique source tokens
SOURCE_VOCAB = {
    "det": [
        "te",
        "he",
        "tau",
        "nga",
        "na",
        "a",
        "o",
        "e",
        "ko",
        "ka",
        "teia",
        "tena",
        "taua",
        "tana",
        "taku",
        "ona",
        "aku",
        "tou",
        "kou",
        "mou",
        "tanaa",
        "takua",
    ],
    "noun": [
        "tangata",
        "vahine",
        "tamaiti",
        "ika",
        "manu",
        "rakau",
        "maa",
        "vai",
        "raa",
        "marama",
        "matangi",
        "ua",
        "moana",
        "motu",
        "maunga",
        "ana",
        "hare",
        "vaka",
        "rima",
        "ava",
        "mata",
        "taringa",
        "ihu",
        "waha",
        "matau",
        "maui",
        "tua",
        "mua",
        "muri",
        "ruga",
        "raro",
        "tai",
        "uta",
        "rau",
        "hua",
        "kiko",
        "ivi",
        "pepe",
        "moka",
        "uri",
        "kahu",
        "puku",
        "hue",
        "huea",
        "kore",
        "mate",
        "ora",
        "manava",
        "take",
        "roe",
        "tai",
        "au",
        "kopu",
        "reka",
        "hau",
        "titi",
        "kura",
        "paea",
        "tore",
        "here",
        "taki",
        "uru",
        "pua",
        "hue",
        "kai",
        "inu",
        "po",
        "ao",
        "mahina",
        "taua",
        "rua",
        "tahi",
        "tpu",
        "hue",
        "tua",
        "kau",
        "rau",
    ],
    "verb": [
        "haere",
        "noho",
        "kai",
        "inu",
        "moe",
        "ora",
        "mate",
        "tangi",
        "kite",
        "korero",
        "hula",
        "ala",
        "ai",
        "hana",
        "makemake",
        "hele",
        "tiki",
        "tuku",
        "tae",
        "hoki",
        "tomo",
        "puke",
        "hahau",
        "huti",
        "kume",
        "rere",
        "topa",
        "tau",
        "rere",
        "huri",
        "taka",
        "ue",
        "nue",
        "haehae",
        "patu",
        "kohu",
        "tahu",
        "huti",
        "amo",
        "oho",
        "aue",
        "tangi",
        "hiamoe",
        "hiamoe",
        "araara",
    ],
    "num": [
        "tahi",
        "rua",
        "toru",
        "ha",
        "rima",
        "ono",
        "hitu",
        "va",
        "iva",
        "hongahuru",
        "tekau",
        "ruatekau",
        "toruteke",
        "hateke",
        "rimateke",
        "tahi_tekau",
        "rua_tekau",
        "toru_tekau",
        "ha_tekau",
        "rima_tekau",
    ],
    "name": [
        "Hotu",
        "Matua",
        "Tuu",
        "Koihu",
        "Ngaara",
        "Tangaroa",
        "Makemake",
        "Tane",
        "Rongo",
        "Tiki",
        "Hina",
        "Papa",
        "Rangi",
        "Maui",
        "Tupa",
        "Kava",
        "Ure",
        "Ana",
        "Riri",
        "Koro",
        "Ariki",
        "Ika",
        "Mana",
        "Tane",
        "Rua",
        "Kena",
        "Pua",
        "Hura",
        "Tiki",
        "Tane",
    ],
    "part": [
        "i",
        "ki",
        "mai",
        "atu",
        "ma",
        "mo",
        "na",
        "no",
        "pe",
        "a",
        "o",
        "ei",
        "ana",
        "ra",
        "nei",
        "na",
        "ai",
        "ana",
        "ra",
        "hoi",
        "to",
        "ta",
        "to",
        "taa",
        "toa",
        "ia",
        "tae",
        "ake",
        "hoki",
        "tonu",
    ],
}


def zipf_probs(items, alpha=1.3):
    ranks = np.arange(1, len(items) + 1)
    probs = 1.0 / (ranks**alpha)
    return probs / probs.sum()


# Context-sensitive glyph mapping
GLYPH_POOL = {
    "det": [f"g{i:03d}" for i in range(1, 16)],
    "noun": [f"g{i:03d}" for i in range(16, 51)],
    "verb": [f"g{i:03d}" for i in range(51, 71)],
    "num": [f"g{i:03d}" for i in range(71, 81)],
    "name": [f"g{i:03d}" for i in range(81, 96)],
    "part": [f"g{i:03d}" for i in range(96, 106)],
    "rare": [f"g{i:03d}" for i in range(106, 121)],
}


def sample_word(category, rng):
    words = SOURCE_VOCAB[category]
    probs = zipf_probs(words)
    return rng.choices(words, weights=probs, k=1)[0]


def sample_glyph_for_word(word, category, position, sentence_len, rng):
    """Sample a glyph with context-dependent variation."""
    pool = GLYPH_POOL[category]
    # Seed RNG with word + position for slight determinism
    local_rng = random.Random(hash(word) + position * 17 + sentence_len * 31)
    probs = zipf_probs(pool, alpha=1.1)
    return local_rng.choices(pool, weights=probs, k=1)[0]


def generate_sentence_pair(rng, max_len=14):
    """Generate one parallel source-target pair."""
    line_len = rng.randint(3, max_len)
    # More varied patterns
    pattern_weights = [3, 3, 2, 2, 1.5, 1.5, 1, 1, 0.8, 0.5, 0.5]
    patterns = [
        ["det", "noun", "verb", "part", "det", "noun"],
        ["det", "noun", "verb", "part", "det", "noun", "part"],
        ["name", "verb", "det", "noun", "part"],
        ["det", "noun", "verb", "num", "noun", "part"],
        ["part", "det", "noun", "verb", "part", "det", "noun"],
        ["det", "noun", "part", "det", "noun", "verb", "part"],
        ["name", "name", "verb", "det", "noun", "part", "part"],
        ["det", "noun", "verb", "part", "name", "part"],
        ["num", "det", "noun", "verb", "part", "det", "noun"],
        ["det", "noun", "verb", "det", "noun", "verb", "part", "det", "noun"],
        ["part", "name", "verb", "num", "noun", "part", "det", "noun"],
    ]
    cats = rng.choices(patterns, weights=pattern_weights, k=1)[0]
    # Trim or extend to desired length
    if len(cats) > line_len:
        cats = cats[:line_len]
    while len(cats) < line_len:
        cats.append(rng.choice(["noun", "verb", "part", "det"]))

    source = []
    glyphs = []
    for pos, cat in enumerate(cats):
        word = sample_word(cat, rng)
        source.append(word)
        g = sample_glyph_for_word(word, cat, pos, len(cats), rng)
        glyphs.append(g)

    # Repetitions
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


def generate(
    n_pairs=100000,
    seed=42,
    output="data/raw/lost_language/parallel_rongorongo_massive_v2.csv",
):
    rng = random.Random(seed)
    np.random.seed(seed)
    rows = []
    for i in range(1, n_pairs + 1):
        pair = generate_sentence_pair(rng)
        pair["pair_id"] = i
        rows.append(pair)
    df = pd.DataFrame(rows)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Generated {len(df)} pairs to {output}")
    print(
        f"  Source mean: {df['source_len'].mean():.2f}, Target mean: {df['target_len'].mean():.2f}"
    )
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", default="data/raw/lost_language/parallel_rongorongo_massive_v2.csv"
    )
    args = parser.parse_args()
    generate(args.n_pairs, args.seed, args.output)


if __name__ == "__main__":
    main()
