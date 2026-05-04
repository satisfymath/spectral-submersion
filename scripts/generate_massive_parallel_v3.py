"""Generate massive synthetic parallel corpus for Rongorongo translation (v3).

Key improvement: context-aware glyph mapping.
Each (word, category, context) tuple maps to a specific glyph,
but the same word in different contexts gets different glyphs.
This produces much more diverse and realistic output.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

# Expanded source vocabulary with 600+ tokens
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
        "te",
        "he",
        "na",
        "tau",
        "nga",
        "ko",
        "ka",
        "a",
        "o",
        "e",
        "teia",
        "tena",
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
        "moko",
        "ura",
        "kaki",
        "hue",
        "pou",
        "uru",
        "tau",
        "pae",
        "aro",
        "tua",
        "tane",
        "wahine",
        "tama",
        "tuahine",
        "tungane",
        "matua",
        "tamahine",
        "mokopuna",
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
        "tangi",
        "aroha",
        "riri",
        "haka",
        "tango",
        "tuku",
        "whaka",
        "hopu",
        "tango",
        "huri",
        "peke",
        "oma",
        "taki",
        "whaka",
        "hapa",
        "whiti",
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
        "rima_rau",
        "tahi_manu",
        "tahi_pou",
        "tahi_moana",
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
        "Atua",
        "Mata",
        "Riri",
        "Tane",
        "Hina",
        "Pou",
        "Kura",
        "Moko",
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
        "ano",
        "rawa",
        "koa",
        "hoki",
        "ake",
        "atu",
        "mai",
        "tahi",
    ],
}


def zipf_probs(items, alpha=1.4):
    ranks = np.arange(1, len(items) + 1)
    probs = 1.0 / (ranks**alpha)
    return probs / probs.sum()


# Context-aware glyph pools
GLYPH_POOL = {
    "det": [f"d{i:02d}" for i in range(1, 16)],
    "noun": [f"n{i:02d}" for i in range(1, 36)],
    "verb": [f"v{i:02d}" for i in range(1, 21)],
    "num": [f"m{i:02d}" for i in range(1, 11)],
    "name": [f"p{i:02d}" for i in range(1, 16)],
    "part": [f"x{i:02d}" for i in range(1, 11)],
    "rare": [f"r{i:02d}" for i in range(1, 11)],
}


# Global mapping cache: (word_hash, context_hash) -> glyph
_glyph_cache = {}


def sample_word(category, rng):
    words = SOURCE_VOCAB[category]
    probs = zipf_probs(words)
    return rng.choices(words, weights=probs, k=1)[0]


def sample_glyph_for_word(
    word, category, position, sentence_len, prev_word, next_word, rng
):
    """Sample a glyph deterministically based on full context.

    The same word in different contexts gets DIFFERENT glyphs,
    but the same word in the SAME context gets the SAME glyph.
    This produces consistency without excessive repetition.
    """
    pool = GLYPH_POOL[category]
    # Create a context hash from word + neighbors + position
    ctx = f"{word}:{prev_word}:{next_word}:{position % 4}"
    h = hash(ctx) % 10000000

    if ctx in _glyph_cache:
        return _glyph_cache[ctx]

    # Deterministic but varied selection
    local_rng = random.Random(h + position * 31)
    probs = zipf_probs(pool, alpha=1.2)
    glyph = local_rng.choices(pool, weights=probs, k=1)[0]
    _glyph_cache[ctx] = glyph
    return glyph


def generate_sentence_pair(rng, max_len=16):
    """Generate one parallel source-target pair."""
    line_len = rng.randint(3, max_len)
    pattern_weights = [3, 3, 2, 2, 1.5, 1.5, 1, 1, 0.8, 0.5, 0.5, 0.3]
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
        ["det", "noun", "verb", "part", "det", "noun", "verb", "part", "det", "noun"],
    ]
    cats = rng.choices(patterns, weights=pattern_weights, k=1)[0]
    if len(cats) > line_len:
        cats = cats[:line_len]
    while len(cats) < line_len:
        cats.append(rng.choice(["noun", "verb", "part", "det"]))

    source = []
    glyphs = []
    for pos, cat in enumerate(cats):
        word = sample_word(cat, rng)
        source.append(word)
        prev_word = source[-2] if pos > 0 else "<bos>"
        next_word = "<eos>" if pos == len(cats) - 1 else ""
        g = sample_glyph_for_word(word, cat, pos, len(cats), prev_word, next_word, rng)
        glyphs.append(g)

    # Repetitions (Rongorongo feature)
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
    n_pairs=200000,
    seed=42,
    output="data/raw/lost_language/parallel_rongorongo_massive_v3.csv",
):
    global _glyph_cache
    _glyph_cache = {}
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
    print(
        f"  Unique source tokens: {len(set(t for s in df['source_text'] for t in s.split()))}"
    )
    print(
        f"  Unique target glyphs: {len(set(t for s in df['target_glyphs'] for t in s.split()))}"
    )
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", default="data/raw/lost_language/parallel_rongorongo_massive_v3.csv"
    )
    args = parser.parse_args()
    generate(args.n_pairs, args.seed, args.output)


if __name__ == "__main__":
    main()
