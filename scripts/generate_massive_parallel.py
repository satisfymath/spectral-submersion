"""Generate massive synthetic parallel corpus for Rongorongo translation.

Generates synthetic source sentences (Rapa Nui-like) aligned with
Rongorongo glyph sequences. Uses template-based generation with
category alignment, producing tens of thousands of parallel pairs.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

# Vocabularies for synthetic Polynesian-like source sentences
SOURCE_VOCAB = {
    "det": ["te", "he", "tau", "nga", "na", "a", "o", "e", "ko", "ka", "teia", "tena"],
    "noun": [
        "tangata",
        "vahine",
        "tamaiti",
        "ika",
        "manu",
        "rakau",
        "ma",
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
    ],
}


def zipf_probs(items, alpha=1.2):
    ranks = np.arange(1, len(items) + 1)
    probs = 1.0 / (ranks**alpha)
    return probs / probs.sum()


def sample_word(category: str, rng: random.Random) -> str:
    words = SOURCE_VOCAB[category]
    probs = zipf_probs(words)
    return rng.choices(words, weights=probs, k=1)[0]


def generate_source_sentence(categories: list[str], rng: random.Random) -> list[str]:
    """Generate a source sentence from a category sequence."""
    return [sample_word(cat, rng) for cat in categories]


def generate_parallel_pair(rng: random.Random, max_line_len: int = 12):
    """Generate one parallel source-target pair."""
    # Random line length
    line_len = rng.randint(3, max_line_len)
    # Random category sequence
    cats = rng.choices(
        ["det", "noun", "verb", "num", "name", "part"],
        weights=[2, 3, 2, 0.5, 1, 1.5],
        k=line_len,
    )
    source = generate_source_sentence(cats, rng)

    # Map to glyphs using same category sequence
    # Use same glyph inventory as Rongorongo v3
    GLYPHS = {
        "det": [f"g{i:03d}" for i in range(1, 16)],
        "noun": [f"g{i:03d}" for i in range(16, 51)],
        "verb": [f"g{i:03d}" for i in range(51, 71)],
        "num": [f"g{i:03d}" for i in range(71, 81)],
        "name": [f"g{i:03d}" for i in range(81, 96)],
        "part": [f"g{i:03d}" for i in range(96, 106)],
        "rare": [f"g{i:03d}" for i in range(106, 121)],
    }

    def _sample_glyph(cat):
        glyphs = GLYPHS[cat]
        probs = zipf_probs(glyphs)
        return rng.choices(glyphs, weights=probs, k=1)[0]

    glyphs = [_sample_glyph(cat) for cat in cats]

    # Apply Rongorongo-style repetitions
    final_glyphs = []
    for g in glyphs:
        final_glyphs.append(g)
        if rng.random() < 0.15:
            final_glyphs.append(g)

    return {
        "source_text": " ".join(source),
        "target_glyphs": " ".join(final_glyphs),
        "source_len": len(source),
        "target_len": len(final_glyphs),
        "categories": " ".join(cats),
    }


def generate_massive_parallel(
    n_pairs: int = 50000,
    seed: int = 42,
    output_path: str = "data/raw/lost_language/parallel_rongorongo_massive.csv",
):
    rng = random.Random(seed)
    np.random.seed(seed)

    rows = []
    for i in range(1, n_pairs + 1):
        pair = generate_parallel_pair(rng)
        pair["pair_id"] = i
        rows.append(pair)

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Generated massive parallel corpus: {len(df)} pairs")
    print(f"  Source mean length: {df['source_len'].mean():.2f}")
    print(f"  Target mean length: {df['target_len'].mean():.2f}")
    print(f"Saved to {output_path}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate massive parallel Rongorongo corpus"
    )
    parser.add_argument("--n-pairs", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", default="data/raw/lost_language/parallel_rongorongo_massive.csv"
    )
    args = parser.parse_args()
    generate_massive_parallel(args.n_pairs, args.seed, args.output)


if __name__ == "__main__":
    main()
