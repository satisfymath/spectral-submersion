"""Generate a large, controlled synthetic lost-language corpus.

This script creates an artificial symbolic system with known structure
(~100 types, ~10k tokens) to serve as a methodological benchmark.
It is NOT a claim about any real lost language; it is a ground-truth
testbed for validating the spectral pipeline, Procrustes alignment,
and anchor recovery.

Grammar: hierarchical phrase-structure with variation.
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

# ------------------------------------------------------------------
# Lexicon: ~100 types across functional classes
# ------------------------------------------------------------------
SYMBOLS = {
    "det": [f"d{i:02d}" for i in range(1, 9)],  # 8 determiners
    "noun": [f"n{i:02d}" for i in range(1, 31)],  # 30 nouns
    "verb": [f"v{i:02d}" for i in range(1, 21)],  # 20 verbs
    "adj": [f"a{i:02d}" for i in range(1, 13)],  # 12 adjectives
    "prep": [f"p{i:02d}" for i in range(1, 7)],  # 6 prepositions
    "conj": [f"c{i:02d}" for i in range(1, 5)],  # 4 conjunctions
    "pron": [f"r{i:02d}" for i in range(1, 9)],  # 8 pronouns
    "num": [f"m{i:02d}" for i in range(1, 7)],  # 6 numerals
    "quant": [f"q{i:02d}" for i in range(1, 5)],  # 4 quantifiers
    "adv": [f"w{i:02d}" for i in range(1, 5)],  # 4 adverbs
    "name": [f"x{i:02d}" for i in range(1, 11)],  # 10 proper names
}

# ------------------------------------------------------------------
# Syntactic rules (probabilistic context-free style)
# ------------------------------------------------------------------
# A sentence is built top-down from a start symbol by expansion.
# We encode a tiny PCFG manually.

EXPANSIONS = {
    "S": [
        (["NP", "VP"], 0.35),
        (["NP", "VP", "PP"], 0.20),
        (["NP", "CONJ", "NP", "VP"], 0.10),
        (["NP", "VP", "CONJ", "S"], 0.10),
        (["DET", "NOUN", "VP", "NP"], 0.15),
        (["PRON", "VP", "NP"], 0.10),
    ],
    "NP": [
        (["DET", "NOUN"], 0.30),
        (["DET", "ADJ", "NOUN"], 0.20),
        (["DET", "NUM", "NOUN"], 0.15),
        (["PRON"], 0.15),
        (["NAME"], 0.10),
        (["DET", "QUANT", "NOUN"], 0.10),
    ],
    "VP": [
        (["VERB", "NP"], 0.40),
        (["VERB", "NP", "PP"], 0.25),
        (["VERB", "ADV"], 0.15),
        (["VERB", "NP", "ADV"], 0.10),
        (["VERB"], 0.10),
    ],
    "PP": [
        (["PREP", "NP"], 0.70),
        (["PREP", "DET", "NOUN"], 0.30),
    ],
    # Terminals (will be resolved by sampling from SYMBOLS)
    "DET": [(["det"], 1.0)],
    "NOUN": [(["noun"], 1.0)],
    "VERB": [(["verb"], 1.0)],
    "ADJ": [(["adj"], 1.0)],
    "PREP": [(["prep"], 1.0)],
    "CONJ": [(["conj"], 1.0)],
    "PRON": [(["pron"], 1.0)],
    "NUM": [(["num"], 1.0)],
    "QUANT": [(["quant"], 1.0)],
    "ADV": [(["adv"], 1.0)],
    "NAME": [(["name"], 1.0)],
}


def _weighted_choice(options: list[tuple]):
    weights = [w for _, w in options]
    items = [it for it, _ in options]
    return random.choices(items, weights=weights, k=1)[0]


def _expand(symbol: str, rng: random.Random) -> list[str]:
    """Expand a non-terminal to a list of terminal symbols."""
    if symbol not in EXPANSIONS:
        raise ValueError(f"Unknown symbol: {symbol}")
    expansion = _weighted_choice(EXPANSIONS[symbol])
    result = []
    for child in expansion:
        if child in SYMBOLS:
            # terminal class: sample concrete token
            result.append(rng.choice(SYMBOLS[child]))
        elif child in EXPANSIONS:
            result.extend(_expand(child, rng))
        else:
            raise ValueError(f"Unknown child symbol: {child}")
    return result


def generate_sentence(rng: random.Random) -> list[str]:
    return _expand("S", rng)


def generate_corpus(n_sentences: int = 2000, seed: int = 42) -> list[list[str]]:
    rng = random.Random(seed)
    return [generate_sentence(rng) for _ in range(n_sentences)]


def corpus_to_df(sequences: list[list[str]], doc_id: str = "synthetic") -> pd.DataFrame:
    rows = []
    for line_id, seq in enumerate(sequences, start=1):
        for pos, tok in enumerate(seq, start=1):
            # Infer POS from prefix for ground-truth annotation
            pos_map = {
                "d": "DET",
                "n": "NOUN",
                "v": "VERB",
                "a": "ADJ",
                "p": "PREP",
                "c": "CONJ",
                "r": "PRON",
                "m": "NUM",
                "q": "QUANT",
                "w": "ADV",
                "x": "NAME",
            }
            pos = pos_map.get(tok[0], "UNKNOWN")
            rows.append(
                {
                    "doc_id": doc_id,
                    "line_id": line_id,
                    "position": pos,
                    "token": tok,
                    "raw_token": tok,
                    "orientation": "normal",
                    "source": "synthetic_v2_pcgrammar",
                    "notes": f"true_pos={pos}",
                }
            )
    return pd.DataFrame(rows)


def save_corpus(
    sequences: list[list[str]], csv_path: str, json_path: str | None = None
):
    df = corpus_to_df(sequences)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} tokens ({len(sequences)} sentences) to {csv_path}")

    if json_path:
        stats = {
            "n_sentences": len(sequences),
            "n_tokens": len(df),
            "vocab_size": df["token"].nunique(),
            "type_token_ratio": round(df["token"].nunique() / len(df), 5),
            "top_10": df["token"].value_counts().head(10).to_dict(),
            "pos_distribution": df["notes"].value_counts().to_dict(),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved stats to {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic lost language corpus v2"
    )
    parser.add_argument("--n-sentences", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-csv", default="data/raw/lost_language/corpus_synthetic_v2.csv"
    )
    parser.add_argument(
        "--output-stats",
        default="data/raw/lost_language/corpus_synthetic_v2_stats.json",
    )
    args = parser.parse_args()

    sequences = generate_corpus(args.n_sentences, args.seed)
    save_corpus(sequences, args.output_csv, args.output_stats)


if __name__ == "__main__":
    main()
