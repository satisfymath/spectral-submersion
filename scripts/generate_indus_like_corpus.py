"""Generate Indus-script-inspired synthetic corpus.

Structural features from published descriptions (Parpola 1994,
Mahadevan 1977, Farmer et al. 2004):
- ~400-500 distinct signs (but many are compounds)
- Very short texts (average 5 signs per seal/inscription)
- High repetition of certain signs (probable numerals, titles)
- Strong positional constraints (signs appear in fixed positions)
- No clear word dividers
- Many texts are just 1-3 signs (very low resource)

This is a methodological benchmark, not a claim about Indus.
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

# Indus-like sign inventory
SIGNS = {
    "title": [
        f"i{i:03d}" for i in range(1, 11)
    ],  # 10 very frequent (possible titles/names)
    "numeral": [f"i{i:03d}" for i in range(11, 21)],  # 10 frequent (possible numerals)
    "object": [
        f"i{i:03d}" for i in range(21, 61)
    ],  # 40 medium (possible objects/animals)
    "rare": [f"i{i:03d}" for i in range(61, 151)],  # 90 rare
}


def generate_inscription(rng: random.Random) -> list[str]:
    """Generate one short inscription with positional constraints."""
    title = SIGNS["title"]
    num = SIGNS["numeral"]
    obj = SIGNS["object"]
    rare = SIGNS["rare"]

    # Patterns inspired by seal compositions:
    # [title] [object] [numeral]
    # [title] [title] [object]
    # [object] [rare]
    # [title] [numeral] [numeral]
    patterns = [
        [rng.choice(title), rng.choice(obj), rng.choice(num)],
        [rng.choice(title), rng.choice(title), rng.choice(obj)],
        [rng.choice(obj), rng.choice(rare)],
        [rng.choice(title), rng.choice(num), rng.choice(num)],
        [rng.choice(title), rng.choice(obj), rng.choice(rare)],
        [rng.choice(title), rng.choice(num)],
    ]
    return rng.choice(patterns)


def generate_corpus(n_inscriptions: int = 2000, seed: int = 42) -> list[list[str]]:
    rng = random.Random(seed)
    return [generate_inscription(rng) for _ in range(n_inscriptions)]


def corpus_to_df(sequences: list[list[str]]) -> pd.DataFrame:
    rows = []
    for line_id, seq in enumerate(sequences, start=1):
        for pos, tok in enumerate(seq, start=1):
            rows.append(
                {
                    "doc_id": f"seal_{(line_id-1)//50 + 1:03d}",
                    "line_id": line_id,
                    "position": pos,
                    "token": tok,
                    "raw_token": tok,
                    "orientation": "normal",
                    "source": "synthetic_indus_like",
                    "notes": "",
                }
            )
    return pd.DataFrame(rows)


def save_corpus(
    sequences: list[list[str]], csv_path: str, json_path: str | None = None
):
    df = corpus_to_df(sequences)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} tokens ({len(sequences)} inscriptions) to {csv_path}")

    if json_path:
        stats = {
            "n_inscriptions": len(sequences),
            "n_tokens": len(df),
            "vocab_size": df["token"].nunique(),
            "type_token_ratio": round(df["token"].nunique() / len(df), 5),
            "top_10": df["token"].value_counts().head(10).to_dict(),
            "mean_length": df.groupby("line_id").size().mean(),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved stats to {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Indus-like synthetic corpus")
    parser.add_argument("--n-inscriptions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-csv", default="data/raw/lost_language/corpus_indus_like.csv"
    )
    parser.add_argument(
        "--output-stats", default="data/raw/lost_language/corpus_indus_like_stats.json"
    )
    args = parser.parse_args()

    sequences = generate_corpus(args.n_inscriptions, args.seed)
    save_corpus(sequences, args.output_csv, args.output_stats)


if __name__ == "__main__":
    main()
