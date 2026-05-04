"""Generate a Rongorongo-inspired synthetic corpus v2 with stronger structure.

This version introduces explicit bigram/trigram patterns to ensure
co-occurrence structure is detectable spectrally, while maintaining
Rongorongo-like surface features (skewed frequencies, doubling,
boustrophedon).

Pattern grammar:
- START -> [func1] [func2]
- BODY -> [common] [rare] [common] | [func] [common] [func]
- END -> [func1] [func2] [func3]
- Doubles/triples inserted at fixed positions
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

# Glyph inventory
GLYPHS = {
    "functional": [f"g{i:03d}" for i in range(1, 21)],
    "common": [f"g{i:03d}" for i in range(21, 61)],
    "rare": [f"g{i:03d}" for i in range(61, 121)],
}


def generate_line_v2(rng: random.Random) -> list[str]:
    """Generate one line with explicit bigram/trigram structure."""
    func = GLYPHS["functional"]
    com = GLYPHS["common"]
    rar = GLYPHS["rare"]

    # Pattern: [func func] [com rar com] [func func func] [com com] [func func]
    # With some variation
    patterns = [
        [
            rng.choice(func),
            rng.choice(func),
            rng.choice(com),
            rng.choice(rar),
            rng.choice(com),
            rng.choice(func),
            rng.choice(func),
            rng.choice(func),
            rng.choice(com),
            rng.choice(com),
            rng.choice(func),
            rng.choice(func),
        ],
        [
            rng.choice(func),
            rng.choice(func),
            rng.choice(func),
            rng.choice(com),
            rng.choice(rar),
            rng.choice(rar),
            rng.choice(com),
            rng.choice(func),
            rng.choice(func),
            rng.choice(com),
            rng.choice(com),
            rng.choice(com),
        ],
        [
            rng.choice(func),
            rng.choice(com),
            rng.choice(func),
            rng.choice(com),
            rng.choice(func),
            rng.choice(com),
            rng.choice(func),
            rng.choice(rar),
            rng.choice(func),
            rng.choice(com),
            rng.choice(func),
            rng.choice(com),
        ],
    ]
    line = rng.choice(patterns)

    # Insert doubles (15%) and triples (5%)
    result = []
    for tok in line:
        r = rng.random()
        if r < 0.05:
            result.extend([tok] * 3)
        elif r < 0.20:
            result.extend([tok] * 2)
        else:
            result.append(tok)
    return result


def generate_tablet_v2(n_lines: int = 20, seed: int = 42) -> list[list[str]]:
    rng = random.Random(seed)
    lines = []
    for line_id in range(1, n_lines + 1):
        line = generate_line_v2(rng)
        if line_id % 2 == 0:
            line = line[::-1]
        lines.append(line)
    return lines


def generate_corpus_v2(
    n_tablets: int = 15, lines_per_tablet: int = 20, seed: int = 42
) -> list[list[str]]:
    all_lines = []
    for t in range(n_tablets):
        tablet_seed = seed + t * 1000
        all_lines.extend(generate_tablet_v2(lines_per_tablet, tablet_seed))
    return all_lines


def corpus_to_df(sequences: list[list[str]]) -> pd.DataFrame:
    rows = []
    for line_id, seq in enumerate(sequences, start=1):
        tablet_id = (line_id - 1) // 20 + 1
        for pos, tok in enumerate(seq, start=1):
            rows.append(
                {
                    "doc_id": f"tablet_{tablet_id:03d}",
                    "line_id": line_id,
                    "position": pos,
                    "token": tok,
                    "raw_token": tok,
                    "orientation": "normal",
                    "source": "synthetic_rongorongo_v2",
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
    print(f"Saved {len(df)} tokens ({len(sequences)} lines) to {csv_path}")

    if json_path:
        stats = {
            "n_lines": len(sequences),
            "n_tokens": len(df),
            "vocab_size": df["token"].nunique(),
            "type_token_ratio": round(df["token"].nunique() / len(df), 5),
            "top_10": df["token"].value_counts().head(10).to_dict(),
            "tablets": df["doc_id"].nunique(),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved stats to {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Rongorongo-like v2 corpus")
    parser.add_argument("--n-tablets", type=int, default=15)
    parser.add_argument("--lines-per-tablet", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-csv", default="data/raw/lost_language/corpus_rongorongo_v2.csv"
    )
    parser.add_argument(
        "--output-stats",
        default="data/raw/lost_language/corpus_rongorongo_v2_stats.json",
    )
    args = parser.parse_args()

    sequences = generate_corpus_v2(args.n_tablets, args.lines_per_tablet, args.seed)
    save_corpus(sequences, args.output_csv, args.output_stats)


if __name__ == "__main__":
    main()
