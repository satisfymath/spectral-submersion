"""Create a synthetic candidate language by transforming the synthetic lost language.

This creates a controlled benchmark where ground-truth correspondences are known:
- A known permutation is applied to the vocabulary.
- Some category collapse may be introduced (e.g., two determiners -> one).
- The result is saved as a candidate corpus CSV.
- Anchor pairs are saved for validation.

This does NOT claim to model a real language; it is a mathematical benchmark.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


def create_transformed_corpus(
    input_csv: str,
    output_csv: str,
    anchor_json: str,
    seed: int = 123,
    collapse_fraction: float = 0.0,
    noise_prob: float = 0.0,
) -> None:
    """Generate a transformed candidate corpus and ground-truth anchors."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    df = pd.read_csv(input_csv)
    vocab = sorted(df["token"].unique())
    n = len(vocab)

    # Create a random bijection (permutation) on the vocabulary
    permuted = vocab.copy()
    rng.shuffle(permuted)
    mapping = {old: new for old, new in zip(vocab, permuted)}

    # Optional: collapse some categories (e.g., d01->d02) to simulate polysemy
    if collapse_fraction > 0:
        n_collapse = max(1, int(n * collapse_fraction))
        for _ in range(n_collapse):
            src = rng.choice(vocab)
            # Collapse to another random token
            tgt = rng.choice(vocab)
            if src != tgt:
                mapping[src] = mapping.get(tgt, tgt)

    # Apply mapping
    df["token"] = df["token"].map(mapping)
    df["raw_token"] = df["token"]
    df["source"] = "synthetic_candidate_transformed"
    df.to_csv(output_csv, index=False)
    print(f"Saved transformed corpus to {output_csv}")

    # Build ground-truth anchors (only for tokens with a 1-to-1 mapping)
    anchors = []
    reverse = {}
    for old, new in mapping.items():
        reverse.setdefault(new, []).append(old)

    for old, new in mapping.items():
        if len(reverse[new]) == 1:
            anchors.append({"lost_token": old, "candidate_token": new})

    # Save anchors
    with open(anchor_json, "w", encoding="utf-8") as f:
        json.dump(anchors, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(anchors)} ground-truth anchors to {anchor_json}")

    # Save full mapping for debugging
    mapping_path = Path(anchor_json).with_suffix(".full_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"Saved full mapping to {mapping_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic candidate language"
    )
    parser.add_argument(
        "--input", default="data/raw/lost_language/corpus_synthetic_v2.csv"
    )
    parser.add_argument(
        "--output", default="data/raw/candidate_languages/synthetic_transformed.csv"
    )
    parser.add_argument(
        "--anchors", default="data/raw/candidate_languages/synthetic_anchors.json"
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--collapse", type=float, default=0.0, help="Fraction of vocab to collapse"
    )
    args = parser.parse_args()

    create_transformed_corpus(
        args.input, args.output, args.anchors, args.seed, args.collapse
    )


if __name__ == "__main__":
    main()
