"""Generate synthetic corpus with explicit positional bias.

This validates the positional bias detection method by creating a
controlled benchmark where certain sign classes have known positional
constraints (title signs at start, numeral signs at end).

Structure (analogous to Indus script descriptions):
- TITLE class: 8 signs, appear at position 0 with probability 0.8
- OBJECT class: 40 signs, appear in middle positions
- NUMERAL class: 8 signs, appear at last position with probability 0.8
- Optional: SIGNATURE class: 4 signs, appear at position -2 (penultimate)

Sequence length: 3-7 tokens (short, like Indus inscriptions).
Total sequences: ~2,000.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd


def zipf_ranks(n: int, alpha: float = 1.2) -> np.ndarray:
    """Generate Zipf-like frequency distribution."""
    ranks = np.arange(1, n + 1)
    probs = 1.0 / (ranks ** alpha)
    return probs / probs.sum()


def generate_positional_corpus(
    n_sequences: int = 2000,
    n_title: int = 8,
    n_object: int = 40,
    n_numeral: int = 8,
    n_signature: int = 4,
    min_len: int = 3,
    max_len: int = 7,
    title_start_prob: float = 0.8,
    numeral_end_prob: float = 0.8,
    signature_penult_prob: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate positional synthetic corpus."""
    random.seed(seed)
    np.random.seed(seed)

    title_signs = [f"T{i:02d}" for i in range(n_title)]
    object_signs = [f"O{i:02d}" for i in range(n_object)]
    numeral_signs = [f"N{i:02d}" for i in range(n_numeral)]
    signature_signs = [f"S{i:02d}" for i in range(n_signature)]

    title_probs = zipf_ranks(n_title)
    object_probs = zipf_ranks(n_object)
    numeral_probs = zipf_ranks(n_numeral)
    signature_probs = zipf_ranks(n_signature)

    rows = []
    seq_id = 0

    for _ in range(n_sequences):
        seq_id += 1
        length = random.randint(min_len, max_len)

        # Build sequence with positional constraints
        seq = []

        # Position 0: TITLE with high probability
        if random.random() < title_start_prob and length >= 1:
            seq.append(np.random.choice(title_signs, p=title_probs))
        else:
            seq.append(np.random.choice(object_signs, p=object_probs))

        # Middle positions: mostly OBJECT
        for pos in range(1, length - 1):
            # Small chance of TITLE or SIGNATURE in middle (noise)
            r = random.random()
            if r < 0.05:
                seq.append(np.random.choice(title_signs, p=title_probs))
            elif r < 0.08:
                seq.append(np.random.choice(signature_signs, p=signature_probs))
            else:
                seq.append(np.random.choice(object_signs, p=object_probs))

        # Last position: NUMERAL with high probability
        if length >= 2:
            if random.random() < numeral_end_prob:
                seq.append(np.random.choice(numeral_signs, p=numeral_probs))
            else:
                seq.append(np.random.choice(object_signs, p=object_probs))

        # If length >= 3 and not occupied, maybe add SIGNATURE at penultimate
        if length >= 3 and random.random() < signature_penult_prob:
            # Insert signature before last
            seq.insert(-1, np.random.choice(signature_signs, p=signature_probs))
            # Truncate to keep length
            seq = seq[:length]

        # Ensure exact length
        while len(seq) < length:
            seq.append(np.random.choice(object_signs, p=object_probs))
        seq = seq[:length]

        for pos, token in enumerate(seq, start=1):
            rows.append({
                "doc_id": f"pos_synthetic",
                "line_id": seq_id,
                "position": pos,
                "token": token,
                "source": "positional_synthetic_v1",
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate positional synthetic corpus")
    parser.add_argument("--output", default="data/raw/lost_language/corpus_positional_synthetic.csv")
    parser.add_argument("--n-sequences", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate_positional_corpus(n_sequences=args.n_sequences, seed=args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    vocab = df["token"].nunique()
    tokens = len(df)
    lines = df["line_id"].nunique()
    mean_len = tokens / lines

    print(f"Generated positional synthetic corpus")
    print(f"Sequences: {lines}")
    print(f"Total tokens: {tokens}")
    print(f"Vocabulary: {vocab}")
    print(f"Mean length: {mean_len:.2f}")
    print(f"Saved to {out_path}")

    # Quick stats by class
    for prefix, label in [("T", "TITLE"), ("O", "OBJECT"), ("N", "NUMERAL"), ("S", "SIGNATURE")]:
        subset = df[df["token"].str.startswith(prefix)]
        if len(subset) > 0:
            first_ratio = (subset["position"] == 1).mean()
            last_ratio = (subset.groupby("line_id")["position"].transform("max") == subset["position"]).mean()
            print(f"  {label}: {len(subset)} tokens, first_pos_ratio={first_ratio:.3f}, last_pos_ratio={last_ratio:.3f}")


if __name__ == "__main__":
    main()
