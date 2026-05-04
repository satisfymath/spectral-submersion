"""Analyze repetition patterns in inscriptions.

Detects consecutive repeated signs (AA, AAA), sign pairs (ABAB), and other
repetition motifs common in symbolic writing systems (Rongorongo double/triple
repetition, Indus potential numeral repetition).

Metrics:
- Consecutive repeat rate: fraction of inscriptions with at least one double
- Triple repeat rate: fraction with at least one triple
- ABAB alternation rate
- Most repeated signs
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from spectral_submersion.tokenization import get_sequences_by_line


def find_repetitions(sequences: list[list[str]]) -> dict:
    """Find repetition patterns across all sequences."""
    stats = {
        "total_inscriptions": len(sequences),
        "double_repeat_count": 0,
        "triple_repeat_count": 0,
        "abab_repeat_count": 0,
        "any_repeat_count": 0,
        "repeat_details": [],
    }

    double_sign_counts = Counter()
    triple_sign_counts = Counter()
    abab_pair_counts = Counter()

    for seq in sequences:
        has_double = False
        has_triple = False
        has_abab = False

        # Consecutive repeats
        for i in range(len(seq) - 1):
            if seq[i] == seq[i + 1]:
                has_double = True
                double_sign_counts[seq[i]] += 1
                if i < len(seq) - 2 and seq[i] == seq[i + 2]:
                    has_triple = True
                    triple_sign_counts[seq[i]] += 1

        # ABAB pattern
        for i in range(len(seq) - 3):
            if (
                seq[i] == seq[i + 2]
                and seq[i + 1] == seq[i + 3]
                and seq[i] != seq[i + 1]
            ):
                has_abab = True
                abab_pair_counts[(seq[i], seq[i + 1])] += 1

        if has_double:
            stats["double_repeat_count"] += 1
        if has_triple:
            stats["triple_repeat_count"] += 1
        if has_abab:
            stats["abab_repeat_count"] += 1
        if has_double or has_abab:
            stats["any_repeat_count"] += 1

    stats["double_repeat_rate"] = stats["double_repeat_count"] / max(
        stats["total_inscriptions"], 1
    )
    stats["triple_repeat_rate"] = stats["triple_repeat_count"] / max(
        stats["total_inscriptions"], 1
    )
    stats["abab_repeat_rate"] = stats["abab_repeat_count"] / max(
        stats["total_inscriptions"], 1
    )
    stats["any_repeat_rate"] = stats["any_repeat_count"] / max(
        stats["total_inscriptions"], 1
    )
    stats["top_double_signs"] = double_sign_counts.most_common(10)
    stats["top_triple_signs"] = triple_sign_counts.most_common(10)
    stats["top_abab_pairs"] = abab_pair_counts.most_common(10)

    return stats


def analyze_by_length(sequences: list[list[str]]) -> pd.DataFrame:
    """Analyze repetition rates by inscription length."""
    rows = []
    for length in sorted(set(len(s) for s in sequences)):
        subset = [s for s in sequences if len(s) == length]
        reps = find_repetitions(subset)
        rows.append(
            {
                "length": length,
                "n_inscriptions": len(subset),
                "double_repeat_rate": reps["double_repeat_rate"],
                "triple_repeat_rate": reps["triple_repeat_rate"],
                "abab_repeat_rate": reps["abab_repeat_rate"],
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze repetition patterns in inscriptions"
    )
    parser.add_argument(
        "--input", default="data/raw/lost_language/corpus_indus_real.csv"
    )
    parser.add_argument(
        "--output-table", default="reports/tables/repetition_analysis.csv"
    )
    parser.add_argument(
        "--output-by-length", default="reports/tables/repetition_by_length.csv"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    sequences = get_sequences_by_line(df)

    stats = find_repetitions(sequences)

    print("=" * 60)
    print("REPETITION PATTERN ANALYSIS")
    print("=" * 60)
    print(f"Total inscriptions: {stats['total_inscriptions']}")
    print(
        f"Any repeat (double or ABAB): {stats['any_repeat_count']} ({stats['any_repeat_rate']:.3f})"
    )
    print(
        f"Double repeat (AA): {stats['double_repeat_count']} ({stats['double_repeat_rate']:.3f})"
    )
    print(
        f"Triple repeat (AAA): {stats['triple_repeat_count']} ({stats['triple_repeat_rate']:.3f})"
    )
    print(
        f"ABAB pattern: {stats['abab_repeat_count']} ({stats['abab_repeat_rate']:.3f})"
    )
    print()
    print("Top signs in double repeats:")
    for sign, count in stats["top_double_signs"]:
        print(f"  {sign}: {count}")
    print("Top signs in triple repeats:")
    for sign, count in stats["top_triple_signs"]:
        print(f"  {sign}: {count}")
    print("Top ABAB pairs:")
    for pair, count in stats["top_abab_pairs"]:
        print(f"  {pair}: {count}")

    # Save summary
    summary_df = pd.DataFrame(
        [
            {
                "total_inscriptions": stats["total_inscriptions"],
                "any_repeat_count": stats["any_repeat_count"],
                "any_repeat_rate": stats["any_repeat_rate"],
                "double_repeat_count": stats["double_repeat_count"],
                "double_repeat_rate": stats["double_repeat_rate"],
                "triple_repeat_count": stats["triple_repeat_count"],
                "triple_repeat_rate": stats["triple_repeat_rate"],
                "abab_repeat_count": stats["abab_repeat_count"],
                "abab_repeat_rate": stats["abab_repeat_rate"],
            }
        ]
    )
    Path(args.output_table).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output_table, index=False)
    print(f"\nSaved summary to {args.output_table}")

    by_length = analyze_by_length(sequences)
    by_length.to_csv(args.output_by_length, index=False)
    print(f"Saved by-length analysis to {args.output_by_length}")
    print(by_length.to_string(index=False))


if __name__ == "__main__":
    main()
