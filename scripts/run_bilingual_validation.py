"""Run bilingual validation experiments on known language pairs.

This script validates the spectral submersion pipeline on language pairs
where we know the ground truth (e.g., English-French translation). The key
question: can the method recover known correspondences under conditions
similar to Rongorongo?

Anchors are TRUE cognates (identical strings in both languages, like
'restaurant' in English and French). We split them into alignment anchors
and evaluation anchors.

Results are saved to runs/bilingual_validation/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from spectral_submersion.bilingual_validation import (
    build_bilingual_corpus,
    restricted_bilingual_experiment,
)

RUN_DIR = Path("runs/bilingual_validation")
FIG_DIR = RUN_DIR / "figures"
TABLE_DIR = RUN_DIR / "tables"


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BILINGUAL VALIDATION: Known-Language Pipeline Test")
    print("Cognate anchors = identical strings (e.g., 'restaurant' in EN/FR)")
    print("=" * 70)

    langs = {
        "en-fr": ("data/raw/candidate_languages/english_tokens.csv",
                   "data/raw/candidate_languages/french_tokens.csv"),
        "en-es": ("data/raw/candidate_languages/english_tokens.csv",
                   "data/raw/candidate_languages/spanish_tokens.csv"),
        "en-de": ("data/raw/candidate_languages/english_tokens.csv",
                   "data/raw/candidate_languages/german_tokens.csv"),
    }

    all_results = {}

    for pair_name, (src_path, tgt_path) in langs.items():
        print(f"\n{'=' * 70}")
        print(f"Language pair: {pair_name}")
        print(f"{'=' * 70}")

        src_df = pd.read_csv(src_path)
        tgt_df = pd.read_csv(tgt_path)

        print(f"Source: {len(src_df)} tokens, {src_df['line_id'].nunique()} lines")
        print(f"Target: {len(tgt_df)} tokens, {tgt_df['line_id'].nunique()} lines")

        conditions = [
            {
                "name": "Full (V=2000, T=50k, 10% anchors)",
                "max_vocab": 2000,
                "max_tokens": 50000,
                "anchor_fraction": 0.10,
            },
            {
                "name": "Medium (V=500, T=20k, 10% anchors)",
                "max_vocab": 500,
                "max_tokens": 20000,
                "anchor_fraction": 0.10,
            },
            {
                "name": "Small (V=200, T=5k, 10% anchors)",
                "max_vocab": 200,
                "max_tokens": 5000,
                "anchor_fraction": 0.10,
            },
            {
                "name": "Indus-like (V=180, T=1k, 5% anchors)",
                "max_vocab": 180,
                "max_tokens": 1000,
                "anchor_fraction": 0.05,
            },
            {
                "name": "RR-like (V=941, T=5.5k, 5% anchors)",
                "max_vocab": 941,
                "max_tokens": 5500,
                "anchor_fraction": 0.05,
            },
            {
                "name": "RR-like few (V=941, T=5.5k, 2% anchors)",
                "max_vocab": 941,
                "max_tokens": 5500,
                "anchor_fraction": 0.02,
            },
        ]

        results = restricted_bilingual_experiment(
            src_df, tgt_df,
            conditions=conditions,
            window_size=3,
            k=16,
            n_bootstrap=50,
            seed=42,
        )

        all_results[pair_name] = results

        print(f"\n{pair_name} Summary:")
        print(f"{'Condition':<55} {'Acc@1':>7} {'Acc@5':>7} {'Acc@10':>8} {'MRR':>7} "
              f"{'Rel_s':>6} {'Rel_t':>6} {'EPC':>7} {'Anch':>5} {'Eval':>5}")
        print("-" * 115)
        for r in results:
            name = r["condition_name"][:53]
            print(f"{name:<55} {r['acc_at_k'][1]:>7.4f} {r['acc_at_k'][5]:>7.4f} "
                  f"{r['acc_at_k'][10]:>8.4f} {r['mrr']:>7.4f} "
                  f"{r['src_spectral_reliability']:>6.3f} {r['tgt_spectral_reliability']:>6.3f} "
                  f"{r['src_epc']:>7.4f} {r['n_alignment_anchors']:>5} {r['n_eval_pairs']:>5}")

    with open(RUN_DIR / "bilingual_validation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {RUN_DIR / 'bilingual_validation_results.json'}")

    print("\n" + "=" * 70)
    print("CROSS-LANGUAGE COMPARISON TABLE")
    print("=" * 70)
    print(
        f"{'Pair':<8} {'Condition':<45} {'Acc@1':>7} {'Acc@5':>7} {'MRR':>7} "
        f"{'Rel_s':>6} {'Anch':>5} {'EPC':>7}"
    )
    print("-" * 95)
    for pair_name, results in all_results.items():
        for r in results:
            name = r["condition_name"][:43]
            print(
                f"{pair_name:<8} {name:<45} {r['acc_at_k'][1]:>7.4f} {r['acc_at_k'][5]:>7.4f} "
                f"{r['mrr']:>7.4f} {r['src_spectral_reliability']:>6.3f} "
                f"{r['n_alignment_anchors']:>5} {r['src_epc']:>7.4f}"
            )

    print("\n" + "=" * 70)
    print("REFERENCE: Synthetic / Rongorongo results from PhD audit v2:")
    print("  PCFG_v2:   V=112, T=24324, Rel_4=0.96, Acc@1=0.089 (17.9% anchors)")
    print("  RR_real:   V=941, T=5460,   Rel_k=0 for all k")
    print("  Indus:     V=182, T=1003,   Rel_4=0.097")
    print("=" * 70)

    print("\nDone.")


if __name__ == "__main__":
    main()