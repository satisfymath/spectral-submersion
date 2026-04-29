"""Run PhD-level experiments: permutation recovery, logosyllabic collapse, boustrophedon, calendar.

Implements the experimental protocols from Sections 17-22 of the guide.
Each experiment produces auditable results with metrics and claim-level limits.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, "src")

from spectral_submersion.io import load_config
from spectral_submersion.tokenization import read_corpus, build_vocab, tokens_to_ids, get_sequences_by_line
from spectral_submersion.synthetic_experiments import (
    experiment_permutation_recovery,
    experiment_logosyllabic_collapse,
    experiment_boustrophedon_direction,
    experiment_calendar_model,
    find_parallel_passages,
    generate_permuted_corpus,
    generate_collapsed_corpus,
)
from spectral_submersion.audit_metrics import negative_control_gap


def main():
    parser = argparse.ArgumentParser(description="PhD experiments")
    parser.add_argument("--config", default="configs/phd_upgrade.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path("runs") / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PhD Synthetic Experiments")
    print("=" * 60)

    all_results = {}

    corpus_configs = [
        ("data/raw/lost_language/corpus_synthetic_v2.csv", "synthetic_v2"),
    ]

    for corpus_path, corpus_name in corpus_configs:
        if not Path(corpus_path).exists():
            print(f"  Skipping {corpus_name}: file not found")
            continue

        print(f"\n--- Corpus: {corpus_name} ---")
        df = read_corpus(corpus_path)
        tokens = df["token"].tolist()
        vocab = build_vocab(tokens)
        vocab_size = len(vocab)
        token_ids = tokens_to_ids(tokens, vocab)
        sequences = get_sequences_by_line(df)

        exp_config = config.get("experiments", {})

        # Experiment 1: Permutation Recovery
        if exp_config.get("permutation_recovery", {}).get("enabled", True):
            print("\n  == Experiment 1: Permutation Recovery ==")
            permuted, perm_map = generate_permuted_corpus(
                sequences, vocab_size, seed=42
            )
            for n_anch in exp_config.get("permutation_recovery", {}).get("n_anchors", [5, 10, 20]):
                print(f"    n_anchors={n_anch}")
                try:
                    result = experiment_permutation_recovery(
                        permuted, sequences, vocab_size, vocab_size,
                        n_anchors=n_anch, seed=42,
                    )
                    all_results[f"perm_n{n_anch}"] = result
                    print(f"    Acc@1={result['acc_at_k'][1]:.3f}, "
                          f"Acc@5={result['acc_at_k'][5]:.3f}, "
                          f"MRR={result['mrr']:.3f}")
                except Exception as e:
                    print(f"    FAILED: {e}")

        # Experiment 4: Boustrophedon Direction
        if exp_config.get("boustrophedon", {}).get("enabled", True):
            print("\n  == Experiment 4: Boustrophedon Direction ==")
            result = experiment_boustrophedon_direction(sequences, vocab_size)
            all_results["boustrophedon"] = {
                "direction_accuracy": result["direction_accuracy"],
                "forward_wins": result["forward_wins"],
                "n_lines": result["n_lines"],
            }
            print(f"    Direction accuracy: {result['direction_accuracy']:.3f}")

        # Experiment 5: Parallel Passages
        if exp_config.get("parallel_passages", {}).get("enabled", True):
            print("\n  == Experiment 5: Parallel Passages ==")
            threshold = exp_config.get("parallel_passages", {}).get(
                "edit_similarity_threshold", 0.7
            )
            parallels = find_parallel_passages(
                sequences, edit_distance_threshold=1.0 - threshold
            )
            all_results["parallel_passages"] = {
                "n_parallels": len(parallels),
                "threshold": threshold,
            }
            if parallels:
                all_results["parallel_passages"]["mean_similarity"] = float(
                    np.mean([p["edit_similarity"] for p in parallels])
                )
            print(f"    Found {len(parallels)} parallel passages")

        # Experiment 6: Calendar Model
        if exp_config.get("calendar_model", {}).get("enabled", True):
            print("\n  == Experiment 6: Calendar Model ==")
            n_phases = exp_config.get("calendar_model", {}).get("n_lunar_phases", 30)
            result = experiment_calendar_model(sequences, vocab_size, n_lunar_phases=n_phases)
            all_results["calendar"] = {
                "ngram_bic": result["ngram_bic"],
                "calendar_bic": result["calendar_bic"],
                "delta_bic": result["delta_bic"],
                "calendar_preferred": result["calendar_preferred"],
            }
            print(f"    n-gram BIC={result['ngram_bic']:.1f}, "
                  f"calendar BIC={result['calendar_bic']:.1f}, "
                  f"delta={result['delta_bic']:.1f}, "
                  f"preferred={'calendar' if result['calendar_preferred'] else 'n-gram'}")

    output_path = output_dir / "experiment_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()