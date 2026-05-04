"""Run PhD-level stability analysis: spectral reliability, co-occurrence coverage, SPPMI sensitivity.

This script implements the mandatory stability checks from Sections 5-7 of the guide:
- SpectralReliability_k for each k
- Co-occurrence concentration analysis
- SPPMI sensitivity sweep
- Spectral rejection rule assessment
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, "src")

from spectral_submersion.io import load_config
from spectral_submersion.tokenization import (
    read_corpus,
    build_vocab,
    tokens_to_ids,
    get_sequences_by_line,
)
from spectral_submersion.cooccurrence import cooccurrence_matrix_from_sequences
from spectral_submersion.pmi import ppmi_matrix
from spectral_submersion.stability import (
    spectral_gap,
    spectral_reliability,
    spectral_stability_bootstrap,
    cooccurrence_coverage,
    expected_pair_count,
    sceptmi_matrix,
    pmi_sensitivity,
    spectral_rejection_rule,
    min_tokens_for_coverage,
)
from spectral_submersion.evaluation import permute_corpus, random_corpus_same_frequency


def main():
    parser = argparse.ArgumentParser(description="PhD stability analysis")
    parser.add_argument("--config", default="configs/phd_upgrade.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path("runs") / "stability"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PhD Stability Analysis")
    print("=" * 60)

    corpus_configs = [
        ("data/raw/lost_language/corpus_synthetic_v2.csv", "synthetic_v2"),
        ("data/raw/lost_language/corpus_rongorongo_v2.csv", "rongorongo_v2"),
    ]

    results = {}

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
        total_tokens = len(token_ids)

        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Total tokens: {total_tokens}")

        for window_size in config.get("cooccurrence", {}).get("window_sizes", [3]):
            C = cooccurrence_matrix_from_sequences(
                sequences, vocab_size, window_size=window_size
            )
            cov = cooccurrence_coverage(C)
            epc = expected_pair_count(total_tokens, window_size, vocab_size)
            min_tokens = min_tokens_for_coverage(vocab_size, window_size)

            coverage_result = {
                "vocab_size": vocab_size,
                "total_tokens": total_tokens,
                "window_size": window_size,
                "cooccurrence_coverage": float(cov),
                "expected_pair_count": float(epc),
                "min_tokens_for_coverage": int(min_tokens),
            }
            print(
                f"  Window={window_size}: coverage={cov:.4f}, EPC={epc:.3f}, min_tokens={min_tokens}"
            )

            results[f"{corpus_name}_w{window_size}_coverage"] = coverage_result

            for alpha in [0.75]:
                PPMI = ppmi_matrix(C, alpha=alpha)

                p_ij = PPMI / (PPMI.sum() + 1e-15)
                p_i = p_ij.sum(axis=1, keepdims=True)
                p_j = p_ij.sum(axis=0, keepdims=True)
                sens = pmi_sensitivity(p_ij, p_i, p_j)
                print(
                    f"  PMI sensitivity: max={sens['max_sensitivity']:.1f}, "
                    f"mean={sens['mean_sensitivity']:.1f}, "
                    f"at_risk={sens['pairs_at_risk']:.4f}"
                )

                results[f"{corpus_name}_w{window_size}_pmi_sensitivity"] = sens

                for epsilon in [0.01, 0.1, 1.0]:
                    SPPMI = sceptmi_matrix(
                        C, epsilon=epsilon, prior_type="marginal_product"
                    )
                    results[f"{corpus_name}_w{window_size}_sppmi_eps{epsilon}"] = {
                        "epsilon": epsilon,
                        "sparsity": float(np.mean(SPPMI == 0)),
                        "mean_nonzero": (
                            float(SPPMI[SPPMI > 0].mean()) if np.any(SPPMI > 0) else 0.0
                        ),
                    }

            for k in config.get("spectral", {}).get("k_values", [8, 16]):
                print(f"  Running bootstrap SVD stability for k={k}...")
                boot_result = spectral_stability_bootstrap(
                    sequences,
                    vocab_size,
                    k=k,
                    window_size=window_size,
                    n_bootstrap=config.get("spectral", {}).get(
                        "bootstrap_samples", 200
                    ),
                    alpha=alpha,
                )
                print(
                    f"    delta_k={boot_result['delta_k_mean']:.4f}, "
                    f"epsilon={boot_result['epsilon_hat']:.4f}, "
                    f"reliability={boot_result['spectral_reliability']:.4f}, "
                    f"stable={boot_result['reliable']}"
                )

                results[f"{corpus_name}_w{window_size}_k{k}_stability"] = boot_result

                sv = np.array(boot_result["singular_values_mean"])
                eps = np.array(boot_result["singular_values_std"])
                rejection = spectral_rejection_rule(sv, eps, k_values=[k])
                results[f"{corpus_name}_w{window_size}_k{k}_rejection"] = rejection

    output_path = output_dir / "stability_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
