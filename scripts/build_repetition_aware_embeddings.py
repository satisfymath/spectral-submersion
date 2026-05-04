"""Build repetition-aware spectral embeddings for a corpus.

Collapses consecutive repetitions (AA, AAA, etc.) into pattern-aware tokens
before building co-occurrence/PPMI/SVD embeddings. This is motivated by
Rongorongo's 66% double-repeat and 18% triple-repeat rates, which dominate
window-based co-occurrence and obscure structural signal.

Produces three embedding sets:
1. collapsed: consecutive repeats collapsed to token_REPn
2. abab_aware: collapsed + ABAB patterns marked as composite tokens
3. pure (baseline): original uncollapsed tokens

Usage:
    PYTHONPATH=src python scripts/build_repetition_aware_embeddings.py \
        --input data/raw/lost_language/corpus_rongorongo_real.xml.csv \
        --output-dir data/processed \
        --k 16 --alpha 0.5 --window 3
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spectral_submersion.cooccurrence import (
    build_vocab,
    cooccurrence_matrix_from_sequences,
    directional_cooccurrence_matrix_from_sequences,
)
from spectral_submersion.pmi import ppmi_matrix
from spectral_submersion.spectral import spectral_embedding, effective_rank
from spectral_submersion.tokenization import (
    get_repetition_aware_sequences,
    get_abab_aware_sequences,
    get_sequences_by_line,
    tokens_to_ids,
)


def build_and_save(
    sequences,
    label,
    output_dir,
    k,
    alpha,
    window,
    seed,
    directional=False,
):
    all_tokens = [t for seq in sequences for t in seq]
    vocab = build_vocab(all_tokens)
    seq_ids = [tokens_to_ids(seq, vocab) for seq in sequences]

    if directional:
        C_left, C_right = directional_cooccurrence_matrix_from_sequences(
            seq_ids, vocab_size=len(vocab), window_size=window, inverse_distance=True
        )
        M_left = ppmi_matrix(C_left, epsilon=1e-9)
        M_right = ppmi_matrix(C_right, epsilon=1e-9)
        M = np.hstack([M_left, M_right])
    else:
        C = cooccurrence_matrix_from_sequences(
            seq_ids, vocab_size=len(vocab), window_size=window, inverse_distance=True
        )
        M = ppmi_matrix(C, epsilon=1e-9)

    E, S, Vt = spectral_embedding(M, k=k, alpha=alpha, random_state=seed)
    r_eff = effective_rank(S)

    np.save(output_dir / f"embeddings_{label}.npy", E)
    np.save(output_dir / f"sv_{label}.npy", S)
    with open(
        output_dir / f"embeddings_{label}.vocab.json", "w", encoding="utf-8"
    ) as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"[{label}] vocab={len(vocab)} shape={E.shape} r_eff={r_eff:.4f}")
    return r_eff


def main():
    parser = argparse.ArgumentParser(
        description="Build repetition-aware spectral embeddings"
    )
    parser.add_argument(
        "--input", required=True, help="Input CSV with doc_id, line_id, position, token"
    )
    parser.add_argument(
        "--output-dir", default="data/processed", help="Output directory"
    )
    parser.add_argument("--k", type=int, default=16, help="Embedding dimension")
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Singular value exponent"
    )
    parser.add_argument(
        "--window", type=int, default=3, help="Co-occurrence window size"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-repeat", type=int, default=4, help="Maximum repeat count to encode"
    )
    parser.add_argument(
        "--directional", action="store_true", help="Use directional co-occurrence"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    collapsed_seqs, pure_seqs = get_repetition_aware_sequences(
        df, max_repeat=args.max_repeat
    )
    r_eff_collapsed = build_and_save(
        collapsed_seqs,
        "rongorongo_real_collapsed",
        output_dir,
        args.k,
        args.alpha,
        args.window,
        args.seed,
        args.directional,
    )
    results["collapsed"] = {
        "r_eff": r_eff_collapsed,
        "vocab_size": len(set(t for s in collapsed_seqs for t in s)),
    }

    abab_seqs = get_abab_aware_sequences(df)
    r_eff_abab = build_and_save(
        abab_seqs,
        "rongorongo_real_abab",
        output_dir,
        args.k,
        args.alpha,
        args.window,
        args.seed,
        args.directional,
    )
    results["abab_aware"] = {
        "r_eff": r_eff_abab,
        "vocab_size": len(set(t for s in abab_seqs for t in s)),
    }

    r_eff_pure = build_and_save(
        pure_seqs,
        "rongorongo_real_pure",
        output_dir,
        args.k,
        args.alpha,
        args.window,
        args.seed,
        args.directional,
    )
    results["pure"] = {
        "r_eff": r_eff_pure,
        "vocab_size": len(set(t for s in pure_seqs for t in s)),
    }

    print("\n=== Embedding Comparison ===")
    print(f"{'Model':<30} {'Vocab':>6} {'r_eff':>8}")
    print("-" * 48)
    for label in ["pure", "collapsed", "abab_aware"]:
        r = results[label]
        print(f"{label:<30} {r['vocab_size']:>6} {r['r_eff']:>8.4f}")

    comparison_path = output_dir / "repetition_aware_comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nComparison saved to {comparison_path}")


if __name__ == "__main__":
    main()
