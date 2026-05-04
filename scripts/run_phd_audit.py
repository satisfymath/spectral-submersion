"""Run PhD-level audit: negative controls, identifiability verification, and metric reports.

Produces the mandatory audit tables from Part V of the guide.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

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
from spectral_submersion.spectral import spectral_embedding
from spectral_submersion.identifiability import (
    verify_non_identifiability,
    anchor_power,
    compute_automorphism_size_upper_bound,
)
from spectral_submersion.evaluation import (
    permute_corpus,
    random_corpus_same_frequency,
    random_corpus_uniform,
    relational_distortion,
)
from spectral_submersion.audit_metrics import negative_control_gap, bootstrap_stability


def main():
    parser = argparse.ArgumentParser(description="PhD audit report")
    parser.add_argument("--config", default="configs/phd_upgrade.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path("runs") / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PhD Audit Report")
    print("=" * 60)

    audit_results = {}

    corpus_configs = [
        ("data/raw/lost_language/corpus_synthetic_v2.csv", "synthetic_v2"),
    ]

    for corpus_path, corpus_name in corpus_configs:
        if not Path(corpus_path).exists():
            continue

        print(f"\n--- Audit for: {corpus_name} ---")
        df = read_corpus(corpus_path)
        tokens = df["token"].tolist()
        vocab = build_vocab(tokens)
        vocab_size = len(vocab)
        token_ids = tokens_to_ids(tokens, vocab)
        sequences = get_sequences_by_line(df)

        # === Section 23: Negative Control Gap ===
        print("\n  == Negative Control Gap ==")
        C_real = cooccurrence_matrix_from_sequences(
            sequences, vocab_size, window_size=3
        )
        M_real = ppmi_matrix(C_real)
        E_real, sv_real, _ = spectral_embedding(M_real, k=16)

        real_score = float(np.sum(sv_real[:4]))
        print(f"    Real corpus score (sum of top-4 SV): {real_score:.4f}")

        neg_scores = []
        for ctrl_type, ctrl_fn in [
            (
                "permuted",
                lambda: permute_corpus([list(map(str, seq)) for seq in sequences]),
            ),
            (
                "same_freq",
                lambda: random_corpus_same_frequency(
                    [list(map(str, seq)) for seq in sequences]
                ),
            ),
        ]:
            print(f"    Running {ctrl_type} negative control...")
            ctrl_sequences = ctrl_fn()
            ctrl_token_ids = [
                tokens_to_ids([t for t in seq], vocab) for seq in ctrl_sequences
            ]
            ctrl_flat = [t for seq in ctrl_token_ids for t in seq if t >= 0]
            if len(ctrl_flat) > 10:
                C_ctrl = cooccurrence_matrix_from_sequences(
                    ctrl_token_ids, vocab_size, window_size=3
                )
                M_ctrl = ppmi_matrix(C_ctrl)
                _, sv_ctrl, _ = spectral_embedding(M_ctrl, k=16)
                ctrl_score = float(np.sum(sv_ctrl[:4]))
                neg_scores.append(ctrl_score)

        if neg_scores:
            gap_result = negative_control_gap(real_score, np.array(neg_scores))
            audit_results["neg_ctrl_gap"] = gap_result
            print(
                f"    NegCtrlGap = {gap_result['gap']:.3f} ({gap_result['interpretation']})"
            )

        # === Section 24: Identifiability Verification ===
        print("\n  == Identifiability Verification ==")

        def structural_stat(c):
            C_test = cooccurrence_matrix_from_sequences(
                [c.tolist()], vocab_size, window_size=2
            )
            s = np.linalg.svd(C_test, compute_uv=False)
            return s[:5]

        ident_result = verify_non_identifiability(
            vocab_size, structural_stat, np.array(token_ids), n_permutations=20, seed=42
        )
        audit_results["identifiability"] = {
            "is_invariant": ident_result["is_invariant"],
            "max_deviation": float(ident_result["max_deviation"]),
        }
        print(f"    Structural statistic invariant: {ident_result['is_invariant']}")
        print(
            f"    Max deviation under permutation: {ident_result['max_deviation']:.2e}"
        )

        # === Automorphism analysis ===
        print("\n  == Automorphism Group Analysis ==")
        aut_size = compute_automorphism_size_upper_bound(C_real)
        audit_results["automorphism_upper_bound"] = int(aut_size)
        print(f"    |Aut(G_X)| upper bound: {aut_size}")

        # === Coverage metrics ===
        from spectral_submersion.stability import (
            cooccurrence_coverage,
            expected_pair_count,
        )

        cov = cooccurrence_coverage(C_real)
        epc = expected_pair_count(sum(len(s) for s in sequences), 3, vocab_size)
        audit_results["coverage"] = {
            "cooccurrence_coverage": float(cov),
            "expected_pair_count": float(epc),
        }
        print(f"\n  Co-occurrence coverage: {cov:.4f}")
        print(f"  Expected pair count: {epc:.3f}")
        if epc < 1.0:
            print(
                "  WARNING: ExpectedPairCount < 1: co-occurrence matrix is statistically weak!"
            )
            print("  PPMI will be noisy; spectral dimensions may be unstable.")

    audit_path = output_dir / "audit_results.json"
    with open(audit_path, "w") as f:
        json.dump(audit_results, f, indent=2, default=str)
    print(f"\nAudit results saved to {audit_path}")

    print("\n" + "=" * 60)
    print("MANDATORY AUDIT CHECKLIST (Section XI):")
    print("=" * 60)
    for item in [
        "1. What is the mathematical object? -> Co-occurrence matrix -> PPMI -> SVD embedding",
        "2. What is the hypothesis space? -> Permutation orbits under Sym(V_X)",
        "3. What symmetries make it non-identifiable? -> Renaming invariance (Theorem 3.2)",
        "4. What anchors break which symmetries? -> See AnchorPower metric",
        "5. What is the stability metric? -> SpectralReliability_k via bootstrap",
        "6. What is the negative control? -> Permuted and same-frequency corpora",
        "7. What is the max claim level permitted? -> C2 without external anchors",
        "8. What happens if window/dimension/smoothing/prior change? -> See sensitivity analysis",
        "9. What evidence contradicts the hypothesis? -> See counterevidence in ledger",
        "10. Where is the reproducible JSON? -> runs/audit/audit_results.json",
    ]:
        print(f"  {item}")


if __name__ == "__main__":
    main()
