"""Generate auditable hypothesis ledger from existing alignment results.

Loads transport coupling matrices and generates hypothesis entries
with full claim-level auditing, evidence, and counterevidence.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from spectral_submersion.io import load_config
from spectral_submersion.generative_model import (
    RongorongoGenerativeModel,
    GenerativeConfig,
)
from spectral_submersion.claims import ClaimLevel


def main():
    parser = argparse.ArgumentParser(description="Generate PhD hypothesis ledger")
    parser.add_argument("--config", default="configs/phd_upgrade.yaml")
    parser.add_argument("--output-dir", default="runs/ledger")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PhD Hypothesis Ledger Generation")
    print("=" * 60)

    claims_config = config.get("claims", {})
    max_level_str = claims_config.get(
        "max_claim_level_without_external_anchor", "C2_FUNCTIONAL"
    )
    max_level = ClaimLevel[max_level_str]

    gen_config = GenerativeConfig(
        max_claim_level=max_level,
        require_negative_controls=claims_config.get("require_negative_control_gap", 2.0)
        > 0,
        block_c5_without_external=claims_config.get(
            "block_c5_without_external_evidence", True
        ),
        min_bootstrap_stability=claims_config.get("require_bootstrap_stability", 0.7),
        min_negative_control_gap=claims_config.get("require_negative_control_gap", 2.0),
    )

    model = RongorongoGenerativeModel(gen_config)

    demo_coupling = np.array(
        [
            [0.7, 0.1, 0.1, 0.05, 0.05],
            [0.1, 0.6, 0.15, 0.1, 0.05],
            [0.05, 0.2, 0.5, 0.15, 0.1],
            [0.05, 0.05, 0.1, 0.6, 0.2],
            [0.05, 0.05, 0.15, 0.25, 0.5],
        ]
    )
    source_tokens = ["200", "076", "380", "010", "052"]
    target_tokens = ["ra", "ki", "ma", "te", "toa"]

    scenarios = [
        {
            "name": "no_anchors",
            "anchor_power": 0.0,
            "stability": 0.3,
            "neg_ctrl_gap": 1.5,
            "spectral_reliability": 0.2,
        },
        {
            "name": "weak_anchors",
            "anchor_power": 0.15,
            "stability": 0.5,
            "neg_ctrl_gap": 2.5,
            "spectral_reliability": 0.4,
        },
        {
            "name": "moderate_anchors",
            "anchor_power": 0.4,
            "stability": 0.7,
            "neg_ctrl_gap": 3.5,
            "spectral_reliability": 0.6,
        },
        {
            "name": "strong_anchors",
            "anchor_power": 0.8,
            "stability": 0.95,
            "neg_ctrl_gap": 5.0,
            "spectral_reliability": 0.85,
        },
        {
            "name": "strong_with_external",
            "anchor_power": 0.9,
            "stability": 0.98,
            "neg_ctrl_gap": 6.0,
            "spectral_reliability": 0.95,
            "external": True,
        },
    ]

    all_results = {}

    for scenario in scenarios:
        print(f"\n--- Scenario: {scenario['name']} ---")
        external = scenario.get("external", False)
        result = model.process_transport_hypotheses(
            coupling_matrix=demo_coupling,
            source_tokens=source_tokens,
            target_tokens=target_tokens,
            anchor_power=scenario["anchor_power"],
            bootstrap_stability=scenario["stability"],
            negative_control_gap=scenario["neg_ctrl_gap"],
            spectral_reliability=scenario["spectral_reliability"],
            external_evidence=external,
        )

        all_results[scenario["name"]] = {
            "hypotheses": result,
            "scenario": scenario,
        }

        for h in result:
            print(
                f"  {h['source_token']}: max_claim={h['max_claim_level']}, "
                f"blocked={h['blocked']}, OCR={h['overclaim_risk']:.3f}"
            )

    model.save_ledger(output_dir / "hypothesis_ledger.jsonl")

    summary = model.ledger_summary()
    print(f"\nLedger Summary:")
    print(f"  Total hypotheses: {summary['total_hypotheses']}")
    print(f"  Level counts: {summary['level_counts']}")
    print(f"  Blocked: {summary['blocked_count']}")
    print(f"  Mean OCR: {summary['mean_overclaim_risk']:.3f}")

    with open(output_dir / "ledger_summary.json", "w") as f:
        json.dump(
            {"summary": summary, "scenarios": all_results}, f, indent=2, default=str
        )

    print(f"\nLedger saved to {output_dir}")


if __name__ == "__main__":
    main()
