"""Run a deterministic iconic-grounding smoke test.

This is not a claim about real Rongorongo glyphs. It exercises the new
iconic-grounding API with synthetic embeddings so the pipeline contract can be
reproduced before heavy vision encoders and external datasets are added.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from spectral_submersion.iconic_grounding import (  # noqa: E402
    IconicClaimEvidence,
    RapaNuiWorld1500,
    assess_c25_admissibility,
    anchor_assignment_stability,
    predict_iconic_anchors,
    rank_iconic_candidates,
    evaluate_anchor_ranking,
)
from spectral_submersion.identifiability import anchor_power  # noqa: E402


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def main() -> None:
    parser = argparse.ArgumentParser(description="Iconic grounding demo")
    parser.add_argument("--output-dir", default="runs/iconic_grounding_demo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-iconicity", type=float, default=0.6)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    world = RapaNuiWorld1500()
    referent_ids = [
        "great_frigatebird",
        "moon",
        "hand",
        "mataa",
        "sweet_potato",
        "octopus",
    ]

    basis = np.eye(len(referent_ids))
    referent_embeddings = {
        referent_id: basis[i] for i, referent_id in enumerate(referent_ids)
    }
    glyph_gold = {
        "RR_demo_600": "great_frigatebird",
        "RR_demo_200": "moon",
        "RR_demo_040": "hand",
        "RR_demo_730": "mataa",
        "RR_demo_900": "sweet_potato",
    }
    glyph_embeddings = {}
    for glyph_id, referent_id in glyph_gold.items():
        ref_vec = referent_embeddings[referent_id]
        glyph_embeddings[glyph_id] = _unit(
            ref_vec + rng.normal(0.0, 0.05, ref_vec.shape)
        )

    ranked = rank_iconic_candidates(
        glyph_embeddings,
        referent_embeddings,
        top_k=args.top_k,
    )
    predicted = predict_iconic_anchors(
        glyph_embeddings,
        referent_embeddings,
        top_k=args.top_k,
        min_iconicity=args.min_iconicity,
    )
    metrics = evaluate_anchor_ranking(ranked, glyph_gold, k_values=(1, 5))

    anchored_count = sum(1 for candidates in predicted.values() if candidates)
    aut_size = math.factorial(len(glyph_embeddings))
    anchored_aut_size = math.factorial(max(len(glyph_embeddings) - anchored_count, 0))
    ap = anchor_power(aut_size, anchored_aut_size)

    bootstrap_assignments = []
    for _ in range(20):
        noisy = {
            glyph_id: _unit(vec + rng.normal(0.0, 0.03, vec.shape))
            for glyph_id, vec in glyph_embeddings.items()
        }
        boot_ranked = rank_iconic_candidates(noisy, referent_embeddings, top_k=1)
        bootstrap_assignments.append(
            {
                glyph_id: candidates[0].referent_id
                for glyph_id, candidates in boot_ranked.items()
            }
        )
    stability = anchor_assignment_stability(bootstrap_assignments)

    world_by_id = world.by_id()
    decisions = {}
    for glyph_id, candidates in predicted.items():
        if not candidates:
            continue
        top = candidates[0]
        referent = world_by_id[top.referent_id]
        decision = assess_c25_admissibility(
            IconicClaimEvidence(
                iota_max=top.score,
                anchor_power=ap,
                bootstrap_stability=stability,
                cross_script_acc_at_5=0.7,
                negative_control_gap=3.5,
                in_world_reconstruction=top.referent_id in world.by_id(),
                bibliographic_sources=referent.source_count,
            )
        )
        decisions[glyph_id] = {
            "referent_id": top.referent_id,
            "score": top.score,
            "admissible": decision.admissible,
            "max_claim_label": decision.max_claim_label,
            "failed_criteria": list(decision.failed_criteria),
        }

    result = {
        "seed": args.seed,
        "note": "Synthetic smoke test; not real Rongorongo evidence.",
        "world_validation": world.validate_min_sources(min_sources=2),
        "metrics": metrics,
        "anchor_power": ap,
        "bootstrap_assignment_stability": stability,
        "ranked_candidates": {
            glyph_id: [candidate.as_dict() for candidate in candidates]
            for glyph_id, candidates in ranked.items()
        },
        "c25_decisions": decisions,
    }

    output_path = output_dir / "iconic_grounding_demo_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("=" * 60)
    print("Iconic Grounding Demo")
    print("=" * 60)
    print(f"Output: {output_path}")
    print(f"World referents validated: {result['world_validation']['valid']}")
    print(f"Accuracy@1: {metrics['accuracy@1']:.3f}")
    print(f"Accuracy@5: {metrics['accuracy@5']:.3f}")
    print(f"MRR: {metrics['mrr']:.3f}")
    print(f"AnchorPower: {ap:.3f}")
    print(f"Bootstrap assignment stability: {stability:.3f}")


if __name__ == "__main__":
    main()
