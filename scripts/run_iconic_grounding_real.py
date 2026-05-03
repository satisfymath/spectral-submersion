"""Run iconic grounding on real Rongorongo SVGs and real referent images.

This pipeline uses:
- RR-corpus XML SVG paths for real Rongorongo glyph instances.
- Local real referent images, typically downloaded with
  ``scripts/download_iconic_referent_images.py``.

It intentionally blocks C2.5 claims until cross-script validation is supplied.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from spectral_submersion.audit_metrics import negative_control_gap  # noqa: E402
from spectral_submersion.iconic_grounding import (  # noqa: E402
    IconicClaimEvidence,
    RapaNuiWorld1500,
    anchor_assignment_stability,
    anchor_power_from_counts,
    assess_c25_admissibility,
    rank_iconic_candidates,
)
from spectral_submersion.iconic_real_data import (  # noqa: E402
    build_rongorongo_glyph_embedding_table,
    export_glyph_svg_audit_files,
    glyph_instance_embedding,
    load_referent_image_embedding_table,
    save_glyph_embedding_table,
    save_referent_embedding_table,
)


def _top1_assignments(ranked):
    return {
        glyph_id: candidates[0].referent_id
        for glyph_id, candidates in ranked.items()
        if candidates
    }


def _dimension_permutation_controls(
    glyph_embeddings: dict[str, np.ndarray],
    referent_embeddings: dict[str, np.ndarray],
    n_controls: int,
    seed: int,
    top_k: int,
) -> list[float]:
    rng = np.random.RandomState(seed)
    neg_scores = []
    glyph_ids = list(glyph_embeddings)
    dim = len(next(iter(glyph_embeddings.values())))
    for _ in range(n_controls):
        permuted = {
            glyph_id: glyph_embeddings[glyph_id][rng.permutation(dim)]
            for glyph_id in glyph_ids
        }
        ranked = rank_iconic_candidates(permuted, referent_embeddings, top_k=top_k)
        neg_scores.append(
            float(np.mean([candidates[0].score for candidates in ranked.values()]))
        )
    return neg_scores


def _bootstrap_assignment_stability(
    glyph_table,
    referent_embeddings: dict[str, np.ndarray],
    n_bootstrap: int,
    seed: int,
    image_size: int,
    grid_size: int,
    top_k: int,
) -> float:
    rng = np.random.RandomState(seed)
    assignments = []
    for _ in range(n_bootstrap):
        boot_embeddings = {}
        for glyph_code, instances in glyph_table.instances_by_code.items():
            if not instances:
                continue
            sample_idx = rng.choice(len(instances), size=len(instances), replace=True)
            sample_embeddings = np.vstack(
                [
                    glyph_instance_embedding(
                        instances[int(i)],
                        image_size=image_size,
                        grid_size=grid_size,
                    )
                    for i in sample_idx
                ]
            )
            from spectral_submersion.iconic_grounding import spherical_mean

            boot_embeddings[glyph_code] = spherical_mean(sample_embeddings)
        ranked = rank_iconic_candidates(
            boot_embeddings,
            referent_embeddings,
            top_k=top_k,
        )
        assignments.append(_top1_assignments(ranked))
    return anchor_assignment_stability(assignments)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real iconic grounding pipeline")
    parser.add_argument("--xml-dir", default="data/external/rongorongo_rr_corpus")
    parser.add_argument(
        "--referent-image-root",
        default="data/external/iconic_referents/rapa_nui_1500",
    )
    parser.add_argument("--output-dir", default="runs/iconic_grounding_real")
    parser.add_argument("--image-type", default="b")
    parser.add_argument("--group-by", default="base_code")
    parser.add_argument(
        "--top-n-glyphs",
        type=int,
        default=0,
        help="Most frequent glyph classes to analyze. Use 0 for all classes.",
    )
    parser.add_argument("--max-instances-per-glyph", type=int, default=20)
    parser.add_argument("--max-images-per-referent", type=int, default=5)
    parser.add_argument("--min-referent-images", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-iconicity", type=float, default=0.6)
    parser.add_argument("--n-controls", type=int, default=100)
    parser.add_argument("--n-bootstrap", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cross-script-acc-at-5",
        type=float,
        default=0.0,
        help="Real cross-script validation score. Default 0 keeps C2.5 blocked.",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    audit_svg_dir = out / "glyph_svg_audit"

    print("=" * 70)
    print("REAL ICONIC GROUNDING PIPELINE")
    print("=" * 70)
    print("Building real Rongorongo glyph embeddings from XML SVG paths...")
    glyph_table = build_rongorongo_glyph_embedding_table(
        xml_dir=args.xml_dir,
        image_type=args.image_type,
        group_by=args.group_by,
        top_n=None if args.top_n_glyphs <= 0 else args.top_n_glyphs,
        max_instances_per_glyph=args.max_instances_per_glyph,
        image_size=args.image_size,
        grid_size=args.grid_size,
    )
    save_glyph_embedding_table(
        glyph_table,
        out / "rongorongo_real_glyph_embeddings.npz",
        out / "rongorongo_real_glyph_metadata.csv",
    )
    exported = export_glyph_svg_audit_files(
        glyph_table.instances_by_code,
        audit_svg_dir,
        max_per_code=3,
    )
    print(f"  Glyph classes: {len(glyph_table.embeddings)}")
    print(f"  SVG audit files: {exported} -> {audit_svg_dir}")

    print("\nBuilding real referent embeddings from local image folders...")
    referent_table = load_referent_image_embedding_table(
        args.referent_image_root,
        min_images=args.min_referent_images,
        max_images_per_referent=args.max_images_per_referent,
        grid_size=args.grid_size,
    )
    if not referent_table.embeddings:
        raise SystemExit(
            "No referent images found. Run scripts/download_iconic_referent_images.py first."
        )
    save_referent_embedding_table(
        referent_table,
        out / "rapa_nui_referent_embeddings.npz",
        out / "rapa_nui_referent_metadata.csv",
    )
    print(f"  Referents with images: {len(referent_table.embeddings)}")

    ranked = rank_iconic_candidates(
        glyph_table.embeddings,
        referent_table.embeddings,
        top_k=args.top_k,
    )
    top_scores = np.array([candidates[0].score for candidates in ranked.values()])
    score_real = float(top_scores.mean())
    neg_scores = _dimension_permutation_controls(
        glyph_table.embeddings,
        referent_table.embeddings,
        n_controls=args.n_controls,
        seed=args.seed,
        top_k=args.top_k,
    )
    neg_gap = negative_control_gap(score_real, np.asarray(neg_scores))

    stability = _bootstrap_assignment_stability(
        glyph_table,
        referent_table.embeddings,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        image_size=args.image_size,
        grid_size=args.grid_size,
        top_k=args.top_k,
    )
    anchored_count = int(np.sum(top_scores >= args.min_iconicity))
    anchor_power = anchor_power_from_counts(
        vocab_size=len(glyph_table.embeddings),
        anchored_count=anchored_count,
    )

    world = RapaNuiWorld1500()
    world_by_id = world.by_id()
    candidate_rows = []
    for glyph_id, candidates in ranked.items():
        glyph_meta = glyph_table.metadata[glyph_id]
        for candidate in candidates:
            referent = world_by_id.get(candidate.referent_id)
            source_count = referent.source_count if referent is not None else 0
            decision = assess_c25_admissibility(
                IconicClaimEvidence(
                    iota_max=candidate.score,
                    anchor_power=anchor_power,
                    bootstrap_stability=stability,
                    cross_script_acc_at_5=args.cross_script_acc_at_5,
                    negative_control_gap=neg_gap["gap"],
                    in_world_reconstruction=referent is not None,
                    bibliographic_sources=source_count,
                )
            )
            candidate_rows.append(
                {
                    "glyph_code": glyph_id,
                    "glyph_instances_total": glyph_meta.n_instances_total,
                    "glyph_instances_used": glyph_meta.n_instances_used,
                    "glyph_dispersion": glyph_meta.dispersion,
                    "referent_id": candidate.referent_id,
                    "rank": candidate.rank,
                    "iconicity": candidate.score,
                    "geodesic_distance": candidate.geodesic_distance,
                    "deiconization_rate": candidate.deiconization_rate,
                    "referent_images": referent_table.metadata[
                        candidate.referent_id
                    ].n_images,
                    "referent_dispersion": referent_table.metadata[
                        candidate.referent_id
                    ].dispersion,
                    "anchor_power_global": anchor_power,
                    "bootstrap_assignment_stability": stability,
                    "negative_control_gap": neg_gap["gap"],
                    "cross_script_acc_at_5": args.cross_script_acc_at_5,
                    "c25_admissible": decision.admissible,
                    "max_claim_label": decision.max_claim_label,
                    "failed_criteria": "|".join(decision.failed_criteria),
                }
            )

    candidates_csv = out / "real_iconic_candidates.csv"
    with open(candidates_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(candidate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(candidate_rows)

    summary = {
        "data_mode": "real_rr_corpus_svg_and_real_referent_images",
        "xml_dir": args.xml_dir,
        "referent_image_root": args.referent_image_root,
        "n_glyph_classes": len(glyph_table.embeddings),
        "n_referents": len(referent_table.embeddings),
        "top_k": args.top_k,
        "min_iconicity": args.min_iconicity,
        "mean_top1_iconicity": score_real,
        "anchored_count_at_threshold": anchored_count,
        "anchor_power": anchor_power,
        "bootstrap_assignment_stability": stability,
        "negative_control_gap": neg_gap,
        "cross_script_acc_at_5": args.cross_script_acc_at_5,
        "c25_candidates_admitted": int(
            sum(1 for row in candidate_rows if row["rank"] == 1 and row["c25_admissible"])
        ),
        "important_note": (
            "C2.5 remains blocked unless real cross-script validation, controls, "
            "and stability all satisfy Definition 13.1."
        ),
    }
    summary_path = out / "real_iconic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    top1_rows = [row for row in candidate_rows if row["rank"] == 1]
    top1_rows.sort(key=lambda row: float(row["iconicity"]), reverse=True)
    report_lines = [
        "# Real Iconic Grounding Run",
        "",
        "This run uses real RR-corpus SVG glyph paths and real local referent images.",
        "It is an iconographic candidate-generation run, not a decipherment.",
        "",
        "## Summary",
        "",
        f"- Glyph classes analyzed: {summary['n_glyph_classes']}",
        f"- Referents with real images: {summary['n_referents']}",
        f"- Mean top-1 iconicity: {score_real:.3f}",
        f"- AnchorPower at threshold {args.min_iconicity}: {anchor_power:.3f}",
        f"- Bootstrap assignment stability: {stability:.3f}",
        f"- NegCtrlGap: {neg_gap['gap']:.3f} ({neg_gap['interpretation']})",
        f"- Cross-script Acc@5 supplied: {args.cross_script_acc_at_5:.3f}",
        f"- C2.5 admitted top-1 candidates: {summary['c25_candidates_admitted']}",
        "",
        "## Top Candidate Rows",
        "",
        "| glyph | referent | iconicity | failed criteria |",
        "|---|---:|---:|---|",
    ]
    for row in top1_rows[:20]:
        report_lines.append(
            f"| `{row['glyph_code']}` | `{row['referent_id']}` | "
            f"{float(row['iconicity']):.3f} | {row['failed_criteria']} |"
        )
    report_lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "C2.5 is intentionally blocked in this run because real cross-script "
            "validation is below the required threshold or another Definition 13.1 "
            "criterion failed. The very high negative-control gap is also flagged "
            "as `very_strong_check_leakage`, so these rankings should be treated "
            "as candidates for inspection and validation, not as semantic claims.",
        ]
    )
    (out / "real_iconic_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Output dir: {out}")
    print(f"Candidates CSV: {candidates_csv}")
    print(f"Glyph classes: {summary['n_glyph_classes']}")
    print(f"Referents: {summary['n_referents']}")
    print(f"Mean top-1 iconicity: {score_real:.3f}")
    print(f"NegCtrlGap: {neg_gap['gap']:.3f} ({neg_gap['interpretation']})")
    print(f"Bootstrap assignment stability: {stability:.3f}")
    print(f"AnchorPower: {anchor_power:.3f}")
    print(f"C2.5 admitted top-1 candidates: {summary['c25_candidates_admitted']}")


if __name__ == "__main__":
    main()
