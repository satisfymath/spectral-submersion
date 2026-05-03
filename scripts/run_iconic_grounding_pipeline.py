"""Run the professional iconic-grounding pipeline end-to-end.

Steps:
1. Cross-script validation on known deciphered/standard signs.
2. Rongorongo real SVG grounding using the measured cross-script Acc@5.
3. Combined machine-readable and Markdown reports.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end iconic grounding")
    parser.add_argument(
        "--referent-image-root",
        default="data/external/iconic_referents/rapa_nui_1500",
    )
    parser.add_argument("--output-dir", default="runs/iconic_grounding_phd")
    parser.add_argument("--top-n-glyphs", type=int, default=0)
    parser.add_argument("--max-instances-per-glyph", type=int, default=8)
    parser.add_argument("--max-images-per-referent", type=int, default=3)
    parser.add_argument("--n-controls", type=int, default=40)
    parser.add_argument("--n-bootstrap", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cross_dir = output_dir / "cross_script"
    rr_dir = output_dir / "rongorongo_real"
    output_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    _run(
        [
            python,
            "scripts/run_cross_script_iconic_validation.py",
            "--referent-image-root",
            args.referent_image_root,
            "--output-dir",
            str(cross_dir),
            "--top-k",
            str(args.top_k),
        ]
    )

    cross_summary = json.loads((cross_dir / "cross_script_summary.json").read_text())
    cross_acc5 = float(cross_summary["metrics"]["accuracy@5"])

    _run(
        [
            python,
            "scripts/run_iconic_grounding_real.py",
            "--referent-image-root",
            args.referent_image_root,
            "--output-dir",
            str(rr_dir),
            "--top-n-glyphs",
            str(args.top_n_glyphs),
            "--max-instances-per-glyph",
            str(args.max_instances_per_glyph),
            "--max-images-per-referent",
            str(args.max_images_per_referent),
            "--n-controls",
            str(args.n_controls),
            "--n-bootstrap",
            str(args.n_bootstrap),
            "--top-k",
            str(args.top_k),
            "--cross-script-acc-at-5",
            str(cross_acc5),
        ]
    )

    rr_summary = json.loads((rr_dir / "real_iconic_summary.json").read_text())
    combined = {
        "cross_script": cross_summary,
        "rongorongo": rr_summary,
        "decision": {
            "c25_unlocked": rr_summary["c25_candidates_admitted"] > 0,
            "reason": (
                "C2.5 remains blocked unless cross-script Acc@5 >= 0.6 "
                "and all Definition 13.1 criteria pass."
            ),
        },
    }
    (output_dir / "iconic_grounding_phd_summary.json").write_text(
        json.dumps(combined, indent=2),
        encoding="utf-8",
    )

    report = [
        "# Iconic Grounding PhD Pipeline",
        "",
        "## Cross-Script Validation",
        "",
        f"- Signs: {cross_summary['n_signs']}",
        f"- Referents: {cross_summary['n_referents']}",
        f"- Acc@1: {cross_summary['metrics']['accuracy@1']:.3f}",
        f"- Acc@5: {cross_summary['metrics']['accuracy@5']:.3f}",
        f"- MRR: {cross_summary['metrics']['mrr']:.3f}",
        "",
        "## Rongorongo Real Grounding",
        "",
        f"- Glyph classes: {rr_summary['n_glyph_classes']}",
        f"- Referents: {rr_summary['n_referents']}",
        f"- Mean top-1 iconicity: {rr_summary['mean_top1_iconicity']:.3f}",
        f"- AnchorPower: {rr_summary['anchor_power']:.3f}",
        f"- Bootstrap stability: {rr_summary['bootstrap_assignment_stability']:.3f}",
        f"- NegCtrlGap: {rr_summary['negative_control_gap']['gap']:.3f}",
        f"- C2.5 admitted top-1 candidates: {rr_summary['c25_candidates_admitted']}",
        "",
        "## Decision",
        "",
        combined["decision"]["reason"],
    ]
    (output_dir / "iconic_grounding_phd_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(f"\nCombined report: {output_dir / 'iconic_grounding_phd_report.md'}")


if __name__ == "__main__":
    main()
