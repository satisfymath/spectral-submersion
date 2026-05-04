"""Run real cross-script iconic validation.

This benchmark renders deciphered/standard signs from real fonts and compares
them against local real referent images. It is designed to feed the C2.5
criterion "cross-script Acc@5" without using Rongorongo labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from spectral_submersion.iconic_cross_script import (  # noqa: E402
    available_known_script_signs,
    build_known_script_embeddings,
    render_known_script_sign,
)
from spectral_submersion.iconic_grounding import (  # noqa: E402
    evaluate_anchor_ranking,
    rank_iconic_candidates,
)
from spectral_submersion.iconic_real_data import (  # noqa: E402
    load_referent_image_embedding_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-script iconic validation")
    parser.add_argument(
        "--referent-image-root",
        default="data/external/iconic_referents/rapa_nui_1500",
    )
    parser.add_argument("--output-dir", default="runs/iconic_cross_script")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rendered_dir = out / "rendered_known_signs"
    rendered_dir.mkdir(exist_ok=True)

    referent_table = load_referent_image_embedding_table(
        args.referent_image_root,
        min_images=1,
        max_images_per_referent=10,
        grid_size=args.grid_size,
    )
    signs = available_known_script_signs(set(referent_table.embeddings))
    if not signs:
        raise SystemExit("No cross-script signs have matching referent images.")

    sign_embeddings = build_known_script_embeddings(
        signs,
        image_size=args.image_size,
        grid_size=args.grid_size,
    )
    ranked = rank_iconic_candidates(
        sign_embeddings,
        referent_table.embeddings,
        top_k=args.top_k,
    )
    gold = {sign.sign_id: sign.referent_id for sign in signs}
    metrics = evaluate_anchor_ranking(ranked, gold, k_values=(1, 5))

    rows = []
    for sign in signs:
        image = render_known_script_sign(sign, image_size=args.image_size)
        image_path = rendered_dir / f"{sign.sign_id}.png"
        image.save(image_path)
        for candidate in ranked[sign.sign_id]:
            rows.append(
                {
                    "sign_id": sign.sign_id,
                    "script": sign.script,
                    "label": sign.label,
                    "gold_referent": sign.referent_id,
                    "candidate_referent": candidate.referent_id,
                    "rank": candidate.rank,
                    "iconicity": candidate.score,
                    "correct": candidate.referent_id == sign.referent_id,
                    "provenance": sign.provenance,
                    "rendered_path": str(image_path),
                }
            )

    with open(
        out / "cross_script_candidates.csv", "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_script = {}
    for script in sorted({sign.script for sign in signs}):
        subset = [sign for sign in signs if sign.script == script]
        by_script[script] = evaluate_anchor_ranking(
            ranked,
            {sign.sign_id: sign.referent_id for sign in subset},
            k_values=(1, 5),
        )

    summary = {
        "n_signs": len(signs),
        "n_referents": len(referent_table.embeddings),
        "metrics": metrics,
        "metrics_by_script": by_script,
        "referent_image_root": args.referent_image_root,
        "note": (
            "Uses real standardized script fonts and real local referent images; "
            "not Rongorongo labels."
        ),
    }
    (out / "cross_script_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    report = [
        "# Cross-Script Iconic Validation",
        "",
        summary["note"],
        "",
        f"- Signs evaluated: {len(signs)}",
        f"- Referents with images: {len(referent_table.embeddings)}",
        f"- Accuracy@1: {metrics['accuracy@1']:.3f}",
        f"- Accuracy@5: {metrics['accuracy@5']:.3f}",
        f"- MRR: {metrics['mrr']:.3f}",
        "",
        "| script | n | Acc@1 | Acc@5 | MRR |",
        "|---|---:|---:|---:|---:|",
    ]
    for script, script_metrics in by_script.items():
        report.append(
            f"| {script} | {int(script_metrics['n'])} | "
            f"{script_metrics['accuracy@1']:.3f} | "
            f"{script_metrics['accuracy@5']:.3f} | "
            f"{script_metrics['mrr']:.3f} |"
        )
    (out / "cross_script_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print("=" * 70)
    print("CROSS-SCRIPT ICONIC VALIDATION")
    print("=" * 70)
    print(f"Output dir: {out}")
    print(f"Signs: {len(signs)}")
    print(f"Referents: {len(referent_table.embeddings)}")
    print(f"Accuracy@1: {metrics['accuracy@1']:.3f}")
    print(f"Accuracy@5: {metrics['accuracy@5']:.3f}")
    print(f"MRR: {metrics['mrr']:.3f}")


if __name__ == "__main__":
    main()
