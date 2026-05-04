"""Batch comparison of all candidate languages against a reference corpus.

Runs compare_alignment_methods.py for every candidate listed in a YAML config.
"""

import argparse
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Batch candidate comparison")
    parser.add_argument("--config", default="configs/candidate_languages.yaml")
    parser.add_argument(
        "--lost-embed", default="data/processed/embeddings_synthetic_v2.npy"
    )
    parser.add_argument("--output-dir", default="reports/tables")
    parser.add_argument(
        "--segmented", action="store_true", help="Use segmented embeddings (_seg.npy)"
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    candidates = config.get("candidates", [])
    results = []

    for cand in candidates:
        name = cand["name"]
        code = cand.get("code", name)
        suffix = "_seg" if args.segmented else ""
        embed_path = f"data/processed/embeddings_{code}{suffix}.npy"
        out_json = f"{args.output_dir}/alignment_comparison_{name}{suffix}.json"

        import subprocess
        import sys

        cmd = [
            sys.executable,
            "scripts/compare_alignment_methods.py",
            "--lost-embed",
            args.lost_embed,
            "--candidate-embed",
            embed_path,
            "--candidate-name",
            name + ("_seg" if args.segmented else ""),
            "--output",
            out_json,
            "--reg",
            "0.5",
        ]
        env = {"PYTHONPATH": "src"}
        print(f"\n>>> Comparing against {name}{suffix} ...")
        subprocess.run(
            cmd, cwd=Path.cwd(), env={**env, "PATH": subprocess.os.environ["PATH"]}
        )

        with open(out_json, "r", encoding="utf-8") as f:
            res = json.load(f)

        ot = res["comparisons"]["ot"]
        results.append(
            {
                "candidate": name + ("_seg" if args.segmented else ""),
                "family": cand.get("family", "unknown"),
                "n_cand": res["n_cand"],
                "geo_dist": ot["geometric_distortion"],
                "rel_dist": ot["relational_distortion"],
                "entropy": ot["coupling_entropy"],
            }
        )

    import pandas as pd

    df = pd.DataFrame(results)
    suffix_str = "_segmented" if args.segmented else ""
    summary_path = (
        Path(args.output_dir) / f"candidate_comparison_summary{suffix_str}.csv"
    )
    df.to_csv(summary_path, index=False)
    print(f"\n{'='*60}")
    print(f"Candidate comparison summary{suffix_str} (all vs synthetic_v2)")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
