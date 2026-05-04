"""Generate master experiment summary from all result tables.

Reads CSV outputs from various experiments and produces a unified
markdown report with cross-references.
"""

import argparse
from pathlib import Path

import pandas as pd


def load_if_exists(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        if p.suffix == ".json":
            import json

            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Flatten nested dict to DataFrame row
            if isinstance(data, dict):
                return pd.DataFrame([data])
            return pd.DataFrame(data)
        return pd.read_csv(p)
    except Exception as e:
        print(f"Warning: could not load {p}: {e}")
        return None


def format_table(df: pd.DataFrame, title: str) -> str:
    lines = [f"### {title}", ""]
    lines.append(df.to_markdown(index=False))
    lines.append("")
    return "\n".join(lines)


def generate_master_report(output_path: str):
    sections = []
    sections.append("# Master Experiment Summary")
    sections.append("")
    sections.append("Generated automatically from result tables.")
    sections.append("")

    # 1. Synthetic corpus controls
    df = load_if_exists("reports/tables/control_comparison.csv")
    if df is not None:
        sections.append(
            format_table(df, "A. Negative Controls (Synthetic Mini, 16 types)")
        )

    df = load_if_exists("reports/tables/control_comparison_v2.csv")
    if df is not None:
        sections.append(
            format_table(df, "B. Negative Controls (Synthetic PCFG, 112 types)")
        )

    df = load_if_exists("reports/tables/control_comparison_rongorongo_v2.csv")
    if df is not None:
        sections.append(
            format_table(df, "C. Negative Controls (Rongorongo-like, 120 types)")
        )

    df = load_if_exists("reports/tables/control_comparison_indus_real.csv")
    if df is not None:
        sections.append(
            format_table(df, "D. Negative Controls (Indus Real, 182 types)")
        )

    # 2. Candidate comparisons
    df = load_if_exists("reports/tables/diverse_comparison_synthetic.csv")
    if df is not None:
        sections.append(
            format_table(df, "E. Multi-Candidate Comparison (Synthetic PCFG)")
        )

    df = load_if_exists("reports/tables/diverse_comparison_rongorongo.csv")
    if df is not None:
        sections.append(
            format_table(df, "F. Multi-Candidate Comparison (Rongorongo-like)")
        )

    df = load_if_exists("reports/tables/diverse_comparison_indus_real.csv")
    if df is not None:
        sections.append(format_table(df, "G. Multi-Candidate Comparison (Indus Real)"))

    # 3. Anchor recovery
    df = load_if_exists("reports/tables/anchor_recovery.json")
    if df is not None:
        sections.append(format_table(df, "H. Anchor Recovery (Synthetic, 20% anchors)"))

    df = load_if_exists("reports/tables/anchor_recovery_polysemy.json")
    if df is not None:
        sections.append(
            format_table(df, "I. Anchor Recovery with Polysemy (15% collapse)")
        )

    # 4. Entropy analysis
    df = load_if_exists("reports/tables/entropy_analysis_indus_real.csv")
    if df is not None:
        sections.append(format_table(df, "J. Conditional Entropy (Indus Real)"))

    df = load_if_exists("reports/tables/entropy_analysis_synthetic.csv")
    if df is not None:
        sections.append(format_table(df, "K. Conditional Entropy (Synthetic PCFG)"))

    # 5. Hyperparameter grid
    df = load_if_exists("reports/tables/hyperparameter_grid.csv")
    if df is not None:
        best = df.nsmallest(5, "effective_rank")
        sections.append(format_table(best, "L. Hyperparameter Grid Top-5 (Synthetic)"))

    df = load_if_exists("reports/tables/hyperparameter_grid_indus_real.csv")
    if df is not None:
        best = df.nsmallest(5, "effective_rank")
        sections.append(format_table(best, "M. Hyperparameter Grid Top-5 (Indus Real)"))

    sections.append("---")
    sections.append("*End of master summary*")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"Saved master summary to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate master experiment summary")
    parser.add_argument(
        "--output", default="reports/final/master_experiment_summary.md"
    )
    args = parser.parse_args()
    generate_master_report(args.output)


if __name__ == "__main__":
    main()
