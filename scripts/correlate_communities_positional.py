"""Correlate network communities with positional bias.

For the Indus corpus, checks if detected communities cluster by
positional role (start-biased, end-biased, middle).
"""
import argparse
from pathlib import Path

import pandas as pd


def correlate_communities_positional(
    communities_csv: str,
    positional_csv: str,
    output_csv: str,
):
    """Join community assignment with positional bias data."""
    comm_df = pd.read_csv(communities_csv)
    pos_df = pd.read_csv(positional_csv)

    # Expand communities to one row per node
    rows = []
    for _, row in comm_df.iterrows():
        comm_id = row["community"]
        nodes = row["nodes"].split()
        for node in nodes:
            rows.append({"token": node, "community": comm_id})

    comm_nodes = pd.DataFrame(rows)

    # Merge with positional data
    merged = comm_nodes.merge(pos_df, on="token", how="left")

    # Aggregate by community
    summary = []
    for comm_id, group in merged.groupby("community"):
        summary.append({
            "community": comm_id,
            "size": len(group),
            "mean_position": group["mean_relative_position"].mean(),
            "mean_first_ratio": group["first_ratio"].mean(),
            "mean_last_ratio": group["last_ratio"].mean(),
            "start_signs": len(group[group["first_ratio"] > 0.5]),
            "end_signs": len(group[group["last_ratio"] > 0.5]),
            "middle_signs": len(group[
                (group["first_ratio"] <= 0.3) & (group["last_ratio"] <= 0.3)
            ]),
            "top_tokens": " ".join(group.sort_values("count", ascending=False).head(5)["token"].tolist()),
        })

    summary_df = pd.DataFrame(summary).sort_values("community")
    summary_df.to_csv(output_csv, index=False)
    print(summary_df.to_string(index=False))

    # Identify pure positional communities
    print("\n=== COMMUNITIES WITH STRONG START BIAS ===")
    start_comms = summary_df[summary_df["mean_first_ratio"] > 0.3]
    if len(start_comms) > 0:
        print(start_comms[["community", "size", "mean_first_ratio", "top_tokens"]].to_string(index=False))
    else:
        print("None found")

    print("\n=== COMMUNITIES WITH STRONG END BIAS ===")
    end_comms = summary_df[summary_df["mean_last_ratio"] > 0.3]
    if len(end_comms) > 0:
        print(end_comms[["community", "size", "mean_last_ratio", "top_tokens"]].to_string(index=False))
    else:
        print("None found")

    print(f"\nSaved correlation table to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Correlate communities with positional bias")
    parser.add_argument("--communities", default="reports/tables/network_communities_indus.csv")
    parser.add_argument("--positional", default="reports/tables/positional_analysis_indus.csv")
    parser.add_argument("--output", default="reports/tables/community_positional_correlation.csv")
    args = parser.parse_args()

    correlate_communities_positional(args.communities, args.positional, args.output)


if __name__ == "__main__":
    main()
