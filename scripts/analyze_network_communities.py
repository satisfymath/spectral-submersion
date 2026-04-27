"""Community detection on co-occurrence network.

Identifies clusters of signs that tend to co-occur, potentially
corresponding to functional classes (titles, numerals, names, etc.).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from spectral_submersion.cooccurrence import build_vocab, cooccurrence_matrix_from_sequences
from spectral_submersion.pmi import ppmi_matrix
from spectral_submersion.tokenization import get_sequences_by_line, tokens_to_ids


def build_network(corpus_csv: str, window: int = 3, min_pmi: float = 0.3):
    """Build weighted graph from PPMI matrix."""
    df = pd.read_csv(corpus_csv)
    sequences = get_sequences_by_line(df)
    vocab = build_vocab([tok for seq in sequences for tok in seq])
    seq_ids = [tokens_to_ids(seq, vocab) for seq in sequences]

    C = cooccurrence_matrix_from_sequences(seq_ids, len(vocab), window_size=window)
    M = ppmi_matrix(C, epsilon=1e-9)

    G = nx.Graph()
    tokens = list(vocab.keys())
    for i, tok in enumerate(tokens):
        G.add_node(tok)
        for j in range(i + 1, len(tokens)):
            if M[i, j] > min_pmi:
                G.add_edge(tok, tokens[j], weight=M[i, j])
    return G


def detect_communities(G: nx.Graph) -> dict:
    """Detect communities using Louvain algorithm."""
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G, weight="weight")
    except ImportError:
        # Fallback to greedy modularity
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(G, weight="weight"))
        partition = {}
        for comm_id, nodes in enumerate(comms):
            for node in nodes:
                partition[node] = comm_id
    return partition


def plot_communities(G: nx.Graph, partition: dict, title: str, output_path: str):
    """Plot network with community colors."""
    fig, ax = plt.subplots(figsize=(16, 12))

    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)

    # Color by community
    communities = sorted(set(partition.values()))
    colors = plt.cm.tab20(np.linspace(0, 1, len(communities)))
    node_colors = [colors[partition.get(n, 0) % len(colors)] for n in G.nodes()]

    # Node sizes by degree
    degrees = dict(G.degree())
    node_sizes = [200 + 100 * degrees[n] for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, font_family="monospace", ax=ax)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved community plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Community detection on co-occurrence network")
    parser.add_argument("--input", default="data/raw/lost_language/corpus_indus_real.csv")
    parser.add_argument("--output-fig", default="reports/figures/network_communities.png")
    parser.add_argument("--output-table", default="reports/tables/network_communities.csv")
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--min-pmi", type=float, default=0.3)
    parser.add_argument("--title", default="Indus Sign Communities")
    args = parser.parse_args()

    G = build_network(args.input, args.window, args.min_pmi)
    partition = detect_communities(G)

    # Summarize communities
    comm_nodes = {}
    for node, comm_id in partition.items():
        comm_nodes.setdefault(comm_id, []).append(node)

    rows = []
    for comm_id, nodes in sorted(comm_nodes.items()):
        rows.append({
            "community": comm_id,
            "size": len(nodes),
            "nodes": " ".join(sorted(nodes)),
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.output_table, index=False)
    print(f"Saved community table to {args.output_table}")
    print(df.to_string(index=False))

    plot_communities(G, partition, args.title, args.output_fig)


if __name__ == "__main__":
    main()
