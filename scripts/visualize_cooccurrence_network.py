"""Visualize co-occurrence network for a corpus.

Creates a graph where nodes are signs/tokens and edges represent
co-occurrences within a window. Edge weights = PMI values.
Useful for identifying clusters of functionally related signs.
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


def build_network(corpus_csv: str, window: int = 3, min_pmi: float = 0.5, max_nodes: int = 50):
    """Build NetworkX graph from PPMI matrix."""
    df = pd.read_csv(corpus_csv)
    sequences = get_sequences_by_line(df)
    vocab = build_vocab([tok for seq in sequences for tok in seq])
    seq_ids = [tokens_to_ids(seq, vocab) for seq in sequences]

    C = cooccurrence_matrix_from_sequences(seq_ids, len(vocab), window_size=window)
    M = ppmi_matrix(C, epsilon=1e-9)

    # Get top N most frequent tokens for visualization
    token_counts = {}
    for seq in sequences:
        for tok in seq:
            token_counts[tok] = token_counts.get(tok, 0) + 1
    top_tokens = sorted(token_counts, key=token_counts.get, reverse=True)[:max_nodes]
    top_indices = [vocab[tok] for tok in top_tokens]

    G = nx.Graph()
    for tok in top_tokens:
        G.add_node(tok, frequency=token_counts[tok])

    for i, tok_i in enumerate(top_tokens):
        for j, tok_j in enumerate(top_tokens):
            if i >= j:
                continue
            idx_i = top_indices[i]
            idx_j = top_indices[j]
            pmi = M[idx_i, idx_j]
            if pmi > min_pmi:
                G.add_edge(tok_i, tok_j, weight=pmi)

    return G


def plot_network(G: nx.Graph, title: str, output_path: str):
    """Plot the co-occurrence network."""
    fig, ax = plt.subplots(figsize=(14, 10))

    pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)

    # Node sizes proportional to frequency
    freqs = [G.nodes[n].get("frequency", 1) for n in G.nodes()]
    node_sizes = [300 + 50 * np.sqrt(f) for f in freqs]

    # Edge widths proportional to PMI
    edges = G.edges(data=True)
    weights = [d["weight"] for (_, _, d) in edges]
    edge_widths = [1 + 3 * (w / max(weights)) if weights else 1 for w in weights]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color="#2E86AB", alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths,
                           edge_color="#A23B72", alpha=0.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="monospace",
                            font_color="#1a1816", ax=ax)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved network plot to {output_path}")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")


def main():
    parser = argparse.ArgumentParser(description="Visualize co-occurrence network")
    parser.add_argument("--input", default="data/raw/lost_language/corpus_indus_real.csv")
    parser.add_argument("--output", default="reports/figures/cooccurrence_network.png")
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--min-pmi", type=float, default=0.5)
    parser.add_argument("--max-nodes", type=int, default=50)
    parser.add_argument("--title", default="Co-occurrence Network")
    args = parser.parse_args()

    G = build_network(args.input, args.window, args.min_pmi, args.max_nodes)
    plot_network(G, args.title, args.output)


if __name__ == "__main__":
    main()
