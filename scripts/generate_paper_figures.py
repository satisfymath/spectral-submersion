"""Generate publication-ready figures for the paper.

Academic style: serif fonts, clean lines, no background color.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["DejaVu Serif"]
matplotlib.rcParams["axes.labelsize"] = 11
matplotlib.rcParams["axes.titlesize"] = 12
matplotlib.rcParams["xtick.labelsize"] = 9
matplotlib.rcParams["ytick.labelsize"] = 9
matplotlib.rcParams["legend.fontsize"] = 9
matplotlib.rcParams["figure.dpi"] = 300


def fig_positional_bias():
    """Figure 1: Positional bias of top Indus signs."""
    df = pd.read_csv("reports/tables/positional_analysis_indus.csv")
    sub = df.head(30).sort_values("mean_relative_position")

    fig, ax = plt.subplots(figsize=(10, 8))

    y = np.arange(len(sub))
    colors = [
        "#2E86AB" if r < 0.33 else "#A23B72" if r > 0.66 else "#F18F01"
        for r in sub["mean_relative_position"]
    ]

    ax.barh(
        y,
        sub["mean_relative_position"],
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(sub["token"], fontfamily="monospace", fontsize=8)
    ax.set_xlabel("Posición Relativa Media (0=inicio, 1=fin)", fontsize=11)
    ax.set_title(
        "Sesgo Posicional de Signos del Indo (corpus Parpola CISI, top 30)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlim(0, 1)
    ax.axvline(1 / 3, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(2 / 3, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    ax.text(
        1 / 6,
        -2.5,
        "INICIO",
        ha="center",
        fontsize=10,
        color="#2E86AB",
        fontweight="bold",
    )
    ax.text(
        0.5, -2.5, "MEDIO", ha="center", fontsize=10, color="#A23B72", fontweight="bold"
    )
    ax.text(
        5 / 6, -2.5, "FIN", ha="center", fontsize=10, color="#F18F01", fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig("reports/figures/paper_fig1_positional_bias.pdf", bbox_inches="tight")
    plt.savefig("reports/figures/paper_fig1_positional_bias.png", bbox_inches="tight")
    print("Saved Figure 1: Positional Bias")


def fig_entropy_curves():
    """Figure 2: Conditional entropy curves."""
    df = pd.read_csv("reports/tables/entropy_analysis_indus_real.csv")

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {
        "real": "#2E86AB",
        "permuted": "#A23B72",
        "random_same_freq": "#F18F01",
        "random_uniform": "#C73E1D",
    }
    labels = {
        "real": "Corpus real",
        "permuted": "Permutado",
        "random_same_freq": "Aleatorio (misma freq)",
        "random_uniform": "Aleatorio (uniforme)",
    }

    x = [1, 2, 3]
    x_labels = [
        "Unigram\n$H(X)$",
        "Bigram\n$H(X|X_{-1})$",
        "Trigram\n$H(X|X_{-2},X_{-1})$",
    ]

    for _, row in df.iterrows():
        variant = row["variant"]
        y = [
            row["h_unconditional"],
            row["h_bigram_conditional"],
            row["h_trigram_conditional"],
        ]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            markersize=8,
            color=colors.get(variant, "gray"),
            label=labels.get(variant, variant),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Entropía Condicional (bits)", fontsize=11)
    ax.set_title(
        "Curvas de Entropía Condicional: Corpus Indo vs. Controles",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=False)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig("reports/figures/paper_fig2_entropy_curves.pdf", bbox_inches="tight")
    plt.savefig("reports/figures/paper_fig2_entropy_curves.png", bbox_inches="tight")
    print("Saved Figure 2: Entropy Curves")


def fig_comparison_bar():
    """Figure 3: Effective rank comparison across corpora."""
    fig, ax = plt.subplots(figsize=(10, 6))

    corpora = [
        "PCFG Sintético",
        "Rongorongo-like",
        "Indus Real",
        "Posicional Sintético",
    ]
    real = [11.18, 15.40, 15.69, 14.74]
    permuted = [14.18, 15.07, 15.66, 14.71]
    random = [14.15, 15.15, 15.35, 14.56]

    x = np.arange(len(corpora))
    width = 0.25

    ax.bar(
        x - width,
        real,
        width,
        label="Real",
        color="#2E86AB",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x,
        permuted,
        width,
        label="Permutado",
        color="#A23B72",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + width,
        random,
        width,
        label="Aleatorio uniforme",
        color="#C73E1D",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_ylabel("Rango Efectivo", fontsize=11)
    ax.set_title(
        "Comparación de Rango Efectivo: Corpus Reales vs. Controles",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(corpora, fontsize=9)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    # Annotate which pass/fail
    for i, (r, rnd) in enumerate(zip(real, random)):
        status = "PASS" if r < rnd else "FAIL"
        color = "green" if r < rnd else "red"
        ax.text(
            i,
            max(r, permuted[i], rnd) + 0.3,
            status,
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=color,
        )

    plt.tight_layout()
    plt.savefig("reports/figures/paper_fig3_comparison.pdf", bbox_inches="tight")
    plt.savefig("reports/figures/paper_fig3_comparison.png", bbox_inches="tight")
    print("Saved Figure 3: Comparison")


def fig_network_communities():
    """Figure 4: Network communities (simplified visualization)."""
    import networkx as nx

    comm_df = pd.read_csv("reports/tables/network_communities_indus.csv")
    pos_df = pd.read_csv("reports/tables/positional_analysis_indus.csv")

    # Build simplified graph
    G = nx.Graph()

    # Add nodes with positional info
    for _, row in pos_df.head(40).iterrows():
        token = row["token"]
        pos = row["mean_relative_position"]
        G.add_node(token, pos=pos)

    # Add edges from communities
    for _, row in comm_df.iterrows():
        nodes = row["nodes"].split()[:10]  # Limit for visibility
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if nodes[i] in G and nodes[j] in G:
                    G.add_edge(nodes[i], nodes[j])

    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G, k=1.2, iterations=50, seed=42)

    # Color by positional bias
    node_colors = []
    for n in G.nodes():
        p = G.nodes[n]["pos"]
        if p < 0.33:
            node_colors.append("#2E86AB")
        elif p > 0.66:
            node_colors.append("#F18F01")
        else:
            node_colors.append("#A23B72")

    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=400, alpha=0.9, ax=ax
    )
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax, width=0.5)
    nx.draw_networkx_labels(G, pos, font_size=7, font_family="monospace", ax=ax)

    ax.set_title(
        "Comunidades de Co-ocurrencia del Indo (top 40 signos)\n"
        "Azul=inicio, Naranja=fin, Púrpura=medio",
        fontsize=12,
        fontweight="bold",
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig("reports/figures/paper_fig4_network.pdf", bbox_inches="tight")
    plt.savefig("reports/figures/paper_fig4_network.png", bbox_inches="tight")
    print("Saved Figure 4: Network Communities")


def main():
    from pathlib import Path

    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    fig_positional_bias()
    fig_entropy_curves()
    fig_comparison_bar()
    fig_network_communities()

    print("\nAll paper figures generated successfully!")


if __name__ == "__main__":
    main()
