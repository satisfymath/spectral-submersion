"""Render glifos Rongorongo estilo Barthel para visualización en paper (v2).

Versión corregida que renderiza directamente en cada subplot.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Ellipse, Polygon
import numpy as np


def draw_glyph_on_axis(ax, code: str):
    """Dibuja un glifo directamente sobre un axis dado."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # Borde
    ax.add_patch(
        patches.Rectangle(
            (0.5, 0.5), 9, 9, fill=False, linewidth=0.5, color="gray", linestyle="--"
        )
    )

    prefix = code[0] if code else "d"
    try:
        variant = int(code[1:]) if len(code) > 1 else 1
    except ValueError:
        variant = 1

    if prefix == "d":
        _draw_determiner(ax, variant)
    elif prefix == "n":
        _draw_noun(ax, variant)
    elif prefix == "v":
        _draw_verb(ax, variant)
    elif prefix == "p":
        _draw_name(ax, variant)
    elif prefix == "x":
        _draw_particle(ax, variant)
    elif prefix == "m":
        _draw_numeral(ax, variant)
    else:
        ax.text(5, 5, code, ha="center", va="center", fontsize=10, fontweight="bold")

    ax.text(5, 0.3, code, ha="center", va="top", fontsize=7, color="gray")


def _draw_determiner(ax, variant):
    if variant == 1:
        ax.add_patch(Circle((5, 5), 2, fill=False, linewidth=2, color="black"))
        ax.add_patch(Circle((5, 5), 0.5, fill=True, color="black"))
    elif variant == 2:
        ax.add_patch(Ellipse((5, 5), 3, 4, fill=False, linewidth=2, color="black"))
    elif variant == 3:
        ax.add_patch(
            FancyBboxPatch(
                (3, 3),
                4,
                4,
                boxstyle="round,pad=0.3",
                fill=False,
                linewidth=2,
                color="black",
            )
        )
    elif variant == 4:
        ax.add_patch(
            Polygon(
                [(5, 7.5), (2.5, 2.5), (7.5, 2.5)],
                fill=False,
                linewidth=2,
                color="black",
            )
        )
    elif variant == 5:
        ax.plot([5, 5], [2.5, 7.5], "k-", linewidth=2)
        ax.plot([2.5, 7.5], [5, 5], "k-", linewidth=2)
    else:
        ax.add_patch(
            Polygon(
                [(5, 8), (2, 5), (5, 2), (8, 5)], fill=False, linewidth=2, color="black"
            )
        )


def _draw_noun(ax, variant):
    if variant == 1:
        ax.add_patch(Ellipse((5, 7.5), 2, 1.5, fill=False, linewidth=2, color="black"))
        ax.add_patch(Circle((3.8, 7.5), 0.4, fill=True, color="black"))
        ax.add_patch(Circle((6.2, 7.5), 0.4, fill=True, color="black"))
        ax.plot([5, 5], [6.5, 4], "k-", linewidth=2)
        ax.plot([3, 7], [5.5, 5.5], "k-", linewidth=2)
        ax.plot([5, 3.5], [4, 2.5], "k-", linewidth=2)
        ax.plot([5, 6.5], [4, 2.5], "k-", linewidth=2)
    elif variant == 2:
        ax.add_patch(
            Ellipse((5, 7.5), 1.8, 1.3, fill=False, linewidth=2, color="black")
        )
        ax.add_patch(Circle((4, 7.5), 0.35, fill=True, color="black"))
        ax.add_patch(Circle((6, 7.5), 0.35, fill=True, color="black"))
        ax.plot([5, 5], [6.5, 4], "k-", linewidth=2)
        ax.plot([3.5, 5, 6.5], [5.5, 6.8, 5.5], "k-", linewidth=2)
        ax.plot([5, 3.5], [4, 2.5], "k-", linewidth=2)
        ax.plot([5, 6.5], [4, 2.5], "k-", linewidth=2)
    elif variant == 3:
        ax.add_patch(
            Polygon(
                [(2, 5), (5, 7), (8, 5), (5, 3)], fill=False, linewidth=2, color="black"
            )
        )
        ax.plot([8, 9], [5, 6.5], "k-", linewidth=2)
        ax.plot([8, 9], [5, 3.5], "k-", linewidth=2)
        ax.add_patch(Circle((3.5, 5.5), 0.3, fill=True, color="black"))
    elif variant == 4:
        ax.add_patch(Ellipse((5, 5), 4, 3, fill=False, linewidth=2, color="black"))
        ax.plot([3, 2], [5, 6.5], "k-", linewidth=2)
        ax.plot([3, 2], [5, 3.5], "k-", linewidth=2)
        ax.add_patch(Circle((4, 5.5), 0.3, fill=True, color="black"))
        ax.add_patch(Circle((6, 5.5), 0.3, fill=True, color="black"))
    elif variant == 5:
        ax.plot([5, 5], [2, 7], "k-", linewidth=2.5)
        for y in [4, 5.5, 6.5]:
            ax.plot([5, 3], [y, y + 1], "k-", linewidth=2)
            ax.plot([5, 7], [y, y + 1], "k-", linewidth=2)
        ax.add_patch(Circle((5, 7.5), 0.5, fill=True, color="black"))
    elif variant == 6:
        ax.add_patch(Ellipse((5, 6), 3, 2, fill=False, linewidth=2, color="black"))
        ax.plot([6.5, 8], [6, 7.5], "k-", linewidth=2)
        ax.add_patch(Circle((4.2, 6.2), 0.25, fill=True, color="black"))
        ax.plot([5, 3], [5, 3.5], "k-", linewidth=2)
        ax.plot([5, 7], [5, 3.5], "k-", linewidth=2)
        ax.plot([3.5, 2.5], [4.5, 5.5], "k-", linewidth=2)
        ax.plot([6.5, 7.5], [4.5, 5.5], "k-", linewidth=2)
    else:
        ax.add_patch(Circle((5, 6), 1.5, fill=False, linewidth=2, color="black"))
        ax.plot([5, 5], [4.5, 2.5], "k-", linewidth=2)
        ax.plot([5, 3], [3.5, 2], "k-", linewidth=2)
        ax.plot([5, 7], [3.5, 2], "k-", linewidth=2)


def _draw_verb(ax, variant):
    if variant == 1:
        ax.add_patch(Circle((5, 6), 1.5, fill=False, linewidth=2, color="black"))
        for angle in [45, 90, 135, 180, 225, 270, 315]:
            rad = np.radians(angle)
            x1, y1 = 5 + 1.5 * np.cos(rad), 6 + 1.5 * np.sin(rad)
            x2, y2 = 5 + 2.5 * np.cos(rad), 6 + 2.5 * np.sin(rad)
            ax.plot([x1, x2], [y1, y2], "k-", linewidth=2)
    elif variant == 2:
        x = np.linspace(2, 8, 50)
        y = 5 + 1.5 * np.sin(x)
        ax.plot(x, y, "k-", linewidth=2)
        y2 = 5 + 1.5 * np.sin(x + 1)
        ax.plot(x, y2, "k-", linewidth=2)
    elif variant == 3:
        ax.arrow(
            2, 5, 6, 0, head_width=1, head_length=1, fc="none", ec="black", linewidth=2
        )
    elif variant == 4:
        theta = np.linspace(0, 4 * np.pi, 100)
        r = 0.3 + 0.2 * theta
        ax.plot(5 + r * np.cos(theta), 5 + r * np.sin(theta), "k-", linewidth=2)
    elif variant == 5:
        ax.plot([2, 3.5, 5, 6.5, 8], [3, 7, 3, 7, 3], "k-", linewidth=2)
    else:
        for _ in range(5):
            x, y = np.random.uniform(2, 8), np.random.uniform(2, 8)
            ax.add_patch(Circle((x, y), 0.3, fill=True, color="black"))


def _draw_name(ax, variant):
    if variant == 1:
        ax.add_patch(
            Ellipse((5, 7.5), 2, 1.5, fill=False, linewidth=2.5, color="black")
        )
        ax.add_patch(Circle((3.8, 7.5), 0.5, fill=True, color="black"))
        ax.add_patch(Circle((6.2, 7.5), 0.5, fill=True, color="black"))
        ax.plot([5, 5], [6.5, 3.5], "k-", linewidth=2.5)
        ax.plot([3, 7], [5, 5], "k-", linewidth=2.5)
        ax.plot([5, 3], [3.5, 2], "k-", linewidth=2)
        ax.plot([5, 7], [3.5, 2], "k-", linewidth=2)
        ax.plot([2, 8], [1.5, 1.5], "k-", linewidth=3)
    elif variant == 2:
        ax.add_patch(Ellipse((4, 7), 1.5, 1.2, fill=False, linewidth=2, color="black"))
        ax.add_patch(Ellipse((6, 7), 1.5, 1.2, fill=False, linewidth=2, color="black"))
        ax.plot([4, 4], [6, 4], "k-", linewidth=2)
        ax.plot([6, 6], [6, 4], "k-", linewidth=2)
        ax.plot([2, 8], [1.5, 1.5], "k-", linewidth=3)
    elif variant == 3:
        ax.add_patch(Ellipse((5, 7), 2, 1.5, fill=False, linewidth=2, color="black"))
        ax.plot([3.5, 6.5], [8, 8], "k-", linewidth=2.5)
        ax.plot([5, 5], [6, 3.5], "k-", linewidth=2)
        ax.plot([2, 8], [1.5, 1.5], "k-", linewidth=3)
    else:
        ax.add_patch(Circle((5, 6), 2, fill=False, linewidth=2.5, color="black"))
        ax.plot([5, 5], [4, 2.5], "k-", linewidth=2.5)
        ax.plot([2, 8], [1.5, 1.5], "k-", linewidth=3)


def _draw_particle(ax, variant):
    if variant == 1:
        ax.add_patch(Circle((5, 5), 0.8, fill=True, color="black"))
    elif variant == 2:
        ax.add_patch(Circle((4, 5), 0.6, fill=True, color="black"))
        ax.add_patch(Circle((6, 5), 0.6, fill=True, color="black"))
    elif variant == 3:
        ax.plot([2, 8], [5, 5], "k-", linewidth=3)
    elif variant == 4:
        ax.plot([2, 5, 8], [4, 6.5, 4], "k-", linewidth=2.5)
    elif variant == 5:
        ax.plot([2, 5, 8], [6, 3.5, 6], "k-", linewidth=2.5)
    else:
        ax.plot([3, 7], [5, 5], "k-", linewidth=2)
        ax.plot([5, 5], [3, 7], "k-", linewidth=2)


def _draw_numeral(ax, variant):
    if variant == 1:
        ax.plot([5, 5], [2.5, 7.5], "k-", linewidth=4)
    elif variant == 2:
        ax.plot([4, 4], [2.5, 7.5], "k-", linewidth=3)
        ax.plot([6, 6], [2.5, 7.5], "k-", linewidth=3)
    elif variant == 3:
        for x in [3.5, 5, 6.5]:
            ax.plot([x, x], [2.5, 7.5], "k-", linewidth=2.5)
    elif variant == 4:
        for x, y in [(4, 6.5), (6, 6.5), (4, 3.5), (6, 3.5)]:
            ax.add_patch(Circle((x, y), 0.5, fill=True, color="black"))
    elif variant == 5:
        for angle in [72 * i for i in range(5)]:
            rad = np.radians(angle - 90)
            ax.plot(
                [5, 5 + 2.5 * np.cos(rad)],
                [5, 5 + 2.5 * np.sin(rad)],
                "k-",
                linewidth=2.5,
            )
    else:
        for i in range(min(variant, 7)):
            x = 2.5 + i * 1.2
            ax.plot([x, x], [3, 7], "k-", linewidth=2)


def render_sequence(glyph_codes: list[str], output_path: str, title: str = ""):
    """Renderiza una secuencia de glifos en una sola imagen."""
    n = len(glyph_codes)
    fig, axes = plt.subplots(1, n, figsize=(1.8 * n, 2.2), dpi=200)
    if n == 1:
        axes = [axes]

    for i, code in enumerate(glyph_codes):
        draw_glyph_on_axis(axes[i], code)

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return output_path


def render_glyph_catalog(output_dir: str = "reports/figures/rongorongo_glyphs_v2"):
    """Genera catálogo y ejemplos."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Ejemplos clave
    examples = [
        (["d03", "n01", "v07", "x02", "d03", "n03"], "te tangata haere ki te moana"),
        (["d01", "n01", "v01", "x02", "d01", "n07"], "he vahine noho i te hare"),
        (["d01", "p01", "p14", "x01", "v01", "n01"], "ko Hotu Matua e kai ika"),
        (["d03", "n06", "v19", "x01", "d01", "n09"], "te manu haere ki te raa"),
        (["d07", "n01", "v01", "x01", "d03", "n26"], "e haere au ki te maunga"),
        (["d03", "n01", "v07", "x02", "d03", "n01"], "te tangata haere ki te tangata"),
    ]

    seq_dir = out / "sequences"
    seq_dir.mkdir(exist_ok=True)
    for glyphs, source in examples:
        render_sequence(
            glyphs, seq_dir / f"seq_{source.replace(' ', '_')}.png", title=f"'{source}'"
        )
        print(f"  Generated: {source}")

    # Glifos individuales representativos
    singles = [
        "d01",
        "d03",
        "n01",
        "n06",
        "v01",
        "v07",
        "p01",
        "p14",
        "x02",
        "x05",
        "m01",
        "m03",
    ]
    single_dir = out / "glyphs"
    single_dir.mkdir(exist_ok=True)
    for code in singles:
        fig, ax = plt.subplots(figsize=(1.5, 1.5), dpi=150)
        draw_glyph_on_axis(ax, code)
        fig.savefig(single_dir / f"{code}.png", bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    print(f"\nGenerated glyph visuals in {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/figures/rongorongo_glyphs_v2")
    args = parser.parse_args()
    render_glyph_catalog(args.output)


if __name__ == "__main__":
    main()
