"""Render glifos Rongorongo estilo Barthel para visualización en paper.

Dibuja glifos sintéticos inspirados en las formas reales de Rongorongo
(humanos, pájaros, peces, plantas, geométricos) usando matplotlib.
Cada glifo tiene un código semántico (d01-d15, n01-n35, etc.)
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Ellipse, Polygon, Arc, Wedge
import numpy as np


class RongorongoGlyphRenderer:
    """Renderiza glifos Rongorongo estilo Barthel."""

    def __init__(self, figsize=(1.5, 1.5), dpi=150):
        self.figsize = figsize
        self.dpi = dpi

    def _create_base(self):
        """Crea figura base con estilo limpio."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        return fig, ax

    def _save(self, fig, path):
        """Guarda figura."""
        fig.savefig(path, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)

    def draw_determiner(self, variant=1):
        """Glifos determinantes: formas simples geométricas."""
        fig, ax = self._create_base()
        if variant == 1:  # d01: círculo simple
            ax.add_patch(Circle((5, 5), 2, fill=False, linewidth=2, color='black'))
            ax.add_patch(Circle((5, 5), 0.5, fill=True, color='black'))
        elif variant == 2:  # d02: óvalo
            ax.add_patch(Ellipse((5, 5), 3, 4, fill=False, linewidth=2, color='black'))
        elif variant == 3:  # d03: rectángulo
            ax.add_patch(FancyBboxPatch((3, 3), 4, 4, boxstyle="round,pad=0.3",
                                         fill=False, linewidth=2, color='black'))
        elif variant == 4:  # d04: triángulo
            triangle = Polygon([(5, 7.5), (2.5, 2.5), (7.5, 2.5)],
                              fill=False, linewidth=2, color='black')
            ax.add_patch(triangle)
        elif variant == 5:  # d05: cruz
            ax.plot([5, 5], [2.5, 7.5], 'k-', linewidth=2)
            ax.plot([2.5, 7.5], [5, 5], 'k-', linewidth=2)
        else:  # d06+: rombo
            diamond = Polygon([(5, 8), (2, 5), (5, 2), (8, 5)],
                             fill=False, linewidth=2, color='black')
            ax.add_patch(diamond)
        return fig, ax

    def draw_noun(self, variant=1):
        """Glifos de nombres: humanos, animales, plantas."""
        fig, ax = self._create_base()
        if variant == 1:  # n01: figura humana sentada (proto-anthropomorfo)
            # Cabeza
            ax.add_patch(Ellipse((5, 7.5), 2, 1.5, fill=False, linewidth=2, color='black'))
            # Protuberancias laterales (ojos/orejas características)
            ax.add_patch(Circle((3.8, 7.5), 0.4, fill=True, color='black'))
            ax.add_patch(Circle((6.2, 7.5), 0.4, fill=True, color='black'))
            # Cuerpo
            ax.plot([5, 5], [6.5, 4], 'k-', linewidth=2)
            # Brazos extendidos
            ax.plot([3, 7], [5.5, 5.5], 'k-', linewidth=2)
            # Piernas
            ax.plot([5, 3.5], [4, 2.5], 'k-', linewidth=2)
            ax.plot([5, 6.5], [4, 2.5], 'k-', linewidth=2)
        elif variant == 2:  # n02: figura humana con brazos levantados
            ax.add_patch(Ellipse((5, 7.5), 1.8, 1.3, fill=False, linewidth=2, color='black'))
            ax.add_patch(Circle((4, 7.5), 0.35, fill=True, color='black'))
            ax.add_patch(Circle((6, 7.5), 0.35, fill=True, color='black'))
            ax.plot([5, 5], [6.5, 4], 'k-', linewidth=2)
            ax.plot([3.5, 5, 6.5], [5.5, 6.8, 5.5], 'k-', linewidth=2)
            ax.plot([5, 3.5], [4, 2.5], 'k-', linewidth=2)
            ax.plot([5, 6.5], [4, 2.5], 'k-', linewidth=2)
        elif variant == 3:  # n03: pez
            body = Polygon([(2, 5), (5, 7), (8, 5), (5, 3)],
                          fill=False, linewidth=2, color='black')
            ax.add_patch(body)
            ax.plot([8, 9], [5, 6.5], 'k-', linewidth=2)
            ax.plot([8, 9], [5, 3.5], 'k-', linewidth=2)
            ax.add_patch(Circle((3.5, 5.5), 0.3, fill=True, color='black'))
        elif variant == 4:  # n04: tortuga/marino
            ax.add_patch(Ellipse((5, 5), 4, 3, fill=False, linewidth=2, color='black'))
            ax.plot([3, 2], [5, 6.5], 'k-', linewidth=2)
            ax.plot([3, 2], [5, 3.5], 'k-', linewidth=2)
            ax.add_patch(Circle((4, 5.5), 0.3, fill=True, color='black'))
            ax.add_patch(Circle((6, 5.5), 0.3, fill=True, color='black'))
        elif variant == 5:  # n05: árbol/planta
            ax.plot([5, 5], [2, 7], 'k-', linewidth=2.5)
            for y in [4, 5.5, 6.5]:
                ax.plot([5, 3], [y, y+1], 'k-', linewidth=2)
                ax.plot([5, 7], [y, y+1], 'k-', linewidth=2)
            ax.add_patch(Circle((5, 7.5), 0.5, fill=True, color='black'))
        elif variant == 6:  # n06: ave/frigate
            ax.add_patch(Ellipse((5, 6), 3, 2, fill=False, linewidth=2, color='black'))
            ax.plot([6.5, 8], [6, 7.5], 'k-', linewidth=2)  # pico
            ax.add_patch(Circle((4.2, 6.2), 0.25, fill=True, color='black'))
            ax.plot([5, 3], [5, 3.5], 'k-', linewidth=2)  # pata
            ax.plot([5, 7], [5, 3.5], 'k-', linewidth=2)
            ax.plot([3.5, 2.5], [4.5, 5.5], 'k-', linewidth=2)  # ala
            ax.plot([6.5, 7.5], [4.5, 5.5], 'k-', linewidth=2)
        else:  # n07+: figura genérica
            ax.add_patch(Circle((5, 6), 1.5, fill=False, linewidth=2, color='black'))
            ax.plot([5, 5], [4.5, 2.5], 'k-', linewidth=2)
            ax.plot([5, 3], [3.5, 2], 'k-', linewidth=2)
            ax.plot([5, 7], [3.5, 2], 'k-', linewidth=2)
        return fig, ax

    def draw_verb(self, variant=1):
        """Glifos de verbos: acciones, movimientos."""
        fig, ax = self._create_base()
        if variant == 1:  # v01: mano/pose
            ax.add_patch(Circle((5, 6), 1.5, fill=False, linewidth=2, color='black'))
            for angle in [45, 90, 135, 180, 225, 270, 315]:
                rad = np.radians(angle)
                x1, y1 = 5 + 1.5*np.cos(rad), 6 + 1.5*np.sin(rad)
                x2, y2 = 5 + 2.5*np.cos(rad), 6 + 2.5*np.sin(rad)
                ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2)
        elif variant == 2:  # v02: ola/movimiento
            x = np.linspace(2, 8, 50)
            y = 5 + 1.5*np.sin(x)
            ax.plot(x, y, 'k-', linewidth=2)
            y2 = 5 + 1.5*np.sin(x + 1)
            ax.plot(x, y2, 'k-', linewidth=2)
        elif variant == 3:  # v03: flecha/dirección
            ax.arrow(2, 5, 6, 0, head_width=1, head_length=1, fc='none', ec='black', linewidth=2)
        elif variant == 4:  # v04: espiral
            theta = np.linspace(0, 4*np.pi, 100)
            r = 0.3 + 0.2*theta
            x = 5 + r*np.cos(theta)
            y = 5 + r*np.sin(theta)
            ax.plot(x, y, 'k-', linewidth=2)
        elif variant == 5:  # v05: zigzag
            x = [2, 3.5, 5, 6.5, 8]
            y = [3, 7, 3, 7, 3]
            ax.plot(x, y, 'k-', linewidth=2)
        else:  # v06+: estrella/puntos
            for _ in range(5):
                x, y = np.random.uniform(2, 8), np.random.uniform(2, 8)
                ax.add_patch(Circle((x, y), 0.3, fill=True, color='black'))
        return fig, ax

    def draw_name(self, variant=1):
        """Glifos de nombres propios: figuras distintivas con marcadores."""
        fig, ax = self._create_base()
        if variant == 1:  # p01: figura humana prominente
            ax.add_patch(Ellipse((5, 7.5), 2, 1.5, fill=False, linewidth=2.5, color='black'))
            ax.add_patch(Circle((3.8, 7.5), 0.5, fill=True, color='black'))
            ax.add_patch(Circle((6.2, 7.5), 0.5, fill=True, color='black'))
            ax.plot([5, 5], [6.5, 3.5], 'k-', linewidth=2.5)
            ax.plot([3, 7], [5, 5], 'k-', linewidth=2.5)
            ax.plot([5, 3], [3.5, 2], 'k-', linewidth=2)
            ax.plot([5, 7], [3.5, 2], 'k-', linewidth=2)
            # Marcador de nombre: línea base
            ax.plot([2, 8], [1.5, 1.5], 'k-', linewidth=3)
        elif variant == 2:  # p02: doble figura
            ax.add_patch(Ellipse((4, 7), 1.5, 1.2, fill=False, linewidth=2, color='black'))
            ax.add_patch(Ellipse((6, 7), 1.5, 1.2, fill=False, linewidth=2, color='black'))
            ax.plot([4, 4], [6, 4], 'k-', linewidth=2)
            ax.plot([6, 6], [6, 4], 'k-', linewidth=2)
            ax.plot([2, 8], [1.5, 1.5], 'k-', linewidth=3)
        elif variant == 3:  # p03: figura con tocado
            ax.add_patch(Ellipse((5, 7), 2, 1.5, fill=False, linewidth=2, color='black'))
            ax.plot([3.5, 6.5], [8, 8], 'k-', linewidth=2.5)  # tocado
            ax.plot([5, 5], [6, 3.5], 'k-', linewidth=2)
            ax.plot([2, 8], [1.5, 1.5], 'k-', linewidth=3)
        else:
            ax.add_patch(Circle((5, 6), 2, fill=False, linewidth=2.5, color='black'))
            ax.plot([5, 5], [4, 2.5], 'k-', linewidth=2.5)
            ax.plot([2, 8], [1.5, 1.5], 'k-', linewidth=3)
        return fig, ax

    def draw_particle(self, variant=1):
        """Glifos de partículas: formas simples, conectores."""
        fig, ax = self._create_base()
        if variant == 1:  # x01: punto
            ax.add_patch(Circle((5, 5), 0.8, fill=True, color='black'))
        elif variant == 2:  # x02: doble punto
            ax.add_patch(Circle((4, 5), 0.6, fill=True, color='black'))
            ax.add_patch(Circle((6, 5), 0.6, fill=True, color='black'))
        elif variant == 3:  # x03: línea
            ax.plot([2, 8], [5, 5], 'k-', linewidth=3)
        elif variant == 4:  # x04: chevrón
            ax.plot([2, 5, 8], [4, 6.5, 4], 'k-', linewidth=2.5)
        elif variant == 5:  # x05: V invertida
            ax.plot([2, 5, 8], [6, 3.5, 6], 'k-', linewidth=2.5)
        else:
            ax.plot([3, 7], [5, 5], 'k-', linewidth=2)
            ax.plot([5, 5], [3, 7], 'k-', linewidth=2)
        return fig, ax

    def draw_numeral(self, variant=1):
        """Glifos numerales: barras, cuentas."""
        fig, ax = self._create_base()
        if variant == 1:  # m01: una barra
            ax.plot([5, 5], [2.5, 7.5], 'k-', linewidth=4)
        elif variant == 2:  # m02: dos barras
            ax.plot([4, 4], [2.5, 7.5], 'k-', linewidth=3)
            ax.plot([6, 6], [2.5, 7.5], 'k-', linewidth=3)
        elif variant == 3:  # m03: tres barras
            for x in [3.5, 5, 6.5]:
                ax.plot([x, x], [2.5, 7.5], 'k-', linewidth=2.5)
        elif variant == 4:  # m04: cuatro puntos
            positions = [(4, 6.5), (6, 6.5), (4, 3.5), (6, 3.5)]
            for x, y in positions:
                ax.add_patch(Circle((x, y), 0.5, fill=True, color='black'))
        elif variant == 5:  # m05: cinco rayos
            for angle in [72*i for i in range(5)]:
                rad = np.radians(angle - 90)
                x2 = 5 + 2.5*np.cos(rad)
                y2 = 5 + 2.5*np.sin(rad)
                ax.plot([5, x2], [5, y2], 'k-', linewidth=2.5)
        else:
            for i in range(variant):
                x = 2.5 + i*1.2
                ax.plot([x, x], [3, 7], 'k-', linewidth=2)
        return fig, ax

    def draw_glyph(self, code: str, output_dir: str):
        """Dibuja un glifo por su código y lo guarda."""
        prefix = code[0]
        try:
            variant = int(code[1:])
        except ValueError:
            variant = 1

        if prefix == 'd':
            fig, ax = self.draw_determiner(variant)
        elif prefix == 'n':
            fig, ax = self.draw_noun(variant)
        elif prefix == 'v':
            fig, ax = self.draw_verb(variant)
        elif prefix == 'p':
            fig, ax = self.draw_name(variant)
        elif prefix == 'x':
            fig, ax = self.draw_particle(variant)
        elif prefix == 'm':
            fig, ax = self.draw_numeral(variant)
        else:
            fig, ax = self._create_base()
            ax.text(5, 5, code, ha='center', va='center', fontsize=14, fontweight='bold')

        out_path = Path(output_dir) / f"{code}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._save(fig, out_path)
        return out_path

    def render_sequence(self, glyph_codes: list[str], output_path: str, title: str = ""):
        """Renderiza una secuencia de glifos en una sola imagen (línea de tableta)."""
        n = len(glyph_codes)
        fig, axes = plt.subplots(1, n, figsize=(1.5*n, 1.8), dpi=150)
        if n == 1:
            axes = [axes]

        for i, code in enumerate(glyph_codes):
            ax = axes[i]
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.set_aspect('equal')
            ax.axis('off')

            # Dibujar borde de glifo
            ax.add_patch(patches.Rectangle((0.5, 0.5), 9, 9, fill=False,
                                           linewidth=0.5, color='gray', linestyle='--'))

            prefix = code[0]
            try:
                variant = int(code[1:])
            except ValueError:
                variant = 1

            if prefix == 'd':
                _, ax_temp = self.draw_determiner(variant)
            elif prefix == 'n':
                _, ax_temp = self.draw_noun(variant)
            elif prefix == 'v':
                _, ax_temp = self.draw_verb(variant)
            elif prefix == 'p':
                _, ax_temp = self.draw_name(variant)
            elif prefix == 'x':
                _, ax_temp = self.draw_particle(variant)
            elif prefix == 'm':
                _, ax_temp = self.draw_numeral(variant)
            else:
                ax.text(5, 5, code, ha='center', va='center', fontsize=10)
                continue

            # Transferir patches y líneas al nuevo axis
            for child in ax_temp.get_children():
                if hasattr(child, 'get_path') or hasattr(child, 'get_xydata'):
                    try:
                        ax.add_artist(child)
                    except Exception:
                        pass
            plt.close(ax_temp.figure)
            ax.text(5, 0.3, code, ha='center', va='top', fontsize=7, color='gray')

        if title:
            fig.suptitle(title, fontsize=10, fontweight='bold')

        plt.tight_layout()
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        return output_path


def generate_glyph_catalog(output_dir: str = "reports/figures/rongorongo_glyphs"):
    """Genera catálogo completo de glifos."""
    renderer = RongorongoGlyphRenderer()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Determinantes d01-d15
    for i in range(1, 16):
        renderer.draw_glyph(f"d{i:02d}", out)
    # Nombres n01-n10 (subset representativo)
    for i in range(1, 11):
        renderer.draw_glyph(f"n{i:02d}", out)
    # Verbos v01-v10
    for i in range(1, 11):
        renderer.draw_glyph(f"v{i:02d}", out)
    # Nombres propios p01-p10
    for i in range(1, 11):
        renderer.draw_glyph(f"p{i:02d}", out)
    # Partículas x01-x10
    for i in range(1, 11):
        renderer.draw_glyph(f"x{i:02d}", out)
    # Numerales m01-m10
    for i in range(1, 11):
        renderer.draw_glyph(f"m{i:02d}", out)

    print(f"Generated glyph catalog in {out}")

    # Generate example sequences
    examples = [
        (["d03", "n01", "v07", "x02", "d03", "n03"], "te tangata haere ki te moana"),
        (["d01", "n01", "v01", "x02", "d01", "n07"], "he vahine noho i te hare"),
        (["d01", "p01", "p14", "x01", "v01", "n01"], "ko Hotu Matua e kai ika"),
        (["d03", "n06", "v19", "x01", "d01", "n09"], "te manu haere ki te raa"),
    ]

    seq_dir = out / "sequences"
    seq_dir.mkdir(exist_ok=True)
    for glyphs, source in examples:
        renderer.render_sequence(glyphs, seq_dir / f"seq_{source.replace(' ', '_')}.png",
                                 title=f"{source}")

    print(f"Generated example sequences in {seq_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/figures/rongorongo_glyphs")
    args = parser.parse_args()
    generate_glyph_catalog(args.output)


if __name__ == "__main__":
    main()
