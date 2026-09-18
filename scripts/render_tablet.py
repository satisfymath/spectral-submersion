"""Render Barthel glyph sequences as a rongorongo tablet image using the real
glyph tracings in data/rongorongo/glyph_svgs (PNG rasters).

Reverse boustrophedon: even lines (2, 4, ...) are rotated 180 degrees, as on
the real tablets. Variant suffixes (041h, 305fs, ...) fall back to the base
code's tracing.

Usage:
    python scripts/render_tablet.py --line "041h 670 580 001 004 004 430 022" \
        --out reports/figures/tablilla_propuesta.png [--no-labels] [--title "..."]
    python scripts/render_tablet.py --lines-file lines.txt --out out.png
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

GLYPH_DIR = Path("data/rongorongo/glyph_svgs")
PARCHMENT = "#d9c59a"


def base(tok):
    m = re.match(r"(\d+)", tok)
    return m.group(1).zfill(3) if m else None


def load_glyph(tok, height_px, ink="#141414"):
    code = base(tok)
    p = GLYPH_DIR / f"{code}.png"
    if code is None or not p.exists():
        return None
    im = Image.open(p).convert("RGBA")
    w = int(im.width * height_px / im.height)
    im = im.resize((max(1, w), height_px), Image.LANCZOS)
    # keep only the dark tracing; recolour to near-black ink on transparent
    arr = np.array(im).astype(float)
    alpha = arr[:, :, 3] / 255.0 * (1 - arr[:, :, :3].mean(axis=2) / 255.0)
    out = np.zeros_like(arr)
    r, gch, b = int(ink[1:3], 16), int(ink[3:5], 16), int(ink[5:7], 16)
    out[:, :, 0], out[:, :, 1], out[:, :, 2] = r, gch, b
    out[:, :, 3] = np.clip(alpha, 0, 1) * 255
    return Image.fromarray(out.astype(np.uint8))


def compose(lines, glyph_h=220, gap=28, margin=120, line_gap=70, labels=True,
            title=None, boustrophedon=True, bg=PARCHMENT, ink="#141414"):
    rendered = []
    missing = []
    for seq in lines:
        row = []
        for tok in seq.split():
            g = load_glyph(tok, glyph_h, ink)
            if g is None:
                missing.append(tok)
                continue
            row.append(g)
        rendered.append(row)
    if missing:
        raise SystemExit(f"No tracing for glyphs: {sorted(set(missing))}")

    line_w = [sum(g.width for g in row) + gap * max(0, len(row) - 1) for row in rendered]
    W = max(line_w) + 2 * margin + (160 if labels else 0)
    H = margin * 2 + len(rendered) * glyph_h + (len(rendered) - 1) * line_gap + (120 if title else 0)
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0) if bg == "none" else bg)

    y = margin + (120 if title else 0)
    x0 = margin + (160 if labels else 0)
    for i, row in enumerate(rendered):
        x = x0
        rotate = boustrophedon and (i % 2 == 1)
        # a rotated line is the whole line turned 180°: reversed order, each glyph upside down
        for g in (row[::-1] if rotate else row):
            gg = g.rotate(180) if rotate else g
            canvas.alpha_composite(gg, (x, y))
            x += g.width + gap
        y += glyph_h + line_gap
    return canvas, (labels, title, glyph_h, line_gap, margin, len(rendered))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", action="append", default=[], help="glyph sequence (repeatable)")
    ap.add_argument("--lines-file", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--no-boustrophedon", action="store_true")
    ap.add_argument("--glyph-height", type=int, default=220)
    ap.add_argument("--bg", default=PARCHMENT, help="background hex, or 'none' for transparent")
    ap.add_argument("--ink", default="#141414")
    args = ap.parse_args()

    lines = list(args.line)
    if args.lines_file:
        lines += [l.strip() for l in Path(args.lines_file).read_text().splitlines() if l.strip()]
    if not lines:
        raise SystemExit("no lines")

    canvas, (labels, title, gh, lg, margin, n) = compose(
        lines, glyph_h=args.glyph_height, labels=not args.no_labels, title=args.title,
        boustrophedon=not args.no_boustrophedon, bg=args.bg, ink=args.ink)

    # labels/title via matplotlib on top of the PIL canvas
    dpi = 100
    fig = plt.figure(figsize=(canvas.width / dpi, canvas.height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(canvas)
    ax.axis("off")
    if title:
        ax.text(canvas.width / 2, margin * 0.6, title, ha="center", va="center",
                fontsize=22, color="#4a3a1e")
    if labels:
        y = margin + (120 if title else 0) + gh / 2
        for i in range(n):
            ax.text(margin * 0.5, y, f"L{i + 1:02d}", ha="left", va="center",
                    fontsize=13, fontweight="bold", color="#4a3a1e")
            y += gh + lg
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor="none" if args.bg == "none" else args.bg, transparent=args.bg == "none")
    print(f"Saved {out} ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    main()
