"""Reading key for the proposal line: glyphs coloured by what we can honestly say."""
import sys
sys.path.insert(0, "scripts")
from render_tablet import load_glyph
from PIL import Image, ImageDraw, ImageFont

SEQ = "002 002 004 004 280 280 450 063 063 004 004 002 002 004 004 430 022".split()
INK, BURDEOS, COBRE, AZUL, PARCH = "#3a2e22", "#8a2f2f", "#c2702a", "#2a6f9e", "#ead9b6"
color = {i: INK for i in range(len(SEQ))}
for i in (4, 5): color[i] = BURDEOS      # 280 280 vahine
color[6] = COBRE                          # 450 verb class
for i in (15, 16): color[i] = AZUL        # 430 022 ora tonu formula
H, GAP, M = 300, 34, 90
gl = [load_glyph(t, H, color[i]) for i, t in enumerate(SEQ)]
W = sum(g.width for g in gl) + GAP * (len(gl) - 1) + 2 * M
canvas = Image.new("RGBA", (W, H + 400), PARCH)
# subtle grain
d = ImageDraw.Draw(canvas)
for y in range(0, canvas.height, 7):
    d.line([(0, y), (W, y)], fill=(120, 90, 50, 6))
x, y0 = M, 100
xs = []
for g in gl:
    canvas.alpha_composite(g, (x, y0)); xs.append((x, x + g.width)); x += g.width + GAP
def font(sz, bold=False, italic=False):
    cands = ["/usr/share/fonts/libertinus/LibertinusSerifDisplay-Regular.otf",
             "/usr/share/fonts/libertinus/LibertinusSerif-Regular.otf"]
    import glob
    fs = glob.glob("/usr/share/fonts/**/LibertinusSerif*Regular*.otf", recursive=True) + glob.glob("/usr/share/fonts/**/DejaVuSerif.ttf", recursive=True)
    if italic: fs = glob.glob("/usr/share/fonts/**/LibertinusSerif*Italic*.otf", recursive=True) + fs
    if bold: fs = glob.glob("/usr/share/fonts/**/LibertinusSerif*Semibold*.otf", recursive=True) + glob.glob("/usr/share/fonts/**/DejaVuSerif-Bold.ttf", recursive=True) + fs
    return ImageFont.truetype(fs[0], sz)
def bracket(i0, i1, col, lines, yb):
    a, b = xs[i0][0], xs[i1][1]; 
    d.line([(a, yb), (a, yb + 14), (b, yb + 14), (b, yb)], fill=col, width=4)
    cx = (a + b) / 2
    yy = yb + 30
    for j, (txt, sz, it) in enumerate(lines):
        f = font(sz, italic=it, bold=(j == 0))
        w = d.textlength(txt, font=f); d.text((cx - w / 2, yy), txt, fill=col, font=f); yy += sz + 8
yb = y0 + H + 18
bracket(0, 3, INK, [("ko kou taku…", 30, True), ("partículas y marcas (serie 001–099)", 22, False)], yb)
bracket(4, 5, BURDEOS, [("vahine · esposa", 34, True), ("dos figuras humanas, serie 200–399 de Barthel:", 22, False), ("la única capa del rongorongo con consenso iconográfico", 22, False)], yb)
bracket(6, 6, COBRE, [("aroha · amar", 30, True), ("clase verbo (400–599)", 22, False)], yb)
bracket(7, 14, INK, [("… e aroha au ki a kou …", 30, True), ("espejo de la apertura: 004 004 · 002 002 · 004 004", 22, False)], yb)
bracket(15, 16, AZUL, [("mo te ora tonu · toda la vida", 34, True), ("430 022: la misma fórmula que cierra", 22, False), ("«quiero conocerte toda la vida» en el poema", 22, False)], yb)
ft = font(40, italic=True); t = "Ko kou taku vahine, e aroha au ki a kou, mo te ora tonu"
d.text(((W - d.textlength(t, font=ft)) / 2, 14), t, fill=INK, font=ft)
f2 = font(26); t2 = "«Tú eres mi esposa; te amo para toda la vida»  ·  hipótesis C2 generada por el modelo, no traducción histórica demostrada"
d.text(((W - d.textlength(t2, font=f2)) / 2, canvas.height - 60), t2, fill="#6a5a48", font=f2)
canvas.convert("RGB").save("reports/figures/clave_lectura_peticion.png", quality=95)
print("saved", canvas.size)
