import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pypdfium2 as pdfium
import verovio
import cairosvg
from pypdf import PdfReader, PdfWriter, Transformation
import _figure_style as C

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "dat")
OUT = os.path.join(ROOT, "fig", "panels")
TMP = os.path.join(OUT, "_fig3_parts")
os.makedirs(TMP, exist_ok=True)

E = json.load(open(os.path.join(DATA, "irish_examples.json")))
ir = C.load()["irish"]

VEROVIO_OPTS = {"pageWidth": 2600, "pageHeight": 330, "scale": 42,
                "adjustPageHeight": True, "adjustPageWidth": False,
                "header": "none", "footer": "none", "svgViewBox": True,
                "breaks": "none", "spacingStaff": 1, "font": "Leipzig"}
tk = verovio.toolkit()
tk.setOptions(VEROVIO_OPTS)
for key, tag in [("functional", "fun"), ("modal", "mod")]:
    tk.loadData(E[key]["abc"])
    cairosvg.svg2pdf(bytestring=tk.renderToSVG(1).encode(),
                     write_to=os.path.join(TMP, f"{tag}.pdf"))

PAD_PT = 2.0

def crop_to_ink(src, dst, scale=6.0):
    pg = pdfium.PdfDocument(src)[0]
    W, H = pg.get_size()
    grey = np.asarray(pg.render(scale=scale).to_pil().convert("L"), float)
    ys, xs = np.where(grey < 246)
    x0, x1 = xs.min() / scale, (xs.max() + 1) / scale
    y0, y1 = H - (ys.max() + 1) / scale, H - ys.min() / scale
    page = PdfReader(src).pages[0]
    page.mediabox.lower_left = (x0 - PAD_PT, y0 - PAD_PT)
    page.mediabox.upper_right = (x1 + PAD_PT, y1 + PAD_PT)
    page.cropbox = page.mediabox
    w = PdfWriter(); w.add_page(page); w.write(dst)

for tag in ("fun", "mod"):
    crop_to_ink(os.path.join(TMP, f"{tag}.pdf"), os.path.join(TMP, f"{tag}_crop.pdf"))

_ref = PdfReader(os.path.join(TMP, "fun_crop.pdf")).pages[0].mediabox
ASPECT = float(_ref.height) / float(_ref.width)

C.style()
W_IN, H_BAR = 4.6, 2.35
AX_RECT_X, AX_RECT_W = 0.135, 0.845
_probe = plt.figure(figsize=(W_IN, 4.0))
_probe.add_axes([AX_RECT_X, 0.2, AX_RECT_W, 0.6])
_probe.canvas.draw()
_bb = _probe.axes[0].get_window_extent().transformed(
    _probe.dpi_scale_trans.inverted())
AX_L, SCORE_W = _bb.x0, _bb.width
plt.close(_probe)

H_SC = SCORE_W * ASPECT
LABEL_H, GAP, BOTGAP = 0.15, 0.14, 0.10
H_TOT = 2 * (LABEL_H + H_SC) + GAP + BOTGAP + 0.06 + H_BAR + 0.55
Y_CAP = [H_TOT - 0.06, H_TOT - 0.06 - (LABEL_H + H_SC) - GAP]

CAPTION = [
    "functional:  " + " \u2013 ".join(E["functional"]["strain1_chords"]).replace("#", "\u266f"),
    "modal:  " + " \u2013 ".join(E["modal"]["strain1_chords"]).replace("#", "\u266f"),
]

GROUPS = ["Major (functional)", "Dorian/Mixolydian (modal)", "Minor"]
DISP = ["Major\n(functional)", "Dorian / Mixolydian\n(modal)", "Minor\n(functional)"]
xs = np.arange(3)
bw = 0.36

_EXP = json.load(open(os.path.join(DATA, "irish_expansion.json")))["step4_errors"]["existing_bars"]
_herr = np.array([[ir[g]["h"] - _EXP[g]["h_ci"][0] for g in GROUPS],
                  [_EXP[g]["h_ci"][1] - ir[g]["h"] for g in GROUPS]])
_merr = np.array([[ir[g]["m"] - _EXP[g]["m_ci"][0] for g in GROUPS],
                  [_EXP[g]["m_ci"][1] - ir[g]["m"] for g in GROUPS]])
_EB = dict(ecolor="#3b4148", elinewidth=0.6, capsize=1.5, capthick=0.6)

fig = plt.figure(figsize=(W_IN, H_TOT))
ax = fig.add_axes([AX_RECT_X, 0.55 / H_TOT, AX_RECT_W, H_BAR / H_TOT])
ax.bar(xs - bw / 2, [ir[g]["h"] for g in GROUPS], bw, color=C.QUAL["major"],
       label="harmonic", zorder=3, yerr=_herr, error_kw=_EB)
ax.bar(xs + bw / 2, [ir[g]["m"] for g in GROUPS], bw, color=C.GREY,
       label="melodic", zorder=3, yerr=_merr, error_kw=_EB)
ax.set_xticks(xs); ax.set_xticklabels(DISP, fontsize=5.8)
ax.set_ylabel(r"$\sigma$ (nats step$^{-1}$)", fontsize=7.5)
ax.tick_params(labelsize=6.4)
ax.legend(frameon=False, loc="upper right", fontsize=6.2)
C.all_spines(ax)
for cap, y in zip(CAPTION, Y_CAP):
    fig.text(AX_L / W_IN, y / H_TOT, cap, fontsize=6.0, va="top", ha="left")

bars_pdf = os.path.join(TMP, "bars.pdf")
with matplotlib.rc_context({"savefig.bbox": "standard", "savefig.pad_inches": 0.0}):
    fig.savefig(bars_pdf)
plt.close(fig)

writer = PdfWriter()
writer.add_page(PdfReader(bars_pdf).pages[0])
page = writer.pages[0]
for tag, y_cap in (("fun", Y_CAP[0]), ("mod", Y_CAP[1])):
    src = PdfReader(os.path.join(TMP, f"{tag}_crop.pdf")).pages[0]
    sc = (SCORE_W * 72) / float(src.mediabox.width)
    page.merge_transformed_page(src, Transformation().scale(sc).translate(
        AX_L * 72 - float(src.mediabox.left) * sc,
        (y_cap - LABEL_H - H_SC) * 72 - float(src.mediabox.bottom) * sc))
writer.write(os.path.join(OUT, "panel_3_irish.pdf"))

pdfium.PdfDocument(os.path.join(OUT, "panel_3_irish.pdf"))[0] \
      .render(scale=300 / 72).to_pil().save(os.path.join(OUT, "panel_3_irish.png"))

_page = PdfReader(os.path.join(OUT, "panel_3_irish.pdf")).pages[0]
_xo = _page.get("/Resources", {}).get("/XObject", {})
_n_raster = sum(1 for _, v in (_xo.items() if hasattr(_xo, "items") else [])
                if v.get_object().get("/Subtype") == "/Image")
print("wrote panel_3_irish  %.2f x %.2f in  raster_images=%d  |  %s"
      % (float(_page.mediabox.width) / 72, float(_page.mediabox.height) / 72, _n_raster,
         ", ".join(f"{g.split()[0]} {ir[g]['h'] / ir[g]['m']:.1f}x" for g in GROUPS)))
assert _n_raster == 0, "engravings must be vector, not embedded bitmaps"
