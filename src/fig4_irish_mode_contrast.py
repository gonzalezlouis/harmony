#!/usr/bin/env python3
"""Figure 4 — the Irish mode contrast, with engraved illustrations

Fig 3 - the Irish mode contrast, with FULLY VECTOR engraved illustrations.

THE MEASUREMENT (the bars). Same repertoire (thesession.org ABC), one thing
changed: is the setting FUNCTIONAL (major/minor) or MODAL (Dorian/Mixolydian)?
Every setting carries BOTH a chord stream (harmonic sigma) and a melody stream
(melodic sigma), so the melodic bars are an internal control.

  Major (functional)          harmonic 0.187 vs melodic 0.0193 -> 9.7x
  Dorian/Mixolydian (modal)   harmonic 0.046 vs melodic 0.0564 -> 0.8x  (COLLAPSES)
  (ratios computed from unrounded values; melodic column shown to 4 dp so the
   quoted ratios reproduce -- rounding melodic to 3 dp gives 6.3x for minor)
  Minor (functional)          harmonic 0.133 vs melodic 0.0205 -> 6.5x

THE ENGRAVINGS ARE AN ILLUSTRATION, NOT EVIDENCE. "The Blarney Pilgrim" (tune_id
5, jig) has two settings whose first strains are character-identical in the source
ABC -- all 36 notes -- over which players wrote genuinely different chords:

  functional (53427, G major)      G  D/F#  G  Cadd9  G  D  G  D/F#  Em
  modal      (28409, D mixolydian) G  D     G  Am     Em C  G  D     G  Am  D

D mixolydian is the G-major scale re-centred on D, which is why the same notes
support either reading. This is the ONLY exactly-matched pair in the corpus where
the chords actually differ; the other exact-melody candidates ("Cup Of Tea",
"Garrett Barry's") have IDENTICAL chord streams and differ only in the declared
K: field -- re-labellings, not re-harmonisations.

NO PER-TUNE SIGMA IS PRINTED, DELIBERATELY, and there is no figure title.
Single-tune sigma is not interpretable: 17.1% of 4,696 chorded tunes score
EXACTLY zero (an out-and-back chord walk zeroes every net current by topology,
and edges with an unobserved reverse must be skipped because log(f/0) diverges).
For this pair it runs the WRONG way (functional 0.072 vs modal 0.093), and the
within-tune paired test over the 66 tunes carrying both harmonisations gives
0.054 vs 0.034 (1.58x), 95% CI [-0.032, +0.074], p = 0.24 -- NOT significant.
The mode effect is a corpus-level statistical claim; the engravings show WHAT a
player changes, the bars show that it matters in aggregate.

HOW THE VECTOR COMPOSITE WORKS (this is why the panel is built by hand rather
than with imshow):
  1. verovio renders each strain to SVG -> cairosvg to PDF. True curves, no raster.
  2. Each engraving PDF is cropped to its ink by measuring the rendered page once
     and mapping the ink bbox back to PDF points.
  3. The bar chart is written to its own PDF page with blank space reserved at the
     top. `savefig.bbox` MUST be overridden to "standard": the project style sets
     it to "tight", which re-crops the page and invalidates the placement maths.
  4. pypdf merges the cropped engravings onto that page with an explicit affine
     transform. Result: zero raster XObjects in the output.
The score column geometry is read from the DRAWN axes box, never from the axes
rect literal -- the y-label consumes width, so the rect's left edge is not where
the plotting area starts, and assuming otherwise pushed the staves off the page.
"""
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

# ---------------------------------------------------------------- 1. engrave
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

# ------------------------------------------------- 2. crop each PDF to its ink
PAD_PT = 2.0

def crop_to_ink(src, dst, scale=6.0):
    pg = pdfium.PdfDocument(src)[0]
    W, H = pg.get_size()
    grey = np.asarray(pg.render(scale=scale).to_pil().convert("L"), float)
    ys, xs = np.where(grey < 246)
    x0, x1 = xs.min() / scale, (xs.max() + 1) / scale
    y0, y1 = H - (ys.max() + 1) / scale, H - ys.min() / scale   # flip to PDF origin
    page = PdfReader(src).pages[0]
    page.mediabox.lower_left = (x0 - PAD_PT, y0 - PAD_PT)
    page.mediabox.upper_right = (x1 + PAD_PT, y1 + PAD_PT)
    page.cropbox = page.mediabox
    w = PdfWriter(); w.add_page(page); w.write(dst)

for tag in ("fun", "mod"):
    crop_to_ink(os.path.join(TMP, f"{tag}.pdf"), os.path.join(TMP, f"{tag}_crop.pdf"))

_ref = PdfReader(os.path.join(TMP, "fun_crop.pdf")).pages[0].mediabox
ASPECT = float(_ref.height) / float(_ref.width)

# --------------------------------------- 3. geometry from the DRAWN axes box
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

# ------------------------------------------------------- 4. bars on own page
GROUPS = ["Major (functional)", "Dorian/Mixolydian (modal)", "Minor"]
DISP = ["Major\n(functional)", "Dorian / Mixolydian\n(modal)", "Minor\n(functional)"]
xs = np.arange(3)
bw = 0.36

# Bootstrap CIs on every bar. Resampling unit is the TUNE (tune_id), not the setting:
# thesession.org carries several settings of the same tune, so setting-level resampling
# would treat correlated transcriptions as independent and understate the interval.
# B = 600, percentile 2.5/97.5. From data/irish_expansion.json (step4_errors.existing_bars),
# which also records that the bar point estimates reproduce final_data.json exactly.
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
# "standard" is REQUIRED: the project style sets savefig.bbox="tight", which would
# re-crop the page and break every coordinate used in the merge below.
with matplotlib.rc_context({"savefig.bbox": "standard", "savefig.pad_inches": 0.0}):
    fig.savefig(bars_pdf)
plt.close(fig)

# ------------------------------------------------- 5. merge engravings (vector)
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

# companion raster for slide decks / inline viewing only; the PDF is the deliverable
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
