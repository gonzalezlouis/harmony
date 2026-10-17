"""Shared figure style, palette and glyph registry for every panel

Shared setup for the final panel scripts.

Each panel_*.py reads the frozen numbers from dat/final_data.json (produced
by ../compute_final.py) and renders ONE panel to PDF+PNG. No panel recomputes;
the estimator run is separated from the drawing so the figures are deterministic
and fast to restyle.

Canonical conventions (do not change per-panel):
  * estimator: sigma_bidirectional (one-way edges excluded, no pseudocount)
  * era palette from the main paper (ERA_COL); Wong-2011 quality palette (QUAL)
  * NO titles, NO in-axis annotations beyond the single headline number
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "dat", "final_data.json")
OUT  = os.path.join(os.path.dirname(HERE), "fig", "panels")
os.makedirs(OUT, exist_ok=True)

ERA_COL = {"Renaissance": "#2D6E5E", "Baroque": "#1F5582", "Classical": "#3D8AAF",
           "Romantic": "#C8860A", "Ragtime": "#C75146", "Modern": "#6A3D8A"}
MODCOL  = "#6A3D8A"
QUAL    = {"major": "#0077BB", "minor": "#009988", "dom7": "#CC3311", "dim": "#7B4F9E"}
GREY    = "#8a9199"

# ---------------------------------------------------------------------------
# PER-CORPUS palette + marker glyph, shared by Fig 3a (panel_4a_matrices) and
# Fig 3c (panel_4c_scaling).  ONE source so a corpus's colour AND shape mean the
# same thing in both panels -- that pairing is what lets a reader tie the marker
# beside a matrix to the point in the scatter without a label anywhere.
#
# The colours are 4a's existing COL dict, moved here unchanged (NOT new colours).
# ERA_COL above is the six-ERA palette for the main paper; this is the eight-CORPUS
# palette, which needs two distinct modern-pop hues that ERA_COL does not carry.
CORPUS_COL = {"Renaissance": "#2E8B70", "Baroque": "#2F5C9E", "Classical": "#4FA8C0",
              "Romantic": "#D9A227", "Ragtime": "#C64A3F",
              "CoCoPops": "#B69AD8", "POP909": "#8C63C4", "Jazz": "#4A2C7A"}
# Shape carries the identity redundantly with hue, so the pairing survives
# greyscale printing and the two purples (CoCoPops/POP909) stay distinguishable.
CORPUS_MARKER = {"Renaissance": "o", "Baroque": "s", "Classical": "^",
                 "Romantic": "D", "Ragtime": "v",
                 "CoCoPops": "P", "POP909": "X", "Jazz": "*"}
# Matplotlib's markersize is a LINEAR dimension, so glyphs of different shape carry
# very different ink at the same value: measured against the circle at ms=14 (200 dpi,
# black on white, pixels below mid-grey), the square reads 1.24x, the triangles 0.68x
# and the Jazz star only 0.44x.  The star therefore looked markedly smaller than every
# other corpus in any panel that used one size for all eight.  These multipliers are
# sqrt(area_circle / area_glyph), so ms * MARKER_SCALE[corpus] equalizes ink area.
MARKER_SCALE = {"Renaissance": 1.000, "Baroque": 0.897, "Classical": 1.211,
                "Romantic": 0.912, "Ragtime": 1.211,
                "CoCoPops": 1.160, "POP909": 1.093, "Jazz": 1.504}


def msize(corpus, base):
    """Point size for `corpus`'s marker that reads at the same ink as a circle of
    size `base`.  Use everywhere a corpus glyph is drawn."""
    return base * MARKER_SCALE.get(corpus, 1.0)


def mscatter(corpus, base):
    """Same correction for scatter(), whose `s` is an AREA in points^2."""
    return (base ** 2) * (MARKER_SCALE.get(corpus, 1.0) ** 2)
CORPUS_ORDER = ["Renaissance", "Baroque", "Classical", "Romantic", "Ragtime",
                "CoCoPops", "POP909", "Jazz"]
# Reader-facing corpus names (the dataset keys are internal). Matches 4a's LABEL dict.
CORPUS_LABEL = {"CoCoPops": "Billboard pop", "POP909": "Mandopop"}
ENGINE_ORDER = ["V7\u2192I cadence", "ii\u2192V cycle", "other desc-5th", "unclassified"]
ENGINE_COL   = [QUAL["major"], QUAL["dom7"], "#C8860A", "#c9ccd1"]

def style():
    """Canonical style for EVERY panel: scienceplots 'science', ALL FOUR SPINES
    visible, one size ladder, sans-serif, no LaTeX.  Load once at the top of each
    panel script so the whole figure set is visually identical."""
    import scienceplots  # noqa: F401  registers the 'science' style
    plt.style.use(["science"])
    mpl.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.family": "sans-serif", "text.usetex": False, "mathtext.fontset": "dejavusans",
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 9,
        "legend.fontsize": 7, "xtick.labelsize": 8, "ytick.labelsize": 8,
        # ALL spines visible (scienceplots hides top/right by default)
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.spines.left": True, "axes.spines.bottom": True,
        "axes.linewidth": 0.6, "xtick.top": True, "ytick.right": True,
    })

def all_spines(ax):
    """Force all four spines visible on an axis (use after any broken-axis edits)."""
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)

def load():
    return json.load(open(DATA))

# ---------------------------------------------------------------- fixed aspect
# Every SUPPLEMENTARY figure is written at one aspect ratio so the set does not
# jump shape from page to page.  Saving with bbox_inches="tight" alone does not
# achieve this: it crops to whatever the artists occupy, so a panel with a long
# tick label comes out a different shape from one without.  Measured on the old
# supplement the set ranged 0.93 to 1.86.  Take the tight bounding box, which by
# construction clips nothing, then expand it symmetrically on the short side.
# MAIN-TEXT figures are unaffected: save() only does this when asked.
SUPP_ASPECT = 3.35 / 2.55          # the single-column supplementary panel, 1.3137

def fixed_aspect_bbox(fig, aspect=SUPP_ASPECT, pad=0.04):
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer()).padded(pad)
    w, h = bb.width, bb.height
    if w / h < aspect:
        w2, h2 = h * aspect, h
    else:
        w2, h2 = w, w / aspect
    cx, cy = 0.5 * (bb.x0 + bb.x1), 0.5 * (bb.y0 + bb.y1)
    return Bbox.from_extents(cx - w2 / 2, cy - h2 / 2, cx + w2 / 2, cy + h2 / 2)


def save(fig, name, aspect=None):
    # dpi= is LOAD-BEARING for the PDF even though vector art ignores it: any artist
    # marked set_rasterized(True) (4a's hexbin) and any imshow'd bitmap (Fig 3's engraved
    # scores) is embedded at THIS resolution. Without it matplotlib uses the figure's
    # default 100 dpi, so those layers went into the PDF visibly blurred while the PNG --
    # written at 300 below -- looked fine. 600 keeps rasterized layers sharp in print.
    _bb = fixed_aspect_bbox(fig, aspect) if aspect else "tight"
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches=_bb, dpi=600)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches=_bb)
    plt.close(fig)


def overlap_report(fig, ax, min_frac=0.02):
    """Flag REAL text collisions in a rendered axes. Returns [] when clean.

    Written after two panels shipped with colliding labels that visual inspection
    passed. Filters, each of which cost a false negative when it was wrong:
      - unfilled patches (guide rings) are skipped
      - a patch is 'huge' (a ring, not a node) only if wider than 40% of the axes;
        an earlier version keyed off text width and discarded real node dots
      - self-containment requires >=90% of the text bbox inside the patch (a node's
        own label); merely having the text CENTRE inside was too loose and ate the
        exact collisions the user had spotted
    ALWAYS negative-control this against a deliberately broken build before trusting
    a clean result -- a detector that cannot catch a known defect is worthless.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ab = ax.get_window_extent(renderer=r)
    T = [(t.get_text(), t.get_window_extent(renderer=r))
         for t in ax.texts if t.get_text().strip()]
    def _inter(b1, b2):
        return (max(0, min(b1.x1, b2.x1) - max(b1.x0, b2.x0)) *
                max(0, min(b1.y1, b2.y1) - max(b1.y0, b2.y0)))
    out = []
    for i, (s1, b1) in enumerate(T):
        for s2, b2 in T[i + 1:]:
            I = _inter(b1, b2)
            if I > 0:
                out.append(("TEXT-TEXT", s1, s2, round(I, 1)))
    for s, b in T:
        area = b.width * b.height
        for p in ax.patches:
            fc = p.get_facecolor()
            if len(fc) == 4 and fc[3] == 0:
                continue
            pb = p.get_window_extent(renderer=r)
            if pb.width > 0.40 * ab.width:
                continue
            I = _inter(b, pb)
            if I <= 0 or I >= 0.90 * area:
                continue
            if I > min_frac * area:
                out.append(("TEXT-DOT", s, round(I, 1)))
    return out
