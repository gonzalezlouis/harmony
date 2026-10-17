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

CORPUS_COL = {"Renaissance": "#2E8B70", "Baroque": "#2F5C9E", "Classical": "#4FA8C0",
              "Romantic": "#D9A227", "Ragtime": "#C64A3F",
              "CoCoPops": "#B69AD8", "POP909": "#8C63C4", "Jazz": "#4A2C7A"}
CORPUS_MARKER = {"Renaissance": "o", "Baroque": "s", "Classical": "^",
                 "Romantic": "D", "Ragtime": "v",
                 "CoCoPops": "P", "POP909": "X", "Jazz": "*"}
MARKER_SCALE = {"Renaissance": 1.000, "Baroque": 0.897, "Classical": 1.211,
                "Romantic": 0.912, "Ragtime": 1.211,
                "CoCoPops": 1.160, "POP909": 1.093, "Jazz": 1.504}


def msize(corpus, base):
    return base * MARKER_SCALE.get(corpus, 1.0)


def mscatter(corpus, base):
    return (base ** 2) * (MARKER_SCALE.get(corpus, 1.0) ** 2)
CORPUS_ORDER = ["Renaissance", "Baroque", "Classical", "Romantic", "Ragtime",
                "CoCoPops", "POP909", "Jazz"]
CORPUS_LABEL = {"CoCoPops": "Billboard pop", "POP909": "Mandopop"}
ENGINE_ORDER = ["V7\u2192I cadence", "ii\u2192V cycle", "other desc-5th", "unclassified"]
ENGINE_COL   = [QUAL["major"], QUAL["dom7"], "#C8860A", "#c9ccd1"]

def style():
    import scienceplots
    plt.style.use(["science"])
    mpl.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.family": "sans-serif", "text.usetex": False, "mathtext.fontset": "dejavusans",
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 9,
        "legend.fontsize": 7, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.spines.left": True, "axes.spines.bottom": True,
        "axes.linewidth": 0.6, "xtick.top": True, "ytick.right": True,
    })

def all_spines(ax):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)

def load():
    return json.load(open(DATA))

SUPP_ASPECT = 3.35 / 2.55

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
    _bb = fixed_aspect_bbox(fig, aspect) if aspect else "tight"
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches=_bb, dpi=600)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches=_bb)
    plt.close(fig)


def overlap_report(fig, ax, min_frac=0.02):
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
