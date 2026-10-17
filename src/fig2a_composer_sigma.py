r"""Figure 2a — sigma per composer, grouped by era

fig2_panel_a.py -- Composer bar chart, standalone vector PDF.

All layout constants are at the top. Change those numbers; nothing else.

STATISTIC: PER-ARTIST POOLED sigma -- the same quantity as every other panel of
Fig 2, evaluated on one artist's pool instead of a whole corpus.  Hence the x
label is simply "sigma (nats step^-1)".

  VERIFIED, 2026-07-31, not assumed.  Every plotted value was recomputed from the
  artist's own score pool both ways.  All 11 art-music rows match POOLED sigma to
  <1e-9 and none matches the per-piece mean:

    artist       plotted   pooled    per-piece mean
    Josquin      0.039483  0.039483  0.175403
    J.S. Bach    0.371218  0.371218  0.162701
    Chopin       0.258302  0.258302  0.135366
    Joplin       0.397383  0.397383  0.179191   (= Ragtime corpus sigma, n=47 both)

  The per-piece mean is nearly FLAT across artists (0.116-0.185) because
  single-piece sigma is dominated by the estimator's censoring, so a per-piece
  panel could not show the Renaissance-to-Baroque contrast this one shows.  The
  three modern rows are likewise corpus-pooled: Mandopop 0.2341557242 and
  Billboard pop 0.1548688676 equal the POP909 and CoCoPops entries of the locked
  canonical list exactly.  Error bars are a 400x piece-level BOOTSTRAP of the
  pooled value (reproduced to 1-4% here), NOT 1.96*SEM over per-piece sigma,
  which would be 2-5x too wide on Bach and Joplin.

  An earlier revision of this docstring described the panel as per-piece mean and
  quoted "Parker 0.044, Ellington 0.058"; the data file holds 1.567 and 0.672,
  which are pooled values.  The docstring was stale, not the data.  Keep this
  panel and panel b on the same estimator: the n-bias of pooled sigma is real and
  is why every bar carries its bootstrap CI and its n.

GRAND MEAN: 0.2377 = piece-weighted mean of the plotted pooled values over the
score-only art-music rows (Renaissance..Ragtime); reproduced to 10 decimals.

Sigma throughout is sigma_bidirectional (pseudocount-free).  The earlier draft of
this panel used the parquet's pseudocount-regularised column, which inflates
sigma ~16-20x by assigning affinity log(1/eps) ~ 13.8 to one-way cadential edges.

Data: dat/composer_sigma.json
Output: ../fig/panels/fig2_panel_a.pdf
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "dat" / "composer_sigma.json"
FIGURES = HERE.parent / "fig" / "panels"
FIGURES.mkdir(exist_ok=True)
plt.rcParams["pdf.fonttype"] = 42

# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT — change these numbers, nothing else
# ═══════════════════════════════════════════════════════════════════════════════

FIGURE_WIDTH_IN  = 2.06   # left column of fig2-complete.ai
FIGURE_HEIGHT_IN = 5.93   # full canvas height of that column

ERA_COL_FRACTION = 0.135    # width fraction for era names + bracket
ERA_LINE_X = 0.82          # colored vertical bracket line (0=left, 1=right)
ERA_TEXT_X = 0.34          # rotated era name text
COMPOSER_NAME_X = -0.014   # composer name x, in bar-axis data coords

ERA_GAP    = 0.42          # gap between era groups, y units
BAR_HEIGHT = 0.60          # bar height, y units (max 1.0)

X_MAX   = 2.45
# 16 of 23 artists sit below 0.35 while Parker reaches 1.57: on a single linear axis the
# classical eras compress into 13% of the width.  A piecewise-linear x-scale gives
# 0-0.35 (every pre-modern composer) 68% of the axis and compresses 0.35-2.45 into the
# rest.  Bars still start at zero and lengths stay monotone in sigma; only the metric
# per inch changes across the break, which the tick spacing makes explicit.
X_BREAK = 0.35
X_FRAC  = 0.68
X_TICKS = [0, 0.1, 0.2, 0.3, 1.0, 2.0]
X_MINOR = [0.05, 0.15, 0.25, 0.5, 1.5]

FS_ERA_LABEL  = 8.0
FS_COMPOSER   = 7.0
FS_N_LABEL    = 5.2
FS_X_LABEL    = 7.5
FS_GRAND_MEAN = 5.0
FS_TICK       = 7.0

# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

# One colour per era — Modern included.  No per-genre colours: the era palette is
# the figure-set convention, and genre is already legible from the artist names.
ERA_COLORS = {
    "Renaissance":   "#3d8b6e",
    "Baroque":       "#3a65b0",
    "Classical":     "#5baabc",
    "Romantic":      "#c9981a",
    "Ragtime": "#c04848",
    "Modern":        "#7252a0",
}
ERA_ORDER = ["Renaissance", "Baroque", "Classical",
             "Romantic", "Ragtime", "Modern"]

_d = json.load(open(DATA))
CURATED = [(r["era"], r["name"], r["sigma"], r["ci"], r["n"]) for r in _d["rows"]]
GMEAN = _d["grand_mean_weighted"]

# ═══════════════════════════════════════════════════════════════════════════════
# DERIVED — do not edit below this line
# ═══════════════════════════════════════════════════════════════════════════════

cursor = 0.0
y_map: dict[str, float] = {}
prev_era = None
for era, name, *_ in CURATED:
    if era != prev_era and prev_era is not None:
        cursor += ERA_GAP
    y_map[name] = cursor
    prev_era = era
    cursor += 1.0
total_y = cursor

era_extents: dict[str, tuple[float, float]] = {}
for era in ERA_ORDER:
    ys = [y_map[nm] for e, nm, *_ in CURATED if e == era]
    if ys:
        era_extents[era] = (min(ys), max(ys))

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         8,
    "axes.linewidth":    0.5,
    "xtick.major.width": 0.5,
    "xtick.major.size":  2.5,
    "ytick.major.size":  0,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
})

fig = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))
gs = gridspec.GridSpec(1, 2, width_ratios=[ERA_COL_FRACTION, 1 - ERA_COL_FRACTION],
                       wspace=0.0, figure=fig)
ax_lbl = fig.add_subplot(gs[0])
ax_bar = fig.add_subplot(gs[1])

ax_lbl.set_xlim(0, 1)
ax_lbl.set_ylim(-0.55, total_y + 0.15)
ax_lbl.invert_yaxis()
ax_lbl.axis("off")

for era, (y0, y1) in era_extents.items():
    col = ERA_COLORS[era]
    ax_lbl.plot([ERA_LINE_X, ERA_LINE_X], [y0 - 0.35, y1 + 0.35],
                color=col, lw=2.0, solid_capstyle="round")
    # A rotated label is taller than a one-row group's span, so a 1-composer era
    # (Ragtime) overflowed into its neighbour. Scale the font to the available span.
    _span = abs(y1 - y0)
    _fs = FS_ERA_LABEL if _span >= 1.8 else FS_ERA_LABEL * 0.78
    ax_lbl.text(ERA_TEXT_X, (y0 + y1) / 2, era, ha="center", va="center",
                fontsize=_fs, fontweight="bold", color=col, rotation=90)

def _fwd(v):
    v = np.asarray(v, dtype=float)
    return np.where(v <= X_BREAK, v / X_BREAK * X_FRAC,
                    X_FRAC + (v - X_BREAK) / (X_MAX - X_BREAK) * (1.0 - X_FRAC))
def _inv(u):
    u = np.asarray(u, dtype=float)
    return np.where(u <= X_FRAC, u / X_FRAC * X_BREAK,
                    X_BREAK + (u - X_FRAC) / (1.0 - X_FRAC) * (X_MAX - X_BREAK))
ax_bar.set_xscale("function", functions=(_fwd, _inv))
ax_bar.set_xlim(-0.34, X_MAX)
ax_bar.set_ylim(-0.55, total_y + 0.15)
ax_bar.invert_yaxis()
for s in ("left", "top", "right"):
    ax_bar.spines[s].set_visible(False)
ax_bar.spines["bottom"].set_bounds(0, X_MAX)
# slash marks on the spine at the scale break, so the change of metric is explicit
# axis break at X_BREAK, positioned by ASKING THE AXIS where that data value lands.
# The previous version placed the marks at X_FRAC (0.68), which is the break's position
# in the piecewise SCALE, not on the drawn axis -- the axis starts at -0.42 to hold the
# label gutter, so the true position is 0.824 and the marks sat 14% too far left.
_bx = ax_bar.transAxes.inverted().transform(
          ax_bar.transData.transform((X_BREAK, 0)))[0]
ax_bar.plot([_bx - 0.014, _bx + 0.014], [0, 0], transform=ax_bar.transAxes,
            color="white", lw=3.0, solid_capstyle="butt", clip_on=False, zorder=11)
for _off in (-0.019, 0.001):
    ax_bar.plot([_bx + _off, _bx + _off + 0.018], [-0.012, 0.012],
                transform=ax_bar.transAxes, color="black", lw=0.9,
                solid_capstyle="round", clip_on=False, zorder=12)

# Grand-mean reference: drawn ONLY in the blank horizontal gaps between era groups,
# so it never crosses a bar, an error bar, a composer name or an n= label.  Each bar
# row occupies +/- BAR_HEIGHT/2 about its y; the era gaps are the space between groups.
_ys = sorted(y_map.values())
_half = BAR_HEIGHT / 2 + 0.14              # bar half-height + clearance for text/error bars
_LABEL_Y = -0.50                            # the "grand mean =" caption row
_segments = []
_top = -0.30                                # start BELOW the caption so it is never grazed
for _y in _ys:                              # gap above each occupied row
    if _y - _half > _top:
        _segments.append((_top, _y - _half))
    _top = max(_top, _y + _half)
if _top < total_y + 0.15:
    _segments.append((_top, total_y + 0.15))
_segments = [(a, b) for a, b in _segments if b - a > 0.05]   # drop slivers
for _y0, _y1 in _segments:
    ax_bar.plot([GMEAN, GMEAN], [_y0, _y1], color="#c8c8c8", lw=0.7,
                ls=(0, (3, 2)), zorder=1, solid_capstyle="butt")
# label sits in the bottom margin, left of the line so it cannot touch it
ax_bar.text(GMEAN + 0.013, -0.50, f"grand mean = {GMEAN:.3f}", ha="left", va="center",
            fontsize=FS_GRAND_MEAN, color="#b8b8b8", clip_on=False)

prev_era, prev_y_v = None, None
for era, name, *_ in CURATED:
    y = y_map[name]
    if era != prev_era and prev_era is not None:
        ax_bar.axhline((prev_y_v + y) / 2, color="#eeeeee", lw=0.4, zorder=0)
    prev_era, prev_y_v = era, y

for era, name, sig, ci, n in CURATED:
    y = y_map[name]
    ax_bar.barh(y, sig, height=BAR_HEIGHT, color=ERA_COLORS[era],
                alpha=0.90, zorder=2, linewidth=0)
    if ci:
        ax_bar.errorbar(sig, y, xerr=ci, fmt="none", ecolor="#444444",
                        elinewidth=0.7, capsize=2.0, capthick=0.7, zorder=3)
    ax_bar.text(COMPOSER_NAME_X, y, name, ha="right", va="center",
                fontsize=FS_COMPOSER, color="#1a1a1a", clip_on=False)
    ax_bar.text(sig + (ci or 0) + 0.012, y, f"n={n:,}", ha="left", va="center",
                fontsize=FS_N_LABEL, color="#b0b0b0", clip_on=False)

ax_bar.set_xlabel(r"$\sigma$ (nats step$^{-1}$)", fontsize=FS_X_LABEL)
# Centre the label on the drawn axis (data 0 -> X_MAX). The default centres it on the
# full xlim, which starts at -0.34 to hold the composer-name gutter, so the label sat
# left of the ticks it belongs to.
_l = ax_bar.transAxes.inverted().transform(ax_bar.transData.transform((0.0, 0)))[0]
_r = ax_bar.transAxes.inverted().transform(ax_bar.transData.transform((X_MAX, 0)))[0]
ax_bar.xaxis.set_label_coords((_l + _r) / 2.0, -0.055)
ax_bar.tick_params(left=False, labelleft=False, labelsize=FS_TICK)
ax_bar.set_xticks(X_TICKS)
ax_bar.set_xticklabels([f"{t:g}" for t in X_TICKS])
ax_bar.set_xticks(X_MINOR, minor=True)
ax_bar.tick_params(axis="x", which="minor", length=1.4, width=0.5)

plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.07)

out = FIGURES / "fig2_panel_a.pdf"
fig.savefig(str(out), dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(str(out.with_suffix(".png")), dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out} | grand mean {GMEAN:.3f} | {len(CURATED)} artists")
