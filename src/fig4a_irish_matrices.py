#!/usr/bin/env python3
"""Figure 4a — Irish net-current matrices by mode class

Fig 4 - net-current transition matrices for the two Irish mode classes.

LEVEL: MODE-CLASS CORPUS, not the engraved tunes. Functional = 2,920 settings of
2,173 distinct tunes (Gmajor/Dmajor/...); modal = 775 settings of 554 tunes
(Dorian + Mixolydian). Single-tune sigma is not interpretable, so nothing here is
computed on "The Blarney Pilgrim".

WHY THE QUALITY DIMENSION IS KEPT (this was the open question). Irish settings are
NOT quality-degenerate: modal settings run 61.6% maj / 36.8% min / 1.6% dom7,
functional 77.3 / 15.6 / 7.0. Collapsing quality and typing on root motion alone
destroys most of the measurement --

  typing                          functional    minor     modal
  (interval, q_from, q_to)          0.1817     0.1285    0.0258   <- canonical, kept
  interval only (12 root motions)   0.1124     0.0468    0.0050
  quality pair only (interval out)  0.0159     0.0047    0.0003
Typing on root motion alone loses 38.1% / 63.6% / 80.8% of sigma respectively --

so a root-motion-only matrix would be the DISHONEST object here, not the honest
one: it would report the modal corpus as ~5x less irreversible than it is. The
canonical (interval, q_from, q_to) typing is retained, exactly as in every other
panel, and drawn as its natural 9 x 12 grid: 3x3 quality pairs by 12 root motions.
The maj/min/dom7 grid drawn here carries 99.6% / 100.0% of all transitions and
98.9% / 100.0% of sigma in the two corpora; dim and aug are < 0.2% and are the
only types omitted.

WHITE CELLS ARE NOT ZEROS. A blank cell is a type the estimator SKIPS -- one
direction unobserved, so log(f/0) diverges. 86/108 cells are bidirectional in the
functional corpus, 70/108 in the modal. Zero current is mid-scale grey, not white.

SHARED SYMLOG SCALE across both panels so they are directly comparable; linear
near zero (linthresh 0.002) so the modal corpus's 0.015-scale structure is visible
next to the functional corpus's 0.065 peak.
"""
import os, sys, json
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import _figure_style as C

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
# config / irreversibility_v2 are siblings in src/
import estimator as ha

PITCH = {n: i for i, n in enumerate(
    ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}
Q3 = ["maj", "min", "dom7"]
QPAIRS = [(a, b) for a in Q3 for b in Q3]
QSHORT = {"maj": "maj", "min": "min", "dom7": "7"}
MCS = ["Major (functional)", "Dorian/Mixolydian (modal)"]
DISP = {"Major (functional)": "functional", "Dorian/Mixolydian (modal)": "modal"}


def mode_class(mode):
    m = mode.lower()
    if "major" in m:                       return "Major (functional)"
    if "dorian" in m or "mixolydian" in m: return "Dorian/Mixolydian (modal)"
    if "minor" in m:                       return "Minor"
    return "other"


def load_pools():
    """Same filters as ha.load_irish_by_mode(): chorded tunes, >= MIN_LEN states."""
    tunes = json.load(open(os.path.join(ROOT, "dat", "irish_tunes.json")))
    out = {}
    for t in tunes:
        if '"' not in t.get("abc", ""):
            continue
        mc = mode_class(t["mode"])
        if mc == "other":
            continue
        h = ha._abc_chords(t["abc"])
        if len(h) >= ha.MIN_LEN:
            out.setdefault(mc, []).append(h)
    return out


def type_counts(seqs):
    """t = (root_interval_mod12, quality_from, quality_to) -- transposition invariant."""
    N = Counter()
    for sq in seqs:
        for a, b in zip(sq[:-1], sq[1:]):
            ra, rb = PITCH.get(a.split(":")[0]), PITCH.get(b.split(":")[0])
            if ra is None or rb is None:
                continue
            N[((rb - ra) % 12, a.split(":")[1], b.split(":")[1])] += 1
    return N


def net_current_matrix(seqs):
    """J(t) = (N_fwd - N_rev)/N_total on the 9 x 12 grid. NaN where the estimator
    skips the type (either direction unobserved)."""
    N = type_counts(seqs)
    tot = sum(N.values())
    M = np.full((9, 12), np.nan)
    for qi, (qf, qt) in enumerate(QPAIRS):
        for iv in range(12):
            f = N.get((iv, qf, qt), 0)
            r = N.get(((-iv) % 12, qt, qf), 0)
            if f > 0 and r > 0:
                M[qi, iv] = (f - r) / tot
    return M


# ------------------------------------------------------------------ compute
POOLS = load_pools()
MATS = {mc: net_current_matrix(POOLS[mc]) for mc in MCS}

# Columns are SIGNED root motion -5..+6 semitones (storage is canonical mod-12).
# +5 is the V->I / bVII->IV fourth-up move; +7 (= -5) is the fifth-up move.
SEMI = list(range(-5, 7))
COLS = [s % 12 for s in SEMI]
ROWLAB = [f"{QSHORT[a]}\u2192{QSHORT[b]}" for a, b in QPAIRS]

# ------------------------------------------------------------------ draw
C.style()
VMAX = max(np.nanmax(np.abs(MATS[mc])) for mc in MCS)
norm = SymLogNorm(linthresh=0.002, vmin=-VMAX, vmax=VMAX, base=10)

# SKIPPED cells must NOT look like zero cells. RdBu_r is white at its centre, and
# matplotlib's default "bad" colour is also white -- so an unobserved-reverse type
# (dropped by the estimator, no defined J) rendered identically to a type with
# genuinely zero net current. Neutral grey separates the two: grey = no measurement,
# white = measured and balanced.
CMAP = plt.get_cmap("RdBu_r").copy()
CMAP.set_bad("#c3c7cb")

# 7.09 in = the shared Fig 4 column width (matches the score, sunburst and spectrum
# panels, so the four stack without ragged edges).  Height scaled to keep the cells
# close to square at 12 columns x 9 rows per matrix.
fig = plt.figure(figsize=(7.09, 2.95))
gs = fig.add_gridspec(1, 2, wspace=0.10, left=0.135, right=0.855,
                      bottom=0.215, top=0.885)
axes = []
for i, mc in enumerate(MCS):
    ax = fig.add_subplot(gs[0, i]); axes.append(ax)
    # pcolormesh, NOT imshow: imshow embeds the grid as a raster XObject in the PDF
    # (3 image objects in the first draft of this panel), and a 108-cell matrix is not
    # a dense density layer that earns rasterisation. pcolormesh emits one vector quad
    # per cell, so the PDF stays fully vector. Masked (skipped) cells are simply not
    # drawn and the axes facecolor shows through as the grey.
    ax.set_facecolor(CMAP.get_bad()[:3] if hasattr(CMAP, "get_bad") else "#c3c7cb")
    im = ax.pcolormesh(np.arange(13), np.arange(10),
                       np.ma.masked_invalid(MATS[mc][:, COLS]),
                       cmap=CMAP, norm=norm, shading="flat",
                       edgecolors="none", linewidth=0, rasterized=False)
    ax.set_ylim(9, 0)          # row 0 at TOP, matching the imshow convention
    ax.set_xlim(0, 12)
    # pcolormesh cell k spans [k, k+1], so ticks belong at CENTRES k+0.5. Using the
    # integer edge positions (the imshow convention) shifted every label half a cell
    # and made the +5 cadence column read as +6.
    _major = [SEMI.index(v) + 0.5 for v in (-4, -2, 0, 2, 4, 6)]
    ax.set_xticks(_major)
    ax.set_xticklabels([f"{v:+d}" if v else "0" for v in (-4, -2, 0, 2, 4, 6)], fontsize=5.8)
    ax.set_xticks(np.arange(12) + 0.5, minor=True)
    ax.tick_params(axis="x", which="minor", length=1.0)
    # +5 semitones = root up a fourth = the V->I cadence. This is the functional corpus's
    # single largest net current, maj->maj J = +0.06521. In the MODAL corpus the same move
    # runs BACKWARD, J = -0.01454: its major-to-major traffic flows the other way, up a
    # fifth (displayed -5), which is bVII->IV. Same column, opposite sign -- the dashed
    # line marks the cadence in both panels, not a shared direction.
    ax.axvline(SEMI.index(5) + 0.5, color="#333333", lw=0.5, ls=(0, (2.2, 1.6)), zorder=4)
    ax.set_yticks(np.arange(9) + 0.5)
    ax.set_yticklabels(ROWLAB if i == 0 else [], fontsize=5.8)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.tick_params(length=1.6, pad=1.2)
    ax.set_title(DISP[mc], fontsize=7.2, pad=2.5)
    if i == 0:
        ax.set_ylabel("chord-quality transition", fontsize=7.2, labelpad=1)
    C.all_spines(ax)
    for s in ax.spines.values():
        s.set_linewidth(0.55)

cb = fig.colorbar(im, ax=axes, fraction=0.030, pad=0.016)
# labelpad clears the -10^-3 / -10^-1 tick labels, which the axis title otherwise sits on
cb.set_label(r"net current  $J(t)$", fontsize=7.0, labelpad=3.5)
cb.ax.tick_params(labelsize=6.0, pad=1.4)
# matplotlib rasterises colorbar solids by DEFAULT (rasterized=True on the QuadMesh),
# which left one image XObject in the PDF after the pcolormesh fix. Force it vector so
# the panel has zero embedded bitmaps.
if cb.solids is not None:
    cb.solids.set_rasterized(False)

_b = [a.get_position() for a in axes]
fig.text((_b[0].x0 + _b[1].x1) / 2, _b[0].y0 - 0.135,
         "root motion  (semitones)", ha="center", va="top", fontsize=7.2)

# key for the grey: 3 words, so it stays in-panel rather than going to the caption
import matplotlib.patches as mpatches
_sw = mpatches.Rectangle((0.862, _b[0].y0 - 0.055), 0.026, 0.055,
                         transform=fig.transFigure, fc="#c3c7cb", ec="#888",
                         lw=0.4, clip_on=False)
fig.patches.append(_sw)
fig.text(0.895, _b[0].y0 - 0.028, "one-way\n(skipped)", fontsize=5.6, va="center", ha="left")

fig.savefig(os.path.join(C.OUT, "panel_4_irish_matrix.pdf"), dpi=600)
fig.savefig(os.path.join(C.OUT, "panel_4_irish_matrix.png"), dpi=300)
plt.close(fig)

print("wrote panel_4_irish_matrix  |  " + "  ".join(
    f"{DISP[mc]}: {int(np.isfinite(MATS[mc]).sum())}/108 bidirectional cells, "
    f"max|J|={np.nanmax(np.abs(MATS[mc])):.5f}" for mc in MCS))
