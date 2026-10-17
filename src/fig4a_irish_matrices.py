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
    N = Counter()
    for sq in seqs:
        for a, b in zip(sq[:-1], sq[1:]):
            ra, rb = PITCH.get(a.split(":")[0]), PITCH.get(b.split(":")[0])
            if ra is None or rb is None:
                continue
            N[((rb - ra) % 12, a.split(":")[1], b.split(":")[1])] += 1
    return N


def net_current_matrix(seqs):
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


POOLS = load_pools()
MATS = {mc: net_current_matrix(POOLS[mc]) for mc in MCS}

SEMI = list(range(-5, 7))
COLS = [s % 12 for s in SEMI]
ROWLAB = [f"{QSHORT[a]}\u2192{QSHORT[b]}" for a, b in QPAIRS]

C.style()
VMAX = max(np.nanmax(np.abs(MATS[mc])) for mc in MCS)
norm = SymLogNorm(linthresh=0.002, vmin=-VMAX, vmax=VMAX, base=10)

CMAP = plt.get_cmap("RdBu_r").copy()
CMAP.set_bad("#c3c7cb")

fig = plt.figure(figsize=(7.09, 2.95))
gs = fig.add_gridspec(1, 2, wspace=0.10, left=0.135, right=0.855,
                      bottom=0.215, top=0.885)
axes = []
for i, mc in enumerate(MCS):
    ax = fig.add_subplot(gs[0, i]); axes.append(ax)
    ax.set_facecolor(CMAP.get_bad()[:3] if hasattr(CMAP, "get_bad") else "#c3c7cb")
    im = ax.pcolormesh(np.arange(13), np.arange(10),
                       np.ma.masked_invalid(MATS[mc][:, COLS]),
                       cmap=CMAP, norm=norm, shading="flat",
                       edgecolors="none", linewidth=0, rasterized=False)
    ax.set_ylim(9, 0)
    ax.set_xlim(0, 12)
    _major = [SEMI.index(v) + 0.5 for v in (-4, -2, 0, 2, 4, 6)]
    ax.set_xticks(_major)
    ax.set_xticklabels([f"{v:+d}" if v else "0" for v in (-4, -2, 0, 2, 4, 6)], fontsize=5.8)
    ax.set_xticks(np.arange(12) + 0.5, minor=True)
    ax.tick_params(axis="x", which="minor", length=1.0)
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
cb.set_label(r"net current  $J(t)$", fontsize=7.0, labelpad=3.5)
cb.ax.tick_params(labelsize=6.0, pad=1.4)
if cb.solids is not None:
    cb.solids.set_rasterized(False)

_b = [a.get_position() for a in axes]
fig.text((_b[0].x0 + _b[1].x1) / 2, _b[0].y0 - 0.135,
         "root motion  (semitones)", ha="center", va="top", fontsize=7.2)

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
