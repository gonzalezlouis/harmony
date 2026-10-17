import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import _figure_style as C

C.style()
DATADIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dat")
Mx = json.load(open(os.path.join(DATADIR, "mechanism.json")))
TM = np.load(os.path.join(DATADIR, "transfer_matrix_canonical.npz"), allow_pickle=True)
MECH, CONC = Mx["mechanism"], Mx["concentration"]
keep = TM["keep"]
_ROWLAB_ALL = [f"{a}\u2192{b}" for a, b in Mx["tm_rows"]]
IVL = Mx["tm_intervals"]

_LAB_TO_IDX = {int(_l.replace("\u2212", "-").replace("+", "")): _i for _i, _l in enumerate(IVL)}
SEMI_ORDER = list(range(-5, 7))
FIFTHS = [_LAB_TO_IDX[s % 12] for s in SEMI_ORDER]

ROW = ["Romantic", "CoCoPops", "POP909", "Jazz"]

ROWLAB = _ROWLAB_ALL
BAND = ["Classical", "Romantic", "CoCoPops", "POP909"]
OTHER = ["Renaissance", "Baroque", "Ragtime", "Jazz"]
LABEL = {"CoCoPops": "Billboard pop", "POP909": "Mandopop", "Ragtime": "Late-\nRom.",
         "Renaissance": "Renais-\nsance"}
COL = C.CORPUS_COL

fig = plt.figure(figsize=(7.20, 3.55))
gs = fig.add_gridspec(1, 4, wspace=0.14)

vmax = max(np.nanmax(np.abs(TM[k])) for k in ROW)
norm = SymLogNorm(linthresh=0.004, vmin=-vmax, vmax=vmax, base=10)
mat_axes = []
for i, k in enumerate(ROW):
    ax = fig.add_subplot(gs[0, i]); mat_axes.append(ax)
    im = ax.imshow(TM[k][np.ix_(keep, FIFTHS)], cmap="RdBu_r", norm=norm,
                   aspect="auto", interpolation="nearest")
    _MAJOR = [SEMI_ORDER.index(v) for v in (-4, -2, 0, 2, 4, 6)]
    ax.set_xticks(_MAJOR)
    ax.set_xticklabels([f"{SEMI_ORDER[i]:+d}" if SEMI_ORDER[i] else "0" for i in _MAJOR],
                       fontsize=5.8)
    ax.set_xticks(range(12), minor=True)
    ax.tick_params(axis="x", which="minor", length=1.0)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.axvline(SEMI_ORDER.index(5), color="#333333", lw=0.5, ls=(0, (2.2, 1.6)), zorder=4)
    ax.set_yticks(range(len(ROWLAB)))
    ax.set_yticklabels(ROWLAB if i == 0 else [], fontsize=5.0)
    ax.tick_params(length=1.6, pad=1.2)
    ax.set_title(
        f"{LABEL.get(k, k)}\n"
        f"{CONC[k]['top3_share']:.0%} concentrated",
        fontsize=7.0, pad=6)
    ax.plot([0.075], [1.062], marker=C.CORPUS_MARKER[k], ls="none",
            markersize=7.0 if C.CORPUS_MARKER[k] == "*" else 5.0,
            markerfacecolor=COL[k], markeredgecolor="#111111", markeredgewidth=0.6,
            transform=ax.transAxes, clip_on=False, zorder=6)
    if i == 0:
        ax.set_ylabel("chord-quality transition", fontsize=7.2, labelpad=1)
    ax.set_xlabel("")
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.55)

cb = fig.colorbar(im, ax=mat_axes, fraction=0.024, pad=0.014)
cb.set_label(r"net current  $J(t)$", fontsize=7.0)
cb.ax.tick_params(labelsize=6.0)

_bb = [a.get_position() for a in mat_axes]
fig.text((_bb[0].x0 + _bb[3].x1) / 2, _bb[0].y0 - 0.045,
         "root motion  (semitones)",
         ha="center", va="top", fontsize=7.2)
C.save(fig, "panel_4a_matrices")
print("wrote panel_4a_matrices | top3 share: " +
      ", ".join(f"{LABEL.get(k, k)} {CONC[k]['top3_share']:.0%}" for k in ROW))
print("  corpus glyphs (shared with 3c): " +
      ", ".join(f"{LABEL.get(k, k)}={C.CORPUS_MARKER[k]}" for k in ROW))
