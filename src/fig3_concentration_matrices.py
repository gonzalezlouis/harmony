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
FIFTHS = [0, 5, 10, 3, 8, 1, 6, 11, 4, 9, 2, 7]
COLS = [IVL[i] for i in FIFTHS]

ROW = ["Romantic", "CoCoPops", "POP909", "Jazz"]

ROWLAB = _ROWLAB_ALL
BAND = ["Classical", "Romantic", "CoCoPops", "POP909"]
OTHER = ["Renaissance", "Baroque", "Ragtime", "Jazz"]
LABEL = {"CoCoPops": "Billboard pop", "POP909": "Mandopop", "Ragtime": "Late-\nRom.",
         "Renaissance": "Renais-\nsance"}
COL = {"Renaissance": "#2E8B70", "Baroque": "#2F5C9E", "Classical": "#4FA8C0",
       "Romantic": "#D9A227", "Ragtime": "#C64A3F",
       "CoCoPops": "#B69AD8", "POP909": "#8C63C4", "Jazz": "#4A2C7A"}

fig = plt.figure(figsize=(7.20, 3.55))
gs = fig.add_gridspec(1, 4, wspace=0.14)

vmax = max(np.nanmax(np.abs(TM[k])) for k in ROW)
norm = SymLogNorm(linthresh=0.004, vmin=-vmax, vmax=vmax, base=10)
mat_axes = []
for i, k in enumerate(ROW):
    ax = fig.add_subplot(gs[0, i]); mat_axes.append(ax)
    im = ax.imshow(TM[k][np.ix_(keep, FIFTHS)], cmap="RdBu_r", norm=norm,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(range(0, len(COLS), 2))
    ax.set_xticklabels([COLS[j] for j in range(0, len(COLS), 2)], fontsize=5.4)
    ax.set_yticks(range(len(ROWLAB)))
    ax.set_yticklabels(ROWLAB if i == 0 else [], fontsize=5.0)
    ax.tick_params(length=1.6, pad=1.2)
    ax.set_title(f"{LABEL.get(k, k)}\n{CONC[k]['n_over_1pct']} cells, "
                 f"top 3 = {CONC[k]['top3_share']:.0%}", fontsize=7.0, pad=3)
    if i == 0:
        ax.set_ylabel("chord-quality transition", fontsize=7.2, labelpad=1)
    ax.set_xlabel("")
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.55)

cb = fig.colorbar(im, ax=mat_axes, fraction=0.024, pad=0.014)
cb.set_label(r"net current  $J(t)$", fontsize=7.0)
cb.ax.tick_params(labelsize=6.0)

_bb = [a.get_position() for a in mat_axes]
fig.text((_bb[0].x0 + _bb[3].x1) / 2, _bb[0].y0 - 0.085,
         "root interval (semitones, fifths order)", ha="center", va="top", fontsize=7.0)
C.save(fig, "panel_4b_poparadox")
print("wrote panel_4b_poparadox | top3 share: " +
      ", ".join(f"{LABEL.get(k, k)} {CONC[k]['top3_share']:.0%}" for k in ROW))
