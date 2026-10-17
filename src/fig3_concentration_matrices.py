#!/usr/bin/env python3
"""Figure 3 — four matrices, most diffuse to most concentrated

Fig 4b - the arrow concentrates: four transfer matrices, most diffuse to most concentrated.

WHY THESE FOUR, IN THIS ORDER.  The top row is the net-current matrix J(t) for Romantic,
Billboard pop, Mandopop and Jazz, on one shared diverging colour scale.  They are ordered
by how concentrated the current is, and the ordering is monotone in the data:

    Romantic        30 cells carry >1% of the current, top 3 cells = 17%
    Billboard pop   28 cells                          top 3       = 19%
    Mandopop        18 cells                          top 3       = 26%
    Jazz            18 cells                          top 3       = 32%

Romantic is the most diffuse body of music in the corpus and Jazz the most concentrated,
so the two ends of the row are the widest contrast available -- a reader sees a grid that
is faintly busy everywhere becoming a grid that is blank except for a few saturated
cells, with no annotation required.  Classical is nearly identical to Romantic (22 cells,
18%) and is dropped so the four columns span the range instead of duplicating one end.

BOTH POP CORPORA APPEAR, and they are not interchangeable.  "Pop" is two independent
datasets built by different means: Billboard pop (CoCoPops, expert hand annotation of US
chart hits) and Mandopop (POP909, chord annotations of Mandarin pop).  They sit at 19%
and 26%, i.e. Billboard falls between Romantic and Mandopop.  Showing only Mandopop would
make a general claim about pop from the more extreme of the two; showing both makes the
weaker, honest claim -- that both pop corpora are more concentrated than the classical
plateau, and that they differ from each other.

WHAT THE MATRIX IS.  Rows are chord-quality transitions (maj->maj, maj->dom7, dom7->maj,
...), columns are root intervals arranged in fifths order, so the descending fifth is one
fixed column in every panel.  Together these are the transfer operator restricted to
transposition-invariant types, and sigma is read off it directly as sum_t J(t) A(t).

BOTTOM ROW: the same thing as numbers, and the honest limit on it.  A_mean (left) is flat
across the classical plateau and both pop corpora -- Classical 0.829, Romantic 0.715,
Billboard 0.806, Mandopop 0.820, CV 6% -- while J_tot (right) rises over those same four,
0.184 -> 0.180 -> 0.192 -> 0.286, CV 21%.  Pop's excess sigma is entirely current.
A_mean is NOT flat across all eras: Renaissance 0.374, Baroque 1.143, Jazz 2.003, CV 46%
over all eight corpora.  Those are plotted in grey outside the shaded band so the
constancy reads as a claim about the plateau and pop, not a universal law.

THE IDENTITY, no fitted parameters: sigma = sum_t J(t) A(t) = J_tot * A_mean, with
J(t) = (N_t - N_t_rev)/N_total and A(t) = log(N_t/N_t_rev).  Mandopop vs Classical:
sigma 1.53x, J_tot 1.55x, A_mean 0.99x, directed types 0.81x -> 1.91x current per type.

Affinities come from each corpus's OWN matrix (currents and forces must come from the
same chain, or the quantity is not sigma).  Invariant typing removes the recording-key
artifact and is independent of that choice.

Data: dat/mechanism.json, dat/transfer_matrix.npz
"""
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

ROW = ["Romantic", "CoCoPops", "POP909", "Jazz"]        # diffuse -> concentrated

# All 20 transition-type rows are kept. The smallest (aug->*, dim->dim) still carry
# |J| up to 0.002-0.009 -- real current, not zeros -- so trimming them would hide data.
# The space comes from panel height instead: at 3.05in the 20 labels were unreadable.
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
# ONE shared scale across all four panels (so they stay directly comparable), but
# symmetric-log so Jazz's 0.165 peak does not flatten Romantic's 0.02-scale structure.
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

# one shared x-label for the matrix row (was repeated x4 and collided with row 2)
_bb = [a.get_position() for a in mat_axes]
fig.text((_bb[0].x0 + _bb[3].x1) / 2, _bb[0].y0 - 0.085,
         "root interval (semitones, fifths order)", ha="center", va="top", fontsize=7.0)
C.save(fig, "panel_4b_poparadox")
print("wrote panel_4b_poparadox | top3 share: " +
      ", ".join(f"{LABEL.get(k, k)} {CONC[k]['top3_share']:.0%}" for k in ROW))
