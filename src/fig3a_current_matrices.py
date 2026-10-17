#!/usr/bin/env python3
"""Figure 3a — net-current matrices by root motion and quality pair

Fig 4b - the arrow concentrates: four transfer matrices, most diffuse to most concentrated.

WHY THESE FOUR, IN THIS ORDER.  The top row is the net-current matrix J(t) for Romantic,
Billboard pop, Mandopop and Jazz, on one shared diverging colour scale.  They are ordered
by how concentrated the current is, and the ordering is monotone in the data:

    Romantic        26 cells carry >1% of the current, top 3 cells = 25%
    Billboard pop   18 cells                          top 3       = 34%
    Mandopop        11 cells                          top 3       = 47%
    Jazz            12 cells                          top 3       = 51%

Romantic is the most diffuse body of music in the corpus and Jazz the most concentrated,
so the two ends of the row are the widest contrast available -- a reader sees a grid that
is faintly busy everywhere becoming a grid that is blank except for a few saturated
cells, with no annotation required.  Classical is nearly identical to Romantic (27 cells,
27%) and is dropped so the four columns span the range instead of duplicating one end.

BOTH POP CORPORA APPEAR, and they are not interchangeable.  "Pop" is two independent
datasets built by different means: Billboard pop (CoCoPops, expert hand annotation of US
chart hits) and Mandopop (POP909, chord annotations of Mandarin pop).  They sit at 34%
and 47%, i.e. Billboard falls between Romantic and Mandopop.  Showing only Mandopop would
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

# X-AXIS = SIGNED ROOT MOTION IN SEMITONES, from -5 to +6.
#
# This is the raw physical quantity: how many semitones the chord root moved. The
# V->I cadence has root motion +5 (G to C is up 5 semitones, equivalently down 7).
# Every column carries its own numeric label; no qualitative anchors.
#
# Earlier drafts used fifths-order because it put the cadence next to home, but
# fifths-steps are not a natural unit -- the user cannot look at "step -1" and know
# it means "5 semitones." Semitones are the unit an interval is defined in.
# Storage is canonical mod-12 (columns labelled "0","+1",...,"+11"). Display uses
# SIGNED semitones -5..+6 for readability (V->I cadence at +5, tritone at +6).
# Map each display value to its storage index: +5 -> index 5, -3 -> index 9, etc.
_LAB_TO_IDX = {int(_l.replace("\u2212", "-").replace("+", "")): _i for _i, _l in enumerate(IVL)}
SEMI_ORDER = list(range(-5, 7))                                   # -5, -4, ..., +5, +6
FIFTHS = [_LAB_TO_IDX[s % 12] for s in SEMI_ORDER]                # signed -> mod-12 index

ROW = ["Romantic", "CoCoPops", "POP909", "Jazz"]        # diffuse -> concentrated

# All 20 transition-type rows are kept. The smallest (aug->*, dim->dim) still carry
# |J| up to 0.002-0.009 -- real current, not zeros -- so trimming them would hide data.
# The space comes from panel height instead: at 3.05in the 20 labels were unreadable.
ROWLAB = _ROWLAB_ALL
BAND = ["Classical", "Romantic", "CoCoPops", "POP909"]
OTHER = ["Renaissance", "Baroque", "Ragtime", "Jazz"]
LABEL = {"CoCoPops": "Billboard pop", "POP909": "Mandopop", "Ragtime": "Late-\nRom.",
         "Renaissance": "Renais-\nsance"}
# Colours and marker shapes now come from _common.CORPUS_COL / CORPUS_MARKER -- the SAME
# registry Fig 3c's centroid markers use, so a glyph means one corpus across both
# panels. The dict that used to live here was byte-identical to CORPUS_COL; it moved
# rather than changed, and this alias keeps the rest of the script untouched.
COL = C.CORPUS_COL

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
    # Tick only the interpretable anchors instead of every other column: the
    # cadence (-1), the tonic (0), and the +-3 / +-6 waypoints. 12 dense ticks
    # invited digit-by-digit reading of a coordinate nobody needed to read that way.
    # Verbal anchors instead of a fifths-step count: "far" (tritone away),
    # "V→I" (the cadence), "home" (root didn't move). A reader who does not
    # Sparse labels every other cell. +5 (the cadence) is not labelled — its column
    # is marked by the dashed line, and labelling it too would collide with +6.
    # Minor ticks on every column keep the 1:1 alignment with cells; only labels are sparse.
    _MAJOR = [SEMI_ORDER.index(v) for v in (-4, -2, 0, 2, 4, 6)]
    ax.set_xticks(_MAJOR)
    ax.set_xticklabels([f"{SEMI_ORDER[i]:+d}" if SEMI_ORDER[i] else "0" for i in _MAJOR],
                       fontsize=5.8)
    ax.set_xticks(range(12), minor=True)
    ax.tick_params(axis="x", which="minor", length=1.0)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    # Dashed line at semitone +5: the V->I cadence (root moves up a fourth, or
    # equivalently down a fifth -- G resolving to C).
    ax.axvline(SEMI_ORDER.index(5), color="#333333", lw=0.5, ls=(0, (2.2, 1.6)), zorder=4)
    ax.set_yticks(range(len(ROWLAB)))
    ax.set_yticklabels(ROWLAB if i == 0 else [], fontsize=5.0)
    ax.tick_params(length=1.6, pad=1.2)
    # ONE number per panel: what share of the total forward-minus-backward
    # traffic is carried by the three most-used move types. "% concentrated"
    # is short enough to fit a 1.75in panel; "top 3 moves = X% of traffic" isn't.
    ax.set_title(
        f"{LABEL.get(k, k)}\n"
        f"{CONC[k]['top3_share']:.0%} concentrated",
        fontsize=7.0, pad=6)   # pad 6 not 3: room for the corpus glyph below
    # CORPUS GLYPH beside the panel name, matching this corpus's centroid marker in
    # Fig 3c. Same shape, same fill, no text -- so a reader who has seen the star sit
    # alone at the top right of 3c can find which matrix that was without either panel
    # having to name the other. Placed in axes coordinates just left of the title, above
    # the matrix, where it overlaps no cell.
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

# one shared x-label for the matrix row (was repeated x4 and collided with row 2)
_bb = [a.get_position() for a in mat_axes]
fig.text((_bb[0].x0 + _bb[3].x1) / 2, _bb[0].y0 - 0.045,
         "root motion  (semitones)",
         ha="center", va="top", fontsize=7.2)
C.save(fig, "panel_4a_matrices")
print("wrote panel_4a_matrices | top3 share: " +
      ", ".join(f"{LABEL.get(k, k)} {CONC[k]['top3_share']:.0%}" for k in ROW))
print("  corpus glyphs (shared with 3c): " +
      ", ".join(f"{LABEL.get(k, k)}={C.CORPUS_MARKER[k]}" for k in ROW))
