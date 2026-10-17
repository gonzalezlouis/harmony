#!/usr/bin/env python3
"""Supplementary — the arrow rides on the descending fifth alone

Fig 3c companion - the arrow rides on the DESCENDING FIFTH and on nothing else.

WHAT THIS PANEL IS FOR.  Fig 3c plots per-piece mean affinity <A> against descending-fifth share
and finds a steep rise.  On its own that is one axis out of twelve possible root
motions, and a reader cannot tell whether the descending fifth is special or whether
ANY measure of "how much the roots move" would have done the same.  This 2x2 is the
control: the same 20,560 pieces, the same y axis, four DIFFERENT x axes.  Only the
first one rises.

THE FOUR CLASSES, and why these four.  All twelve mod-12 root motions were tested
against per-piece mean affinity <A> with none privileged in advance:

    semitones  +5  desc. fifth   pearson +0.6402   mean share 0.2186
               +3                        -0.3163               0.0669
               -5  asc. fifth            -0.2492               0.1636
               -4                        -0.2330               0.0626
               -2                        -0.2175               0.0744
               +4                        -0.1777               0.0612
               -1                        -0.1276               0.0394
               +6  tritone               -0.0425               0.0260
               +1                        -0.0367               0.0477
                0  root retained         -0.0035               0.0758
               -3                        -0.0075               0.0815
               +2                        +0.0066               0.0823

The descending fifth is the ONLY motion with a positive correlation of any size; its
+0.64 is 2.0x the largest competitor in magnitude and the competitors all have the
WRONG SIGN.  The four plotted classes are then chosen to be an exhaustive-ish partition
of the interval circle by musical kind rather than a shortlist of the winners:

    descending fifth   {+5}              mean share 0.2186   r +0.6402
    ascending fifth    {-5}              mean share 0.1636   r -0.2492
    step motion        {+-1, +-2}        mean share 0.2438   r -0.1672
    third motion       {+-3, +-4}        mean share 0.2721   r -0.2781

Together they cover 10 of 12 intervals and 89.8% of all chord moves (the remainder is
the tritone at 2.6% and root-retained at 7.6%, both r ~ 0 and both plotted as classes
in dat/rootmotion_2x2.json but omitted here to keep the grid at four).  Ascending
fifth is the sharpest control available -- it is the SAME interval traversed the other
way, so it holds interval size fixed and varies only direction.  That is the panel's
strongest single statement: 0.6402 one way, -0.2492 the other.

WHY THE NEGATIVE CORRELATIONS ARE NOT THREE MORE FINDINGS.  Shares sum to 1, so a
piece with more descending fifths necessarily has fewer of everything else; a
compositional constraint alone forces the other classes negative.  Controlling for
descending-fifth share linearly, the residual relations are small and change sign --
ascending fifth -0.5845, third +0.2435, tritone +0.1655, root-retained +0.1452, step
+0.1085 -- i.e. the other classes carry no independent positive relation to <A>,
and the ascending fifth's apparent negative slope is largely the mirror of the
descending fifth's positive one.  Conversely the descending fifth's relation survives
controlling for any single other class (+0.63 to +0.77).  The honest reading of the
2x2 is "one axis rises, three do not", not "three axes fall".

WHAT THE BINNED MEDIAN DOES ACROSS THE FOUR (fixed-width 0.05 bins, >=30 pieces):
    descending fifth  -0.005 -> +0.597 over x 0.03..0.73   monotone, 15 bins
    ascending fifth   +0.060 -> -0.009 over x 0.03..0.53   11 bins
    step motion       +0.032 -> +0.021 over x 0.03..0.63   13 bins, flat within 0.03
    third motion      +0.043 -> -0.005 over x 0.03..0.73   15 bins
Measured as the FULL RANGE of each median curve (max minus min, which does not assume
monotonicity -- three of the four are not monotone), the descending fifth travels 0.603
nats step^-1 while the other three travel 0.069, 0.028 and 0.069, a factor of 8.7 to 21.8.
That ratio in median travel is the effect, stated without a fitted slope.

SHARED AXES, AND WHY THE RIGHT THIRD LOOKS EMPTY.  All four subpanels use the SAME x
limits and the SAME y limits, so a gradient in one means what the same gradient means in
another -- the entire point of a 2x2, and something per-panel autoscaling would destroy.
The shared x range runs to 1.0 because step and third motion genuinely reach it: 26
pieces sit above a 0.9 share (15 step, 11 third) and 56 above 0.8.  Those are real
pieces, so the limit is data extent as in 3c, not a round number -- cropping to 0.75
would tidy the grid by deleting 99 pieces from it.  The descending-fifth and
ascending-fifth panels top out at 0.80 and 0.65, and the visible difference in how far
each class's cloud extends is itself informative: no piece is built mostly of ascending
fifths, while pieces built almost entirely of step or third motion exist.

Data: dat/rootmotion_2x2.json, dat/rootmotion_2x2.npz
      (built by ../compute_rootmotion.py, which re-derives the per-piece arrays from
       the sequence cache and reproduces emergent_full.npz element-for-element)
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import _figure_style as C

C.style()
DATADIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dat")
T = json.load(open(os.path.join(DATADIR, "rootmotion_2x2.json")))
Z = np.load(os.path.join(DATADIR, "rootmotion_2x2.npz"), allow_pickle=True)
arrow = Z["arrow"]
CLASSES = T["classes"]
X = {k: Z["x_%d" % i] for i, k in enumerate(CLASSES)}

# Axis labels stay under ~15 characters of musical name plus the word "share"; the
# interval sets they stand for go in the caption, not in the panel.
XLAB = {"descending fifth": "descending-fifth share",
        "ascending fifth":  "ascending-fifth share",
        "step motion":      "step-motion share",
        "third motion":     "third-motion share"}

fig, axes = plt.subplots(2, 2, figsize=(6.90, 5.60), sharex=True, sharey=True)
axes = axes.ravel()

# SHARED limits from the data extent across ALL FOUR classes, so a gradient in one
# subpanel means the same thing as the same gradient in another.
_xhi = max(float(X[k].max()) for k in CLASSES)
_xpad, _ypad = 0.03 * _xhi, 0.03 * (arrow.max() - arrow.min())
hb = None
for ax, k in zip(axes, CLASSES):
    ax.set_facecolor("#EFEFF2")
    hb = ax.hexbin(X[k], arrow, gridsize=46, cmap="magma_r", bins="log",
                   mincnt=1, linewidths=0, zorder=2,
                   extent=(0.0, _xhi, arrow.min(), arrow.max()))
    # Same rasterization discipline as 3c: as vector paths the low-count hexes below
    # zero drop out of the PDF stream. Image layer at 600 dpi; axes and lines vector.
    # set_rasterized() on the artist ONLY -- NOT ax.set_rasterization_zorder(), which
    # rasterizes the axis text along with everything below the threshold (see 3c).
    hb.set_rasterized(True)

    b = T["bins"][k]
    ax.fill_between(b["x"], b["q25"], b["q75"], facecolor="white", alpha=0.26,
                    edgecolor="white", linewidth=0.9, zorder=3)
    ax.plot(b["x"], b["med"], color="white", lw=1.9, zorder=4,
            path_effects=[pe.Stroke(linewidth=3.0, foreground="#222222"), pe.Normal()])
    ax.scatter(b["x"], b["med"], s=11, color="white", edgecolor="#222222",
               linewidth=0.55, zorder=5)
    ax.axhline(0.0, color="#555555", lw=0.8, ls=(0, (4, 2)), zorder=3.5)

    # ONE number per subpanel: the Pearson r that IS the comparison being made.
    # Six characters at 7 pt. Everything else about the class is in the caption.
    r = T["class_corr"][k]["pearson"]
    ax.text(0.965, 0.945, "$r=%+.2f$" % r, transform=ax.transAxes, ha="right",
            va="top", fontsize=7.4,
            bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                      edgecolor="#222222", linewidth=0.7))
    ax.set_xlabel(XLAB[k], fontsize=8.2)
    C.all_spines(ax)

for ax in axes:
    ax.set_xlim(-_xpad, _xhi + _xpad)
    ax.set_ylim(arrow.min() - _ypad, arrow.max() + _ypad)
# ONE y-label for the whole grid, centred on the shared axis. Per-axes labels at this
# figure height overlapped each other across the row gap (the upper one ran into the
# lower panel's tick labels) -- and repeating an identical label twice on a sharey grid
# says nothing the single label does not.
# Same correction as panel_4c_scaling: this is a signed FORCE (mean affinity), not an
# entropy production, because 20.0% of pieces are negative and entropy production cannot
# be.  A single piece samples an incomplete set of forward/reverse pairs, so the
# non-negativity bound -- which comes from pairing t with its reverse -- does not apply
# to it.  Every panel here carries per-piece hexes only, no pooled markers.
fig.supylabel(r"mean affinity  $\langle A \rangle$  (nats step$^{-1}$)",
              fontsize=8.6, x=0.045)

cb = fig.colorbar(hb, ax=axes.tolist(), fraction=0.026, pad=0.016)
cb.set_label("pieces per cell", fontsize=7.4)
cb.ax.tick_params(labelsize=6.2)

C.save(fig, "panel_4c_rootmotion_2x2", aspect=C.SUPP_ASPECT)
print("wrote panel_4c_rootmotion_2x2 | n=%d" % T["n"])
for k in CLASSES:
    d = T["class_corr"][k]
    print("  %-18s pearson %+.4f [%+.4f,%+.4f]  spearman %+.4f [%+.4f,%+.4f]  share %.4f"
          % (k, d["pearson"], d["ci"][0], d["ci"][1],
             d["spearman"], d["sci"][0], d["sci"][1], d["mean_share"]))
print("  xlim %.3f..%.3f  ylim %.3f..%.3f (shared)"
      % (*axes[0].get_xlim(), *axes[0].get_ylim()))
