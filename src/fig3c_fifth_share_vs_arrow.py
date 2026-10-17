#!/usr/bin/env python3
"""Figure 3c — per-work arrow against descending-fifth share

Fig 3c - the arrow of a piece rises with its descending-fifth content.
FULL CORPUS: 20,560 distinct pieces, every dataset, no era labels used.

WHAT THE Y AXIS IS, AND WHY IT IS CALLED sigma_i.  The plotted quantity is the mean
affinity over one piece's steps, A(t) taken from the WHOLE-CORPUS reference matrix:

    sigma_i = (1/m_i) sum over piece i's steps of A(t)      m_i = # steps in piece i

That is not the corpus-pooled sigma of a single piece -- per-piece
sigma_bidirectional is censoring-prone on short lead sheets (a piece of 20 chords has
too few reverse observations for its own log-ratios) -- but it is also not an
unrelated statistic.  It is an EXACT additive decomposition of the pooled sigma:
weighting each piece by its step count m_i and averaging recovers the pooled value to
1.9e-7 (0.049821296 vs 0.049821104 nats step^-1 over the 20,560-piece pool).  So each
point is one piece's share of the corpus's entropy production, in the same units, and
the axis is honestly labeled sigma_i with the definition in the caption.

The label deliberately does NOT read plain "sigma": a per-piece point uses corpus-wide
affinities, so it inherits the pool's forces rather than measuring its own, and no
single point should be quoted as that piece's sigma.  Subscript i and the caption carry
that restriction.  (An era's sigma_i centroid is correspondingly NOT its own sigma:
Jazz sits at 0.443 here against 1.307 measured on its own matrix, because a corpus
computes its affinities from itself while these points read them off the pool.  The
markers show WHERE a corpus sits in this plane, not how much arrow it has.  Both numbers
are carried in dat/corpus_centroids.json, per corpus, so the two can be compared but
never conflated.)

AXIS LIMITS ARE THE DATA EXTENT, not round numbers.  Earlier limits (x 0..0.72,
y -0.30..0.72) cropped 58 pieces out of the frame: 41 above x = 0.72, 25 above
y = 0.72, 3 below y = -0.30.  Limits now come from the arrays with a 3% margin, so
nothing plotted falls outside the axes.

WHY THE FULL CORPUS IS CORRECT HERE.  Earlier drafts of this panel used only 6,229
score-notation and expert-annotated pieces, excluding the chordified datasets by
extraction provenance.  That exclusion exists to protect ERA-LEVEL sigma comparisons
(performance-MIDI chordify suppresses sigma to roughly a third of the score value, so
mixing methods manufactures historical trends).  This panel makes no era comparison
and reads no era label -- it is a per-piece relation -- so the confound does not apply
and the exclusion was unjustified.  Likewise the "Various (Polish)" bucket is dropped
elsewhere for a wrong ERA LABEL, which this panel never uses.

Only genuine data defects are excluded:
    ASAP (959 rows)  all resolve to ONE distinct cached sequence through the
                     basename join -- degenerate, a pipeline bug logged in
                     dat/coverage.json (it produced a spurious pearson of
                     exactly +1.000 before being caught)
    duplicate sequences  removed corpus-wide, so repeated performances of the same
                     score are not counted twice (KernScores 7%, GiantMIDI 7%)

Composition: KernScores 7,658 | GiantMIDI 9,678 | MAESTRO 1,263 | POP909 904 |
Jazz 847 | CoCoPops 210.

UNITS ARE NATS (natural log), matching sigma.

WHY THE X-AXIS IS NOT A CHOICE WE MADE.  Fig 4b tests all twelve root-motion
intervals against the per-piece arrow with no interval privileged in advance.
Interval 5 semitones -- the descending fifth, carrying both ii->V and V->I -- wins at
r = +0.63, more than double the next largest (interval 3 at -0.31).  The data selects
this axis.  (An earlier draft used PCA loadings for this; a direct per-interval
correlation says the same thing without sign conventions or a variance-share to
interpret.)

THE RELATION.

    pearson  +0.6402     spearman  +0.4798     n = 20,560

WHY THE CORPUS MARKERS ARE HERE.  The hexbin answers "does cadential harmony carry
arrow" but not "which music is where", so a reader cannot see that the steep upper
right is Jazz.  Eight markers give each corpus's centroid in this plane (mean over its
pieces, 95% CI by resampling pieces, 2,000 draws).  Jazz is highest on BOTH axes --
fifth share 0.5339 [0.5234, 0.5440], mean affinity 0.4427 [0.4301, 0.4546] -- against a
next-highest of 0.3003 (POP909 / Mandopop) and 0.1380 (Ragtime), so the marker cloud reproduces
the trend line across corpora as well as within pieces.  Values here are quoted to 4
decimals, matching dat/corpus_centroids.json exactly, so the file is the arbiter and
no digit is a rounding of a rounding.  Color and shape come from
_common.CORPUS_COL / CORPUS_MARKER, the SAME registry Fig 3a draws its marker from, so
a glyph identifies one corpus across both panels with no text in either.

NO CROWD CORPUS IS ON THIS AXIS.  Chordonomicon, the one crowd-transcribed and
triad-flattened body, is excluded from this panel entirely and keeps its own figure;
the eight markers are score-notation (KernScores), performance-MIDI-chordified
(GiantMIDI, MAESTRO) and expert-annotated (CoCoPops, POP909, Jazz) sources only.  All
eight may legitimately share this axis because every marker is read off the SAME pooled
reference matrix, so no marker's position depends on its own corpus's counts.  Marker
positions are therefore below own-matrix sigma throughout -- CoCoPops 0.0549 against
0.1549, POP909 0.0485 against 0.2342 -- by construction, not by disagreement.

It survives splitting by extraction pipeline -- score/expert +0.67 (n=9,619),
chordified +0.43 (n=10,941) -- and holds in every dataset separately: KernScores
+0.38, GiantMIDI +0.43, MAESTRO +0.47, POP909 +0.45, Jazz +0.61, CoCoPops +0.47.
It is therefore not an artifact of any one corpus or extraction method.

Individual pieces are the data; the binned median and interquartile band are drawn
over them so the trend is read from the distribution rather than asserted.  Bins are
FIXED WIDTH (0.05) across the whole x-range, not equal-count: equal-count quantile
bins lumped everything above a 0.5 share into one point and hid the steepest part of
the rise.  The median is strictly monotone over all 15 bins, from -0.003 to +0.639
nats per step.  Color
marks extraction pipeline only, and the legend sits outside the axes so it cannot
cover data.

Data: dat/emergent_full.json, dat/emergent_full.npz
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import _figure_style as C

C.style()
DATADIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dat")
T = json.load(open(os.path.join(DATADIR, "emergent_full.json")))
Z = np.load(os.path.join(DATADIR, "emergent_full.npz"), allow_pickle=True)
i5, arrow, pipe = Z["i5"], Z["arrow"], Z["pipe"]

PCOL = {"score / expert": "#2f6f9f", "chordified": "#c98a2e"}

fig, ax = plt.subplots(figsize=(3.60, 3.10))

# 4,122 pieces (20.1% of the corpus) have a NEGATIVE arrow, reaching -0.56: pieces
# with little cadential motion have no reliable direction.  That mass must be visible,
# so every occupied cell is drawn (mincnt=1) and the axes are tinted so that pale
# single-digit-count hexes read as filled rather than vanishing into white.
ax.set_facecolor("#EFEFF2")
hb = ax.hexbin(i5, arrow, gridsize=52, cmap="magma_r", bins="log",
               mincnt=1, linewidths=0, zorder=2)
# Rasterize the density layer. As vector paths the ~900 low-count hexes below zero were
# dropped from the PDF stream (the PNG kept them), so the PDF truncated at y=0 and the
# 20% of pieces with a negative arrow vanished. As an embedded image at 600 dpi the PDF
# is pixel-identical to the PNG; axes, text and the median curve stay vector.
hb.set_rasterized(True)
# NO ax.set_rasterization_zorder() -- it used to be here and it was RASTERIZING THE
# AXIS TEXT.  set_rasterization_zorder(z) rasterizes the whole below-z group, and
# matplotlib draws the x/y axis artists (ticks, tick labels, axis labels) at zorder 2.5,
# so they were swept into the image layer: pdftotext recovered the legend and colorbar
# strings from the PDF but NOT the axis labels or the numeric ticks, which had been
# flattened to pixels.  Dropping the call leaves the hexbin rasterized exactly as before
# (set_rasterized(True) on the artist is what does that, and is sufficient on its own --
# same 2 image objects either way) while the text goes back to being vector, which the
# figure rules require.  Verified with pypdfium2 text extraction on both variants.
cb = fig.colorbar(hb, ax=ax, fraction=0.043, pad=0.02)
cb.set_label("pieces per cell", fontsize=7)
cb.ax.tick_params(labelsize=6)

b = T["bins"]
ax.fill_between(b["x"], b["q25"], b["q75"], facecolor="white", alpha=0.26,
                edgecolor="white", linewidth=0.9, zorder=3)
ax.plot(b["x"], b["med"], color="white", lw=2.0, zorder=4,
        path_effects=[pe.Stroke(linewidth=3.1, foreground="#222222"), pe.Normal()])
ax.scatter(b["x"], b["med"], s=13, color="white", edgecolor="#222222",
           linewidth=0.6, zorder=5)
ax.axhline(0.0, color="#555555", lw=0.8, ls=(0, (4, 2)), zorder=3.5)

# ---- CORPUS CENTROID MARKERS (Step 3) ------------------------------------
# Drawn BEFORE the limits are set so autoscale is not consulted -- limits below are
# taken from the piece arrays, and every centroid is inside the piece cloud by
# construction (a mean cannot exceed its own sample's range).
CENT = json.load(open(os.path.join(DATADIR, "corpus_centroids.json")))
for k in C.CORPUS_ORDER:
    c = CENT[k]
    # 95% CI in both directions.  MEASURED bar lengths at this figure size are
    # 0.3-15 pt (x) and 0.3-4.8 pt (y) against a 6-8.5 pt marker, so an earlier
    # version that drew them capless BENEATH the glyph (zorder 6 under 7) hid every
    # error bar inside its own marker -- the author correctly asked "where are the
    # errors?".  They are now drawn ON TOP with caps, so even a sub-marker interval
    # is visible as a cross projecting from the glyph.  Where a bar is genuinely
    # shorter than the marker that is the honest width and it should look tight.
    ax.plot(c["fifth"], c["arrow"], marker=C.CORPUS_MARKER[k], ls="none",
            markersize=8.5 if C.CORPUS_MARKER[k] == "*" else 6.0,
            markerfacecolor=C.CORPUS_COL[k], markeredgecolor="#FFFFFF",
            markeredgewidth=0.9, zorder=6,
            path_effects=[pe.withStroke(linewidth=1.1, foreground="#111111")])
    ax.errorbar(c["fifth"], c["arrow"],
                xerr=[[c["fifth"] - c["fifth_ci"][0]], [c["fifth_ci"][1] - c["fifth"]]],
                yerr=[[c["arrow"] - c["arrow_ci"][0]], [c["arrow_ci"][1] - c["arrow"]]],
                fmt="none", ecolor="#111111", elinewidth=0.55, capsize=1.3,
                capthick=0.55, zorder=8)

# AXES = DATA EXTENT.  Both limits come from the plotted arrays with a 3% margin, so
# no piece is cropped (the previous fixed limits dropped 58) and no empty frame is
# shown beyond the data.
_xpad = 0.03 * (i5.max() - i5.min())
_ypad = 0.03 * (arrow.max() - arrow.min())
ax.set_xlim(i5.min() - _xpad, i5.max() + _xpad)
ax.set_ylim(arrow.min() - _ypad, arrow.max() + _ypad)
ax.set_xlabel("descending-fifth share of chord transitions", fontsize=8.5)
# sigma_i, NOT plain sigma: this is the per-piece mean affinity under the pooled
# reference matrix, an exact additive decomposition of the corpus sigma (see the
# module docstring). The definition belongs in the caption, not in the axis.
# THE Y AXIS IS AN AFFINITY, NOT AN ENTROPY PRODUCTION.  Entropy production cannot be
# negative, and 20.0% of these points are -- so calling the per-piece quantity sigma was
# wrong (the author caught it).  What a piece carries is the MEAN AFFINITY of its steps,
# <A>_i = (1/m_i) sum_steps log(N_fwd/N_rev): a thermodynamic FORCE, which is signed by
# construction.  A negative value means the piece runs against the pool's preferred
# direction, which is a perfectly ordinary thing for a force to do.
#   Entropy production is force x current summed over COMPLETE edge pairs.  Pairing t
# with its reverse gives (p_f - p_r) log(f/r) >= 0 because both factors share a sign, so
# sigma >= 0 -- but that pairing only closes when the sum runs over a whole chain.  A
# single piece samples an incomplete set of pairs, so its mean affinity has no such
# bound.  The step-weighted mean over a corpus DOES close the pairs and is an entropy
# production: all eight corpora come out positive (0.0204 to 0.4427).
#   UNITS.  A(t) = log(N_f/N_r) is the entropy produced by ONE step, so A is nats
# step^-1; J(t) = (N_f - N_r)/M is a net FRACTION of steps and is dimensionless.  Hence
# sigma = sum_t J A carries nats step^-1, and so must <A>, because a weighted mean
# cannot change units and the step-weighted mean of these very values IS sigma
# (0.049821296, asserted in compute_rootmotion.py).  A bare "(nats)" label -- used
# briefly in an earlier version of this script -- was wrong for exactly that reason.
ax.set_ylabel(r"mean affinity  $\langle A \rangle$  (nats step$^{-1}$)", fontsize=8.5)

# LEGEND: the median curve, plus the eight corpus glyphs so nothing in the panel is
# unexplained. Corpus names only -- no numbers, no commentary. It sits LOWER RIGHT,
# which honest limits leave empty: only 1 of 20,560 pieces falls at x>0.45 & y<-0.10.
# That region exists because 3 pieces reach y = -0.56 and stretch the frame; putting
# the legend there spends the dead space instead of covering the rise past x = 0.5,
# which is the feature the panel is for.  Upper-left is emptier still (0 pieces above
# y = 0.55) but sits directly above the steep segment.
_med_h = plt.Line2D([0], [0], color="white", lw=2.0,
                    path_effects=[pe.Stroke(linewidth=3.1, foreground="#222222"),
                                  pe.Normal()],
                    label="median, IQR  (n=%s)" % f"{T['n']:,}")
_corp_h = [plt.Line2D([0], [0], marker=C.CORPUS_MARKER[k], ls="none",
                      markersize=7.5 if C.CORPUS_MARKER[k] == "*" else 5.4,
                      markerfacecolor=C.CORPUS_COL[k], markeredgecolor="#111111",
                      markeredgewidth=0.7, label=C.CORPUS_LABEL.get(k, k))
           for k in C.CORPUS_ORDER]
leg = ax.legend(handles=[_med_h] + _corp_h, fontsize=5.9, frameon=True,
                loc="lower right", ncol=2, handletextpad=0.4, columnspacing=0.8,
                labelspacing=0.26, borderpad=0.35)
leg.get_frame().set_edgecolor("#222222")
leg.get_frame().set_linewidth(0.8)
leg.get_frame().set_facecolor("white")
leg.get_frame().set_alpha(1.0)
leg.set_zorder(8)
C.all_spines(ax)

# ---- MAGNIFIER INSET on the seven-corpus cluster --------------------------
# Seven of the eight centroids fall inside x 0.18-0.31, y 0.02-0.14, closer together
# than the markers are wide: Baroque-Classical and Baroque-Romantic are 1.1 pt apart
# and CoCoPops-POP909 (Billboard pop - Mandopop) 2.3 pt, against 6 pt markers. Dataset
# keys are used in code and data files; Table 1 and the figures use the display names.
# In the main axes they overlap
# into one blob, so the legend promises eight glyphs and only four are readable.
# Nudging them apart would misreport their positions; magnifying the region reports
# them exactly. The inset sits upper-left, where 2 of 20,560 pieces fall.
# The connector rectangle + lines are what make it self-explanatory -- no text.
# x0 raised from 0.055 to 0.135: the y-tick labels are drawn OUTSIDE the inset's own
# box, to the left of it, so at x0=0.055 they ran past the main axes' left spine and
# were clipped at the figure edge (the author caught this).  0.135 of a 3.76 in axes
# is 0.51 in of clearance, against ~0.16 in of label + pad at 5.0 pt.
axi = ax.inset_axes([0.135, 0.605, 0.315, 0.345])
axi.set_facecolor("#FFFFFF")
for k in C.CORPUS_ORDER:
    c = CENT[k]
    # Same ordering fix as the main axes: glyph first, CI on top with caps, so the
    # magnified view actually shows the intervals it exists to resolve.
    axi.plot(c["fifth"], c["arrow"], marker=C.CORPUS_MARKER[k], ls="none",
             markersize=8.0 if C.CORPUS_MARKER[k] == "*" else 5.6,
             markerfacecolor=C.CORPUS_COL[k], markeredgecolor="#FFFFFF",
             markeredgewidth=0.8, zorder=6,
             path_effects=[pe.withStroke(linewidth=1.0, foreground="#111111")])
    axi.errorbar(c["fifth"], c["arrow"],
                 xerr=[[c["fifth"] - c["fifth_ci"][0]], [c["fifth_ci"][1] - c["fifth"]]],
                 yerr=[[c["arrow"] - c["arrow_ci"][0]], [c["arrow_ci"][1] - c["arrow"]]],
                 fmt="none", ecolor="#111111", elinewidth=0.55, capsize=1.2,
                 capthick=0.55, zorder=8)
# Window = the seven-corpus bbox (x 0.187-0.300, y 0.020-0.138) plus a margin wide
# enough that no marker is clipped by the inset frame -- Renaissance sits at the
# lower-left corner and a tighter window cut its glyph in half.
axi.set_xlim(0.163, 0.325)
axi.set_ylim(-0.008, 0.158)
axi.set_xticks([0.20, 0.25, 0.30]); axi.set_yticks([0.00, 0.05, 0.10, 0.15])
axi.tick_params(labelsize=5.0, length=1.6, pad=0.8, top=False, right=False)
axi.set_xlabel(""); axi.set_ylabel("")
C.all_spines(axi)
for s in axi.spines.values():
    s.set_linewidth(0.55)
ax.indicate_inset_zoom(axi, edgecolor="#111111", linewidth=0.6, alpha=0.85)

C.save(fig, "panel_4c_scaling")
print("wrote panel_4c_scaling | n=%d pearson=%+.4f spearman=%+.4f"
      % (T["n"], T["pearson"], T["spearman"]))
print("  xlim %.4f..%.4f  ylim %.4f..%.4f (data extent, 3%% pad)"
      % (*ax.get_xlim(), *ax.get_ylim()))
print("  centroids: " + "  ".join("%s(%.3f,%.3f)" % (k, CENT[k]["fifth"], CENT[k]["arrow"])
                                  for k in C.CORPUS_ORDER))
