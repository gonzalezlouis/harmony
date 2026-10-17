#!/usr/bin/env python3
"""Figure 4 — weighted loop network for the Irish settings

Fig 4 - weighted harmonic loop networks for the two Irish mode classes.

LEVEL: MODE-CLASS CORPUS. Functional = 2,920 settings of 2,173 tunes; modal = 775
settings of 554 tunes. Nothing here is computed on the engraved tunes -- single-tune
sigma is not interpretable and a single tune's chord walk is usually an out-and-back
that zeroes every net current by topology.

WHAT THE BRIEF EXPECTED vs WHAT THE DATA SAYS. The expectation was a directed cycle
for the functional corpus and near-cancellation for the modal one. THE MODAL CORPUS
ALSO CIRCULATES, and this panel is drawn to say so:

  functional  I -> VIm -> IV -> V -> I     cycle affinity A_C = +2.760 [+2.526,+3.032]
  modal       Im -> bIII -> bVII -> Vm -> Im            A_C = +2.240 [+1.752,+2.758]

All four edges run forward in 100.0% (functional) and 99.5% (modal) of 600
tune-resampled bootstrap replicates, so neither loop is a cancellation. The modal
loop is a genuine directed cycle around the double tonic, not a wash.

The 7.03x sigma gap [4.32, 9.72] is therefore NOT "loop vs no loop". It factorises
exactly as sigma = J_tot * A_mean:
    J_tot   0.2380 vs 0.0737  ->  3.23x    (less net traffic on the modal loop)
    A_mean  0.763  vs 0.351   ->  2.18x    (weaker force per type)
    product                       7.03x
Modal settings also spend more of their steps oscillating: 42.09% [40.11,44.17] of
interior steps are immediate A->B->A reversals, against 34.30% [33.26,35.19]
functional -- the double-tonic see-saw is real, it just does not cancel the loop.

HOW FAR ABOVE ITS OWN NOISE FLOOR EACH LOOP SITS (B=300 order-shuffles per corpus,
floor = 95th percentile of the shuffled null; data/irish_axis_nulls.json):
    functional harmonic  sigma 0.1867  floor 0.00601  ->  30.9x above null
    modal      harmonic  sigma 0.0458  floor 0.01292  ->   3.5x above null
    minor      harmonic  sigma 0.1335  floor 0.01640  ->   8.0x above null
This is the number that reconciles this panel with the Fig 4 bars. Both loops are
real, but the modal loop clears its own null by only 3.5x against functional's 30.9x,
so "modal barely circulates" is defensible while "modal does not circulate" is not.

DO NOT claim the MELODIC axis is loop-free. Measured (same nulls file): melodic sigma
is 145.9x / 120.0x / 54.5x above its own floor for functional / modal / minor -- the
melody carries a robust arrow too. Those ratios look enormous only because melody runs
~5.6x more transitions over 7 pitch-class states instead of 52 chord states, which
collapses the shuffled floor ~46x; in ABSOLUTE terms melodic sigma is 0.019-0.056, an
order of magnitude under functional harmonic sigma. The defensible harmonic-vs-melodic
statement is the RATIO: functional harmony carries 9.68x its own melody's arrow while
modal harmony carries 0.81x -- less than its own melody.

DRAWING. Arrow width is LINEAR in measured net current J(t) on a scale SHARED
between the two networks, so the modal loop's thinner arrows are the measurement,
not a style choice. A width key resolves the scale. Reverse arrows are drawn thin,
dashed, grey wherever the reverse type is observed (it always is here -- that is
what makes these edges admissible to the estimator at all). Node fill = chord
QUALITY on the canonical QUAL_FILL palette. Values are NOT printed on the panel;
they live in data/irish_expansion.json and the caption spec.
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _figure_style as C

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXP = json.load(open(os.path.join(ROOT, "dat", "irish_expansion.json")))
LOOPS = EXP["step2_loop"]["loops"]

# canonical quality palette (panel_2c_networks.QUAL_FILL / _common.QUAL)
QUAL_FILL = {"maj": "#0077BB", "min": "#009988", "dom7": "#CC3311", "dim": "#7B4F9E"}

MCS = ["Major (functional)", "Dorian/Mixolydian (modal)"]
DISP = {"Major (functional)": "functional", "Dorian/Mixolydian (modal)": "modal"}
# node quality per loop, in the same order as LOOPS[mc]["cycle"]
NODE_Q = {"Major (functional)":        ["maj", "min", "maj", "maj"],
          "Dorian/Mixolydian (modal)": ["min", "maj", "maj", "min"]}

# shared width scale across BOTH networks
JMAX = max(max(LOOPS[mc]["J"]) for mc in MCS)
LW_MIN, LW_SPAN = 0.55, 3.05


def lw_of(J):
    return LW_MIN + LW_SPAN * (J / JMAX)


C.style()
fig = plt.figure(figsize=(4.60, 2.42))
gs = fig.add_gridspec(1, 2, wspace=0.02, left=0.01, right=0.99,
                      bottom=0.115, top=0.90)

R_RING, R_NODE, RAD, BDY = 1.16, 0.40, 0.30, 13

for i, mc in enumerate(MCS):
    ax = fig.add_subplot(gs[0, i])
    ax.set_xlim(-1.80, 1.80); ax.set_ylim(-1.90, 1.72)
    ax.axis("off"); ax.set_aspect("equal")

    names = LOOPS[mc]["cycle"]
    J = LOOPS[mc]["J"]
    n = len(names)
    # start at top, run CLOCKWISE so the drawn circulation matches reading order
    ang = [np.pi / 2 - k * 2 * np.pi / n for k in range(n)]
    P = {names[k]: R_RING * np.array([np.cos(ang[k]), np.sin(ang[k])]) for k in range(n)}

    def bdy(src, dst, deg):
        v = P[dst] - P[src]
        th = np.arctan2(v[1], v[0]) + np.radians(deg)
        return P[src] + R_NODE * np.array([np.cos(th), np.sin(th)])

    for k in range(n):
        a, b = names[k], names[(k + 1) % n]
        p0, p1 = bdy(a, b, +BDY), bdy(b, a, -BDY)
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", color="#111111",
                     lw=lw_of(J[k]), mutation_scale=8 + 5.5 * (J[k] / JMAX),
                     connectionstyle=f"arc3,rad={RAD}", zorder=3,
                     joinstyle="miter", capstyle="butt"))
        # reverse traffic: observed on every one of these edges (required by the estimator)
        ax.add_patch(FancyArrowPatch(p1, p0, arrowstyle="-|>", color="#9AA5B1",
                     lw=0.7, mutation_scale=6.5, linestyle=(0, (2.2, 1.6)),
                     connectionstyle=f"arc3,rad={RAD}", zorder=2,
                     joinstyle="miter", capstyle="butt"))

    for k, nm in enumerate(names):
        col = QUAL_FILL[NODE_Q[mc][k]]
        ax.add_patch(plt.Circle(P[nm], R_NODE, fc=col, ec="white", lw=1.7,
                                zorder=4, clip_on=False))
        ax.text(P[nm][0], P[nm][1], nm, ha="center", va="center", fontsize=8.2,
                color="white", fontweight="bold", zorder=5)

    ax.set_title(DISP[mc], fontsize=7.2, pad=1.0)

    # width key, once per panel, in the dead space below the ring
    if i == 0:
        for m, yk in ((JMAX, -1.52), (JMAX / 4, -1.80)):
            ax.add_patch(FancyArrowPatch((-1.62, yk), (-0.92, yk), arrowstyle="-|>",
                         color="#111111", lw=lw_of(m),
                         mutation_scale=8 + 5.5 * (m / JMAX), zorder=3))
            ax.text(-0.84, yk, f"{m:.2f}", fontsize=5.6, va="center", ha="left")
        ax.text(-0.30, -1.66, r"$J(t)$", fontsize=6.2, va="center", ha="left")
    else:
        ax.add_patch(FancyArrowPatch((-1.62, -1.52), (-0.92, -1.52), arrowstyle="-|>",
                     color="#9AA5B1", lw=0.7, mutation_scale=6.5,
                     linestyle=(0, (2.2, 1.6)), zorder=3))
        ax.text(-0.84, -1.52, "reverse", fontsize=5.6, va="center", ha="left")
        for q, lab, yk in (("maj", "major", -1.80), ("min", "minor", -1.80)):
            pass
        ax.add_patch(plt.Circle((-1.50, -1.80), 0.10, fc=QUAL_FILL["maj"],
                                ec="white", lw=0.8, zorder=4))
        ax.text(-1.35, -1.80, "major", fontsize=5.6, va="center", ha="left")
        ax.add_patch(plt.Circle((-0.62, -1.80), 0.10, fc=QUAL_FILL["min"],
                                ec="white", lw=0.8, zorder=4))
        ax.text(-0.47, -1.80, "minor", fontsize=5.6, va="center", ha="left")

fig.savefig(os.path.join(C.OUT, "panel_4_irish_loop.pdf"), dpi=600)
fig.savefig(os.path.join(C.OUT, "panel_4_irish_loop.png"), dpi=300)
plt.close(fig)

print("wrote panel_4_irish_loop  |  " + "  ".join(
    f"{DISP[mc]}: A_C={LOOPS[mc]['A_C']:+.3f} "
    f"[{LOOPS[mc]['A_C_ci'][0]:+.3f},{LOOPS[mc]['A_C_ci'][1]:+.3f}] "
    f"Jmin={min(LOOPS[mc]['J']):.5f} Jmax={max(LOOPS[mc]['J']):.5f}" for mc in MCS))
