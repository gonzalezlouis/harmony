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

QUAL_FILL = {"maj": "#0077BB", "min": "#009988", "dom7": "#CC3311", "dim": "#7B4F9E"}

MCS = ["Major (functional)", "Dorian/Mixolydian (modal)"]
DISP = {"Major (functional)": "functional", "Dorian/Mixolydian (modal)": "modal"}
NODE_Q = {"Major (functional)":        ["maj", "min", "maj", "maj"],
          "Dorian/Mixolydian (modal)": ["min", "maj", "maj", "min"]}

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
