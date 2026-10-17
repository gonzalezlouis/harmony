import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _figure_style as C
from fig2c_networks import FIFTHS, QUAL_FILL, QUAL_RADIUS

C.style()

QCOL = QUAL_FILL
CHORD_TONES = {"maj": [0, 4, 7], "min": [0, 3, 7], "dom7": [0, 4, 7, 10],
               "dim": [0, 3, 6], "aug": [0, 4, 8]}
SYMBOL = {"maj": "C", "min": "Cm", "dom7": "C7", "dim": "C\u00b0", "aug": "C+"}
QORDER = ["maj", "dom7", "min", "dim", "aug"]
QLABEL = {"maj": "maj", "dom7": "dom7", "min": "min", "dim": "dim", "aug": "aug"}

WHITE_SEMI  = [0, 2, 4, 5, 7, 9, 11, 12]
BLACK_SEMI  = [1, 3, 6, 8, 10]
BLACK_AFTER = {1: 0, 3: 1, 6: 3, 8: 4, 10: 5}
KH, BH, BW  = 4.2, 2.6, 0.58

_ORD  = sorted(QUAL_RADIUS, key=lambda q: -QUAL_RADIUS[q])
RING_R = {q: r for q, r in zip(_ORD, [1.30, 1.075, 0.85, 0.625, 0.40])}
REST_COL = "#8a9199"


def _assert_61_visible(ax, drawn):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    inside = lambda p: x0 <= p.center[0] <= x1 and y0 <= p.center[1] <= y1
    marks = [p for p in ax.patches
             if isinstance(p, Circle) and (p.get_gid() or "").startswith("state:")
             and inside(p)]
    filled = [p for p in marks if p.get_gid() == "state:measured"]
    hollow = [p for p in marks if p.get_gid() == "state:rest"]
    assert len(filled) == 60, f"{len(filled)} measured markers visible, expected 60"
    assert len(hollow) == 1, f"{len(hollow)} hollow markers visible, expected 1 (REST)"
    assert len(drawn) == len(set(drawn)) == 61, f"{len(drawn)} labels, {len(set(drawn))} unique"
    return len(filled) + len(hollow)


def _semitone_xy(s):
    if s in WHITE_SEMI:
        return WHITE_SEMI.index(s) + 0.5, 0.80
    return BLACK_AFTER[s] + 1.0, BH - 0.62 + (KH - BH)


def keyboard(q):
    fig, ax = plt.subplots(figsize=(1.10, 0.70))
    ax.set_aspect("equal"); ax.axis("off")
    for i in range(len(WHITE_SEMI)):
        ax.add_patch(Rectangle((i, 0), 1, KH, fc="white", ec="#333333", lw=0.5, zorder=1))
    for s in BLACK_SEMI:
        ax.add_patch(Rectangle((BLACK_AFTER[s] + 1 - BW / 2, KH - BH), BW, BH,
                               fc="#1A1A1A", ec="#1A1A1A", lw=0.4, zorder=2))
    for s in CHORD_TONES[q]:
        x, y = _semitone_xy(s)
        ax.add_patch(Circle((x, y), 0.30, fc=QCOL[q], ec="white", lw=0.5, zorder=4))
    ax.text(4.0, KH + 0.42, SYMBOL[q], ha="center", va="bottom", fontsize=6.5,
            fontweight="bold", color=QCOL[q])
    ax.set_xlim(-0.15, 8.15); ax.set_ylim(-0.15, KH + 1.35)
    C.save(fig, f"fig1b_key_{q}")


def _ang(fi):
    return np.pi / 2 - fi * 2 * np.pi / 12


MINORS = ['a', 'e', 'b', r'f$\sharp$', r'c$\sharp$', r'g$\sharp$', r'd$\sharp$',
          r'a$\sharp$', 'f', 'c', 'g', 'd']


def circle_of_fifths(minors=False, name="fig1b_circle_of_fifths", size=2.2):
    fig, ax = plt.subplots(figsize=(size, size))
    ax.set_aspect("equal"); ax.axis("off")
    RL = 1.0
    ax.add_patch(Circle((0, 0), RL, fill=False, ec="#b9bec4", lw=0.6, zorder=0))
    if minors:
        ax.add_patch(Circle((0, 0), 0.62, fill=False, ec="#d3d7db", lw=0.5, ls=(0, (3, 2)), zorder=0))
    for fi, lbl in enumerate(FIFTHS):
        a = _ang(fi)
        r = RL
        ax.text(r * np.cos(a), r * np.sin(a), lbl, ha="center", va="center",
                fontsize=7.5 if not minors else 7.0, fontweight="bold",
                color="#1A1A1A", zorder=4,
                bbox=dict(boxstyle="circle,pad=0.30", fc="white", ec=QCOL["maj"], lw=0.7))
        if minors:
            ax.text(0.62 * np.cos(a), 0.62 * np.sin(a), MINORS[fi], ha="center",
                    va="center", fontsize=5.6, color="#1A1A1A", zorder=4,
                    bbox=dict(boxstyle="circle,pad=0.22", fc="white",
                              ec=QCOL["min"], lw=0.55))
    m = 1.30 if not minors else 1.30
    ax.set_xlim(-m, m); ax.set_ylim(-m, m)
    C.save(fig, name)


def state_space(name="fig1b_state_space", size=2.75):
    fig, ax = plt.subplots(figsize=(size, size))
    ax.set_aspect("equal"); ax.axis("off")
    drawn = []
    for fi in range(12):
        a = _ang(fi)
        ax.plot([RING_R["aug"] * np.cos(a), RING_R["maj"] * np.cos(a)],
                [RING_R["aug"] * np.sin(a), RING_R["maj"] * np.sin(a)],
                color="#d9dde1", lw=0.35, zorder=0)
    for q in QORDER:
        ax.add_patch(Circle((0, 0), RING_R[q], fill=False, ec="#e2e6ea", lw=0.35, zorder=0))
    for fi in range(12):
        a = _ang(fi)
        for q in QORDER:
            r = RING_R[q]
            ax.add_patch(Circle((r * np.cos(a), r * np.sin(a)), 0.072, fc=QCOL[q],
                                ec="white", lw=0.45, zorder=3, gid="state:measured"))
            drawn.append(f"{fi}:{q}")
        ax.text(1.52 * np.cos(a), 1.52 * np.sin(a), FIFTHS[fi], ha="center",
                va="center", fontsize=6.2, fontweight="bold", color="#1A1A1A", zorder=4)
    for q in QORDER:
        ax.text(0.105, RING_R[q], QLABEL[q], ha="left", va="center", fontsize=5.0,
                color="#3a4046", zorder=5)
    rx, ry = -1.44, -1.44
    ax.add_patch(Circle((rx, ry), 0.072, fc="none", ec=REST_COL, lw=0.7, zorder=3,
                        gid="state:rest"))
    drawn.append("REST")
    ax.text(rx + 0.13, ry, "REST", ha="left", va="center", fontsize=5.6,
            color="#3a4046", zorder=5)
    ax.set_xlim(-1.66, 1.66); ax.set_ylim(-1.66, 1.66)
    n = _assert_61_visible(ax, drawn)
    C.save(fig, name)
    return n


def state_grid(name="_fig1b_alt_grid"):
    fig, ax = plt.subplots(figsize=(2.75, 1.35))
    ax.set_aspect("equal"); ax.axis("off")
    drawn = []
    for ci in range(12):
        for ri, q in enumerate(QORDER):
            ax.add_patch(Circle((ci, -ri), 0.34, fc=QCOL[q], ec="white", lw=0.5,
                                zorder=3, gid="state:measured"))
            drawn.append(f"{ci}:{q}")
        ax.text(ci, 0.85, FIFTHS[ci], ha="center", va="center", fontsize=6.0,
                fontweight="bold", color="#1A1A1A")
    for ri, q in enumerate(QORDER):
        ax.text(-0.95, -ri, QLABEL[q], ha="right", va="center", fontsize=6.0, color="#3a4046")
    ax.add_patch(Circle((13.1, -2.0), 0.34, fc="none", ec=REST_COL, lw=0.7, zorder=3,
                        gid="state:rest"))
    drawn.append("REST")
    ax.text(13.1, -2.72, "REST", ha="center", va="center", fontsize=6.0, color="#3a4046")
    ax.set_xlim(-2.6, 14.2); ax.set_ylim(-4.9, 1.4)
    n = _assert_61_visible(ax, drawn)
    C.save(fig, name)
    return n


if __name__ == "__main__":
    for q in QORDER:
        keyboard(q)
    circle_of_fifths(minors=False)
    circle_of_fifths(minors=True, name="_fig1b_cof_minors")
    n_ring = state_space()
    n_grid = state_grid()
    print("states drawn ring/grid:", n_ring, n_grid)
    print("FIFTHS:", [f for f in FIFTHS])
    print("ring order:", _ORD, "radii:", RING_R)
