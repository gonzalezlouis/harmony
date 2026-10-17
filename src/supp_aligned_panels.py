import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
AA = os.path.dirname(HERE)
OUT = os.path.join(AA, "fig", "supp")

import _figure_style as C
import canonical_pools as CP
import compute_mechanism as CM
import supp_panel_library as SPL

ORD8 = ["Renaissance", "Baroque", "Classical", "Romantic", "Ragtime",
        "CoCoPops", "POP909", "Jazz"]
NM = SPL.NM


def pools():
    P = CP.build(HERE);        P = P[0] if isinstance(P, tuple) else P
    M = CP.build_modern(HERE); M = M[0] if isinstance(M, tuple) else M
    return {**P, **M}


def _sigma(seqs):
    return float(CM.sigma_invariant(seqs)[0])


def panel_palindrome(ALL, corpus="Jazz", seed=0):
    rng = np.random.default_rng(seed)
    sq = ALL[corpus]
    vals = {
        "as measured":          _sigma(sq),
        "each work\ntime-reversed": _sigma([s[::-1] for s in sq]),
        "each work\npalindromized": _sigma([s + s[::-1] for s in sq]),
        "order\nshuffled":      _sigma([list(rng.permutation(s)) for s in sq]),
    }
    fig, ax = plt.subplots(figsize=(SPL.W1 * 1.28, SPL.H1 * 1.10))
    xs = np.arange(len(vals))
    ax.bar(xs, list(vals.values()), width=0.62, color=C.CORPUS_COL[corpus],
           edgecolor="white", linewidth=0.6)
    ax.set_xticks(xs); ax.set_xticklabels(list(vals), fontsize=6.0)
    ax.set_ylabel(r"$\sigma$  (nats step$^{-1}$)")
    ax.set_ylim(0, max(vals.values()) * 1.22)
    for xx, v in zip(xs, vals.values()):
        ax.annotate(f"{v:.3f}", (xx, v), textcoords="offset points", xytext=(0, 2),
                    ha="center", va="bottom", fontsize=6.2)
    return SPL._fin(fig, ax, "aligned_palindrome", OUT), vals


def _current_shares(seqs):
    Ct = CM.invariant_counts(seqs)
    J = []
    for t, f in Ct.items():
        r = Ct.get(CM._rev(t), 0)
        if r > 0 and f > r:
            J.append(f - r)
    J = np.array(sorted(J, reverse=True), float)
    return J / J.sum() if J.sum() else J


def panel_concentration(ALL):
    fig, ax = plt.subplots(figsize=(SPL.W1 * 1.28, SPL.H1 * 1.10))
    ax.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.7, ls=(0, (3, 2)), zorder=1)
    for k in ORD8:
        sh = _current_shares(ALL[k])
        x = np.arange(1, len(sh) + 1) / len(sh)
        ax.plot(np.r_[0, x], np.r_[0, np.cumsum(sh)], color=C.CORPUS_COL[k], lw=1.0,
                label=NM(k), zorder=3)
    ax.set_xlabel("fraction of transition types, ranked by current")
    ax.set_ylabel("cumulative share of net current")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="lower right", frameon=False, fontsize=5.6, ncol=2,
              handlelength=1.3, handletextpad=0.5, labelspacing=0.2, borderpad=0.0)
    return SPL._fin(fig, ax, "aligned_concentration", OUT)


def _fifth_share(seqs):
    Ct = CM.invariant_counts(seqs)
    tot = fifth = 0.0
    for t, f in Ct.items():
        r = Ct.get(CM._rev(t), 0)
        if r > 0 and f > r:
            tot += f - r
            if int(t[0]) == 5:
                fifth += f - r
    return fifth / tot if tot else float("nan")


def panel_fifth_share(ALL):
    fig, ax = plt.subplots(figsize=(SPL.W1 * 1.20, SPL.H1 * 1.16))
    ys = np.arange(len(ORD8))[::-1]
    for y, k in zip(ys, ORD8):
        ax.plot([_fifth_share(ALL[k])], [y], marker=C.CORPUS_MARKER[k], ms=5.2,
                color=C.CORPUS_COL[k], mec="white", mew=0.6, ls="none", zorder=3)
    ax.axvline(1 / 12, color="#888888", lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(ys); ax.set_yticklabels([NM(k) for k in ORD8], fontsize=6.4)
    ax.set_xlabel("share of net current on descending-fifth motion")
    ax.set_xlim(0, 0.8); ax.set_ylim(-0.7, len(ORD8) - 0.3)
    ax.legend(handles=[Line2D([], [], ls=(0, (3, 2)), color="#888888", lw=0.8,
                              label="1/12, equal share over motions")],
              loc="lower right", frameon=False, fontsize=6.0, borderpad=0.0)
    return SPL._fin(fig, ax, "aligned_fifth_share", OUT)


if __name__ == "__main__":
    C.style()
    ALL = pools()
    made = []
    nm, vals = panel_palindrome(ALL); made.append(nm)
    print("palindrome:", {k.replace(chr(10), " "): round(v, 4) for k, v in vals.items()})
    made.append(panel_concentration(ALL))
    made.append(panel_fifth_share(ALL))
    print("fifth share:", {NM(k): round(_fifth_share(ALL[k]), 3) for k in ORD8})
    print("MADE:", made)
