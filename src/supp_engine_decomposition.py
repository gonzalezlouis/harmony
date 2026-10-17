"""Supplementary — net current by harmonic mechanism

Regenerate the 25x12 invariant-type current matrix from the canonical pools, then
render the engine decomposition on it.

The stored transfer_matrix.npz reproduces Table 1's J_tot for six of eight corpora but
runs 11% high for ragtime and 2.4% high for Billboard pop, so it is stale for those two.
Recomputing from canonical_pools reproduces all eight, and that is what is used here.

Engine classes follow the original fig_engine_eras design, now identifiable because the
row labels are the 25 ordered quality pairs:
  V7->I cadence      row dom7->maj at the descending fifth
  ii->V cycle        row min->dom7  at the descending fifth
  other fifth        any other quality pair at the descending fifth
  unclassified       every other root motion
"""
import os, sys, json
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
AA = os.path.dirname(_HERE)          # project root
DAT = os.path.join(AA, "dat")
FIGSUPP = os.path.join(AA, "fig", "supp")
sys.path.insert(0, _HERE)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _figure_style as C
from compute_mechanism import invariant_counts, _rev
from canonical_pools import build, build_modern

QUAL = ["maj", "min", "dom7", "dim", "aug"]
ROWS = [f"{a}->{b}" for a in QUAL for b in QUAL]
ORD = ["Renaissance", "Baroque", "Classical", "Romantic", "Ragtime",
       "CoCoPops", "POP909", "Jazz"]
T1J = {"Renaissance": 0.095, "Baroque": 0.264, "Classical": 0.185, "Romantic": 0.191,
       "Ragtime": 0.320, "CoCoPops": 0.192, "POP909": 0.285, "Jazz": 0.653}
FIFTH = 5


def matrix(seqs):
    Ct = invariant_counts(seqs); N = sum(Ct.values())
    M = np.full((25, 12), np.nan)
    for t, f in Ct.items():
        r = Ct.get(_rev(t), 0)
        if r == 0:
            continue
        d, qa, qb = t
        try:
            i = ROWS.index(f"{qa}->{qb}")
        except ValueError:
            continue
        M[i, d] = (f - r) / N
    return M


if __name__ == "__main__":
    pools, _ = build(AA); pools.update(build_modern(AA))
    MATS, chk = {}, {}
    print(f"{'corpus':13s} {'pos-sum':>9s} {'Table 1 J':>10s} {'agree':>6s}")
    for k in ORD:
        M = matrix(pools[k]); MATS[k] = M
        pos = float(M[np.isfinite(M) & (M > 0)].sum())
        ok = abs(pos - T1J[k]) < 0.002
        chk[k] = dict(pos_sum=round(pos, 4), table1_J=T1J[k], agree=bool(ok))
        print(f"{k:13s} {pos:9.4f} {T1J[k]:10.3f} {'yes' if ok else 'NO':>6s}")
    np.savez(os.path.join(DAT, "transfer_matrix_canonical.npz"),
             rows=np.array(ROWS), **MATS)

    # ---- engine decomposition ----
    iC, iII = ROWS.index("dom7->maj"), ROWS.index("min->dom7")
    ENG = {}
    for k in ORD:
        M = MATS[k]; tot = M[np.isfinite(M) & (M > 0)].sum()
        col = M[:, FIFTH]
        pos = lambda v: float(v) if np.isfinite(v) and v > 0 else 0.0
        cad, ii = pos(col[iC]), pos(col[iII])
        oth = sum(pos(col[i]) for i in range(25) if i not in (iC, iII))
        ENG[k] = dict(cadence=cad / tot, ii_v=ii / tot, other_fifth=oth / tot,
                      unclassified=1 - (cad + ii + oth) / tot, sigma_J_total=float(tot))
    LBL = [("cadence", r"V$^7\!\to$I"), ("ii_v", r"ii$\to$V"),
           ("other_fifth", "other fifth"), ("unclassified", "other motion")]
    COLS = ["#1F5582", "#3D8AAF", "#8FBFD4", "#D5D5DB"]
    C.style()
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    y = np.arange(len(ORD))[::-1]
    left = np.zeros(len(ORD))
    for (key, lab), c in zip(LBL, COLS):
        v = np.array([ENG[k][key] for k in ORD])
        ax.barh(y, v, left=left, height=0.62, color=c, label=lab, zorder=3,
                edgecolor="white", linewidth=0.4)
        left += v
    ax.set_yticks(y); ax.set_yticklabels([C.CORPUS_LABEL.get(k, k) for k in ORD])
    ax.set_xlabel("share of net current")
    ax.set_xlim(0, 1); ax.set_ylim(-0.7, len(ORD) - 0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=4, frameon=False,
              fontsize=5.8, handlelength=1.1, columnspacing=0.9, handletextpad=0.4)
    fig.tight_layout(pad=0.3); fig.canvas.draw()
    fig.savefig(os.path.join(FIGSUPP, "aligned_engine.pdf"),
                bbox_inches=C.fixed_aspect_bbox(fig))
    print("\nengine shares (fraction of net current):")
    print(f"{'corpus':13s} {'V7->I':>7s} {'ii->V':>7s} {'other5':>7s} {'other':>7s}")
    for k in ORD:
        e = ENG[k]
        print(f"{k:13s} {e['cadence']:7.3f} {e['ii_v']:7.3f} {e['other_fifth']:7.3f} {e['unclassified']:7.3f}")
    json.dump({"engine": ENG, "matrix_verification": chk,
               "stale_npz":"transfer_matrix.npz is stale for Ragtime (0.3549 vs 0.320) and CoCoPops "
                           "(0.1967 vs 0.192); panel_4a_matrices.py and panel_4b_poparadox.py read it"},
              open(os.path.join(DAT, "engine_decomposition.json"), "w"), indent=1)
