#!/usr/bin/env python3
r"""Figure 2c inset — the symmetrized-null network

Fig 2c companion -- what sigma = 0 looks like ON THE SAME STATE SPACE.

Takes BAROQUE's own bigram counts and symmetrizes them, C' = (C + C.T)/2, then
renders with fig2c_networks.render and fig2c_networks._xy -- imported, not
reimplemented -- so the node vocabulary, radii, spokes, sizes, fonts and legend
are byte-for-byte the same machinery as the six directed wheels.

WHY SYMMETRIZATION IS THE RIGHT NULL HERE.  It preserves the chord vocabulary,
the transition support and the stationary distribution pi (a symmetric C has the
same pi as its own reversed chain), and annihilates exactly one thing: the net
current J_ij = pi_i P_ij - pi_j P_ji.  Baroque's real wheel and this one
therefore differ in NOTHING except circulation, which is the comparison the
figure needs to make.  Measured: sigma 0.302030 -> 0.00000000, max |J| 8.82e-03
-> 2.78e-17 (float round-off), same 60 states, pi unchanged to 1.1e-03, with 254
transposition-invariant terms still ENTERING the sum.

WHY THAT LAST NUMBER MATTERS.  A one-way circle-of-fifths wheel ALSO prints
sigma = 0.0000, but for the opposite reason: every reverse count is zero, so
log(N_fwd/0) diverges, all terms are SKIPPED, and the estimator returns an empty
sum (0 terms) on the most directed configuration possible (J_tot = 1.000, its
maximum; add-one-regularised sigma = 2.74, 9.1x Baroque).  Here all 254 terms are
evaluated and each is individually zero.  Same printed value, opposite physics.
See dat/null_models.json.

DRAWING.  With J == 0 there is no arrow to draw and an empty wheel would read as
a rendering failure, so render(neutral_edges=True) draws the UNDIRECTED support
instead: same edge set, above the same 75th-percentile cut, no arrowheads, one
neutral grey, one uniform width.  Connections exist; circulation does not.

Output: ../fig/panels/fig2c_null_network.pdf (+ .png), sized to ONE cell of
the 2x3 grid in panel_2c_networks.py.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig2c_networks as P2C
from irreversibility_v2 import count_bigrams, stationary

OUT = P2C.OUT


def _sym_field(seqs):
    """Baroque's own counts, symmetrized. Returns (labels, pi, J, sigma, S) where
    J is identically zero and S is the undirected flow support pi_i P_ij + pi_j P_ji
    (used for node ranking and for the grey edges, since |J| is all zeros)."""
    C, alpha = count_bigrams(seqs)
    act = (C.sum(0) + C.sum(1)) > 0
    idx = np.where(act)[0]
    labels = [alpha[i] for i in idx]
    Cs = C[np.ix_(idx, idx)].astype(float)
    Cs = 0.5 * (Cs + Cs.T)                      # <- the null: forward/reverse averaged
    rs = Cs.sum(1, keepdims=True); rs[rs == 0] = 1
    P = Cs / rs
    pi = stationary(P)
    F = pi[:, None] * P
    J = F - F.T                                 # identically zero, up to float eps
    S = F + F.T
    return labels, pi, J, 0.0, S


if __name__ == "__main__":
    P = P2C._pools()
    baroque = P[("Baroque", "KernScores")]

    # ---- audit: verify the null actually is one, don't assert it ----
    lb, pib, Jb, sb = P2C._net_field(baroque)
    ln, pin, Jn, sn, Sn = _sym_field(baroque)
    print(f"Baroque real : sigma {P2C._sigma_invariant(baroque):.6f}  "
          f"max|J| {np.abs(Jb).max():.3e}  states {len(lb)}")
    print(f"symmetrized  : sigma {sn:.8f}  max|J| {np.abs(Jn).max():.3e}  states {len(ln)}")
    print(f"vocabulary identical: {lb == ln}   "
          f"pi max abs diff: {np.abs(pib - pin).max():.3e}")

    # Neutral grey title: the era colour would imply this is the Baroque MEASUREMENT.
    # 15 chars at 5.6pt, and no n= (a null has no sample size to report).
    P2C.render([("null", baroque, "#6b7075")],
               os.path.join(OUT, "fig2c_null_network"),
               field_fn=_sym_field,
               title_fn=lambda name, seqs: "symmetrized null\n" r"$\sigma$ = 0",
               neutral_edges=True)
    print("wrote fig2c_null_network")
