import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig2c_networks as P2C
from irreversibility_v2 import count_bigrams, stationary

OUT = P2C.OUT


def _sym_field(seqs):
    C, alpha = count_bigrams(seqs)
    act = (C.sum(0) + C.sum(1)) > 0
    idx = np.where(act)[0]
    labels = [alpha[i] for i in idx]
    Cs = C[np.ix_(idx, idx)].astype(float)
    Cs = 0.5 * (Cs + Cs.T)
    rs = Cs.sum(1, keepdims=True); rs[rs == 0] = 1
    P = Cs / rs
    pi = stationary(P)
    F = pi[:, None] * P
    J = F - F.T
    S = F + F.T
    return labels, pi, J, 0.0, S


if __name__ == "__main__":
    P = P2C._pools()
    baroque = P[("Baroque", "KernScores")]

    lb, pib, Jb, sb = P2C._net_field(baroque)
    ln, pin, Jn, sn, Sn = _sym_field(baroque)
    print(f"Baroque real : sigma {P2C._sigma_invariant(baroque):.6f}  "
          f"max|J| {np.abs(Jb).max():.3e}  states {len(lb)}")
    print(f"symmetrized  : sigma {sn:.8f}  max|J| {np.abs(Jn).max():.3e}  states {len(ln)}")
    print(f"vocabulary identical: {lb == ln}   "
          f"pi max abs diff: {np.abs(pib - pin).max():.3e}")

    P2C.render([("null", baroque, "#6b7075")],
               os.path.join(OUT, "fig2c_null_network"),
               field_fn=_sym_field,
               title_fn=lambda name, seqs: "symmetrized null\n" r"$\sigma$ = 0",
               neutral_edges=True)
    print("wrote fig2c_null_network")
