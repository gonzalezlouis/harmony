"""Corrected irreversibility estimators.

Replaces the parts of irreversibility.py that were measuring the estimator
rather than the music. What changed and why:

1. epr() with EPS_PSEUDOCOUNT
   An edge seen forward but never backward gets
       A_ij = log(pi_i P_ij + eps) - log(0 + eps) ~= log(pi_i P_ij) + log(1/eps)
   With eps=1e-6 that is +13.8 nats of pure pseudocount. 94-98% of the old
   per-piece sigma came from such edges, so sigma scaled with log(1/eps) and
   its units were arbitrary.
   -> use sigma_bidirectional(): sum only over edges observed in BOTH
      directions. Pooled, this drops <1% of the probability flow.

2. _markov_order_epr()
   Builds super-states (X_t..X_{t+k-1}) and computes the EPR of the shift.
   But the shift's reverse (b,c)->(a,b) EXISTS only when c==a, so most
   super-edges are one-way BY CONSTRUCTION. The estimator was measuring the
   irreversibility of its own scaffolding, and eps inflated it: I(2)=4.3 with
   a shuffled null of 4.0.
   -> use sigma_word(): the actual definition of irreversibility for a
      symbolic sequence,  sigma_k = sum_w p(w) log[p(w)/p(w~)],  w~ = reversed.
      Well posed at every order; rises and saturates past the correlation
      length.

3. window_curve()
   Coarse-grains by "longest chord in the window wins", which discards the
   rest of the window and degenerates on ties. A shuffled corpus reproduces
   85-95% of the published I(w) rise.
   -> not used. saturation_curve() below uses word order, not time windows.

4. fundamental_cycles_top()
   Sums over SIMPLE cycles, which share edges, so the same current is charged
   to many cycles. Overcounts sigma by 9.7x (Baroque) to 15.5x (Romantic).
   -> use flux_peel(): Hill/Kalpazidou peeling. Assigns each unit of current
      to exactly one cycle; contributions sum to sigma exactly.

Everything here is eps-free. Nothing has a tunable knob.
"""
from __future__ import annotations

import numpy as np
from collections import Counter


# ==================================================================
# Alphabet reducers
# ==================================================================
# The 61-state space has 61^k order-k words. At k>=4 that outruns the corpus,
# so the saturation plateau is only visible on a coarser alphabet. Use the
# full alphabet for sigma, a coarse one for the saturation curve.

def to_full(s):
    return s


def to_root(s):
    """'C#:dom7' -> 'C#'.  13 symbols."""
    return "REST" if s == "REST" else s.split(":")[0]


def to_quality(s):
    """'C#:dom7' -> 'dom7'.  6 symbols."""
    return "REST" if s == "REST" else s.split(":")[1]


def drop_repeats(seq):
    """Collapse held chords: ['C','C','G'] -> ['C','G']."""
    return [x for i, x in enumerate(seq) if i == 0 or seq[i] != seq[i - 1]]


def recode(seq, fn):
    """Map to a coarser alphabet, then re-collapse repeats it creates."""
    return drop_repeats([fn(x) for x in seq])


# ==================================================================
# 1.  Bigram sigma, eps-free
# ==================================================================
def stationary(P, n_iter=5000):
    """Stationary distribution via the dominant left eigenvector."""
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1.0))])
    if pi.sum() < 0:
        pi = -pi
    pi = np.clip(pi, 0.0, None)
    if pi.sum() > 0:
        return pi / pi.sum()
    pi = np.ones(P.shape[0]) / P.shape[0]
    for _ in range(n_iter):
        pi = pi @ P
    return pi / pi.sum()


def sigma_bidirectional(C):
    """Schnakenberg EPR over edges observed in BOTH directions. No eps.

    Parameters
    ----------
    C : (N,N) transition counts, self-loops already dropped.

    Returns
    -------
    sigma        : float, nats per transition
    frac_dropped : fraction of probability flow on one-way edges (excluded).
                   Pooled over a corpus this is <1%; per piece it is large,
                   which is exactly why per-piece sigma was unusable.
    """
    act = (C.sum(0) + C.sum(1)) > 0
    if act.sum() < 2:
        return np.nan, np.nan
    Cs = C[np.ix_(act, act)].astype(float)
    rsum = Cs.sum(1, keepdims=True)
    rsum[rsum == 0] = 1.0
    P = Cs / rsum
    pi = stationary(P)
    flow = pi[:, None] * P

    both = (Cs > 0) & (Cs.T > 0)
    J = flow - flow.T
    with np.errstate(divide="ignore", invalid="ignore"):
        logf = np.log(np.where(flow > 0, flow, 1.0))
        A = np.where(both, logf - logf.T, 0.0)
    sigma = 0.5 * np.sum(np.where(both, J * A, 0.0))

    oneway = (Cs > 0) & (Cs.T == 0)
    frac = flow[oneway].sum() / flow.sum() if flow.sum() > 0 else np.nan
    return float(sigma), float(frac)


def sigma_dirichlet(C, a=0.5):
    """Robustness check: Dirichlet posterior instead of a pseudocount.

    P_ij = (C_ij + a) / (N_i + a*K).  a=0.5 Jeffreys, a=1 Laplace.
    A principled prior, not a tunable eps. Should agree with
    sigma_bidirectional to within ~15%.
    """
    act = (C.sum(0) + C.sum(1)) > 0
    if act.sum() < 2:
        return np.nan
    Cs = C[np.ix_(act, act)].astype(float)
    K = Cs.shape[0]
    P = (Cs + a) / (Cs.sum(1, keepdims=True) + a * K)
    pi = stationary(P)
    flow = pi[:, None] * P
    J = flow - flow.T
    A = np.log(flow) - np.log(flow.T)
    np.fill_diagonal(A, 0.0)
    return float(0.5 * np.sum(J * A))


def count_bigrams(seqs, alphabet=None):
    """Pooled bigram counts. No cross-piece transitions. Self-loops dropped."""
    if alphabet is None:
        alphabet = sorted({x for s in seqs for x in s})
    idx = {s: i for i, s in enumerate(alphabet)}
    C = np.zeros((len(alphabet), len(alphabet)))
    for s in seqs:
        for a, b in zip(s[:-1], s[1:]):
            if a != b and a in idx and b in idx:
                C[idx[a], idx[b]] += 1
    return C, alphabet


# ==================================================================
# 2.  Multi-order sigma: word-reversal KL  (this is the collapse)
# ==================================================================
def sigma_word(seqs, k):
    """Irreversibility at word order k.

        sigma_k = sum_w p(w) * log[ p(w) / p(w~) ],   w~ = w reversed

    This is the KL divergence rate between the forward process and its time
    reverse, restricted to k-words. At k=2 it reduces to the bigram measure.
    It RISES with k and SATURATES once k passes the correlation length --
    that saturation is the collapse.

    Words whose reversal was never observed are excluded (no eps), and their
    share of probability is returned so you can report where the estimate
    stops being trustworthy. Keep k in the range where frac_dropped < ~5%.

    Returns
    -------
    sigma_k, frac_dropped, n_distinct_words
    """
    c = Counter()
    for s in seqs:
        for t in range(len(s) - k + 1):
            c[tuple(s[t:t + k])] += 1
    N = sum(c.values())
    if N == 0:
        return np.nan, np.nan, 0
    sigma, dropped = 0.0, 0.0
    for w, n in c.items():
        p = n / N
        rev = c.get(w[::-1])
        if not rev:
            dropped += p
            continue
        sigma += p * np.log(p / (rev / N))
    return float(sigma), float(dropped), len(c)


def shuffle_seqs(seqs, rng):
    """Null: permute symbols within each piece. Preserves length and
    marginals, destroys temporal structure. True sigma of the result is 0
    at every order, so it measures the estimator's bias."""
    out = []
    for s in seqs:
        a = list(s)
        rng.shuffle(a)
        out.append(a)
    return out


def saturation_curve(seqs, kmax=7, alphabet_fn=to_quality, n_null=3, seed=0):
    """The collapse curve: null-corrected sigma_k for k = 2..kmax.

    Use a coarse alphabet (to_quality, 6 symbols) so the plateau is reachable.
    The full 61-state alphabet only supports k<=3, where you see the rise but
    not the saturation.

    Returns a list of dicts, one per order:
        k, sigma_real, sigma_null, signal, frac_dropped, n_words
    The saturation order (where `signal` stops rising) is the correlation
    length -- the honest replacement for the fitted tau.
    """
    rng = np.random.default_rng(seed)
    coded = [recode(s, alphabet_fn) for s in seqs]
    out = []
    for k in range(2, kmax + 1):
        real, drop, nw = sigma_word(coded, k)
        nulls = [sigma_word(shuffle_seqs(coded, rng), k)[0] for _ in range(n_null)]
        null = float(np.mean(nulls))
        out.append({"k": k, "sigma_real": real, "sigma_null": null,
                    "signal": real - null, "frac_dropped": drop,
                    "n_words": nw})
    return out


# ==================================================================
# 3.  Additive cycle decomposition (flux peeling)
# ==================================================================
def flux_peel(C, labels=None, tol=1e-12):
    """Hill/Kalpazidou peeling: split sigma into cycles that ADD UP.

    Each pass finds a cycle in the positive net-current field, assigns it its
    bottleneck flux, records flux * (affinity around the loop), and subtracts
    that flux from every edge in the loop -- so the bottleneck edge is used up
    and cannot be charged twice. Contributions sum to sigma exactly.

    (fundamental_cycles_top enumerates SIMPLE cycles, which share edges, and
    overcounts by 9.7x-15.5x.)

    Returns (cycles, sigma) where each cycle is a dict:
        states (labels), flux, affinity, contribution, share
    """
    import networkx as nx

    act = (C.sum(0) + C.sum(1)) > 0
    idx_map = np.where(act)[0]
    Cs = C[np.ix_(act, act)].astype(float)
    if labels is not None:
        labels = [labels[i] for i in idx_map]
    rsum = Cs.sum(1, keepdims=True)
    rsum[rsum == 0] = 1.0
    P = Cs / rsum
    pi = stationary(P)
    flow = pi[:, None] * P
    J = flow - flow.T
    both = (Cs > 0) & (Cs.T > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        logf = np.log(np.where(flow > 0, flow, 1.0))
        A = np.where(both, logf - logf.T, 0.0)
    sigma = 0.5 * np.sum(np.where(both, J * A, 0.0))

    # peel only where affinity is defined (bidirectional edges)
    Jw = np.where(both, J, 0.0).copy()
    cycles = []
    while True:
        pos = np.argwhere(Jw > tol)
        if len(pos) == 0:
            break
        G = nx.DiGraph()
        G.add_edges_from([(int(i), int(j)) for i, j in pos])
        try:
            cyc = nx.find_cycle(G, orientation="original")
        except nx.NetworkXNoCycle:
            break
        edges = [(u, v) for u, v, *_ in cyc]
        flux = min(Jw[u, v] for u, v in edges)
        aff = sum(A[u, v] for u, v in edges)
        nodes = [u for u, _ in edges]
        cycles.append({
            "states": [labels[n] for n in nodes] if labels else nodes,
            "flux": float(flux), "affinity": float(aff),
            "contribution": float(flux * aff),
            "share": float(flux * aff / sigma) if sigma else np.nan,
        })
        for u, v in edges:
            Jw[u, v] -= flux
    cycles.sort(key=lambda d: -d["contribution"])
    return cycles, float(sigma)
