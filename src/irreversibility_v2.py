from __future__ import annotations

import numpy as np
from collections import Counter


def to_full(s):
    return s


def to_root(s):
    return "REST" if s == "REST" else s.split(":")[0]


def to_quality(s):
    return "REST" if s == "REST" else s.split(":")[1]


def drop_repeats(seq):
    return [x for i, x in enumerate(seq) if i == 0 or seq[i] != seq[i - 1]]


def recode(seq, fn):
    return drop_repeats([fn(x) for x in seq])


def stationary(P, n_iter=5000):
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
    if alphabet is None:
        alphabet = sorted({x for s in seqs for x in s})
    idx = {s: i for i, s in enumerate(alphabet)}
    C = np.zeros((len(alphabet), len(alphabet)))
    for s in seqs:
        for a, b in zip(s[:-1], s[1:]):
            if a != b and a in idx and b in idx:
                C[idx[a], idx[b]] += 1
    return C, alphabet


def sigma_word(seqs, k):
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
    out = []
    for s in seqs:
        a = list(s)
        rng.shuffle(a)
        out.append(a)
    return out


def saturation_curve(seqs, kmax=7, alphabet_fn=to_quality, n_null=3, seed=0):
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


def flux_peel(C, labels=None, tol=1e-12):
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
