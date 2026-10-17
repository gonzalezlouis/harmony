#!/usr/bin/env python3
"""Compute the mechanism decomposition on invariant transition types

Freeze the per-piece root-motion arrays for Fig 3c and its 2x2 companion.

Writes data/rootmotion_2x2.{json,npz} and data/corpus_centroids.json.  Nothing here is
drawn; the panels read the frozen files, as everywhere else in src.

VALIDATION THIS SCRIPT PERFORMS ON ITSELF (all three must pass or it raises):

  1. It reproduces data/emergent_full.npz as an EXACT MULTISET of joint
     (i5, arrow, pipeline) triples: same 20,560 pieces, every value identical to 0.0 at
     12 decimals, with each piece's three columns still attached to each other.  Row
     ORDER differs, because both scripts walk zipfile.namelist() and that order is not
     stable across runs -- so a row-by-row comparison is not the available check, and
     a per-column sorted comparison would be too weak (it could pass while columns were
     mismatched between pieces).  Comparing sorted TRIPLES is the strongest test the
     frozen file supports.  That file was frozen by an earlier run of a script no longer
     in the tree, so re-deriving it from the sequence cache is the only way to add
     columns (the other eleven root motions) without trusting a reimplementation.
     Getting there required matching three non-obvious choices:
       - dedup is by SEQUENCE CONTENT, not by basename (basename dedup gives 18,223)
       - the "Various (Polish)" bucket is KEPT here (it is dropped elsewhere for a
         wrong ERA label, which this panel never reads) -- dropping it gives 20,213
       - ASAP is dropped (959 rows collapse to 1 distinct sequence: pipeline bug)

  2. The affinity reference is TRANSPOSITION-INVARIANT typing, t = (root interval
     mod 12, quality_from, quality_to), NOT state-pair typing.  Both were tried: the
     invariant form reproduces emergent_full.npz exactly, while state-pair affinities
     (whether from raw counts or from the stationary flow pi_i P_ij) give pearson
     +0.6309 instead of +0.6402 and an arrow range of -0.442..1.363 instead of
     -0.556..0.983.  Invariant typing is also the locked convention for sigma.

  3. sigma IDENTITY.  Weighting each piece's mean affinity by its step count and
     averaging must return the pooled invariant sigma, because sum_t p(t) A(t) with
     both from the same corpus IS that sum.  Checked to 1e-6; it comes out 1.9e-7.
     This is what licenses calling the y axis sigma_i rather than an unnamed index.

Per-corpus sigma computed here on each corpus's OWN counts reproduces the canonical
eight values exactly (Renaissance 0.0354 ... Jazz 1.3072).  Note that ha.sigma_pooled
does NOT -- it uses state-pair typing and returns 0.0479/1.4061 for those two -- so the
invariant estimator below is the one that matches the paper.
"""
import os, sys, json, zipfile, ntpath, posixpath
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "dat")
sys.path.insert(0, HERE)
# config / irreversibility_v2 are siblings in src/
import estimator as ha

B_BOOT = 1000          # bootstrap draws for correlation CIs
B_CENT = 2000          # bootstrap draws for centroid CIs
BINW = 0.05            # fixed-width bins for the median curve

CLASSES = {"descending fifth": [5], "ascending fifth": [7],
           "step motion": [1, 2, 10, 11], "third motion": [3, 4, 8, 9],
           "tritone": [6], "root retained": [0]}
PLOT_CLASSES = ["descending fifth", "ascending fifth", "step motion", "third motion"]
CANON = {"Renaissance": 0.0354, "Baroque": 0.3020, "Classical": 0.1538,
         "Romantic": 0.1371, "Ragtime": 0.3974, "CoCoPops": 0.1549,
         "POP909": 0.2342, "Jazz": 1.3072}
CORPUS_ORDER = ["Renaissance", "Baroque", "Classical", "Romantic", "Ragtime",
                "CoCoPops", "POP909", "Jazz"]
ERAFIX = {"Late-Romantic": "Ragtime"}


def _bn(p):
    return ntpath.basename(p) if "\\" in p else posixpath.basename(p)


def _rq(state):
    """(pitch-class of root, quality) or (None, None)."""
    if ":" not in state:
        return None, None
    r, q = state.split(":")
    return ha.PCIDX.get(r, ha._NOTE.get(r)), q


def load_pieces():
    """Two views of the same cache, built in one pass over the zip.

    pieces : [(dataset, era, sequence, composer)] -- the 20,560 rows of
             emergent_full.npz.  CONTENT-deduped, ASAP dropped, Polish KEPT.  This is
             the per-piece cloud; it reads no era label so it needs no era exclusions.
    pools  : {(era, dataset): [sequence, ...]} -- BASENAME-deduped, Polish dropped.
             This is panel_2c_networks._pools(), the pool the canonical eight sigma
             values are defined on.  It is a DIFFERENT pool and must be: Baroque has
             1,541 pieces here against 1,017 content-distinct ones, and computing the
             canonical sigma on the content-deduped subset gives 0.2050, not 0.3020.
             Repeated performances of the same score are legitimate independent
             observations for a corpus-level transition count; they are not independent
             POINTS in a per-piece scatter.  Hence two views, one pass.
    """
    recs = pd.read_parquet(os.path.join(REPO, "dat/output/all_records_usable.parquet"))
    era_of = {_bn(r.midi_path): r.era for r in recs.itertuples()}
    ds_of = {_bn(r.midi_path): r.dataset for r in recs.itertuples()}
    comp_of = {_bn(r.midi_path): r.composer for r in recs.itertuples()}
    zf = zipfile.ZipFile(os.path.join(REPO, "dat/output/sequences.zip"))
    seen, seqseen, out, pools = set(), set(), [], {}
    for name in zf.namelist():
        if not name.endswith(".json") or name.startswith("__MACOSX"):
            continue
        try:
            d = json.loads(zf.read(name))
        except Exception:
            continue
        k = _bn(d["midi_path"])
        era, ds = era_of.get(k), ds_of.get(k)
        if era is None or k in seen:
            continue
        seen.add(k)
        s = ha.drop_repeats([e[0] for e in d["sequence"] if e[0] != "REST"])
        if len(s) < 8:
            continue
        if comp_of.get(k) != "Various (Polish)":
            pools.setdefault((ERAFIX.get(era, era), ds), []).append(s)
        if ds == "ASAP":                      # 959 rows -> 1 distinct sequence
            continue
        t = tuple(s)
        if t in seqseen:                      # dedup by CONTENT, not basename
            continue
        seqseen.add(t)
        out.append((ds, era, s, comp_of.get(k)))
    cc = ha.load_cocopops()
    for s in cc:
        out.append(("CoCoPops", "Modern", s, None))
    pools[("Modern", "CoCoPops")] = cc
    return out, pools


def invariant_counts(seqs):
    """Counter over t = (root interval mod 12, quality_from, quality_to)."""
    Ct = Counter()
    for s in seqs:
        for a, b in zip(s, s[1:]):
            pa, qa = _rq(a)
            pb, qb = _rq(b)
            if pa is None or pb is None:
                continue
            Ct[((pb - pa) % 12, qa, qb)] += 1
    return Ct


def _rev(t):
    d, qa, qb = t
    return ((-d) % 12, qb, qa)


def affinities(Ct):
    """A(t) = log(N_fwd/N_rev), SKIPPING any t whose reverse is unobserved."""
    return {t: np.log(f / Ct[_rev(t)]) for t, f in Ct.items() if Ct.get(_rev(t), 0) > 0}


def sigma_invariant(seqs):
    """(sigma, J_tot) on a corpus's OWN counts.  sigma = 0.5 sum_t A(t) J(t)."""
    Ct = invariant_counts(seqs)
    N = sum(Ct.values())
    sig = jtot = 0.0
    for t, f in Ct.items():
        r = Ct.get(_rev(t), 0)
        if r > 0:
            J = (f - r) / N
            sig += 0.5 * np.log(f / r) * J
            if J > 0:
                jtot += J
    return sig, jtot


def per_piece(pieces, A):
    """arrow (mean A over the piece's typed steps), per-interval share, step count."""
    n = len(pieces)
    arrow = np.zeros(n)
    share = np.zeros((n, 12))
    nstep = np.zeros(n)
    for k, (_, _, s, _) in enumerate(pieces):
        vals, cnt, tot = [], np.zeros(12), 0
        for a, b in zip(s, s[1:]):
            pa, qa = _rq(a)
            pb, qb = _rq(b)
            if pa is None or pb is None:
                continue
            d = (pb - pa) % 12
            cnt[d] += 1
            tot += 1
            v = A.get((d, qa, qb))
            if v is not None:
                vals.append(v)
        arrow[k] = np.mean(vals) if vals else 0.0
        share[k] = cnt / tot if tot else 0.0
        nstep[k] = len(vals)
    return arrow, share, nstep


def binstats(x, y, w=BINW, minn=30):
    r = {"x": [], "med": [], "q25": [], "q75": [], "n": []}
    for lo in np.arange(0.0, 1.0 + w, w):
        m = (x >= lo) & (x < lo + w)
        if m.sum() >= minn:
            r["x"].append(float(lo + w / 2))
            r["med"].append(float(np.median(y[m])))
            r["q25"].append(float(np.percentile(y[m], 25)))
            r["q75"].append(float(np.percentile(y[m], 75)))
            r["n"].append(int(m.sum()))
    return r


def partial_r(x, y, z):
    """Pearson r of x,y after linearly residualizing both on z."""
    Zm = np.column_stack([np.ones_like(z), z])
    rx = x - Zm @ np.linalg.lstsq(Zm, x, rcond=None)[0]
    ry = y - Zm @ np.linalg.lstsq(Zm, y, rcond=None)[0]
    return float(pearsonr(rx, ry)[0])


def main():
    pieces, pools = load_pieces()
    seqs = [p[2] for p in pieces]
    A = affinities(invariant_counts(seqs))
    arrow, share, nstep = per_piece(pieces, A)
    n = len(pieces)

    # -- CHECK 1: multiset of joint triples vs the frozen arrays -------------
    Zf = np.load(os.path.join(DATA, "emergent_full.npz"), allow_pickle=True)
    assert len(Zf["arrow"]) == n, "piece count %d != frozen %d" % (n, len(Zf["arrow"]))
    pipe = np.where(np.isin(np.array([p[0] for p in pieces]), ["GiantMIDI", "MAESTRO"]),
                    "chordified", "score / expert")
    mine = sorted(zip(np.round(share[:, 5], 12), np.round(arrow, 12), pipe))
    frz = sorted(zip(np.round(Zf["i5"], 12), np.round(Zf["arrow"], 12),
                     [str(x) for x in Zf["pipe"]]))
    assert mine == frz, "does NOT reproduce emergent_full.npz as a triple multiset"
    # Sorted-column diffs, reported for the record now that the triple test has passed.
    d_arrow = float(np.abs(np.sort(Zf["arrow"]) - np.sort(arrow)).max())
    d_i5 = float(np.abs(np.sort(Zf["i5"]) - np.sort(share[:, 5])).max())
    # And the published correlations, which are order-independent.
    assert abs(pearsonr(share[:, 5], arrow)[0] - 0.6402360231903856) < 1e-12
    assert abs(spearmanr(share[:, 5], arrow)[0] - 0.4797722293911997) < 1e-12

    # -- CHECK 2: sigma identity --------------------------------------------
    sig_pool, _ = sigma_invariant(seqs)
    wmean = float(np.sum(nstep * arrow) / nstep.sum())
    assert abs(wmean - sig_pool) < 1e-6, \
        "step-weighted per-piece mean %.9f != pooled sigma %.9f" % (wmean, sig_pool)

    CS = {k: share[:, v].sum(1) for k, v in CLASSES.items()}
    rng = np.random.default_rng(0)
    idxs = [rng.integers(0, n, n) for _ in range(B_BOOT)]

    out = {"n": n, "classes": PLOT_CLASSES,
           "class_intervals": {k: CLASSES[k] for k in CLASSES},
           "bootstrap": {"B_corr": B_BOOT, "B_centroid": B_CENT, "seed": 0},
           "reproduces_emergent_full": {"triple_multiset_identical": True,
                                        "max_abs_diff_arrow_sorted": d_arrow,
                                        "max_abs_diff_i5_sorted": d_i5},
           "sigma_identity": {"sigma_pool_invariant": sig_pool,
                              "stepweighted_mean_perpiece": wmean,
                              "abs_diff": abs(wmean - sig_pool)},
           "per_interval": {}, "class_corr": {}, "partial_ctrl_fifth": {},
           "bins": {k: binstats(CS[k], arrow) for k in PLOT_CLASSES},
           "extent": {"x_desc5": [float(CS["descending fifth"].min()),
                                  float(CS["descending fifth"].max())],
                      "y": [float(arrow.min()), float(arrow.max())],
                      "x_all_classes_max": float(max(CS[k].max() for k in PLOT_CLASSES))}}

    for d in range(12):
        out["per_interval"][str(d)] = {
            "pearson": float(pearsonr(share[:, d], arrow)[0]),
            "spearman": float(spearmanr(share[:, d], arrow)[0]),
            "mean_share": float(share[:, d].mean())}

    f5 = CS["descending fifth"]
    for k, x in CS.items():
        bs = np.array([pearsonr(x[i], arrow[i])[0] for i in idxs])
        bss = np.array([spearmanr(x[i], arrow[i])[0] for i in idxs])
        out["class_corr"][k] = {
            "pearson": float(pearsonr(x, arrow)[0]),
            "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "spearman": float(spearmanr(x, arrow)[0]),
            "sci": [float(np.percentile(bss, 2.5)), float(np.percentile(bss, 97.5))],
            "mean_share": float(x.mean()), "n": n}
        if k != "descending fifth":
            out["partial_ctrl_fifth"][k] = partial_r(x, arrow, f5)
    # the mirror test: does the fifth survive controlling for each other class?
    out["fifth_partial_ctrl_other"] = {k: partial_r(f5, arrow, CS[k])
                                       for k in CS if k != "descending fifth"}
    # the direction test: fifth share MINUS its retrograde
    out["net_fifth_asymmetry"] = {
        "pearson": float(pearsonr(share[:, 5] - share[:, 7], arrow)[0]),
        "spearman": float(spearmanr(share[:, 5] - share[:, 7], arrow)[0])}

    json.dump(out, open(os.path.join(DATA, "rootmotion_2x2.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(DATA, "rootmotion_2x2.npz"), arrow=arrow,
                        classes=np.array(PLOT_CLASSES),
                        **{("x_%d" % i): CS[k] for i, k in enumerate(PLOT_CLASSES)})

    # -- corpus centroids in the (fifth share, sigma_i) plane ---------------
    ds = np.array([p[0] for p in pieces])
    era = np.array([ERAFIX.get(p[1], p[1]) for p in pieces])
    comp = np.array(["" if p[3] is None else p[3] for p in pieces])
    cent = {}
    for nm in CORPUS_ORDER:
        if nm in ("CoCoPops", "POP909", "Jazz"):
            m = ds == nm
        else:
            # KernScores score notation only, and the mislabeled Polish bucket out --
            # a CENTROID is labelled by era, so here the era label does matter and the
            # exclusions that the per-piece cloud does not need do apply.
            m = (ds == "KernScores") & (era == nm) & (comp != "Various (Polish)")
        nn = int(m.sum())
        x, y, w = CS["descending fifth"][m], arrow[m], nstep[m]
        # STEP-WEIGHTED, not unweighted.  sigma_i is a per-STEP quantity: the pooled
        # sigma of any set of pieces is sum_i (m_i * sigma_i) / sum_i m_i, weighting each
        # piece by its typed-step count m_i.  An unweighted mean of sigma_i is therefore
        # NOT sigma of anything -- over all 20,560 pieces it gives 0.0572 where the
        # step-weighted mean gives 0.049821, which IS the pooled sigma of the whole
        # scatter set (asserted in CHECK 2 above).  The earlier unweighted version made
        # every marker a statistic with no interpretation as an entropy production,
        # and it is what made these markers disagree with Fig 2 by 2-5x.
        # Both coordinates are weighted the same way so the marker is one consistent
        # point: x is the step-weighted fifth share, i.e. the fraction of THIS corpus's
        # typed transitions that are descending fifths.
        _wsum = w.sum()
        bi = [rng.integers(0, nn, nn) for _ in range(B_CENT)]
        bx = np.array([float((x[i] * w[i]).sum() / w[i].sum()) for i in bi])
        by = np.array([float((y[i] * w[i]).sum() / w[i].sum()) for i in bi])
        # sigma on this corpus's OWN counts, from the CANONICAL basename-deduped pool
        # (not the content-deduped scatter subset -- see load_pieces).  Stored beside
        # the centroid so the two are never confused: they are different quantities.
        pk = ("Modern", nm) if nm in ("CoCoPops", "POP909", "Jazz") else (nm, "KernScores")
        own, jtot = sigma_invariant(pools[pk])
        assert abs(own - CANON[nm]) < 5e-4, \
            "%s own-matrix sigma %.4f != canonical %.4f" % (nm, own, CANON[nm])
        cent[nm] = {"n": nn,
                    "fifth": float((x * w).sum() / _wsum),
                    "arrow": float((y * w).sum() / _wsum),
                    "fifth_ci": [float(np.percentile(bx, 2.5)),
                                 float(np.percentile(bx, 97.5))],
                    "arrow_ci": [float(np.percentile(by, 2.5)),
                                 float(np.percentile(by, 97.5))],
                    # kept for reference so the two are never conflated in prose
                    "arrow_unweighted_DEPRECATED": float(y.mean()),
                    "n_typed_steps": float(_wsum),
                    "sigma_own_matrix": float(own),
                    "A_mean_own": float(own / jtot),   # sigma/J_tot, current-weighted
                    "sigma_canonical": CANON[nm]}
    json.dump(cent, open(os.path.join(DATA, "corpus_centroids.json"), "w"), indent=1)

    print("reproduced emergent_full.npz: triple multiset identical, "
          "sorted-column diffs arrow %g i5 %g" % (d_arrow, d_i5))
    print("sigma identity: pooled %.9f  step-weighted per-piece %.9f  diff %.2e"
          % (sig_pool, wmean, abs(wmean - sig_pool)))
    for k in CLASSES:
        c = out["class_corr"][k]
        print("  %-18s r %+.4f [%+.4f,%+.4f]  rho %+.4f  share %.4f"
              % (k, c["pearson"], c["ci"][0], c["ci"][1], c["spearman"], c["mean_share"]))
    for nm in CORPUS_ORDER:
        c = cent[nm]
        print("  %-12s n=%5d  fifth %.4f  sigma_i %+.4f  own sigma %.4f (canon %.4f)"
              % (nm, c["n"], c["fifth"], c["arrow"], c["sigma_own_matrix"],
                 c["sigma_canonical"]))


if __name__ == "__main__":
    main()
