#!/usr/bin/env python3
"""Compute pooled sigma for every corpus

compute_final.py - regenerate data/final_data.json for the final figure panels.

Separates the (slow, stochastic) estimator run from the (fast, deterministic)
drawing in panels/*.py.  Run this once; then each panel script just reads the JSON.

Canonical decisions baked in here:
  * estimator = sigma_bidirectional (one-way edges excluded; NO pseudocount).
    The parquet I_inf/I_w/tau columns are pseudocount-inflated ~20-50x -- never
    used for reported magnitudes.
  * classical trajectory = KernScores SCORE notation only (one consistent
    music21-chordify extraction across all eras).  Performance MIDI (GiantMIDI,
    MAESTRO, ASAP) is EXCLUDED from the trajectory: chordify + sustain pedal
    suppresses sigma to ~0.32x of the score value, and the perf/score mix ratio
    worsens over time, which manufactured a spurious "plateau descent".
  * modern magnitude anchors = EXPERT annotations (POP909, Jazz iRb, CoCoPops).
  * Chordonomicon = crowd/triad-flattened -> RELATIVE-only, its own panel.
  * 'Various (Polish)' bucket dropped (early-Baroque mislabeled Romantic).
  * dedup by piece basename (the sequence cache double-stores POP909).
"""
import os, sys, json, glob, zipfile, ntpath, posixpath
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
# config / irreversibility_v2 are siblings in src/
sys.path.insert(0, HERE)
import estimator as ha
from irreversibility_v2 import count_bigrams, sigma_bidirectional, drop_repeats

SEQZIP  = os.path.join(REPO, "dat/output/sequences.zip")
RECORDS = os.path.join(REPO, "dat/output/all_records_usable.parquet")
_bn = lambda p: ntpath.basename(p) if "\\" in p else posixpath.basename(p)

def sig(seqs):
    C, _ = count_bigrams(seqs); return sigma_bidirectional(C)[0]

def boot(seqs, B=300, seed=0):
    rng = np.random.default_rng(seed); n = len(seqs)
    v = [sig([seqs[i] for i in rng.integers(0, n, n)]) for _ in range(B)]
    return float(np.mean(v)), float(np.std(v))

def pool_by_era_dataset():
    """{(era,dataset): [seq,...]}, dedup by basename, Polish dropped."""
    recs = pd.read_parquet(RECORDS)
    era_of  = {_bn(r.midi_path): r.era      for r in recs.itertuples()}
    ds_of   = {_bn(r.midi_path): r.dataset  for r in recs.itertuples()}
    comp_of = {_bn(r.midi_path): r.composer for r in recs.itertuples()}
    zf = zipfile.ZipFile(SEQZIP); seen, pools = set(), {}
    for name in zf.namelist():
        if not name.endswith(".json") or name.startswith("__MACOSX"): continue
        try: d = json.loads(zf.read(name))
        except Exception: continue
        k = _bn(d["midi_path"]); era = era_of.get(k); ds = ds_of.get(k)
        if era is None or k in seen or comp_of.get(k) == "Various (Polish)": continue
        seen.add(k)
        s = drop_repeats([e[0] for e in d["sequence"] if e[0] != "REST"])
        if len(s) >= ha.MIN_LEN: pools.setdefault((era, ds), []).append(s)
    return pools

def main():
    pools = pool_by_era_dataset()
    ks = lambda era: pools.get((era, "KernScores"), [])
    D = {}

    # ---- Fig 2a: trajectory (classical=KernScores score; modern=expert) ----
    traj = {}
    for era in ["Renaissance", "Baroque", "Classical", "Romantic", "Late-Romantic"]:
        sq = ks(era)
        if sq:
            _, sd = boot(sq); traj[era] = dict(sigma=sig(sq), sd=sd, n=len(sq), src="KernScores(score)")
    cocopops = ha.load_cocopops()
    pop  = pools[("Modern", "POP909")]
    jazz = pools[("Modern", "Jazz")]
    for nm, sq in [("CoCoPops (Billboard)", cocopops), ("POP909 (Mandopop)", pop), ("Jazz (iRb)", jazz)]:
        _, sd = boot(sq); traj[nm] = dict(sigma=sig(sq), sd=sd, n=len(sq), src="expert")
    D["trajectory"] = traj

    # ---- chordify performance-suppression ratio (Fig 2 caption fact) ----
    ratios = {}
    for era in ["Baroque", "Classical", "Romantic", "Late-Romantic"]:
        perf = sum((pools.get((era, ds), []) for ds in ["GiantMIDI", "MAESTRO", "ASAP"]), [])
        if ks(era) and perf:
            ratios[era] = sig(perf) / sig(ks(era))
    D["chordify_suppression"] = dict(ratios=ratios, mean=float(np.mean(list(ratios.values()))))

    # ---- Fig 2b: Chordonomicon genre breadth (relative-only) ----
    crowd = ha.load_chordonomicon_by_genre(
        ["pop", "rock", "country", "alternative", "punk", "metal",
         "rap", "soul", "jazz", "electronic", "reggae"], sample=1500)
    D["chordonomicon"] = {g: dict(sigma=sig(s), sd=boot(s, B=150, seed=1)[1]) for g, s in crowd.items()}

    # ---- Fig 3: Irish controlled experiment ----
    irish = ha.load_irish_by_mode()
    D["irish"] = {mc: dict(h=sig(d["harmonic"]), m=sig(d["melodic"]),
                           nh=len(d["harmonic"]), nm=len(d["melodic"])) for mc, d in irish.items()}

    # ---- Fig 4a: palindrome on the three modern corpora ----
    chord_all = [s for g in crowd.values() for s in g]
    def palin(seqs, seed=0):
        rng = np.random.default_rng(seed)
        return dict(real=sig(seqs), palindromized=sig([s + s[::-1] for s in seqs]),
                    shuffled=sig([list(rng.permutation(s)) for s in seqs]))
    D["palindrome"] = dict(pop909=palin(pop, 909), jazz=palin(jazz, 1), chordonomicon=palin(chord_all, 2))

    # ---- Fig 4b: engine decomposition (exact edge J*A), incl. Jazz ----
    D["engine"] = {nm: ha.collapse_engines(ha.engine_decomposition(sq)[0]) for nm, sq in
                   [("Baroque", ks("Baroque")), ("Classical", ks("Classical")),
                    ("Romantic", ks("Romantic")), ("POP909", pop), ("Jazz", jazz)]}

    out = os.path.join(REPO, "dat", "final_data.json")
    json.dump(D, open(out, "w"), indent=2, default=float)
    print("wrote", out)
    print("trajectory:", {k: round(v["sigma"], 3) for k, v in D["trajectory"].items()})

if __name__ == "__main__":
    main()
