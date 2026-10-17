import os, sys, json, zipfile, ntpath, posixpath
from collections import Counter

def build(AA):
    sys.path.insert(0, AA); sys.path.insert(0, os.path.join(AA, "panels"))
    import pandas as pd
    import estimator as ha
    ERA_FIX = {"Late-Romantic": "Ragtime"}
    SCORE = ["Renaissance", "Baroque", "Classical", "Romantic", "Ragtime"]
    recs = pd.read_parquet(ha.ALL_RECORDS)
    bn = lambda p: ntpath.basename(p) if "\\" in p else posixpath.basename(p)
    era_of = {bn(r.midi_path): ERA_FIX.get(r.era, r.era) for r in recs.itertuples()}
    ds_of = {bn(r.midi_path): r.dataset for r in recs.itertuples()}
    cp_of = {bn(r.midi_path): r.composer for r in recs.itertuples()}
    pools, seen = {k: [] for k in SCORE}, set()
    zf = zipfile.ZipFile(ha.SEQUENCES_ZIP)
    for name in zf.namelist():
        if not name.endswith(".json") or "__MACOSX" in name or bn(name).startswith("._"):
            continue
        try:
            d = json.loads(zf.read(name).decode())
        except Exception:
            continue
        k = bn(d.get("midi_path", "") or "")
        era, ds = era_of.get(k), ds_of.get(k)
        if era not in pools or ds != "KernScores":
            continue
        if cp_of.get(k) == "Various (Polish)":
            continue
        if (k, ds) in seen:
            continue
        seen.add((k, ds))
        s = ha.drop_repeats([e[0] for e in d["sequence"] if e[0] != "REST"])
        if len(s) >= ha.MIN_LEN:
            pools[era].append(s)
    return pools, ha

if __name__ == "__main__":
    AA = os.environ["AA"]
    pools, ha = build(AA)
    from compute_mechanism import sigma_invariant
    T1N = {"Renaissance": 2036, "Baroque": 1541, "Classical": 375, "Romantic": 210, "Ragtime": 47}
    T1S = {"Renaissance": 0.0354, "Baroque": 0.3020, "Classical": 0.1538,
           "Romantic": 0.1371, "Ragtime": 0.3974}
    print(f"{'corpus':13s} {'n':>6s} {'Table 1 n':>10s} {'inv sigma':>10s} {'Table 1':>9s} {'match':>6s} {'state':>9s}")
    out = {}
    for k in T1N:
        sq = pools[k]
        si, _ = sigma_invariant(sq); ss = ha.sigma_pooled(sq)
        okn = len(sq) == T1N[k]; oks = abs(si - T1S[k]) < 6e-4
        out[k] = dict(n=len(sq), n_expected=T1N[k], invariant=si, state=ss,
                      table1=T1S[k], n_match=bool(okn), sigma_match=bool(oks))
        print(f"{k:13s} {len(sq):6d} {T1N[k]:10d} {si:10.4f} {T1S[k]:9.4f} "
              f"{'yes' if oks else 'NO':>6s} {ss:9.4f}")
    json.dump(out, open(os.environ["OUTJ"], "w"), indent=1)


def build_modern(AA):
    import pandas as pd, estimator as ha
    bn = lambda p: ntpath.basename(p) if "\\" in p else posixpath.basename(p)
    recs = pd.read_parquet(ha.ALL_RECORDS)
    ds_of = {bn(r.midi_path): r.dataset for r in recs.itertuples()}
    pools, seen = {"POP909": [], "Jazz": []}, set()
    zf = zipfile.ZipFile(ha.SEQUENCES_ZIP)
    for name in zf.namelist():
        if not name.endswith(".json") or "__MACOSX" in name or bn(name).startswith("._"):
            continue
        try:
            d = json.loads(zf.read(name).decode())
        except Exception:
            continue
        k = bn(d.get("midi_path", "") or "")
        ds = ds_of.get(k)
        if ds not in pools or k in seen:
            continue
        seen.add(k)
        s = ha.drop_repeats([e[0] for e in d["sequence"] if e[0] != "REST"])
        if len(s) >= ha.MIN_LEN:
            pools[ds].append(s)
    pools["CoCoPops"] = [s for s in ha.load_cocopops() if len(ha.drop_repeats(s)) >= ha.MIN_LEN]
    pools["CoCoPops"] = [ha.drop_repeats(s) for s in pools["CoCoPops"]]
    return pools
