import os, re, glob, sys
import numpy as np
import pandas as pd
from collections import Counter

HERE      = os.path.dirname(os.path.abspath(__file__))
HARMONY   = os.path.dirname(HERE)
DATA      = os.path.join(HARMONY, "dat")
HARM_EPR  = HERE
POP909    = os.path.join(HARMONY, "dat/input/pop909")
SEQUENCES_ZIP = os.path.join(HARMONY, "dat/output/sequences.zip")
ALL_RECORDS   = os.path.join(HARMONY, "dat/output/all_records_usable.parquet")
IRISH_JSON    = os.path.join(DATA, "irish_tunes.json")
CHORDONOMICON = os.path.join(DATA, "chordonomicon_v2.csv")

if HARM_EPR not in sys.path:
    sys.path.insert(0, HARM_EPR)

from config import PITCH_NAMES
from irreversibility_v2 import (
    count_bigrams, sigma_bidirectional, stationary, drop_repeats)

PCIDX = {n: i for i, n in enumerate(PITCH_NAMES)}
MIN_LEN = 8

ERA_ORDER = ["Renaissance", "Baroque", "Classical",
             "Romantic", "Late-Romantic", "Modern"]
ERA_COL = {
    "Renaissance":   "#2D6E5E",
    "Baroque":       "#1F5582",
    "Classical":     "#3D8AAF",
    "Romantic":      "#C8860A",
    "Late-Romantic": "#C75146",
    "Modern":        "#6A3D8A",
}
QUAL_COL = {
    "major": "#0077BB", "minor": "#009988", "dom7": "#CC3311", "dim": "#7B4F9E",
}
ENGINE_ORDER = ["V7\u2192I cadence", "ii\u2192V cycle", "other desc-5th", "unclassified"]
ENGINE_COL   = [QUAL_COL["major"], QUAL_COL["dom7"], "#C8860A", "#c9ccd1"]

_NOTE = {'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'Fb':4,'F':5,'E#':5,
         'F#':6,'Gb':6,'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11,'Cb':11}
_NOTE_CH = dict(_NOTE, Cs=1, Ds=3, Es=5, Fs=6, Gs=8, As=10, Bs=0)

_POP909_Q = {
    "maj":"maj","maj6":"maj","maj7":"maj","maj9":"maj","maj(9)":"maj","maj6/9":"maj",
    "sus2":"maj","sus4":"maj","1":"maj",
    "min":"min","min6":"min","min7":"min","min9":"min","min(9)":"min","minmaj7":"min","hdim7":"min",
    "dom7":"dom7","7":"dom7","9":"dom7","11":"dom7","13":"dom7","sus4(b7)":"dom7",
    "dim":"dim","dim7":"dim","hdim":"dim","aug":"aug"}

def _quality_from_suffix(suffix):
    s = suffix.lower()
    if "dim" in s:                              return "dim"
    if "aug" in s or s == "+":                  return "aug"
    if s.startswith("maj7") or s.startswith("maj9"): return "maj"
    if s.startswith("min") or (s.startswith("m") and not s.startswith("maj")): return "min"
    if s.startswith(("7", "9", "11", "13")):    return "dom7"
    return "maj"

def parse_pop909_label(label):
    label = label.strip()
    if label in ("N", "X", "") or ":" not in label:
        return "REST"
    root, rest = label.split(":", 1)
    pc, q = _NOTE.get(root), _POP909_Q.get(rest.split("/")[0])
    return f"{PITCH_NAMES[pc]}:{q}" if (pc is not None and q) else "REST"

def parse_chord_symbol(sym, note_map=_NOTE_CH):
    sym = sym.split("/")[0]
    m = re.match(r"^([A-G][sb#]?)(.*)$", sym)
    if not m:
        return None
    pc = note_map.get(m.group(1))
    if pc is None:
        return None
    return f"{PITCH_NAMES[pc]}:{_quality_from_suffix(m.group(2))}"

def _basename(p):
    import ntpath, posixpath
    return ntpath.basename(p) if "\\" in p else posixpath.basename(p)

def load_all_datasets(dedup=True, drop_polish=True):
    import zipfile, json
    recs = pd.read_parquet(ALL_RECORDS)
    era_of  = {_basename(r.midi_path): r.era      for r in recs.itertuples()}
    comp_of = {_basename(r.midi_path): r.composer for r in recs.itertuples()}
    zf = zipfile.ZipFile(SEQUENCES_ZIP)
    seen, pools = set(), {}
    for name in zf.namelist():
        if not name.endswith(".json") or name.startswith("__MACOSX"):
            continue
        try:
            d = json.loads(zf.read(name))
        except Exception:
            continue
        key = _basename(d["midi_path"])
        era = era_of.get(key)
        if era is None:
            continue
        if drop_polish and comp_of.get(key) == "Various (Polish)":
            continue
        if dedup and key in seen:
            continue
        seen.add(key)
        s = drop_repeats([e[0] for e in d["sequence"] if e[0] != "REST"])
        if len(s) >= MIN_LEN:
            pools.setdefault(era, []).append(s)
    return pools

_ABC_MEL = {"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}

def _abc_chords(abc):
    syms = re.findall(r'"([^"]+)"', abc)
    return drop_repeats([x for x in (parse_chord_symbol(s, _NOTE) for s in syms) if x])

def _abc_melody(abc):
    body = re.sub(r'"[^"]*"', "", abc)
    body = re.sub(r"[^A-Ga-g]", " ", body)
    return drop_repeats([str(_ABC_MEL[c.upper()]) for c in body if c.upper() in _ABC_MEL])

def _irish_mode_class(mode):
    m = mode.lower()
    if "major" in m:                          return "Major (functional)"
    if "dorian" in m or "mixolydian" in m:    return "Dorian/Mixolydian (modal)"
    if "minor" in m:                          return "Minor"
    return "other"

def load_irish_by_mode():
    import json
    tunes = [t for t in json.load(open(IRISH_JSON)) if '"' in t.get("abc", "")]
    out = {}
    for t in tunes:
        mc = _irish_mode_class(t["mode"])
        if mc == "other":
            continue
        h, m = _abc_chords(t["abc"]), _abc_melody(t["abc"])
        d = out.setdefault(mc, {"harmonic": [], "melodic": []})
        if len(h) >= MIN_LEN: d["harmonic"].append(h)
        if len(m) >= MIN_LEN: d["melodic"].append(m)
    return out

COCOPOPS = os.path.join(DATA, "cocopops_billboard")
_HNOTE = {'C':0,'C#':1,'D-':1,'D':2,'D#':3,'E-':3,'E':4,'F-':4,'F':5,'E#':5,
          'F#':6,'G-':6,'G':7,'G#':8,'A-':8,'A':9,'A#':10,'B-':10,'B':11,'C-':11}

def _harte_to_state(tok):
    tok = tok.strip()
    if tok in (".", "N", "", "*") or ":" not in tok:
        return None
    root, qual = tok.split(":", 1)
    pc = _HNOTE.get(root)
    if pc is None:
        return None
    q = qual.lower()
    if "dim" in q or "hdim" in q:                 Q = "dim"
    elif "aug" in q:                              Q = "aug"
    elif q.startswith("maj"):                     Q = "maj"
    elif q.startswith("min"):                     Q = "min"
    elif q.startswith(("7", "9", "11", "13")):    Q = "dom7"
    else:                                         Q = "maj"
    return f"{PITCH_NAMES[pc]}:{Q}"

def load_cocopops():
    out = []
    for path in sorted(glob.glob(os.path.join(COCOPOPS, "*.hum"))):
        lines = [l.rstrip("\n") for l in open(path, encoding="utf-8", errors="ignore")]
        hidx = None
        for l in lines:
            if l.startswith("**"):
                cols = l.split("\t")
                if "**harte" in cols:
                    hidx = cols.index("**harte"); break
        if hidx is None:
            continue
        seq = []
        for l in lines:
            if not l or l[0] in "*!=":
                continue
            cols = l.split("\t")
            if len(cols) > hidx:
                st = _harte_to_state(cols[hidx])
                if st:
                    seq.append(st)
        seq = drop_repeats(seq)
        if len(seq) >= MIN_LEN:
            out.append(seq)
    return out

_TAG = re.compile(r"<[^>]+>")

def _parse_chordonomicon_row(chord_field):
    toks = _TAG.sub(" ", chord_field).split()
    return drop_repeats([x for x in (parse_chord_symbol(t) for t in toks) if x])

def load_chordonomicon_by_genre(genres, sample=1500, seed=0):
    df = pd.read_csv(CHORDONOMICON, engine="python", quotechar='"', on_bad_lines="skip")
    rng = np.random.default_rng(seed)
    out = {}
    for g in genres:
        seqs = [_parse_chordonomicon_row(c) for c in df.loc[df.main_genre == g, "chords"]]
        seqs = [s for s in seqs if len(s) >= MIN_LEN]
        if len(seqs) > sample:
            seqs = [seqs[i] for i in rng.choice(len(seqs), sample, replace=False)]
        if len(seqs) >= 50:
            out[g] = seqs
    return out

def edge_field(seqs):
    C, alphabet = count_bigrams(seqs)
    act = (C.sum(0) + C.sum(1)) > 0
    labels = [alphabet[i] for i in np.where(act)[0]]
    Cs = C[np.ix_(act, act)].astype(float)
    rsum = Cs.sum(1, keepdims=True); rsum[rsum == 0] = 1.0
    P = Cs / rsum
    pi = stationary(P)
    flow = pi[:, None] * P
    both = (Cs > 0) & (Cs.T > 0)
    J = flow - flow.T
    with np.errstate(divide="ignore", invalid="ignore"):
        logf = np.log(np.where(flow > 0, flow, 1.0))
        A = np.where(both, logf - logf.T, 0.0)
    sigma = 0.5 * float(np.sum(np.where(both, J * A, 0.0)))
    return dict(labels=labels, J=J, A=A, both=both, sigma=sigma)

def positive_edges(field):
    J, A, both = field["J"], field["A"], field["both"]
    N = J.shape[0]
    for i in range(N):
        for j in range(N):
            if both[i, j] and J[i, j] > 0:
                yield i, j, J[i, j] * A[i, j]

def _root_quality(state):
    if state == "REST" or ":" not in state:
        return None, None
    r, q = state.split(":")
    return PCIDX.get(r, _NOTE.get(r)), q

def classify_edge(src, dst):
    pa, qa = _root_quality(src)
    pb, qb = _root_quality(dst)
    if pa is None or pb is None:
        return "other"
    step = (pb - pa) % 12
    if pa == pb and qb == "dom7" and qa in ("maj", "min"):
        return "V7-build (same-root)"
    if step == 5:
        if qa == "dom7":          return "dominant resolution (V7->I)"
        if qa in ("min", "min7"): return "ii->V (desc-5th)"
        return "desc-5th (other)"
    if step == 7 and qa == "maj":
        return "plagal/other (asc-5th)"
    return "other"

def engine_decomposition(seqs):
    field = edge_field(seqs)
    labels = field["labels"]
    eng = Counter()
    for i, j, contrib in positive_edges(field):
        eng[classify_edge(labels[i], labels[j])] += contrib
    return dict(eng), field["sigma"]

def collapse_engines(eng):
    return {
        "V7\u2192I cadence": eng.get("dominant resolution (V7->I)", 0) + eng.get("V7-build (same-root)", 0),
        "ii\u2192V cycle":   eng.get("ii->V (desc-5th)", 0),
        "other desc-5th":    eng.get("desc-5th (other)", 0) + eng.get("plagal/other (asc-5th)", 0),
        "unclassified":      eng.get("other", 0),
    }

def edge_concentration(seqs):
    contribs = np.array(sorted((c for *_, c in positive_edges(edge_field(seqs))),
                               reverse=True))
    total = contribs.sum()
    cum = np.cumsum(contribs) / total
    n50 = int(np.searchsorted(cum, 0.5)) + 1
    top5 = float(cum[4]) if len(cum) > 4 else 1.0
    return dict(cum=cum, n50=n50, top5=top5, n_edges=len(contribs), sigma=float(total))

def descending_fifth_share(seqs):
    field = edge_field(seqs)
    labels = field["labels"]
    sig_d5 = sig_all = n_d5 = n_all = 0.0
    for i, j, contrib in positive_edges(field):
        is_d5 = (_root_quality(labels[j])[0] - _root_quality(labels[i])[0]) % 12 == 5
        sig_all += contrib; n_all += 1
        if is_d5:
            sig_d5 += contrib; n_d5 += 1
    sigma_share = sig_d5 / sig_all if sig_all else np.nan
    edge_share  = n_d5 / n_all if n_all else np.nan
    return dict(sigma_share=sigma_share, edge_share=edge_share,
                enrichment=sigma_share / edge_share if edge_share else np.nan,
                sigma=field["sigma"])

def sigma_pooled(seqs):
    C, _ = count_bigrams(seqs)
    return sigma_bidirectional(C)[0]

def palindrome_test(seqs, seed=0):
    rng = np.random.default_rng(seed)
    return dict(
        real=sigma_pooled(seqs),
        reversed=sigma_pooled([s[::-1] for s in seqs]),
        palindromized=sigma_pooled([s + s[::-1] for s in seqs]),
        shuffled=sigma_pooled([list(rng.permutation(s)) for s in seqs]),
    )

def boot_sigma_ci(seqs, B=300, seed=0):
    rng = np.random.default_rng(seed); n = len(seqs); vals = []
    for _ in range(B):
        samp = [seqs[i] for i in rng.integers(0, n, n)]
        vals.append(sigma_pooled(samp))
    v = np.array(vals, float)
    return (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)),
            float(np.nanstd(v)))

def triad_flatten(seqs):
    out = []
    for s in seqs:
        fs = []
        for st in s:
            if ":" not in st:
                fs.append(st); continue
            r, q = st.split(":")
            q = {"dom7": "maj", "dim": "min", "aug": "maj"}.get(q, q)
            fs.append(f"{r}:{q}")
        out.append(drop_repeats(fs))
    return out

def flattening_suppression(reference_pools):
    out = {}
    for name, seqs in reference_pools.items():
        full = sigma_pooled(seqs)
        flat = sigma_pooled(triad_flatten(seqs))
        out[name] = flat / full if full else np.nan
    return out

def multi_order_sigma(seqs, orders=(2, 3, 4), alphabet_fn=None, n_null=3, seed=0):
    from irreversibility_v2 import sigma_word, to_root, recode, shuffle_seqs
    if alphabet_fn is None:
        alphabet_fn = to_root
    coded = [recode(s, alphabet_fn) for s in seqs]
    rng = np.random.default_rng(seed)
    out = []
    for k in orders:
        real = sigma_word(coded, k)[0]
        null = float(np.mean([sigma_word(shuffle_seqs(coded, rng), k)[0] for _ in range(n_null)]))
        out.append(float(real - null))
    return out
