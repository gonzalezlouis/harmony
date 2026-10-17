import os, sys, zipfile, json, ntpath, posixpath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import estimator as ha
import _figure_style as _C
from irreversibility_v2 import count_bigrams, sigma_bidirectional, stationary, drop_repeats

OUT = os.path.join(REPO, "fig", "panels"); os.makedirs(OUT, exist_ok=True)
_C.style()

DISPLAY = {"POP909": "Mandopop", "CoCoPops": "Billboard pop",
           "Ragtime": "Ragtime"}
QUAL_FILL = {"maj": "#0077BB", "min": "#009988", "dom7": "#CC3311", "dim": "#7B4F9E", "aug": "#E070B0"}
FIFTHS = ['C', 'G', 'D', 'A', 'E', 'B', r'F$\sharp$', r'C$\sharp$',
          r'G$\sharp$', r'D$\sharp$', r'A$\sharp$', 'F']
_PC2FIFTH = {'C': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4, 'B': 5, 'F#': 6, 'C#': 7,
             'G#': 8, 'D#': 9, 'A#': 10, 'F': 11}
QUAL_RADIUS = {"maj": 1.42, "dom7": 1.22, "min": 1.02, "dim": 0.82, "aug": 0.62}

def _xy(state):
    if state == "REST" or ":" not in state: return None
    note, q = state.split(":")
    fi = _PC2FIFTH.get(note)
    if fi is None: return None
    ang = np.pi/2 - fi * 2*np.pi/12
    ang += {"maj": 0.0, "dom7": 0.05, "min": -0.05, "dim": 0.07, "aug": -0.07}.get(q, 0.0)
    r = QUAL_RADIUS.get(q, 0.62)
    return np.array([r*np.cos(ang), r*np.sin(ang)]), q


def _check_nodes(ax, pos, sz, name=""):
    ks = list(pos)
    xy = ax.transData.transform(np.array([pos[k] for k in ks]))
    rr = np.array([np.sqrt(sz[k] / np.pi) for k in ks]) * ax.figure.dpi / 72.0
    bad = []
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            d = float(np.hypot(*(xy[i] - xy[j])))
            if d < max(rr[i], rr[j]):
                bad.append((ks[i], ks[j], round(d, 1), round(max(rr[i], rr[j]), 1)))
    if bad:
        raise AssertionError("%s: %d overlapping node pair(s), e.g. %s"
                             % (name, len(bad), bad[:4]))
    return len(ks)

def _short(state):
    root, q = state.split(":")
    suf = {"min": "m", "dom7": "7", "dim": r"$\circ$", "aug": "+", "maj": ""}.get(q, "")
    return root + suf

_INV = json.load(open(os.path.join(os.path.dirname(HERE) if "HERE" in dir() else
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dat", "mechanism.json")))["mechanism"]
INV_KEY = {"Renaissance": "Renaissance", "Baroque": "Baroque", "Classical": "Classical",
           "Romantic": "Romantic", "CoCoPops": "CoCoPops", "POP909": "POP909",
           "Jazz": "Jazz"}

_PITCH = {n: i for i, n in enumerate(
    ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}

def _sigma_invariant(seqs):
    from collections import Counter
    N = Counter()
    for sq in seqs:
        for a, b in zip(sq[:-1], sq[1:]):
            if ":" not in a or ":" not in b:
                continue
            ra, rb = _PITCH.get(a.split(":")[0]), _PITCH.get(b.split(":")[0])
            if ra is None or rb is None:
                continue
            N[((rb - ra) % 12, a.split(":")[1], b.split(":")[1])] += 1
    tot = sum(N.values())
    if not tot:
        return float("nan")
    out = 0.0
    for t, f in N.items():
        r = N.get(((-t[0]) % 12, t[2], t[1]), 0)
        if f > 0 and r > 0:
            out += np.log(f / r) * (f - r) / tot
    return out / 2.0

def _net_field(seqs):
    C, alpha = count_bigrams(seqs)
    act = (C.sum(0) + C.sum(1)) > 0
    idx = np.where(act)[0]; labels = [alpha[i] for i in idx]
    Cs = C[np.ix_(idx, idx)].astype(float)
    rs = Cs.sum(1, keepdims=True); rs[rs == 0] = 1
    P = Cs / rs; pi = stationary(P)
    J = pi[:, None]*P - (pi[:, None]*P).T
    sigma = sigma_bidirectional(C)[0]
    return labels, pi, J, sigma

def render(subsets, path, field_fn=None, title_fn=None, neutral_edges=False,
           cell_field_fn=None, cell_title_fn=None, cell_neutral=None):
    _spare = None
    field_fn = field_fn or _net_field
    cell_field_fn = cell_field_fn or {}
    cell_title_fn = cell_title_fn or {}
    cell_neutral = cell_neutral or set()
    n = len(subsets)
    if n == 1:
        _W, _H = 5.30/3, 3.62/2
        _AXW, _AXH, _TTL = 1.3207, 1.4748, 0.24
        _b, _l = (_H - _TTL - _AXH), (_W - _AXW)/2
        fig, axes = plt.subplots(1, 1, figsize=(_W, _H))
        fig.subplots_adjust(left=_l/_W, right=(_l+_AXW)/_W,
                            bottom=_b/_H, top=(_b+_AXH)/_H)
    elif n == 2:
        fig, axes = plt.subplots(1, 2, figsize=(5.6, 3.2), gridspec_kw={"wspace": 0.06})
    elif n == 3:
        fig, axes = plt.subplots(2, 2, figsize=(5.20, 3.96),
                                 gridspec_kw={"wspace": 0.05, "hspace": 0.16})
        _spare = axes.flat[3]
    elif n == 8:
        fig, axes = plt.subplots(2, 4, figsize=(7.10, 3.62),
                                 gridspec_kw={"hspace": 0.16, "wspace": 0.055,
                                              "bottom": 0.075, "top": 0.955})
    elif n == 4:
        fig, axes = plt.subplots(2, 2, figsize=(3.62, 3.68),
                                 gridspec_kw={"hspace": 0.16, "wspace": 0.055,
                                              "bottom": 0.075, "top": 0.955})
    else:
        fig, axes = plt.subplots(2, 3, figsize=(5.30, 3.62),
                                 gridspec_kw={"hspace": 0.16, "wspace": 0.055,
                                              "bottom": 0.075, "top": 0.955})
    for _ci, (ax, (name, seqs, color)) in enumerate(zip(np.atleast_1d(axes).flat, subsets)):
        ax.set_aspect("equal"); ax.set_xlim(-1.70, 1.70); ax.set_ylim(-1.62, 1.74); ax.axis("off")
        _ffn = cell_field_fn.get(_ci, field_fn)
        _neutral = (_ci in cell_neutral) or neutral_edges
        _f = _ffn(seqs)
        labels, pi, J, sigma = _f[:4]
        S = _f[4] if len(_f) > 4 else np.abs(J)
        mass = S.sum(0) + S.sum(1)
        order = np.argsort(mass)[::-1]
        keep = set(order[:max(10, int(np.searchsorted(np.cumsum(mass[order]), 0.85*mass.sum()))+1)])
        for fi in range(12):
            a = np.pi/2 - fi*2*np.pi/12
            ax.plot([0, 1.44*np.cos(a)], [0, 1.44*np.sin(a)], color="#cccccc", lw=0.3, zorder=0)
        for r, ls in [(1.30, "-"), (0.98, "--"), (0.66, "--"), (0.34, ":")]:
            ax.add_patch(plt.Circle((0, 0), r, fill=False, color="#aaaaaa", lw=0.3, ls=ls, zorder=0))
        for fi, lbl in enumerate(FIFTHS):
            a = np.pi/2 - fi*2*np.pi/12
            ax.text(1.60*np.cos(a), 1.60*np.sin(a), lbl, ha="center", va="center", fontsize=4.2, color="#666")
        pos, sz = {}, {}
        for li, st in enumerate(labels):
            xy = _xy(st)
            if xy is None: continue
            pos[li] = xy[0]; sz[li] = 9 + 62*float(np.sqrt(pi[li]/pi.max()))
        if _neutral:
            Sa = S.copy(); np.fill_diagonal(Sa, 0)
            snz = Sa[Sa > 0]
            sthr = np.percentile(snz, 75) if len(snz) else np.inf
            drawn = set()
            for i in list(pos):
                for j in list(pos):
                    if i >= j or i not in keep or j not in keep:
                        continue
                    if Sa[i, j] > sthr and (i, j) not in drawn:
                        drawn.add((i, j))
                        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                                color="#9aa0a6", lw=0.45, alpha=0.85,
                                solid_capstyle="round", zorder=2)
        Ja = np.abs(J.copy()); np.fill_diagonal(Ja, 0)
        nz = Ja[Ja > 0]
        if len(nz) and not _neutral:
            thr, mx = np.percentile(nz, 75), nz.max()
            for i in list(pos):
                for j in list(pos):
                    if i in keep and j in keep and J[i, j] > thr:
                        t = J[i, j]/mx
                        ax.annotate("", xy=pos[j], xytext=pos[i],
                                    arrowprops=dict(arrowstyle="-|>", color=color,
                                                    lw=0.2+0.8*t, alpha=0.35+0.6*t, mutation_scale=5,
                                                    shrinkA=2.5, shrinkB=3, connectionstyle="arc3,rad=0.14"),
                                    zorder=2)
        for li in pos:
            if li not in keep: continue
            st = labels[li]; q = st.split(":")[1]
            ax.scatter(*pos[li], s=sz[li], c=QUAL_FILL.get(q, "#aaa"),
                       edgecolors="#FFFFFF", linewidths=0.45, zorder=4)
            tc = "white" if q in ("maj", "dom7", "aug") else "#1A1A1A"
            ax.text(*pos[li], _short(st), ha="center", va="center", fontsize=2.6,
                    fontweight="bold", color=tc, zorder=5)
        _check_nodes(ax, pos, sz, name=os.path.basename(str(path)))
        _tfn = cell_title_fn.get(_ci, title_fn)
        if _tfn is not None:
            _t = _tfn(name, seqs)
        else:
            _t = (f"{DISPLAY.get(name, name)}\n"
                  fr"$\sigma$ = {_sigma_invariant(seqs):.2f}   (n = {len(seqs):,})")
        ax.set_title(_t, fontweight="bold", color=color, fontsize=5.6, pad=0.5,
                     linespacing=1.2)
    if n > 1:
        leg = [mpatches.Patch(fc=QUAL_FILL[q], ec="#888", lw=0.05, label=l)
               for q, l in [("maj", "major"), ("dom7", "dom7"), ("min", "minor"), ("dim", "dim")]]
        leg.append(mpatches.Patch(fc="none", ec=QUAL_FILL["aug"], lw=0.7, label="aug"))
        if _spare is not None:
            _spare.axis("off")
            _spare.legend(handles=leg, loc="center", ncol=1, fontsize=6.4,
                          frameon=False, labelspacing=0.9, handlelength=1.4,
                          handletextpad=0.7, borderpad=0.0)
        else:
            fig.legend(handles=leg, loc="lower center", ncol=5, fontsize=5.0,
                       frameon=False, bbox_to_anchor=(0.5, -0.005))
        _bb = _C.fixed_aspect_bbox(fig)
        fig.savefig(path+".pdf", bbox_inches=_bb)
        fig.savefig(path+".png", dpi=300, bbox_inches=_bb)
    else:
        with plt.rc_context({"savefig.bbox": None}):
            _bb = _C.fixed_aspect_bbox(fig)
            fig.savefig(path+".pdf", bbox_inches=_bb)
            fig.savefig(path+".png", dpi=300, bbox_inches=_bb)
    plt.close(fig)

def _pools():
    recs = pd.read_parquet(os.path.join(REPO, "dat/output/all_records_usable.parquet"))
    bn = lambda p: ntpath.basename(p) if "\\" in p else posixpath.basename(p)
    _ERA_FIX = {"Late-Romantic": "Ragtime"}
    era_of = {bn(r.midi_path): _ERA_FIX.get(r.era, r.era) for r in recs.itertuples()}
    ds_of = {bn(r.midi_path): r.dataset for r in recs.itertuples()}
    comp_of = {bn(r.midi_path): r.composer for r in recs.itertuples()}
    zf = zipfile.ZipFile(os.path.join(REPO, "dat/output/sequences.zip")); seen, pools = set(), {}
    for name in zf.namelist():
        if not name.endswith(".json") or name.startswith("__MACOSX"): continue
        try: d = json.loads(zf.read(name))
        except Exception: continue
        k = bn(d["midi_path"]); era = era_of.get(k); ds = ds_of.get(k)
        if era is None or k in seen or comp_of.get(k) == "Various (Polish)": continue
        seen.add(k); s = drop_repeats([e[0] for e in d["sequence"] if e[0] != "REST"])
        if len(s) >= 8: pools.setdefault((era, ds), []).append(s)
    return pools

if __name__ == "__main__":
    P = _pools(); cc = ha.load_cocopops()
    subsets = [("Renaissance", P[("Renaissance", "KernScores")], "#3d8b6e"),
               ("Baroque", P[("Baroque", "KernScores")], "#3a65b0"),
               ("Classical", P[("Classical", "KernScores")], "#5baabc"),
               ("Romantic", P[("Romantic", "KernScores")], "#c9981a"),
               ("CoCoPops", cc, "#8e6fc4"),
               ("POP909", P[("Modern", "POP909")], "#6a3d9a"),
               ("Jazz", P[("Modern", "Jazz")], "#4a2c7a")]
    from fig2c_null_network import _sym_field as _null_field
    main6 = [("null", subsets[1][1], "#6b7075"),
             subsets[0], subsets[1],
             subsets[3], subsets[6], subsets[5]]
    render(main6, os.path.join(OUT, "panel_2c_networks"),
           cell_field_fn={0: _null_field},
           cell_title_fn={0: lambda name, seqs: "symmetrized null\n" r"$\sigma$ = 0"},
           cell_neutral={0})
    def _chord(era):
        return P.get((era, "GiantMIDI"), []) + P.get((era, "MAESTRO"), [])
    pipe = []
    for era, col in [("Baroque", "#3a65b0"), ("Classical", "#5baabc"),
                     ("Romantic", "#c9981a"), ("Ragtime", "#c04848")]:
        pipe.append((f"{era}\nscore", P[(era, "KernScores")], col))
    for era, col in [("Baroque", "#3a65b0"), ("Classical", "#5baabc"),
                     ("Romantic", "#c9981a"), ("Ragtime", "#c04848")]:
        pipe.append((f"{era}\nchordified", _chord(era), col))
    render(pipe, os.path.join(OUT, "suppfig_2c_pipeline"))
    print("wrote panel_2c_networks (2x3 chronological) and suppfig_2c_pipeline "
          "(2x4 score vs chordified)")
