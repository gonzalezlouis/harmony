#!/usr/bin/env python3
"""Figure 2c — corpus networks on the circle of fifths

Fig 2c - chord-transition networks on the circle of fifths (2x3 grid).

Each panel is one corpus's net probability-current field J_ij = pi_i P_ij -
pi_j P_ji, drawn as directed edges on the circle of fifths; node fill = chord
QUALITY (Wong palette), node size proportional to stationary probability pi.
Only |J| above the 75th percentile is drawn, so the dominant circulation reads.

Reconciled with the rest of the figure set:
  * classical corpora = KernScores SCORE notation (Baroque/Classical/Romantic)
  * modern corpora = EXPERT annotations (CoCoPops, POP909, Jazz)
  * title sigma = sigma_bidirectional (honest); NOT the pseudocount-inflated
    parquet I_inf column the original fig6_network.py used.

Rebuilds its pools directly from sequences.zip (dedup, Polish dropped), so it is
self-contained; expect ~1-2 min for the zip read.
"""
import os, sys, zipfile, json, ntpath, posixpath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# config / irreversibility_v2 are siblings in src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # panels/ for _common
# harmony_arrow is a sibling in src/
import estimator as ha
import _figure_style as _C
from irreversibility_v2 import count_bigrams, sigma_bidirectional, stationary, drop_repeats

OUT = os.path.join(REPO, "fig", "panels"); os.makedirs(OUT, exist_ok=True)
_C.style()   # scienceplots + shared fonts/colors (network panels are axis-off, no spines)

DISPLAY = {"POP909": "Mandopop", "CoCoPops": "Billboard pop",
           "Ragtime": "Ragtime"}
QUAL_FILL = {"maj": "#0077BB", "min": "#009988", "dom7": "#CC3311", "dim": "#7B4F9E", "aug": "#E070B0"}
FIFTHS = ['C', 'G', 'D', 'A', 'E', 'B', r'F$\sharp$', r'C$\sharp$',
          r'G$\sharp$', r'D$\sharp$', r'A$\sharp$', 'F']
_PC2FIFTH = {'C': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4, 'B': 5, 'F#': 6, 'C#': 7,
             'G#': 8, 'D#': 9, 'A#': 10, 'F': 11}
# quality rings spread WIDE so same-root different-quality nodes don't overlap
# Rings are spaced over an ANNULUS, not down to the hub.  At r = 0.34 the arc
# between adjacent roots is 2*pi*r/12 = 0.18 data units, which at this scale is
# about one node diameter, and at r = 0.12 it is a third of one -- so the dim and
# aug rings piled adjacent roots on top of each other with illegible labels.
# The innermost ring is now 0.62, where the arc is 0.32 units, and _check_nodes
# below asserts the result rather than trusting it.
QUAL_RADIUS = {"maj": 1.42, "dom7": 1.22, "min": 1.02, "dim": 0.82, "aug": 0.62}

def _xy(state):
    if state == "REST" or ":" not in state: return None
    note, q = state.split(":")
    fi = _PC2FIFTH.get(note)
    if fi is None: return None
    ang = np.pi/2 - fi * 2*np.pi/12
    # small angular nudge per quality so same-root nodes don't stack on one spoke
    # Nudges shrink with the ring gap: at the old amplitudes a dim node was pushed
    # a sixth of a spoke toward its neighbour, which is what closed the gap.
    ang += {"maj": 0.0, "dom7": 0.05, "min": -0.05, "dim": 0.07, "aug": -0.07}.get(q, 0.0)
    r = QUAL_RADIUS.get(q, 0.62)
    return np.array([r*np.cos(ang), r*np.sin(ang)]), q


def _check_nodes(ax, pos, sz, name=""):
    """Assert no two drawn nodes overlap, measured in display coordinates.

    The text-overlap audit cannot see this: the collision is between MARKERS, so a
    wheel whose dim ring piled four roots on one another reported clean.  s is in
    points squared, so the marker radius is sqrt(s/pi) points.
    """
    ks = list(pos)
    xy = ax.transData.transform(np.array([pos[k] for k in ks]))
    rr = np.array([np.sqrt(sz[k] / np.pi) for k in ks]) * ax.figure.dpi / 72.0
    bad = []
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            d = float(np.hypot(*(xy[i] - xy[j])))
            # Criterion: no node may cover another node's CENTRE.  Nodes touching at
            # the rim is normal in this wheel and main Fig. 2c does it too; a node
            # swallowing its neighbour's centre is what made labels illegible.
            # Measured: at the old radii the closest pair was 10.5 px apart with
            # 19.8 px markers and 12 pairs covered a centre; at these radii the
            # closest pair is 35.5 px and no pair covers a centre.
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
    """The paper's canonical estimator: transposition-invariant typing, own-matrix
    affinities. Used for EVERY title so no panel silently reports a different
    quantity (the previous name-lookup fell back to absolute-name sigma)."""
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
    """Draw one wheel per subset.

    The three FIGURE-WIDE hooks exist so panel_2c_null.py can render the sigma = 0
    reference on IDENTICAL geometry (same radii, spokes, node placement, sizes,
    fonts, legend) without duplicating any of this code.  All three default to the
    behaviour the main 2x3 figure has always had, so that figure is unchanged:
      field_fn(seqs)  -> (labels, pi, J, sigma); default _net_field
      title_fn(name, seqs) -> title string; default corpus name + sigma + n
      neutral_edges   -> when the net-current field is identically zero there is
                         nothing directed to draw, so draw the UNDIRECTED support
                         in grey at uniform weight instead: connections exist,
                         circulation does not.

    The three PER-CELL hooks (cell_field_fn, cell_title_fn, cell_neutral) take a
    dict/set keyed by cell INDEX and override the figure-wide value for just those
    cells. They exist because the main 2x3 grid now holds five measured corpora plus
    ONE null cell, so the null's field function, title and neutral-edge treatment
    have to apply to that cell alone rather than to the whole figure.
    """
    _spare = None
    field_fn = field_fn or _net_field
    cell_field_fn = cell_field_fn or {}
    cell_title_fn = cell_title_fn or {}
    cell_neutral = cell_neutral or set()
    n = len(subsets)
    if n == 1:
        # ONE cell of the 2x3 grid: canvas 5.30/3 x 3.62/2 in and the AXES inside it
        # exactly 1.3207 x 1.4748 in, the cell size measured from that figure, so this
        # drops in beside the six without rescaling in Illustrator.
        #
        # The fractions below are derived from those inches, not guessed: the axes is
        # centred horizontally and 0.24 in is reserved at the TOP for the two-line
        # title (5.6pt, linespacing 1.2, pad 0.5). The grid version can leave top=0.955
        # because it is saved with bbox_inches="tight", which expands the canvas to fit
        # overflowing titles; this one is saved untight to keep the cell match, so the
        # title space has to be inside the canvas or it clips.
        _W, _H = 5.30/3, 3.62/2
        _AXW, _AXH, _TTL = 1.3207, 1.4748, 0.24
        _b, _l = (_H - _TTL - _AXH), (_W - _AXW)/2
        fig, axes = plt.subplots(1, 1, figsize=(_W, _H))
        fig.subplots_adjust(left=_l/_W, right=(_l+_AXW)/_W,
                            bottom=_b/_H, top=(_b+_AXH)/_H)
    elif n == 2:
        fig, axes = plt.subplots(1, 2, figsize=(5.6, 3.2), gridspec_kw={"wspace": 0.06})
    elif n == 3:
        # 2x2 with the legend in the fourth cell, NOT a 1x3 row.  Three wheels side by
        # side make a figure of aspect ~2.2, and every supplementary figure is saved at
        # 1.3137, so the row had to be padded with a band of empty canvas above and
        # below amounting to about 40% of the page area.  This arrangement fills it.
        fig, axes = plt.subplots(2, 2, figsize=(5.20, 3.96),
                                 gridspec_kw={"wspace": 0.05, "hspace": 0.16})
        _spare = axes.flat[3]
    elif n == 8:
        # 2x4 pipeline control: same era on each COLUMN, score (top) vs chordified
        # (bottom). Near-square axes so the equal-aspect wheels leave no dead width.
        fig, axes = plt.subplots(2, 4, figsize=(7.10, 3.62),
                                 gridspec_kw={"hspace": 0.16, "wspace": 0.055,
                                              "bottom": 0.075, "top": 0.955})
    elif n == 4:
        # lower-right slot of fig/fig2-complete.ai is 4.39 x 3.57 in, but with
        # aspect="equal" a wider-than-tall axes leaves the circle height-limited and
        # wastes ~40% of each axes' width. Sized so the four axes are near-square: the
        # panel is narrower than the slot and has no interior gaps.
        fig, axes = plt.subplots(2, 2, figsize=(3.62, 3.68),
                                 gridspec_kw={"hspace": 0.16, "wspace": 0.055,
                                              "bottom": 0.075, "top": 0.955})
    else:
        # six panels in the lower-right slot of fig/fig2-complete.ai (4.39 x 3.57 in);
        # near-square axes so the equal-aspect wheels leave no interior whitespace
        fig, axes = plt.subplots(2, 3, figsize=(5.30, 3.62),
                                 gridspec_kw={"hspace": 0.16, "wspace": 0.055,
                                              "bottom": 0.075, "top": 0.955})
    for _ci, (ax, (name, seqs, color)) in enumerate(zip(np.atleast_1d(axes).flat, subsets)):
        ax.set_aspect("equal"); ax.set_xlim(-1.70, 1.70); ax.set_ylim(-1.62, 1.74); ax.axis("off")
        _ffn = cell_field_fn.get(_ci, field_fn)
        _neutral = (_ci in cell_neutral) or neutral_edges
        _f = _ffn(seqs)
        # field_fn may return a 5th element S: the UNDIRECTED flow support, used for
        # node ranking and grey edges when J is identically zero (there is then no
        # net current to rank states by, but the same states are still visited).
        labels, pi, J, sigma = _f[:4]
        S = _f[4] if len(_f) > 4 else np.abs(J)
        # keep states carrying 85% of mass, min 10
        mass = S.sum(0) + S.sum(1)
        order = np.argsort(mass)[::-1]
        keep = set(order[:max(10, int(np.searchsorted(np.cumsum(mass[order]), 0.85*mass.sum()))+1)])
        # guide circles + spokes
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
            # gentler sqrt scaling + smaller cap so adjacent nodes don't collide
            pos[li] = xy[0]; sz[li] = 9 + 62*float(np.sqrt(pi[li]/pi.max()))
        if _neutral:
            # sigma = 0 reference: every net current is zero, so there is no arrow to
            # draw and an empty wheel would read as a rendering failure. Draw the
            # UNDIRECTED support -- same edges, no arrowheads, one neutral grey, one
            # uniform weight -- above the same 75th-percentile cut the directed
            # panels use, so the edge SET is comparable and only the direction is gone.
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
        # two-line title: corpus name (bold, larger) over the sigma / n line
        _tfn = cell_title_fn.get(_ci, title_fn)
        if _tfn is not None:
            _t = _tfn(name, seqs)
        else:
            _t = (f"{DISPLAY.get(name, name)}\n"
                  fr"$\sigma$ = {_sigma_invariant(seqs):.2f}   (n = {len(seqs):,})")
        ax.set_title(_t, fontweight="bold", color=color, fontsize=5.6, pad=0.5,
                     linespacing=1.2)
    # LEGEND, five qualities. FILLED swatch = quality that has at least one node drawn;
    # HOLLOW swatch = quality present in the measurement but with no node above the
    # per-node display threshold. Augmented is the only hollow one, and it is hollow in
    # every corpus: aug occurs everywhere (0.12-3.87% of typed transitions; 14.6% of
    # Classical sigma) but its edge mass is split across up to 12 absolute-root aug
    # states, so the strongest single aug state ranks 26th-48th by edge mass and clearing
    # the 85% cumulative-mass rule would require raising it to 93.4-99.9% (see
    # data/aug_audit.json). Dropping aug from the legend entirely would read as "aug does
    # not occur"; drawing it filled would read as "aug nodes are on this wheel". Neither
    # is true, so the fill state carries the distinction and the caption states the rule.
    if n > 1:
        leg = [mpatches.Patch(fc=QUAL_FILL[q], ec="#888", lw=0.05, label=l)
               for q, l in [("maj", "major"), ("dom7", "dom7"), ("min", "minor"), ("dim", "dim")]]
        leg.append(mpatches.Patch(fc="none", ec=QUAL_FILL["aug"], lw=0.7, label="aug"))
        if _spare is not None:
            # The legend occupies the empty grid cell rather than a strip under the
            # figure, so the three wheels keep their full height.
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
        # SINGLE-CELL drop-in: no legend (the reader has it from the six-wheel panel,
        # and at this size it would outweigh the wheel), and tight cropping disabled so
        # the canvas keeps the declared figsize -- tight would shrink-wrap the art and
        # destroy the cell match.
        #
        # THE RC PARAM IS THE ONLY WAY TO DISABLE IT HERE. _common.style() loads
        # scienceplots, which sets savefig.bbox = "tight" GLOBALLY. Two things that do
        # NOT work, both measured: omitting the argument, and passing bbox_inches=None
        # -- None is savefig's own signature default, so it falls through to the rc
        # param and still cropped to 1.4232 x 1.6186 in. (bbox_inches="standard" raises
        # AttributeError in mpl 3.11.) Overriding the rc param gives the declared
        # 1.7667 x 1.8100 in exactly, so the axes inside is 1.3207 x 1.4748 in --
        # identical to one cell of the 2x3 grid.
        # The declared canvas is then PADDED to the single supplementary aspect
        # ratio.  Padding adds margin without rescaling anything, so the axes stays
        # 1.3207 x 1.4748 in and still matches one cell of the 2x3 grid exactly.
        with plt.rc_context({"savefig.bbox": None}):
            _bb = _C.fixed_aspect_bbox(fig)
            fig.savefig(path+".pdf", bbox_inches=_bb)
            fig.savefig(path+".png", dpi=300, bbox_inches=_bb)
    plt.close(fig)

def _pools():
    recs = pd.read_parquet(os.path.join(REPO, "dat/output/all_records_usable.parquet"))
    bn = lambda p: ntpath.basename(p) if "\\" in p else posixpath.basename(p)
    # The parquet's "Late-Romantic" bucket is 47/47 Scott Joplin ragtime (score side).
    # Relabel at read time so every panel says RAGTIME; the parquet is left untouched.
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
    # Palette IDENTICAL to Fig 2a and 2b (panel_2b_violins.COLS): a hue means the same
    # corpus in every panel of Fig 2. 2c previously used its own era hues and a
    # teal/red modern family, so Mandopop was teal here and purple two panels up.
    subsets = [("Renaissance", P[("Renaissance", "KernScores")], "#3d8b6e"),
               ("Baroque", P[("Baroque", "KernScores")], "#3a65b0"),
               ("Classical", P[("Classical", "KernScores")], "#5baabc"),
               ("Romantic", P[("Romantic", "KernScores")], "#c9981a"),
               ("CoCoPops", cc, "#8e6fc4"),
               ("POP909", P[("Modern", "POP909")], "#6a3d9a"),
               ("Jazz", P[("Modern", "Jazz")], "#4a2c7a")]
    # FINAL Fig 2c = 2x3 (six panels), ordered Renaissance -> Jazz.
    #
    # WHAT THE ROW SHOWS: harmony gets MORE ELABORATE without getting MORE IRREVERSIBLE.
    # The two are measurably orthogonal. Per piece (n = 6,127), the number of distinct
    # chords a piece uses explains 1.0% of the variance in its directedness (r = -0.098);
    # within corpora, 1.4-6.9% (R^2). Across corpora the entropy rate is uncorrelated
    # with sigma (r = -0.47, p = 0.29, n = 7 corpora).
    #
    # Careful with the wording: complexity is NOT monotone in time either. Entropy rate
    # rises Renaissance -> Romantic (2.22 -> 2.75 nats/step) then FALLS in the modern
    # corpora (Jazz 2.14, Billboard 2.26, Mandopop 2.34). So the defensible claim is
    # "elaboration and irreversibility are decoupled", not "music kept getting more
    # complex while sigma stayed flat". Complexity peaks at Romantic; sigma peaks at Jazz.
    #
    # RENAISSANCE IS THE VISUAL NULL that makes this legible: 78 directed transition
    # types against Mandopop's 86 -- comparable vocabulary -- but 0.15x the arrow
    # (0.035 vs 0.234). Its arrows run both ways (45.3% of steps backward, the highest
    # of any corpus) with forward/backward magnitudes nearly equal (0.135 vs 0.132), so
    # they cancel. A wheel can look busy and still have no net direction.
    #
    # WHY CLASSICAL IS OUT rather than Renaissance: Classical (0.153, 106 directed types,
    # 40.7% backward) and Romantic (0.128, 108, 40.8%) are near-duplicates on every
    # measure, so one of the two is redundant in a six-panel row.
    #
    # WHY BILLBOARD POP IS OUT (author decision) and the sigma = 0 NULL takes its cell:
    # an earlier version of this comment kept both pop corpora on the grounds that they
    # "do not agree" (0.155 vs 0.234). That was argued from point estimates only. With
    # bootstrap CIs (data/mechanism_ci.json) the two intervals OVERLAP -- CoCoPops
    # [0.1227,0.2477] vs POP909 [0.2098,0.2668] -- so they are not in statistical
    # conflict and one pop corpus does represent the other within error. Mandopop is
    # kept over Billboard because it is the larger sample (904 vs 210 pieces) and
    # carries the tighter interval. Billboard pop remains in Fig 2a and Fig 2b, so
    # nothing is lost from the paper; only this six-cell grid drops it.
    # The freed cell is far better spent on the NULL, which gives the reader a
    # zero-circulation baseline in the SAME visual language as the six real wheels --
    # previously it was a separate standalone panel a reader had to be pointed at.
    #
    # ORDERING: chronological left to right, top to bottom, for the five REAL corpora.
    # The null has no date, so it goes FIRST -- it is the baseline the eye should
    # calibrate against before reading any real wheel, and putting it last would read
    # as the endpoint of a historical progression, which it is not.
    #   null (sigma = 0)     | Renaissance (c.1500) | Baroque (c.1700)
    #   Romantic (c.1830)    | Jazz (c.1930)        | Mandopop (1990s-)
    # The arrow does NOT increase monotonically along the real path (Jazz 1.31 is the
    # maximum and sits fifth), which is the point: the trajectory in Fig 2b is a
    # history of harmonic practice, not a ladder.
    #
    # The null cell is Baroque's own counts symmetrized, C' = (C + C.T)/2: identical
    # 60-state vocabulary and node positions, stationary distribution unchanged to
    # 1.1e-03, max|J| collapsing 8.82e-03 -> 2.78e-17, sigma 0.302030 -> 0.00000000.
    # It is rendered through this same render() with cell_neutral={0}, so its edge
    # SET is comparable to Baroque's and only the direction is gone.
    # The null cell reuses panel_2c_null._sym_field rather than a second
    # symmetrization written here, so there is exactly one implementation of the null.
    from fig2c_null_network import _sym_field as _null_field
    main6 = [("null", subsets[1][1], "#6b7075"),    # Baroque's counts, symmetrized
             subsets[0], subsets[1],                # Renaissance, Baroque
             subsets[3], subsets[6], subsets[5]]    # Romantic, Jazz, Mandopop
    render(main6, os.path.join(OUT, "panel_2c_networks"),
           cell_field_fn={0: _null_field},
           cell_title_fn={0: lambda name, seqs: "symmetrized null\n" r"$\sigma$ = 0"},
           cell_neutral={0})
    # SUPPLEMENT: pipeline control. Same era on each column, score (top row) vs
    # chordified (bottom row). Chordified extraction recovers only 10-41% of score
    # sigma (mean 26%), so the two rows should look visibly different -- that IS the
    # figure's claim, and it is why the Fig 2 trajectory uses score corpora only.
    # An all-six-eras network grid was considered and rejected as redundant:
    # Classical (sigma 0.153, 106 types) is a near-duplicate of Romantic (0.128, 108)
    # which is already in the main grid, so it would repeat rather than control.
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
