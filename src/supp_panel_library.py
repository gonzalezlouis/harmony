
"""Supplementary panel library — every supplementary panel builder

suppfig_panels.py -- every SUPPLEMENTARY panel, one file per panel.

STYLE: C.style() from _common.py -- scienceplots 'science', all four spines,
sans-serif, the paper's 8/9 font ladder.  Identical to every main-text panel.
PALETTE: C.CORPUS_COL / C.CORPUS_MARKER -- a corpus has one colour and one glyph
in the SI exactly as it does in Figs 2-4.  The earlier SI used a private palette
with ZERO overlap with the paper's, so Classical/Mandopop/Jazz were three
different colours in the SI than in the main figures.
NO BAR PLOTS: the three bar panels of the old SI pages are redrawn as
level-and-marker comparisons.  Histograms are kept (a histogram is a density,
not a category bar).
Designs are the old SI pages' designs, re-laid-out one panel per figure so the
tiny 5.6-7.2 pt overrides needed to fit 7 panels on a page are no longer used.
"""
import numpy as np, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import _figure_style as C

W1, H1 = 3.35, 2.55      # single-column panel
DISP = {"CoCoPops": "Billboard pop", "POP909": "Mandopop"}
NM = lambda k: DISP.get(k, k)


from matplotlib.text import Text as _Text

def _text_box(t, rend):
    """Extent of the TEXT ONLY.

    Annotation.get_window_extent returns the union of the text and its leader line,
    so a label placed 200 pt away reports a box stretching all the way back to its
    data point.  Measured that way ten placed labels covered every remaining
    candidate position and the search reported the axes full when 3% of its area was
    in use.  A leader line is meant to cross the plot; only the text must be clear.
    """
    return _Text.get_window_extent(t, renderer=rend)


def place_labels(ax, xs, ys, labels, sizes=None, prefer=None, fontsize=6.2,
                 color="#333333"):
    """Direct-label a scatter without any label touching a marker or another label.

    Hand-tuned per-point offsets do not survive a data change, and the offsets that
    shipped here put five of eleven genre labels on top of plotted points.  This tries
    candidate offsets in order and measures each against the REAL renderer: the drawn
    text extent versus the drawn marker extents and versus labels already placed.  The
    first candidate that is clear is kept, so placement is deterministic and re-derives
    itself whenever the data move.  A label pushed beyond 13 pt gets a leader line.
    """
    # An expanding ring rather than a fixed list: a point in a crowded neighbourhood
    # needs to reach further out than one on the edge, and a hard-coded fallback
    # position simply moves the collision somewhere else.
    CAND = []
    # The ring must be able to reach the empty parts of the axes, not just the
    # label's own neighbourhood: at a 92 pt maximum every one of the 130 positions
    # clear of markers for 'alternative' still fell inside the cluster its
    # neighbours had already taken.
    for rad in (11, 15, 20, 26, 33, 41, 50, 62, 76, 92, 115, 145, 180, 225):
        for ang in range(0, 360, 20):
            th = np.radians(ang)
            CAND.append((round(rad * np.cos(th)), round(rad * np.sin(th))))
    from matplotlib.transforms import Bbox
    fig = ax.figure
    fig.canvas.draw(); rend = fig.canvas.get_renderer()
    # PER-MARKER boxes.  A PathCollection's window extent is the bounding box of the
    # WHOLE scatter, so testing against it marks every candidate inside the data cloud
    # as blocked and every label ends up parked in the margin.  Build one box per point
    # from its own centre and its own area instead (s is in points squared).
    xy = ax.transData.transform(np.column_stack([np.asarray(xs, float),
                                                 np.asarray(ys, float)]))
    if sizes is None:
        sizes = np.full(len(xy), 25.0)
    rr = np.sqrt(np.asarray(sizes, float) / np.pi) * fig.dpi / 72.0
    # Circle-versus-rectangle, not the marker's bounding SQUARE.  The square is ~27%
    # larger in area and over-constrains the search: with eleven labels the greedy
    # placement exhausted every candidate for two of them.
    discs = list(zip(xy[:, 0], xy[:, 1], rr))

    def _hits_marker(b):
        for px, py, r in discs:
            cx = min(max(px, b.x0), b.x1); cy = min(max(py, b.y0), b.y1)
            if (px - cx) ** 2 + (py - cy) ** 2 < r * r:
                return True
        return False
    # check_panel counts a label as covering data when its box CONTAINS any plotted
    # vertex -- including every vertex of a fitted line and every errorbar segment
    # end, not just the scatter centres.  Mirror that obstacle set exactly, or the
    # placement search reports clear on a panel the check then rejects.
    verts = []
    for col in ax.collections:
        try:
            o = col.get_offsets()
        except Exception:
            continue
        if o is not None and len(o):
            verts.append(ax.transData.transform(np.asarray(o, float)))
    for ln in ax.lines:
        xd, yd = ln.get_xdata(), ln.get_ydata()
        if len(xd):
            try:
                verts.append(ax.transData.transform(
                    np.column_stack([np.asarray(xd, float), np.asarray(yd, float)])))
            except Exception:
                pass
    verts = np.vstack(verts) if verts else np.empty((0, 2))
    # Place the most constrained label FIRST.  Greedy placement in data order let the
    # roomy points take the good positions and left a crowded one -- reggae, with 54 of
    # 84 candidates clear on its own -- with nothing at all by the time its turn came.
    # Score each label by how many candidates are clear ignoring other labels, then
    # place in ascending order of that freedom.
    def _free(xx, yy, g):
        k = 0
        for dx, dy in CAND:
            t = ax.annotate(g, (xx, yy), textcoords="offset points", xytext=(dx, dy),
                            fontsize=fontsize,
                            ha="left" if dx > 0 else ("right" if dx < 0 else "center"),
                            va="center" if abs(dy) < 8 else ("bottom" if dy > 0 else "top"))
            fig.canvas.draw(); b = _text_box(t, rend); t.remove()
            if not (_hits_marker(b) or
                    (len(verts) and np.any((verts[:, 0] > b.x0) & (verts[:, 0] < b.x1) &
                                           (verts[:, 1] > b.y0) & (verts[:, 1] < b.y1)))):
                k += 1
        return k
    order = sorted(range(len(labels)), key=lambda i: _free(xs[i], ys[i], labels[i]))
    placed, out = [], {}
    for g, xx, yy in [(labels[i], xs[i], ys[i]) for i in order]:
        chosen = None
        for dx, dy in ([prefer[g]] if prefer and g in prefer else []) + CAND:
            t = ax.annotate(g, (xx, yy), textcoords="offset points", xytext=(dx, dy),
                            fontsize=fontsize, color=color, zorder=6,
                            ha="left" if dx > 0 else ("right" if dx < 0 else "center"),
                            va="center" if abs(dy) < 8 else ("bottom" if dy > 0 else "top"),
                            arrowprops=dict(arrowstyle="-", lw=0.45, color="#AAAAAA",
                                            shrinkA=0.5, shrinkB=3.0)
                            if abs(dx) + abs(dy) > 13 else None)
            fig.canvas.draw()
            b = _text_box(t, rend)
            covers = bool(len(verts)) and bool(
                np.any((verts[:, 0] > b.x0) & (verts[:, 0] < b.x1) &
                       (verts[:, 1] > b.y0) & (verts[:, 1] < b.y1)))
            if not (covers or _hits_marker(b)
                    or any(b.overlaps(q) for q in placed)):
                chosen = (dx, dy); placed.append(b); break
            t.remove()
        if chosen is None:
            raise AssertionError(
                "place_labels: no clear position for %r after %d candidates. The axes "
                "are too crowded to direct-label; use a legend instead." % (g, len(CAND)))
        out[g] = chosen
    return out


def check_panel(fig, ax):
    """Overlap check covering EVERY text artist: annotations, tick labels, axis labels
    and legend entries, pairwise, plus annotation text sitting on plotted markers.  An
    earlier version compared ax.texts against ax.texts only and reported zero on panels
    whose tick labels collided."""
    fig.canvas.draw(); rend = fig.canvas.get_renderer()
    items = []
    for t in ax.texts:
        if t.get_text().strip():
            # TEXT only: an annotation's own extent includes its leader line, and a
            # leader line is supposed to cross the plot.  Including it made every
            # direct-labelled panel unsatisfiable.
            items.append(("annot", t.get_text()[:24], _text_box(t, rend)))
    # MINOR tick labels are included: get_xticklabels() returns majors only, and a log
    # axis under two decades wide also labels its minors, which is how a collided
    # x-axis on the provenance panel passed this check.
    for t in (list(ax.get_xticklabels()) + list(ax.get_yticklabels())
              + list(ax.get_xticklabels(minor=True)) + list(ax.get_yticklabels(minor=True))):
        if t.get_text().strip():
            items.append(("tick", t.get_text()[:14], t.get_window_extent(rend)))
    for nm, t in [("xlabel", ax.xaxis.label), ("ylabel", ax.yaxis.label)]:
        if t.get_text().strip():
            items.append((nm, t.get_text()[:24], t.get_window_extent(rend)))
    lg = ax.get_legend()
    if lg:
        # The WHOLE legend box, not just its label texts: a handle sits left of its
        # label, so testing only the text missed a legend whose marker column covered
        # a data point.  Verified with a negative control (legend at loc="center" over
        # a centred point) -- the text-only version reported clean.
        try:
            items.append(("legend", "<box>", lg.get_window_extent(rend)))
        except Exception:
            pass
        for t in lg.get_texts():
            items.append(("legend", t.get_text()[:24], t.get_window_extent(rend)))
    bad = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            k1, n1, b1 = items[i]; k2, n2, b2 = items[j]
            if k1 == "legend" and k2 == "legend":
                continue          # the box necessarily encloses its own labels
            if b1.overlaps(b2):
                bad.append("%s:%r x %s:%r" % (k1, n1, k2, n2))
    for k, n, b in items:
        if k in ("tick", "xlabel", "ylabel"):
            continue
        hit = False
        for col in ax.collections:          # scatter / hexbin offsets
            try:
                pts = col.get_offsets()
            except Exception:
                continue
            if pts is None or len(pts) == 0:
                continue
            px = ax.transData.transform(np.asarray(pts, float))
            if any(b.contains(x_, y_) for x_, y_ in px):
                hit = True; break
        if not hit:
            # ax.lines too: markers drawn with ax.plot are Line2D, not collections,
            # and an earlier version of this check ignored them -- which let a legend
            # sit directly on top of the Jazz star in the pooling-bias panel.
            for ln in ax.lines:
                xd, yd = ln.get_xdata(), ln.get_ydata()
                if len(xd) == 0:
                    continue
                try:
                    px = ax.transData.transform(np.column_stack([np.asarray(xd, float),
                                                                 np.asarray(yd, float)]))
                except Exception:
                    continue
                if any(b.contains(x_, y_) for x_, y_ in px):
                    hit = True; break
        if hit:
            bad.append("%s:%r covers plotted data" % (k, n))
    return bad

def _fin(fig, ax, name, OUT):
    if isinstance(ax, (list, tuple)):
        for a in ax: C.all_spines(a)
    else:
        C.all_spines(ax)
    fig.tight_layout(pad=0.3)
    _a0 = ax[0] if isinstance(ax, (list, tuple)) else ax
    _bad = check_panel(fig, _a0)
    if _bad:
        raise AssertionError("%s: %d text defect(s): %s" % (name, len(_bad), "; ".join(_bad[:6])))
    # Every text artist must lie inside the canvas.  This was previously masked by
    # bbox_inches="tight", which simply grew the output to swallow anything hanging
    # over the edge; the panel then shipped at whatever aspect the overhang implied.
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches=_fixed_aspect_bbox(fig))
    return name


# Every supplementary panel is written at ONE aspect ratio, ASPECT below.  Saving
# with bbox_inches="tight" alone does not do this: it crops to whatever the artists
# happen to occupy, so a panel with a long tick label came out a different shape
# from one without, and the set ranged 0.93 to 1.86.  Measured on the old outputs,
# several panels had tick labels 20-31 pt outside the declared canvas and "tight"
# simply grew the file to swallow them.  So take the tight bounding box, which by
# construction clips nothing, and expand it symmetrically on the short side until
# it reaches ASPECT.  The result is a fixed shape with no clipping.
ASPECT = W1 / H1                                            # 3.35 / 2.55 = 1.3137

def _fixed_aspect_bbox(fig, pad=0.04):
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer()).padded(pad)
    w, h = bb.width, bb.height
    if w / h < ASPECT:
        w2, h2 = h * ASPECT, h
    else:
        w2, h2 = w, w / ASPECT
    cx, cy = 0.5 * (bb.x0 + bb.x1), 0.5 * (bb.y0 + bb.y1)
    return Bbox.from_extents(cx - w2 / 2, cy - h2 / 2, cx + w2 / 2, cy + h2 / 2)

# ---------------------------------------------------------------- A1 surrogates
# superseded below

# ------------------------------------------------------------- A2 floor scaling
# superseded below

# ------------------------------------------------------- A3 pooling floor/signal
# replaced below

# ------------------------------------------------- A4 pooled vs per-work bias
def p_a4_bias(A3, S, ORD8, OUT):
    """The two estimator floors against the measured values they must clear, and
    which corpora the per-work floor swamps.  Drawn horizontally: eight corpus
    names on a rotated x-axis collided with each other in a panel this wide, and
    names on the y-axis need no rotation at all.  Marker sizes go through
    C.msize so the Jazz star carries the same ink as the other glyphs."""
    pool = A3["pooled_bias_zero_truth"]; pw = A3["per_piece_bias_zero_truth"]
    fig, ax = plt.subplots(figsize=(W1, H1 * 1.18))
    y = np.arange(len(ORD8))[::-1]
    ax.axvspan(1e-4, pool, color="#6E9E7E", alpha=0.16, zorder=0)
    ax.axvline(pool, color="#6E9E7E", lw=1.0, zorder=2)
    ax.axvline(pw, color="#C06060", lw=1.0, ls=(0, (3, 2)), zorder=2)
    for i, k in enumerate(ORD8):
        yy = y[i]
        if S[k] < pw:
            ax.plot([S[k]], [yy], "o", ms=11.0, mfc="none", mec="#C06060", mew=0.9, zorder=3)
        ax.plot([S[k]], [yy], C.CORPUS_MARKER[k], ms=C.msize(k, 5.4),
                color=C.CORPUS_COL[k], mec="white", mew=0.6, zorder=4)
    ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels([NM(k) for k in ORD8])
    ax.set_xlabel(r"$\sigma$  (nats step$^{-1}$)")
    ax.set_xlim(pool * 0.40, 6.0); ax.set_ylim(-0.8, len(ORD8) - 0.2)
    ax.legend(handles=[Line2D([], [], color="#6E9E7E", lw=1.0,
                              label=f"pooled floor, {pool:.4f}"),
                       Line2D([], [], color="#C06060", lw=1.0, ls=(0, (3, 2)),
                              label=f"per-work floor, {pw:.3f}"),
                       Line2D([], [], marker="o", ls="", mfc="none", mec="#C06060",
                              ms=8, mew=0.9, label="measured below it")],
              loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=False,
              handletextpad=0.45, borderpad=0.0, columnspacing=1.1, fontsize=5.5)
    return _fin(fig, ax, "suppfig_a4_pooling_bias", OUT)

# ----------------------------------------------------- A5 per-tune distribution
# replaced below

# ------------------------------------------------------------ A6 the same zero
def p_a6_zero_kinds(pt, ntv, OUT):
    """Was a two-bar chart.  Redrawn as a single stacked strip: the two
    meanings partition one population, which a stacked proportion shows and
    two free-standing bars do not."""
    fig, ax = plt.subplots(figsize=(W1, H1 * 0.62))
    nz0 = int((ntv[pt == 0] == 0).sum()); cz = int((ntv[pt == 0] > 0).sum())
    tot = nz0 + cz
    ax.barh([0], [nz0 / tot], height=0.42, color="#C0A060", edgecolor="white", lw=0.6)
    ax.barh([0], [cz / tot], left=[nz0 / tot], height=0.42, color="#6E9E7E",
            edgecolor="white", lw=0.6)
    ax.annotate(f"no terms enter\n(all one-way)\n{nz0:,}  ({nz0/tot:.0%})",
                (nz0 / tot / 2, -0.30), ha="center", va="top", fontsize=6.4,
                color="#8a6a20", linespacing=1.2)
    ax.annotate(f"terms cancel\n(truly balanced)\n{cz:,}  ({cz/tot:.0%})",
                (nz0 / tot + cz / tot / 2, 0.30), ha="center", va="bottom", fontsize=6.4,
                color="#3f6b4f", linespacing=1.2)
    ax.set_xlim(0, 1); ax.set_ylim(-1.05, 1.05); ax.set_yticks([])
    ax.set_xlabel(f"share of the {tot:,} tunes reading $\\sigma = 0$")
    return _fin(fig, ax, "suppfig_a6_zero_kinds", OUT)

# ------------------------------------------------------ A7 resolvability vs types
def p_a7_resolvability(pt, ntv, OUT):
    fig, ax = plt.subplots(figsize=(W1, H1))
    ax.plot(ntv, pt, ".", ms=1.1, color="#8A8A92", alpha=0.45, zorder=2, rasterized=True)
    ed = np.array([0, 2, 4, 8, 16, 32, 64, 128])
    med = [np.median(pt[(ntv >= ed[i]) & (ntv < ed[i+1])])
           if ((ntv >= ed[i]) & (ntv < ed[i+1])).sum() > 8 else np.nan
           for i in range(len(ed) - 1)]
    ax.plot((ed[:-1] + ed[1:]) / 2, med, "o-", ms=3.4, lw=1.1, color="#C04848",
            zorder=4, mec="white", mew=0.5, label="median")
    ax.set_xscale("symlog", linthresh=2)
    ax.set_xlabel("types entering the sum")
    ax.set_ylabel(r"single-tune $\sigma$")
    ax.set_ylim(0, 1.35)
    ax.legend(loc="upper left", frameon=False, handletextpad=0.6, borderpad=0.15)
    return _fin(fig, ax, "suppfig_a7_resolvability", OUT)


FUN, MOD = C.ERA_COL["Baroque"], C.MODCOL

# ------------------------------------------------------------ B1 mode scope
# replaced below

# ------------------------------------------------------- B2 paired difference
def p_b2_paired(WT, OUT):
    fig, ax = plt.subplots(figsize=(W1, H1 * 0.58))
    diff = WT["functional"] - WT["modal"]; ci = WT["ci95"]
    ax.axvline(0, color="#C04848", lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.errorbar([diff], [0], xerr=[[diff - ci[0]], [ci[1] - diff]], fmt="o", ms=5.6,
                color=FUN, mec="white", mew=0.6, ecolor="#333333", capsize=2.6,
                lw=1.0, zorder=3)
    ax.annotate("CI spans 0", (0, 0.62), textcoords="offset points", xytext=(4, 0),
                fontsize=6.6, color="#C04848", va="center", ha="left")
    ax.annotate(f"$P = {WT['p_one_sided']:.2f}$,  $n = {WT['n_tunes']}$",
                (diff, -0.62), textcoords="offset points", xytext=(0, 0),
                fontsize=6.8, ha="center", va="center")
    ax.set_yticks([]); ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(ci[0] - 0.016, ci[1] + 0.016)
    ax.set_xlabel(r"within-tune $\sigma_{\rm func} - \sigma_{\rm modal}$")
    return _fin(fig, ax, "suppfig_b2_paired_difference", OUT)

# ------------------------------------------------------ D1 composer jackknife
def p_d1_jackknife(JCI, ERAS, OUT):
    """Per-composer sigma is already Fig 2a of the main text; this panel adds
    only what Fig 2a cannot show -- each era recomputed WITHOUT its largest
    contributor."""
    fig, ax = plt.subplots(figsize=(W1, H1))
    for i, k in enumerate(ERAS):
        j = JCI[k]; col = C.CORPUS_COL[k]
        ax.plot([i - 0.15, i + 0.15], [j["full"], j["drop"]], color=col, lw=0.8,
                alpha=0.55, zorder=2)
        ax.errorbar(i - 0.15, j["full"],
                    yerr=[[j["full"] - j["full_ci"][0]], [j["full_ci"][1] - j["full"]]],
                    fmt=C.CORPUS_MARKER[k], ms=5.0, color=col, mec="white", mew=0.6,
                    capsize=2.2, lw=1.0, zorder=4)
        ax.errorbar(i + 0.15, j["drop"],
                    yerr=[[j["drop"] - j["drop_ci"][0]], [j["drop_ci"][1] - j["drop"]]],
                    fmt=C.CORPUS_MARKER[k], ms=4.6, mfc="white", mec=col, mew=0.9,
                    color=col, capsize=2.2, lw=1.0, zorder=4)
        lab = j["dropped"].split()[-1] if "Various" not in j["dropped"] else "unattrib."
        last = (i == len(ERAS) - 1)
        ax.annotate(lab, (i + 0.15, j["drop"]), textcoords="offset points",
                    xytext=(-9 if last else 9, 0), fontsize=6.4,
                    ha="right" if last else "left", va="center", color="#555555")
    ax.set_xticks(range(len(ERAS)))
    ax.set_xticklabels(["Renais.", "Baroque", "Classical", "Romantic"])
    ax.set_ylabel(r"$\sigma$  (nats step$^{-1}$)")
    ax.set_ylim(0, 0.42); ax.set_xlim(-0.45, len(ERAS) - 0.42)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color="#555555", ms=5.0,
                              mec="white", mew=0.6, label="all contributors"),
                       Line2D([], [], marker="o", ls="", mfc="white", mec="#555555",
                              ms=4.6, mew=0.9, label="largest removed")],
              loc="upper right", frameon=False, handletextpad=0.5, borderpad=0.15,
              labelspacing=0.22)
    return _fin(fig, ax, "suppfig_d1_composer_jackknife", OUT)

def p_a3_floor_signal(A3, ORD_A, DA, OUT):
    """Glyphs are named inline on the top row instead of in a legend box: the
    box could only sit over the Jazz row, whose clearance label is the widest."""
    fig, ax = plt.subplots(figsize=(W1, H1 * 1.30))
    y = np.arange(len(ORD_A))[::-1]
    for i, k in enumerate(ORD_A):
        yy = y[i]; kk = "Ragtime" if k == "Late-Rom." else k
        col = C.CORPUS_COL[kk]; mk = C.CORPUS_MARKER[kk]
        ax.plot([A3["pooling_floor"][k], A3["real_pooled"][k]], [yy, yy],
                color="#C8C8CE", lw=1.5, zorder=1)
        ax.plot([A3["pooling_floor"][k]], [yy], marker="|", ms=7,
                color="#6E6E76", mew=1.1, zorder=3)
        ax.plot([A3["real_pooled"][k]], [yy], marker=mk, ms=5.0, color=col,
                mec="white", mew=0.7, zorder=4)
        ax.annotate(f"{A3['signal_to_floor'][k]:.0f}x", (A3["real_pooled"][k], yy),
                    textcoords="offset points", xytext=(7, -0.5), fontsize=6.4,
                    color=col, va="center")
    top = y[0]
    ax.annotate("floor", (A3["pooling_floor"][ORD_A[0]], top), textcoords="offset points",
                xytext=(0, 9), fontsize=6.4, color="#6E6E76", ha="center")
    ax.annotate("measured", (A3["real_pooled"][ORD_A[0]], top), textcoords="offset points",
                xytext=(0, 9), fontsize=6.4, color=C.CORPUS_COL[ORD_A[0]], ha="center")
    ax.set_yticks(y); ax.set_yticklabels([DA.get(k, k) for k in ORD_A])
    ax.set_xscale("log"); ax.set_xlim(1e-3, 6.0); ax.set_ylim(-0.7, len(ORD_A) + 0.25)
    ax.set_xlabel(r"$\sigma$  (nats step$^{-1}$)")
    return _fin(fig, ax, "suppfig_a3_floor_vs_signal", OUT)

# superseded below

def p_b1_mode_levels(CO, WT, OUT):
    """Was a two-bar chart of the pooled contrast alone.  Both contrasts now sit
    on one axis, because the point is that the within-tune one is far smaller."""
    from matplotlib.ticker import FixedLocator, NullFormatter, FixedFormatter
    fig, ax = plt.subplots(figsize=(W1, H1))
    lv = [("pooled\n2,920 vs 775", CO["functional_h"], CO["modal_h"]),
          ("within-tune\n66 tunes", WT["functional"], WT["modal"])]
    for i, (lab, f_, m_) in enumerate(lv):
        ax.plot([i - 0.14, i + 0.14], [f_, m_], color=C.GREY, lw=0.8, zorder=2)
        ax.plot([i - 0.14], [f_], "o", ms=5.6, color=FUN, mec="white", mew=0.6, zorder=4)
        ax.plot([i + 0.14], [m_], "s", ms=5.2, color=MOD, mec="white", mew=0.6, zorder=4)
        ax.annotate(f"{f_/m_:.1f}x", (i, max(f_, m_)), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7.2)
    ax.set_yscale("log")
    tk = [0.03, 0.05, 0.1, 0.2, 0.3]
    ax.yaxis.set_major_locator(FixedLocator(tk))
    ax.yaxis.set_major_formatter(FixedFormatter([f"{v:g}" for v in tk]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([0, 1]); ax.set_xticklabels([l for l, _, _ in lv])
    ax.set_ylabel(r"harmonic $\sigma$  (nats step$^{-1}$)")
    ax.set_xlim(-0.55, 1.55); ax.set_ylim(0.024, 0.42)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=FUN, ms=5.6, mec="white",
                              mew=0.6, label="functional"),
                       Line2D([], [], marker="s", ls="", color=MOD, ms=5.2, mec="white",
                              mew=0.6, label="modal")],
              loc="lower left", frameon=False, handletextpad=0.5, borderpad=0.15,
              labelspacing=0.22)
    return _fin(fig, ax, "suppfig_b1_mode_levels", OUT)


# ------------------------------------------------- B3 melody-matched pairs
# superseded

def _corpus_legend(ax, ORD8, loc="lower right", ncol=2):
    ax.legend(handles=[Line2D([], [], marker=C.CORPUS_MARKER[k], ls="", ms=4.6,
                              color=C.CORPUS_COL[k], mec="white", mew=0.5, label=NM(k))
                       for k in ORD8],
              loc=loc, frameon=False, ncol=ncol, handletextpad=0.35,
              columnspacing=0.9, borderpad=0.15, labelspacing=0.22, fontsize=6.0)

# superseded

def p_a2_floor(rows, NT, SL, IC, OUT):
    fig, ax = plt.subplots(figsize=(W1, H1 * 1.16))
    xx = np.logspace(np.log10(NT.min() * 0.7), np.log10(NT.max() * 1.5), 50)
    ax.plot(xx, 10 ** IC * xx ** SL, color="#555555", lw=0.9, ls=(0, (3, 2)), zorder=2)
    for r in rows:
        k = r["corpus"]
        ax.plot(r["n_trans"], r["mean"], C.CORPUS_MARKER[k], ms=5.0,
                color=C.CORPUS_COL[k], mec="white", mew=0.6, zorder=3,
                label=NM(k))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("transitions in corpus"); ax.set_ylabel("surrogate floor")
    ax.text(0.03, 0.06, rf"fit $\propto n^{{{SL:.2f}}}$", transform=ax.transAxes,
            fontsize=6.6, color="#555555", ha="left", va="bottom")
    _corpus_legend(ax, [r["corpus"] for r in rows], loc="upper right", ncol=2)
    return _fin(fig, ax, "suppfig_a2_floor_scaling", OUT)

def p_a5_per_tune(pt, OUT):
    fig, ax = plt.subplots(figsize=(W1, H1))
    pos = pt[pt > 0]
    bins = np.logspace(np.log10(max(pos.min(), 1e-4)), np.log10(pt.max() * 1.02), 46)
    ax.hist(pos, bins=bins, color="#9AA0AA", edgecolor="white", linewidth=0.25,
            label=f"$\\sigma > 0$  ({len(pos):,})")
    ax.set_xscale("log")
    nz = int((pt == 0).sum())
    ax.hist([bins[0] * 0.6] * nz, bins=[bins[0] * 0.45, bins[0] * 0.80],
            color="#C04848", edgecolor="white", linewidth=0.4, zorder=3,
            label=f"$\\sigma = 0$ exactly  ({nz:,}, {nz/len(pt):.0%})")
    med = float(np.median(pt))
    ax.set_ylim(0, nz * 1.30)
    ax.axvline(med, color="#333333", lw=0.8, ls=(0, (3, 2)), zorder=4,
               label=f"median  {med:.3f}")
    ax.set_xlabel(r"single-tune $\sigma$  (nats step$^{-1}$)")
    ax.set_ylabel("tunes")
    ax.legend(loc="upper right", frameon=False, handletextpad=0.6, borderpad=0.15,
              labelspacing=0.25, fontsize=6.2)
    return _fin(fig, ax, "suppfig_a5_per_tune", OUT)


def p_a1_surrogates(rows, DRAWS, OUT):
    """Legend and the ratio note sit in a headroom strip added above the top row;
    every other region of the axes carries either a density or a marker."""
    fig, ax = plt.subplots(figsize=(W1, H1 * 1.42))
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i; k = r["corpus"]; d_ = DRAWS[k]
        vp = ax.violinplot([d_], positions=[y], orientation="horizontal",
                           widths=0.74, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor("#B8B8BE"); b.set_edgecolor("none"); b.set_alpha(0.85)
        ax.plot([d_.min(), d_.max()], [y, y], color="#6E6E76", lw=0.7, zorder=2)
        ax.plot([r["sigma"]], [y], marker=C.CORPUS_MARKER[k], ms=5.0,
                color=C.CORPUS_COL[k], mec="white", mew=0.7, zorder=4)
        ax.annotate(f"{r['ratio']:.0f}x", (r["sigma"], y), textcoords="offset points",
                    xytext=(7, -0.5), fontsize=6.4, color=C.CORPUS_COL[k], va="center")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([NM(r["corpus"]) for r in rows][::-1])
    ax.set_xscale("log"); ax.set_xlim(3e-4, 9.0)
    ax.set_ylim(-0.7, len(rows) + 1.15)
    ax.set_xlabel(r"$\sigma$  (nats step$^{-1}$)")
    ax.legend(handles=[Line2D([], [], lw=5, color="#B8B8BE", alpha=0.85,
                              label="300 order-randomized surrogates"),
                       Line2D([], [], lw=0.7, color="#6E6E76", label="surrogate range"),
                       Line2D([], [], marker="D", ls="", color="#555555", ms=5.0,
                              mec="white", mew=0.7,
                              # Identity only.  "label = ratio to mean" explained the
                              # encoding inside the axes; that belongs in the caption.
                              label="measured")],
              loc="upper left", frameon=False, handletextpad=0.5, borderpad=0.0,
              labelspacing=0.20, fontsize=6.0)
    return _fin(fig, ax, "suppfig_a1_surrogates", OUT)

def p_b3_exact_pairs(EMP, OUT):
    """The five informative melody-matched pairs, plus the two candidates whose two
    settings carry BYTE-IDENTICAL chord streams (mode LABELS differ, harmony does
    not) and which therefore cannot inform the sign."""
    fig, ax = plt.subplots(figsize=(W1, H1 * 1.10))
    inf = [p for p in EMP["pairs"] if abs(p["sigma_fun"] - p["sigma_mod"]) > 1e-9]
    deg = [p for p in EMP["pairs"] if abs(p["sigma_fun"] - p["sigma_mod"]) <= 1e-9]
    OK, NO, GR = "#2F5C9E", "#C64A3F", "#9AA0AA"
    for p in deg:
        ax.plot([0, 1], [p["sigma_fun"], p["sigma_mod"]], "o--", ms=4.0, lw=0.9,
                color=GR, mec="white", mew=0.5, zorder=2)
    for p in inf:
        good = p["sigma_fun"] > p["sigma_mod"]
        ax.plot([0, 1], [p["sigma_fun"], p["sigma_mod"]], "o-", ms=4.4, lw=1.1,
                color=OK if good else NO, mec="white", mew=0.5, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["functional", "modal"])
    ax.set_xlim(-0.30, 1.30); ax.set_ylim(-0.03, 0.72)
    ax.set_ylabel(r"$\sigma$  (nats step$^{-1}$)")
    nk = len(inf) - EMP["n_correct_sign"]
    ax.legend(handles=[Line2D([], [], marker="o", ls="-", color=OK, ms=4.4, lw=1.1,
                              mec="white", mew=0.5,
                              label=f"functional larger ({EMP['n_correct_sign']} pairs)"),
                       Line2D([], [], marker="o", ls="-", color=NO, ms=4.4, lw=1.1,
                              mec="white", mew=0.5,
                              label=f"modal larger or equal ({nk} pairs)"),
                       Line2D([], [], marker="o", ls="--", color=GR, ms=4.0, lw=0.9,
                              mec="white", mew=0.5,
                              label=f"identical chord stream ({len(deg)}, excluded)")],
              loc="upper left", frameon=False, handletextpad=0.5, borderpad=0.0,
              labelspacing=0.20, fontsize=6.0)
    ax.text(0.98, 0.90, f"$P = {EMP['binomial_p']:.2f}$ (sign test)", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.6)
    return _fin(fig, ax, "suppfig_b3_exact_pairs", OUT)


# ------------------------------------------------- E1 Chordonomicon genre breadth
# superseded

def p_e2_decoupling(DEC, S, OUT):
    """Entropy rate against sigma, per corpus: if sigma were a proxy for harmonic
    complexity the points would lie on a rising line; they do not.

    sigma comes from the CANONICAL table, not from decoupling.json, whose Classical
    (0.1528) and Romantic (0.1283) entries predate the dedup/Polish-drop fix.  The
    canonical values are 0.1538 and 0.1371, and the stored sigma CIs bracket THOSE and
    not the stored point estimates.  Spearman is identical either way (rank order
    unchanged), and equals the rho = -0.32, P = 0.48 quoted in the main text.

    Intervals are segments, not errorbars: the full-sample entropy rate lies slightly
    ABOVE its bootstrap interval for Romantic, Billboard pop and jazz.  Entropy rate is
    downward-biased at small n, so resamples -- containing fewer distinct transition
    types -- return lower H than the full sample.  The direction is the same in all
    three cases and is a property of the estimator, not of the data.

    Both left corners hold a marker (jazz top-left, Renaissance bottom-left), so the
    legend goes in a headroom band opened above the data and the statistic sits
    bottom-right, which is empty.  Verified against the plotted coordinates."""
    from scipy import stats as _st
    ks = [k for k in C.CORPUS_ORDER if k in DEC]
    H = np.array([DEC[k]["H_rate"] for k in ks]); Sg = np.array([S[k] for k in ks])
    rho, P = _st.spearmanr(H, Sg)
    fig, ax = plt.subplots(figsize=(W1, H1 * 1.24))
    for k in ks:
        v = DEC[k]; col = C.CORPUS_COL[k]; s_ = S[k]
        ax.plot(v["H_ci"], [s_, s_], color=col, lw=0.7, alpha=0.75, zorder=2)
        ax.plot([v["H_rate"]] * 2, v["sigma_ci"], color=col, lw=0.7, alpha=0.75, zorder=2)
        ax.plot([v["H_rate"]], [s_], C.CORPUS_MARKER[k], ms=5.0, color=col,
                mec="white", mew=0.6, zorder=4)
    ax.set_yscale("log")
    ax.set_ylim(0.024, 5.2)
    ax.set_xlabel(r"chord-transition entropy rate (nats step$^{-1}$)")
    ax.set_ylabel(r"$\sigma$  (nats step$^{-1}$)")
    ax.text(0.975, 0.035, rf"Spearman $\rho = {rho:+.2f}$,  $P = {P:.2f}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8)
    ax.legend(handles=[Line2D([], [], marker=C.CORPUS_MARKER[k], ls="", ms=4.6,
                              color=C.CORPUS_COL[k], mec="white", mew=0.5, label=NM(k))
                       for k in ks],
              loc="upper center", frameon=False, ncol=4, handletextpad=0.35,
              columnspacing=0.8, borderpad=0.0, labelspacing=0.20, fontsize=6.0)
    return _fin(fig, ax, "suppfig_e2_decoupling", OUT), float(rho), float(P)


def p_e1_chordonomicon(CH, OUT):
    """Was a bar chart.  Redrawn as a ranked dot plot with bootstrap SD, because the
    claim is an ORDERING within one corpus, not a set of magnitudes: Chordonomicon is
    crowd-transcribed and triad-flattened (dom7 3.7-4.0% against 9.4% for the expert
    CoCoPops on the same repertoire), so its sigma is a lower bound of unknown tightness
    and never shares an axis with the expert corpora.  That caveat is in the caption,
    not on the face -- it does not fit in ~15 characters.  The legend sits in a headroom
    row opened above the top genre; every data row is occupied out to the SD whisker."""
    order = sorted(CH, key=lambda k: CH[k]["sigma"])
    fig, ax = plt.subplots(figsize=(W1, H1 * 1.42))
    for i, k in enumerate(order):
        s_, e_ = CH[k]["sigma"], CH[k]["sd"]
        ax.plot([0, s_], [i, i], color="#DCDCE2", lw=1.4, zorder=1)
        ax.errorbar([s_], [i], xerr=[e_], fmt="o", ms=4.6, color=C.MODCOL,
                    mec="white", mew=0.6, ecolor="#6E6E76", capsize=2.0, lw=0.9, zorder=3)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
    ax.set_ylim(-0.7, len(order) + 0.55)
    ax.set_xlim(0, max(CH[k]["sigma"] + CH[k]["sd"] for k in CH) * 1.10)
    ax.set_xlabel(r"relative $\sigma$  (nats step$^{-1}$)")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C.MODCOL, ms=4.6,
                              mec="white", mew=0.6, label=r"genre $\sigma$"),
                       Line2D([], [], color="#6E6E76", lw=0.9, label="bootstrap SD")],
              loc="upper right", frameon=False, ncol=2, handletextpad=0.5,
              columnspacing=1.0, borderpad=0.0, labelspacing=0.20, fontsize=6.2)
    return _fin(fig, ax, "suppfig_e1_chordonomicon", OUT)


# --------------------------------------------- F1 pipeline / dataset robustness
def p_f1_provenance(FLOOR, OUT):
    """sigma by extraction route, with the composer held fixed at Bach.

    The earlier version of this panel pooled by era and compared four collections
    that were different repertoire -- Baroque KernScores is 80% Bach while
    GiantMIDI Baroque is 54% Scarlatti -- so its ratio mixed route with
    repertoire.  Holding the composer fixed removes that.

    Each row is a floor tick at the 95th percentile of the pool's own
    order-shuffled null and a marker at the measured value, joined by the span
    between them, which is the idiom used by the estimator panels.  The span
    length IS the evidence: score clears its floor 83-fold, the two performance
    routes 5.7- and 6.8-fold.

    ASAP, the score-as-MIDI route, is not drawn.  At 42 works its floor has risen
    to 0.099 against a measurement of 0.139, a clearance of 1.4, so it cannot
    support a claim and no score-format effect is inferred from it.

    No confidence intervals: resampling works with replacement duplicates works,
    which manufactures one-way transitions and biases sigma upward by 12-33% of
    the estimate at these sample sizes.
    """
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    rows = [("KernScores", "notated score\n(voices read)",    C.CORPUS_COL["Baroque"], "o"),
            ("MAESTRO",    "performance MIDI\n(competition)", C.CORPUS_COL["Ragtime"], "s"),
            ("GiantMIDI",  "performance MIDI\n(transcribed)", C.CORPUS_COL["Ragtime"], "s")]
    fig, ax = plt.subplots(figsize=(3.42, 2.20))
    y = np.arange(len(rows))[::-1]
    for i, (key, _lab, col, mk) in enumerate(rows):
        yy, v = y[i], FLOOR[key]
        ax.plot([v["null_p95"], v["sigma"]], [yy, yy], color=col, lw=1.4,
                solid_capstyle="butt", zorder=2)
        ax.plot([v["null_p95"]], [yy], marker="|", ms=7, color="#6E6E76",
                mew=1.1, zorder=3)
        ax.plot([v["sigma"]], [yy], marker=mk, ms=5.6, color=col, mec="white",
                mew=0.7, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab, _, _ in rows], fontsize=6.2,
                       linespacing=1.15)
    ax.set_xscale("log")
    ax.set_xlim(0.0030, 0.62)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    tk = [0.005, 0.01, 0.05, 0.1, 0.5]
    ax.xaxis.set_major_locator(FixedLocator(tk))
    ax.xaxis.set_major_formatter(FixedFormatter(["%g" % v for v in tk]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"$\sigma$  (nats step$^{-1}$)")
    ax.legend(handles=[
        Line2D([], [], marker="|", ls="", color="#6E6E76", ms=7, mew=1.1,
               label="shuffled-order floor"),
        Line2D([], [], marker="o", ls="", color=C.CORPUS_COL["Baroque"], ms=5.0,
               mec="white", mew=0.6, label="score"),
        Line2D([], [], marker="s", ls="", color=C.CORPUS_COL["Ragtime"], ms=4.8,
               mec="white", mew=0.6, label="performance")],
        loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False,
        handletextpad=0.4, borderpad=0.1, columnspacing=1.0, fontsize=5.8)
    return _fin(fig, ax, "suppfig_f1_provenance", OUT)


def p_g1_rootmotion_sweep(PI, IVN, OUT):
    """Fig. 3c plots mean affinity against DESCENDING-FIFTH share for 20,560 works.  This
    is the same statistic for all twelve root motions, so the fifth is visibly not one
    arbitrary choice among twelve.  Marker area encodes share of repertoire: the fifth is
    both the strongest predictor and the most common motion.  Legend sits upper-left,
    where the near-zero small markers leave the space empty."""
    ds = sorted(PI, key=int)
    r = np.array([PI[k]["pearson"] for k in ds]); sh = np.array([PI[k]["mean_share"] for k in ds])
    fig, ax = plt.subplots(figsize=(W1 * 1.34, H1 * 1.06))
    ax.axhline(0, color="#333333", lw=0.7, zorder=1)
    HL = C.ERA_COL["Baroque"]; OT = "#9AA0AA"
    for i, k in enumerate(ds):
        hot = (int(k) == 5)
        ax.plot([i, i], [0, r[i]], color=HL if hot else OT, lw=1.1, zorder=2)
        ax.scatter([i], [r[i]], s=28 + 620 * sh[i], color=HL if hot else OT,
                   edgecolor="white", linewidth=0.6, zorder=4)
    ax.annotate("descending fifth", (5, r[5]), textcoords="offset points", xytext=(10, -1),
                fontsize=6.8, color=HL, ha="left", va="center")
    ax.set_xticks(np.arange(len(ds)))
    ax.set_xticklabels([IVN[int(k)] for k in ds], rotation=45, ha="right", fontsize=5.8)
    ax.set_ylabel("correlation of work affinity\nwith share of this motion")
    ax.set_ylim(-0.44, 0.80); ax.set_xlim(-0.7, len(ds) - 0.3)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=HL, ms=6.0, mec="white",
                              mew=0.6, label="descending fifth"),
                       Line2D([], [], marker="o", ls="", color=OT, ms=4.6, mec="white",
                              mew=0.6, label="all other motions"),
                       Line2D([], [], marker="o", ls="", color="#CCCCCC", ms=7.5,
                              mec="#666666", mew=0.6, label="area = share of repertoire")],
              loc="upper left", frameon=False, handletextpad=0.6, borderpad=0.0,
              labelspacing=0.28, fontsize=6.2)
    return _fin(fig, ax, "suppfig_g1_rootmotion_sweep", OUT)

# ---------------------------- E1b Chordonomicon mechanism: sevenths drive genre sigma
# superseded


def p_e1b_chord_mechanism(MECH, r_, p_, OUT, OFF=None):
    """Why sigma varies across modern genres, on 312,554 crowd-transcribed songs.  x is
    the fraction of chord tokens that are sevenths; the seventh carries the tritone whose
    semitone resolution makes the cadence one-way.  Independent corroboration of the main
    text's mechanism on a corpus three orders of magnitude larger than Table 1, and a
    RELATIVE statement only: Chordonomicon is triad-flattened, so no value here shares an
    axis with the expert corpora.

    Label offsets are set per genre with leader lines rather than by a uniform rule: rap
    and pop sit 0.0014 apart in x and 0.003 in sigma, and country/alternative/reggae fall
    within one marker radius of each other."""
    OFF = OFF or {"punk": (10, -2), "metal": (11, 2), "electronic": (11, 3), "jazz": (11, 3), "soul": (11, -4), "reggae": (9, 12), "alternative": (11, 0), "country": (-11, -7), "rock": (9, -12), "pop": (0, 12), "rap": (11, -7)}
    gs = sorted(MECH, key=lambda g: MECH[g]["seventh_frac"])
    x = np.array([MECH[g]["seventh_frac"] for g in gs])
    y = np.array([MECH[g]["sigma"] for g in gs])
    e = np.array([MECH[g]["sd"] for g in gs])
    n = np.array([MECH[g]["n"] for g in gs], float)
    # Eleven direct labels need room: at 1.30x the placement search exhausts 84
    # candidate positions for reggae.  The saved aspect is fixed regardless.
    fig, ax = plt.subplots(figsize=(W1 * 1.45, H1 * 1.34))
    b, a = np.polyfit(x, y, 1)
    # Two vertices, not 40: the fit is a straight line, and check_panel treats every
    # vertex as plotted data, so a 40-point guide line walls off the whole panel and
    # no direct label can be placed anywhere near the trend.
    xf = np.linspace(0.004, x.max() * 1.10, 2)
    ax.plot(xf, a + b * xf, color="#555555", lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.errorbar(x, y, yerr=e, fmt="none", ecolor="#9AA0AA", elinewidth=0.7,
                capsize=1.8, zorder=2)
    ax.scatter(x, y, s=20 + 260 * (n / n.max()) ** 0.5, color=C.MODCOL,
               edgecolor="white", linewidth=0.6, zorder=4)
    place_labels(ax, x, y, gs, sizes=20 + 260 * (n / n.max()) ** 0.5, prefer=OFF)
    ax.set_xlabel("fraction of chords that are sevenths")
    ax.set_ylabel(r"relative $\sigma$  (nats step$^{-1}$)")
    ax.set_xlim(0, x.max() * 1.22); ax.set_ylim(0.055, 0.245)
    # the "0.00" x tick sat on the "0.050" y tick in the corner
    ax.set_xticks([t for t in ax.get_xticks() if t > 0])
    ax.text(0.035, 0.965, rf"$r = {r_:+.2f}$,  $P = {p_:.3f}$", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.8)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C.MODCOL, ms=5.0,
                              mec="white", mew=0.6, label="genre (area = songs)"),
                       Line2D([], [], color="#9AA0AA", lw=0.7, label="bootstrap SD")],
              loc="lower right", frameon=False, handletextpad=0.5, borderpad=0.0,
              labelspacing=0.22, fontsize=6.2)
    return _fin(fig, ax, "suppfig_e1b_chord_mechanism", OUT)


# ------------------------- B1new: per-setting distributions (2,920 vs 775 settings)
def p_b1_setting_dist(per_set, sf, sm, OUT):
    """Replaces a four-point summary.  Every one of the 3,695 settings is drawn.  The
    zero bar is separated because a zero means 'no reversible chord move was observed in
    this setting', which is a different statement from a small positive value."""
    F, M = per_set["F"], per_set["M"]
    fig, ax = plt.subplots(figsize=(W1 * 1.20, H1 * 1.08))
    bins = np.logspace(np.log10(4e-3), np.log10(2.1), 34)
    for v, col, lab, sty in [(F, FUN, f"functional, {len(F):,} settings", "-"),
                             (M, MOD, f"modal, {len(M):,} settings", "--")]:
        h, _ = np.histogram(v[v > 0], bins=bins)
        ax.step(bins[:-1], h / len(v), where="post", color=col, lw=1.2, ls=sty, zorder=3,
                label=lab)
    for v, col, x0 in [(F, FUN, 1.9e-3), (M, MOD, 2.6e-3)]:
        ax.bar([x0], [(v == 0).mean()], width=x0 * 0.42, color=col, alpha=0.55,
               edgecolor="white", linewidth=0.4, zorder=2)
    ax.annotate(f"no one-way move\nobserved\n({(F==0).mean():.0%} / {(M==0).mean():.0%})",
                (2.2e-3, max((F == 0).mean(), (M == 0).mean())), textcoords="offset points",
                xytext=(3, 6), fontsize=6.0, color="#555555", linespacing=1.25,
                ha="left", va="bottom")
    for s_, col in [(sf, FUN), (sm, MOD)]:
        ax.axvline(s_, color=col, lw=0.8, ls=(0, (1, 1.6)), zorder=4)
    ax.set_xscale("log"); ax.set_xlim(1.4e-3, 2.6); ax.set_ylim(0, 0.30)
    # The log locator also creates tick labels for the decades just OUTSIDE xlim
    # (10^-3 and 10^1 here).  Those artists are clipped in the render but still laid
    # out, and they land in the bottom corners, where 10^-3 overlapped the "0.00" y
    # tick.  Pinning the locator to the visible decades removes them.  Minor labels
    # are suppressed for the same reason: under three decades matplotlib labels them.
    import matplotlib.ticker as _mt
    ax.set_xticks([1e-2, 1e-1, 1e0])
    ax.xaxis.set_minor_formatter(_mt.NullFormatter())
    ax.set_xlabel("how one-way one setting's chord changes are\n"
                  r"($\sigma$, nats per chord change; log scale)")
    ax.set_ylabel("share of settings")
    ax.legend(loc="upper right", frameon=False, handletextpad=0.7, borderpad=0.0,
              labelspacing=0.26, fontsize=6.2)
    return _fin(fig, ax, "suppfig_b1_setting_dist", OUT)


# ------------------- B2new: the 66 per-tune differences (replaces a single point)
def p_b2_tune_differences(pdiff, pooled, OUT):
    """Replaces one point with an error bar.  Each of the 66 tunes carrying both kinds of
    setting is a dot.  The point of the panel is the disagreement between the two
    summaries: the pooled statistic is positive, yet more individual tunes fall on the
    modal side than the functional one, because pooling weights a tune by how many chord
    changes it contributes.  That disagreement is why the paired test does not resolve."""
    pos = int((pdiff > 0).sum()); neg = int((pdiff < 0).sum()); zer = int((pdiff == 0).sum())
    fig, ax = plt.subplots(figsize=(W1 * 1.20, H1 * 0.86))
    rng = np.random.default_rng(4)
    y = rng.normal(0, 0.052, len(pdiff))
    col = [FUN if d > 0 else (MOD if d < 0 else "#AAAAAA") for d in pdiff]
    ax.axvline(0, color="#333333", lw=0.8, zorder=1)
    ax.scatter(pdiff, y, s=17, c=col, edgecolor="white", linewidth=0.4, alpha=0.92, zorder=3)
    ax.plot([pooled], [-0.20], marker="v", ms=6.5, color="#222222", zorder=5)
    # No sentences inside the axes.  The two side counts, the pooled value and the
    # zero count were previously drawn as four blocks of running text over the data;
    # they are identity and context, so they belong in the legend and the caption.
    # Side membership is already carried by the point colour, which the legend keys.
    ax.set_yticks([]); ax.set_ylim(-0.30, 0.30)
    ax.set_xlabel("change in one-way-ness when the same tune is reharmonized\n"
                  r"($\sigma_{\rm functional} - \sigma_{\rm modal}$, nats per chord change)")
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.legend(handles=[Line2D([], [], ls="none", marker="o", ms=4.2, mfc=FUN, mec="white",
                              mew=0.4, label=f"functional side ({pos})"),
                       Line2D([], [], ls="none", marker="o", ms=4.2, mfc=MOD, mec="white",
                              mew=0.4, label=f"modal side ({neg})"),
                       Line2D([], [], ls="none", marker="o", ms=4.2, mfc="#AAAAAA",
                              mec="white", mew=0.4, label=f"no change ({zer})"),
                       Line2D([], [], ls="none", marker="v", ms=4.6, color="#222222",
                              label=f"pooled  {pooled:+.3f}")],
              loc="upper left", frameon=False, handletextpad=0.5, borderpad=0.0,
              labelspacing=0.22, fontsize=6.0, ncol=2, columnspacing=1.0)
    return _fin(fig, ax, "suppfig_b2_tune_differences", OUT)
