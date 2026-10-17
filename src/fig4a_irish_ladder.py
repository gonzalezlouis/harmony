#!/usr/bin/env python3
"""Figure 4a — one tune under two harmonizations, bar for bar

Fig 4 ladder - two SEQUENTIAL STRIPS of the same Irish melody, harmonised twice.

ILLUSTRATION ONLY. No sigma value is printed anywhere on this panel and none is
computed here. Single-tune sigma is not interpretable (out-and-back walks zero by
topology, one-way progressions read zero because the reverse count is zero, 17.1%
of chorded tunes score exactly 0, and the within-tune paired test over 66 tunes
gave p=0.24). The corpus-level bars carry the measurement; these strips only show
WHAT the two harmonisations do, position for position.

WHAT IS DRAWN. "The Blarney Pilgrim" (jig, 6/8), first strain, in two settings
whose melody is character-identical in the source ABC (verified on the raw text
after stripping the "..." chord annotations -- NOT on an extracted pitch-class
stream, which collapses repeats and can make different melodies look equal):

    setting 53427  K:G      functional  (I/V/IV in G, D/F# leading-tone bass)
    setting 28409  K:Dmix   modal       (double tonic G<->D, no leading tone)

ALIGNMENT. The two settings carry DIFFERENT numbers of chord slots -- 9 functional
vs 11 modal -- so an event-indexed strip would put unrelated chords above one
another. Every chord onset in both settings lands on a multiple of three eighths,
i.e. a half-bar, so the strips are drawn on a common metric grid of 16 half-bar
slots (8 bars x 2). Each slot shows the chord in force over those three eighths;
a chord held across two slots appears in both. Position i therefore covers the
same three melody notes in both rows, which is what makes the comparison honest.

VISUAL LANGUAGE. Deliberately identical to draw_strip() in
../../harmony-epr/make_fig1_panels.py, the project's established way of drawing a
sequence of states in time (Fig 1a/1b/1c): a row of rounded FancyBboxPatch boxes,
white edge lw 0.6, bold centred label, position ticks 1..N along the bottom,
left/top/right spines hidden. Two departures, both required by this figure:
  * colour keys CHORD QUALITY (_common.QUAL), not scale degree or chord root, so
    the boxes mean the same thing as every other chord colour in the paper
  * no ax.set_title -- titles and panel letters go on in Illustrator. The rows are
    distinguished by a horizontal left-side row label (rotation-0 ylabel), which
    sits outside the boxes and is not a title.
Labels are root + quality symbol, the same convention make_fig1_panels uses for
chord states, so "D/F#" prints as "D" and "Cadd9" as "C". The bass note and the
added ninth are visible in the engraved score above the panel; at a 16-slot pitch
they would not fit inside a box.
"""
import os, sys, json, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _figure_style as C

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(os.path.dirname(HERE), "dat", "irish_examples.json")
NAME = "panel_4_irish_ladder"

# ---- draw_strip geometry, copied unchanged from make_fig1_panels -------------
BAR_W, ROUND = 0.80, 0.10
Y_BOT, Y_TOP = 0.10, 0.92

# quality palette: canonical project colours (_common.QUAL) + the two it omits
QCOL = {"maj": C.QUAL["major"], "min": C.QUAL["minor"], "dom7": C.QUAL["dom7"],
        "dim": C.QUAL["dim"], "aug": "#E070B0"}
QSYM = {"maj": "", "min": "m", "dom7": "7", "dim": "\u00b0", "aug": "+"}

SLOTS_PER_BAR = 2          # half-bar grid; every chord onset falls on one
EIGHTHS_PER_SLOT = 3       # 6/8 bar, L:1/8


def label_text_color(c):
    """Black or white label depending on how dark the box is (draw_strip's rule)."""
    r, g, b = matplotlib.colors.to_rgb(c)
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.60 else "#2E3440"


# ---- ABC -> (label, colour) per metric slot ---------------------------------
def parse_chord(sym):
    """ABC chord symbol -> (root, quality). Bass note and colour tones dropped:
    quality is what the palette keys on, and the engraving carries the rest."""
    s = sym.split("/")[0].strip()                      # D/F# -> D
    m = re.match(r"^([A-G][b#]?)(.*)$", s)
    root, rest = m.group(1), m.group(2)
    rest = re.sub(r"(add\d+|sus\d*|\(.*\)|9|11|13|6)", "", rest)   # Cadd9 -> C
    low = rest.lower()
    if low.startswith("dim") or "\u00b0" in rest:      qual = "dim"
    elif low.startswith("aug") or rest.startswith("+"): qual = "aug"
    elif low.startswith("maj7"):                        qual = "maj"
    elif low.startswith("m"):                           qual = "min"
    elif "7" in rest:                                   qual = "dom7"
    else:                                               qual = "maj"
    return root, qual


def chord_onsets(abc):
    """[(onset_in_eighths, chord_symbol)] and the strain's total length, read off
    the RAW ABC so the strip cannot drift from the engraved score."""
    body = abc.split("K:")[1].split("\n", 1)[1]
    body = body.replace("|:", "|").replace(":|", "|").replace("]", "")
    out, t = [], 0
    for tok in re.finditer(r'"([^"]*)"|([A-Ga-g])(\d*)', body):
        ch, note, dur = tok.groups()
        if ch is not None:
            out.append((t, ch))
        else:
            t += int(dur) if dur else 1
    return out, t


def slot_states(abc):
    """One (label, colour) per half-bar slot; a held chord repeats across slots."""
    onsets, total = chord_onsets(abc)
    n_slots = total // EIGHTHS_PER_SLOT
    bad = [o for o, _ in onsets if o % EIGHTHS_PER_SLOT]
    assert not bad, f"chord onsets off the half-bar grid: {bad}"
    states, cur = [], None
    by_slot = {o // EIGHTHS_PER_SLOT: c for o, c in onsets}
    for i in range(n_slots):
        cur = by_slot.get(i, cur)
        assert cur is not None, "strain starts without a chord"
        root, qual = parse_chord(cur)
        states.append((root + QSYM[qual], QCOL[qual]))
    return states, n_slots


# ---- drawing: draw_strip, minus the title, plus a row label ------------------
def draw_strip(ax, states, n, row_label, xticks=True):
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    for i, (lab, col) in enumerate(states, 1):
        ax.add_patch(FancyBboxPatch(
            (i - BAR_W / 2, Y_BOT), BAR_W, Y_TOP - Y_BOT,
            boxstyle=f"round,pad=0.0,rounding_size={ROUND}",
            fc=col, ec="white", lw=0.6, zorder=3))
        ax.text(i, (Y_BOT + Y_TOP) / 2, lab, ha="center", va="center",
                fontsize=6.0, color=label_text_color(col),
                fontweight="bold", zorder=4)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    # the house style turns xtick.top/ytick.right on and scienceplots adds minor
    # ticks; both leak stray marks along the hidden spines of a strip, so kill
    # ticks on every side except the bottom, MAJOR AND MINOR (which="both" is
    # load-bearing -- tick_params defaults to major only and the minor marks stayed).
    ax.tick_params(which="both", left=False, right=False, top=False)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks(range(1, n + 1))
    if xticks:
        ax.set_xticklabels(range(1, n + 1), fontsize=6)
        ax.set_xlabel("Position", fontsize=7.5, labelpad=2)
    else:
        ax.set_xticklabels([])
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(which="both", bottom=False)
    # row label: a text element at the axes' left, outside the boxes. NOT a title.
    ax.set_ylabel(row_label, rotation=0, ha="right", va="center",
                  fontsize=7.5, fontweight="bold", labelpad=4)


def build(break_it=False):
    C.style()
    d = json.load(open(SRC))
    f_states, nf = slot_states(d["functional"]["abc"])
    m_states, nm = slot_states(d["modal"]["abc"])
    assert nf == nm, f"metric grids differ: {nf} vs {nm}"

    fig, axes = plt.subplots(2, 1, figsize=(4.62, 1.66),
                             gridspec_kw=dict(hspace=0.38))
    draw_strip(axes[0], f_states, nf, "Functional", xticks=False)
    draw_strip(axes[1], m_states, nm, "Modal", xticks=True)
    if break_it:   # negative control: force labels wide enough to collide
        for ax in axes:
            for t in ax.texts:
                t.set_fontsize(22)
    return fig, axes, f_states, m_states, nf


if __name__ == "__main__":
    fig, axes, fs, ms, n = build()
    rep = [C.overlap_report(fig, ax) for ax in axes]
    print("overlap:", rep)
    C.save(fig, NAME)
    print("functional labels:", " ".join(l for l, _ in fs))
    print("modal labels     :", " ".join(l for l, _ in ms))
    print("slots:", n)
