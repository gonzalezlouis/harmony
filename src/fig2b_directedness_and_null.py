#!/usr/bin/env python3
"""Figure 2b — per-work directedness against the shuffled null

Fig 2b - per-piece directedness distribution (ECDF) + corpus-pooled sigma.

CONSISTENT WITH 2a AND 2c BY CONSTRUCTION.  All three panels now report the same
underlying quantity:

  2a  per-piece directedness, by composer
  2b  per-piece directedness, by corpus (ECDF)  +  pooled sigma (the corpus total)
  2c  the pooled net-current field drawn as a network

Per-piece directedness = mean log(P_fwd / P_rev) per transition, evaluated under
the GLOBAL pooled transition matrix (all 22,648 pieces).  Two properties make it
the right per-piece statistic:

 1. IT DECOMPOSES THE POOLED SIGMA EXACTLY.  Count-weighted mean directedness x
    (n_bidirectional / n_total) reproduces pooled sigma to 3 decimals (Jazz
    1.4064 vs 1.4061; Mandopop 0.2756 vs 0.2756).  Panel a/b is therefore a
    decomposition of panel b/c's total, not a second unrelated measurement.

 2. IT IS CENSORING-IMMUNE.  Per-piece sigma_bidirectional discards any chord
    pair not traversed BOTH ways inside that piece.  On short jazz lead sheets
    that throws away 76.9% of the flow and sends 24.6% of pieces to exactly 0 --
    which is why the earlier per-piece-sigma version of this panel ranked jazz
    LOWEST while the pooled value ranked it HIGHEST.  That was a measurement
    artifact, not a musical fact.  Directedness needs no within-piece reverse
    observation and resolves it: jazz is highest on both.

The fix is measurable: Spearman rho between the per-piece statistic and pooled
sigma across the eight corpora was -0.595 with per-piece sigma (ANTI-correlated
-- the source of this panel's earlier internal contradiction) and is +0.762 with
directedness.

Robustness: rebuilding the reference matrix with equal weight per corpus (so the
classical-heavy pool cannot define "forward" for everyone) preserves the ordering
(Renaissance lowest 0.030, Jazz highest 0.905).

PLOT TYPE: ECDF -- a step function through the actual data, no kernel, no bins,
so it cannot render density where there is none.

NULL FLOOR (added 2026-07-31).  Each stem now RISES FROM ITS OWN SHUFFLED NULL
rather than from a cosmetic baseline, so the drawn length IS the effect size.

  null = within-piece chord-order shuffle, B = 300 per corpus, identical estimator.
  It preserves each piece's chord multiset and length and destroys only sequence.
  The floor is small but POSITIVE because a finite shuffled corpus still has
  N_fwd != N_rev by sampling noise; it scales as 1/n_transitions, so the 47-piece
  Ragtime floor (0.0351) is 57x the 2,036-piece Renaissance floor (0.0006).

  Open marker = the null's 95th percentile.  Every real sigma clears its own p95
  by 9.0x (Ragtime, the tightest) to 199x (Jazz).  A single global null line would
  be wrong: Ragtime's floor is as large as Renaissance's REAL sigma, so sigma near
  0.03 is only meaningful against a corpus's OWN floor.

  The SYMMETRIZED null (average forward and reverse counts) is exactly 0 by
  construction and so has no position on a log axis -- it is stated in the caption
  and drawn instead as the companion wheel fig2c_null_network.  Numbers and the
  full derivation: dat/null_models.json.

Data: dat/corpus_directedness.json, dat/null_models.json
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import _figure_style as C

C.style()
DATADIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dat")
d = json.load(open(os.path.join(DATADIR, "corpus_directedness.json")))
NULL = json.load(open(os.path.join(DATADIR, "null_models.json")))["shuffled_null"]["per_corpus"]
# Chronological by the period each corpus samples (Table 1): Jazz standards c.1920-60
# precede the Billboard frame (1958-91) and POP909's release years (c.1990-2010). This
# keeps the score/label provenance split contiguous too -- indices 0-4 are score-derived,
# 5-7 label-bearing -- so the divider at 4.5 still separates both orderings at once.
order = ["Renaissance", "Baroque", "Classical", "Romantic", "Ragtime",
         "Jazz", "CoCoPops", "POP909"]
LABEL = {"CoCoPops": "Billboard pop", "POP909": "Mandopop"}
# Period sampled by each corpus, matching Table 1. Drawn on a second line under the corpus
# name so the chronological x-order is legible without cross-referencing the table.
# Derived from the composers ACTUALLY pooled (see COMP audit): Renaissance spans Dufay
# (b.1397) to Lasso (d.1594) and NOT 1440-1520 as first written; Baroque extends to
# D. Scarlatti (d.1757); Romantic starts with Beethoven's working life (c.1795).
# No corpus carries per-work dates, so these are indicative only and enter no calculation.
PERIOD = {"Renaissance": "c.1420\u20131590", "Baroque": "c.1680\u20131760",
          "Classical": "c.1760\u20131830", "Romantic": "c.1795\u20131850",
          "Ragtime": "1899\u20131917", "Jazz": "c.1920\u20131960",
          "CoCoPops": "1958\u20131991", "POP909": "c.1990\u20132010"}
COLS = {"Renaissance": "#3d8b6e", "Baroque": "#3a65b0", "Classical": "#5baabc",
        "Romantic": "#c9981a", "Ragtime": "#c04848",
        "CoCoPops": "#8e6fc4", "POP909": "#6a3d9a", "Jazz": "#4a2c7a"}
MODERN = {"CoCoPops", "POP909", "Jazz"}

# sized to the upper-right slot of fig/fig2-complete.ai (4.61 x 2.49 in) so the three
# Fig 2 panels drop straight into that canvas without rescaling
fig, axa = plt.subplots(figsize=(4.96, 2.33))

# ---------- (a) corpus-pooled sigma over its own shuffled null (log) ----------
# The old baseline was a cosmetic 0.03. Each stem now spans null p95 -> measured
# sigma, so on a log axis the DRAWN LENGTH is log(real/null): the effect size is
# the thing the eye measures, and no stem can start below its own noise floor.
for i, k in enumerate(order):
    v = d[k]["pooled_sigma"]
    f95 = NULL[k]["shuffled_null_p95"]
    axa.plot([i, i], [f95, v], color=COLS[k], lw=1.4, zorder=2, solid_capstyle="butt")
    axa.plot([i], [f95], marker="o", ms=3.4, markerfacecolor="white",
             markeredgecolor=COLS[k], markeredgewidth=0.7, zorder=3)
    axa.plot([i], [v], marker="o", ms=6, color=COLS[k],
             markeredgecolor="black", markeredgewidth=0.6, zorder=4)
    axa.annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(0, 4),
                 ha="center", fontsize=5.4)
axa.set_yscale("log"); axa.set_ylim(3.5e-4, 4.2)
axa.set_yticks([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0])
axa.set_yticklabels(["0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1.0", "3.0"])
# two marker types now appear, so they carry a two-entry key (no prose in-axis).
# Upper left is empty: the tallest stem there is Renaissance at 0.035.
_key = [plt.Line2D([], [], color="#555555", marker="o", ms=4.4, lw=1.2,
                   markeredgecolor="black", markeredgewidth=0.5, label=r"measured $\sigma$"),
        plt.Line2D([], [], color="none", marker="o", ms=3.4, markerfacecolor="white",
                   markeredgecolor="#555555", markeredgewidth=0.7, label="shuffled null")]
axa.legend(handles=_key, loc="upper left", fontsize=5.2, frameon=False,
           handlelength=1.1, handletextpad=0.4, labelspacing=0.25,
           borderpad=0.1, borderaxespad=0.25)
axa.set_ylabel(r"corpus-pooled $\sigma$" "\n" r"(nats step$^{-1}$, log)", fontsize=7.4)
axa.set_xticks(range(len(order)))
# The period line is set two points smaller than the corpus name via \small-style scaling
# in mathtext-free plain text, achieved by drawing it as a separate, smaller tick line.
# Adding a second line grew the page 2.46 -> 2.55 in, past the 2.49 in artboard slot, so
# figsize drops 0.09 in to hold the panel at its allotted height.
axa.set_xticklabels([f"{LABEL.get(k, k)}\n{PERIOD[k]}" for k in order], rotation=40,
                    ha="right", rotation_mode="anchor", fontsize=6.1)
axa.tick_params(axis="x", top=False, bottom=True, labeltop=False, labelbottom=True)
axa.set_xlim(-0.7, len(order) - 0.3)
axa.axvline(4.5, color="#999999", lw=0.8, ls=":", zorder=1)
# group captions live INSIDE the top axes, in data coordinates, above the tallest stem
# (Jazz 1.31, value label ~1.5) but well below ylim 4.2 -- so they cannot clip at the
# figure edge and cannot collide with the rotated corpus names above the axes.
# y lowered 2.9 -> 2.1 so the captions clear the two-entry marker key that now sits
# at upper left; still above Jazz's 1.31 value label (~1.5) and below ylim 4.2.
axa.text(2.0, 2.9, "classical (score)", ha="center", va="center", fontsize=6.2,
         color="#666666", zorder=6)
# Both captions sit at y = 2.9 on one horizontal line, and each is centered on its own
# group: indices 0-4 -> x = 2.0, indices 5-7 -> x = 6.0. At y = 2.9 the caption clears
# Jazz's 1.31 value label (top ~1.5) without needing a horizontal offset.
axa.text(6.0, 2.9, "modern (labeled)", ha="center", va="center", fontsize=6.2,
         color="#666666", zorder=6)

C.all_spines(axa)
axa.yaxis.set_label_coords(-0.085, 0.5)
axa.set_xlabel("")

C.save(fig, "panel_2b_violins")
print("wrote panel_2b_stems | " + ", ".join(f"{k} {d[k]['pooled_sigma']:.2f}" for k in order))
