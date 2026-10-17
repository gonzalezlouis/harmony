#!/usr/bin/env python3
"""Figure 3b — constant affinity, rising current

Fig 4c - constant force, rising current.

Split out of panel_4b_poparadox, which was trying to carry four 20x12 transfer
matrices AND these two summary axes in one 3.9in panel; that is what made both
unreadable. The matrices keep 4b; the decomposition gets its own panel.

sigma = sum_t J(t) A(t), so an arrow can grow two ways: raise the per-step
thermodynamic force A, or push more net current J around the same edges. Across
the classical plateau and both pop corpora the force is flat (coloured points,
shaded band) while the current rises -- pop reorganises circulation at fixed
force rather than inventing stronger cadential pull.

The grey points are every other corpus, plotted so the reader can see the claim
is SCOPED: A is not constant across all eras (Renaissance 0.37, Jazz 2.00). Both
groups are labelled in-axis; nothing here is unexplained.

ERROR BARS (added): 95% percentile bootstrap CIs from data/mechanism_ci.json,
resampling PIECES with replacement within each corpus (B = 2000). Pieces are the
independent observations; transitions are not, so resampling transitions would
understate the width badly. Each replicate recomputes type counts, sigma, J_tot
and A = sigma/J_tot from scratch. The intervals are ASYMMETRIC about the point
estimate (thinning rare types inflates |log(N_fwd/N_rev)|, so the resampling
distribution is right-skewed in most corpora) -- so bars are drawn from the
percentiles directly, not as +/- SE, and the marker stays at the full-sample
point estimate. CoCoPops and Ragtime are visibly the widest: 210 and 47 pieces.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import _figure_style as C

C.style()
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "dat")
Mx = json.load(open(os.path.join(DATA, "mechanism.json")))
MECH = Mx["mechanism"]
CI = json.load(open(os.path.join(DATA, "mechanism_ci.json")))["per_corpus"]

def err(k, key):
    """(lower, upper) bar lengths from the percentile CI, relative to the point."""
    lo, hi = CI[k][key + "_ci"]
    v = MECH[k][key]
    return np.array([[max(v - lo, 0.0)], [max(hi - v, 0.0)]])

BAND = ["Classical", "Romantic", "CoCoPops", "POP909"]   # the constant-force group
OTHER = ["Renaissance", "Baroque", "Ragtime", "Jazz"]  # reference
LABEL = {"CoCoPops": "Billboard pop", "POP909": "Mandopop", "Ragtime": "Ragtime"}
COL = {"Classical": "#5baabc", "Romantic": "#c9981a", "CoCoPops": "#8e6fc4",
       "POP909": "#6a3d9a", "Renaissance": "#BBBBBB", "Baroque": "#BBBBBB",
       "Ragtime": "#BBBBBB", "Jazz": "#BBBBBB"}

fig, (axA, axJ) = plt.subplots(2, 1, figsize=(3.45, 3.30),
                               gridspec_kw=dict(hspace=0.12))

def corpus_axis(ax, key, ylab, ymax, connect):
    xs = np.arange(len(BAND))
    vals = [MECH[k][key] for k in BAND]
    if connect:
        ax.plot(xs, vals, color="#555555", lw=1.0, zorder=2)
    else:
        ax.axhspan(min(vals), max(vals), color="#EDEDED", zorder=1)
    # Bars go ON TOP of the markers, in black. Several CIs (Classical A, Mandopop A,
    # Baroque/Jazz A and J) are NARROWER THAN THE MARKER -- a 46pt^2 disc is ~3.4pt in
    # radius and Classical's lower A arm is 1.2pt -- so a same-colour bar behind the
    # disc is invisible and the reader would read "no error bar" as "no uncertainty".
    # Drawn over the fill, the short black stroke with caps stays legible at 7pt and
    # the smallness of the interval is itself the message.
    for x, k in zip(xs, BAND):
        ax.scatter(x, MECH[k][key], s=46, color=COL[k], edgecolor="black",
                   linewidth=0.5, zorder=4)
        ax.errorbar(x, MECH[k][key], yerr=err(k, key), color="black", lw=0.8,
                    capsize=1.7, capthick=0.8, zorder=5)
    for j, k in enumerate(OTHER):
        ax.scatter(len(BAND) + 0.85 + j, MECH[k][key], s=26, color="#BBBBBB",
                   edgecolor="#777777", linewidth=0.5, zorder=3)
        ax.errorbar(len(BAND) + 0.85 + j, MECH[k][key], yerr=err(k, key),
                    color="#555555", lw=0.7, capsize=1.5, capthick=0.7, zorder=5)
    ax.set_xticks(list(xs) + [len(BAND) + 0.85 + j for j in range(len(OTHER))])
    ax.set_xticklabels([LABEL.get(k, k) for k in BAND + OTHER],
                       fontsize=5.4, rotation=38, ha="right", rotation_mode="anchor")
    for lbl, k in zip(ax.get_xticklabels(), BAND + OTHER):
        lbl.set_color(COL[k] if k in BAND else "#888888")
    ax.axvline(len(BAND) - 0.5 + 0.42, color="#CCCCCC", lw=0.7, ls=":", zorder=1)
    ax.set_ylabel(ylab, fontsize=6.8, labelpad=2)
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.7, len(BAND) + 0.85 + len(OTHER) - 0.3)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.tick_params(axis="y", labelsize=6.0)
    C.all_spines(ax)

corpus_axis(axA, "A", r"force per step" "\n" r"$\bar{A}$ (nats step$^{-1}$)",
            2.72, connect=False)
axA.tick_params(axis="x", labelbottom=False)           # shared with the axis below
axA.text((len(BAND) - 1) / 2.0, 2.50, "constant force, rising current",
         ha="center", va="center", fontsize=5.8, color="#444444")
axA.text(len(BAND) + 0.85 + (len(OTHER) - 1) / 2.0, 2.50,
         "reference: force differs", ha="center", va="center",
         fontsize=5.8, color="#8A8A8A")
corpus_axis(axJ, "J", r"net current" "\n" r"$J_{\rm tot}$", 0.76, connect=True)

C.save(fig, "panel_4b_forcecurrent")
print("wrote panel_4b_forcecurrent | A over the band (point [95% CI]): " +
      "; ".join(f"{LABEL.get(k, k)} {MECH[k]['A']:.3f} "
                f"[{CI[k]['A_ci'][0]:.3f},{CI[k]['A_ci'][1]:.3f}]" for k in BAND))
