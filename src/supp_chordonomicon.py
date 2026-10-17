#!/usr/bin/env python3
"""Supplementary — Chordonomicon modern-genre breadth

Fig 2b - Chordonomicon modern-genre breadth (SEPARATE, RELATIVE-only).

Chordonomicon is crowd-transcribed and triad-flattened (dom7 3.7-4.0% vs the
expert CoCoPops 9.4% on the same repertoire), so its sigma is a lower bound of
unknown tightness -- it is NEVER placed on the magnitude axis with the expert
corpora.  This panel makes only the WITHIN-corpus relative statement: sigma
tracks harmonic richness (jazz/soul high, punk/rock low).  Error bars are
statistical bootstrap SD only; no label correction (the flattened fraction is
not identifiable, so a "restored" bar would be fabricated)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scienceplots
import matplotlib.pyplot as plt
import _figure_style as C

plt.style.use('science')

C.style()
g = C.load()["chordonomicon"]
order = sorted(g, key=lambda k: -g[k]["sigma"])
y = [g[k]["sigma"] for k in order]
e = [g[k]["sd"] for k in order]

fig, ax = plt.subplots(figsize=(4.6, 2.9))
ax.bar(range(len(order)), y, yerr=e, color=C.MODCOL, width=0.72,
       capsize=2, error_kw=dict(lw=0.7), zorder=3)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=45, ha="right", fontsize=6)
ax.set_ylabel(r"relative $\sigma$ (nats step$^{-1}$)")
C.save(fig, "panel_2b_chordonomicon", aspect=C.SUPP_ASPECT)
print("wrote panel_2b_chordonomicon")
