"""Supplementary network figures on the circle of fifths

Three figures, all through fig2c_networks.render so the wheel geometry is identical
to main Fig. 2c and the panels are directly comparable:

  suppfig_networks_unshown   Classical, Ragtime, Billboard pop -- the only three
                             corpora with no network anywhere in the main text.
  suppfig_2c_networks_all6   the six-corpus sheet SI.tex has always included.
  fig2c_null_network         the symmetrized null, sigma = 0 by construction.

They are written here rather than ad hoc so the supplement can be rebuilt from
src/ alone, and they pick up the single supplementary aspect ratio from
fig2c_networks' save path.
"""
import os, sys, matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
AA = os.path.dirname(HERE)
OUT = os.path.join(AA, "fig", "supp")

import _figure_style as C
import fig2c_networks as FN
import supp_aligned_panels as SAP

UNSHOWN = ["Classical", "Ragtime", "CoCoPops"]
ALL6 = ["Baroque", "Classical", "Romantic", "CoCoPops", "POP909", "Jazz"]

if __name__ == "__main__":
    C.style()
    ALL = SAP.pools()
    made = []
    for name, keys in [("suppfig_networks_unshown", UNSHOWN),
                       ("suppfig_2c_networks_all6", ALL6)]:
        items = [(k, ALL[k], C.CORPUS_COL[k]) for k in keys]
        FN.render(items, os.path.join(OUT, name))
        made.append(name)
    print("MADE:", made)
    print("corpora with no network in the main text:", UNSHOWN,
          "-- main Fig. 2c shows null, Renaissance, Baroque, Romantic, Jazz, Mandopop")
