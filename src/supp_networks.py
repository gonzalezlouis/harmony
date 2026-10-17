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
