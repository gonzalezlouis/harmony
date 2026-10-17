import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import _figure_style as C

C.style()
DATADIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dat")
d = json.load(open(os.path.join(DATADIR, "corpus_directedness.json")))
NULL = json.load(open(os.path.join(DATADIR, "null_models.json")))["shuffled_null"]["per_corpus"]
order = ["Renaissance", "Baroque", "Classical", "Romantic", "Ragtime",
         "Jazz", "CoCoPops", "POP909"]
LABEL = {"CoCoPops": "Billboard pop", "POP909": "Mandopop"}
PERIOD = {"Renaissance": "c.1420\u20131590", "Baroque": "c.1680\u20131760",
          "Classical": "c.1760\u20131830", "Romantic": "c.1795\u20131850",
          "Ragtime": "1899\u20131917", "Jazz": "c.1920\u20131960",
          "CoCoPops": "1958\u20131991", "POP909": "c.1990\u20132010"}
COLS = {"Renaissance": "#3d8b6e", "Baroque": "#3a65b0", "Classical": "#5baabc",
        "Romantic": "#c9981a", "Ragtime": "#c04848",
        "CoCoPops": "#8e6fc4", "POP909": "#6a3d9a", "Jazz": "#4a2c7a"}
MODERN = {"CoCoPops", "POP909", "Jazz"}

fig, axa = plt.subplots(figsize=(4.96, 2.33))

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
_key = [plt.Line2D([], [], color="#555555", marker="o", ms=4.4, lw=1.2,
                   markeredgecolor="black", markeredgewidth=0.5, label=r"measured $\sigma$"),
        plt.Line2D([], [], color="none", marker="o", ms=3.4, markerfacecolor="white",
                   markeredgecolor="#555555", markeredgewidth=0.7, label="shuffled null")]
axa.legend(handles=_key, loc="upper left", fontsize=5.2, frameon=False,
           handlelength=1.1, handletextpad=0.4, labelspacing=0.25,
           borderpad=0.1, borderaxespad=0.25)
axa.set_ylabel(r"corpus-pooled $\sigma$" "\n" r"(nats step$^{-1}$, log)", fontsize=7.4)
axa.set_xticks(range(len(order)))
axa.set_xticklabels([f"{LABEL.get(k, k)}\n{PERIOD[k]}" for k in order], rotation=40,
                    ha="right", rotation_mode="anchor", fontsize=6.1)
axa.tick_params(axis="x", top=False, bottom=True, labeltop=False, labelbottom=True)
axa.set_xlim(-0.7, len(order) - 0.3)
axa.axvline(4.5, color="#999999", lw=0.8, ls=":", zorder=1)
axa.text(2.0, 2.9, "classical (score)", ha="center", va="center", fontsize=6.2,
         color="#666666", zorder=6)
axa.text(6.0, 2.9, "modern (labeled)", ha="center", va="center", fontsize=6.2,
         color="#666666", zorder=6)

C.all_spines(axa)
axa.yaxis.set_label_coords(-0.085, 0.5)
axa.set_xlabel("")

C.save(fig, "panel_2b_violins")
print("wrote panel_2b_stems | " + ", ".join(f"{k} {d[k]['pooled_sigma']:.2f}" for k in order))
