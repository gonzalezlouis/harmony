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
    lo, hi = CI[k][key + "_ci"]
    v = MECH[k][key]
    return np.array([[max(v - lo, 0.0)], [max(hi - v, 0.0)]])

BAND = ["Classical", "Romantic", "CoCoPops", "POP909"]
OTHER = ["Renaissance", "Baroque", "Ragtime", "Jazz"]
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
axA.tick_params(axis="x", labelbottom=False)
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
