import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import _figure_style as C

C.style()
DATADIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dat")
T = json.load(open(os.path.join(DATADIR, "emergent_full.json")))
Z = np.load(os.path.join(DATADIR, "emergent_full.npz"), allow_pickle=True)
i5, arrow, pipe = Z["i5"], Z["arrow"], Z["pipe"]

PCOL = {"score / expert": "#2f6f9f", "chordified": "#c98a2e"}

fig, ax = plt.subplots(figsize=(3.60, 3.10))

ax.set_facecolor("#EFEFF2")
hb = ax.hexbin(i5, arrow, gridsize=52, cmap="magma_r", bins="log",
               mincnt=1, linewidths=0, zorder=2)
hb.set_rasterized(True)
cb = fig.colorbar(hb, ax=ax, fraction=0.043, pad=0.02)
cb.set_label("pieces per cell", fontsize=7)
cb.ax.tick_params(labelsize=6)

b = T["bins"]
ax.fill_between(b["x"], b["q25"], b["q75"], facecolor="white", alpha=0.26,
                edgecolor="white", linewidth=0.9, zorder=3)
ax.plot(b["x"], b["med"], color="white", lw=2.0, zorder=4,
        path_effects=[pe.Stroke(linewidth=3.1, foreground="#222222"), pe.Normal()])
ax.scatter(b["x"], b["med"], s=13, color="white", edgecolor="#222222",
           linewidth=0.6, zorder=5)
ax.axhline(0.0, color="#555555", lw=0.8, ls=(0, (4, 2)), zorder=3.5)

CENT = json.load(open(os.path.join(DATADIR, "corpus_centroids.json")))
for k in C.CORPUS_ORDER:
    c = CENT[k]
    ax.plot(c["fifth"], c["arrow"], marker=C.CORPUS_MARKER[k], ls="none",
            markersize=8.5 if C.CORPUS_MARKER[k] == "*" else 6.0,
            markerfacecolor=C.CORPUS_COL[k], markeredgecolor="#FFFFFF",
            markeredgewidth=0.9, zorder=6,
            path_effects=[pe.withStroke(linewidth=1.1, foreground="#111111")])
    ax.errorbar(c["fifth"], c["arrow"],
                xerr=[[c["fifth"] - c["fifth_ci"][0]], [c["fifth_ci"][1] - c["fifth"]]],
                yerr=[[c["arrow"] - c["arrow_ci"][0]], [c["arrow_ci"][1] - c["arrow"]]],
                fmt="none", ecolor="#111111", elinewidth=0.55, capsize=1.3,
                capthick=0.55, zorder=8)

_xpad = 0.03 * (i5.max() - i5.min())
_ypad = 0.03 * (arrow.max() - arrow.min())
ax.set_xlim(i5.min() - _xpad, i5.max() + _xpad)
ax.set_ylim(arrow.min() - _ypad, arrow.max() + _ypad)
ax.set_xlabel("descending-fifth share of chord transitions", fontsize=8.5)
ax.set_ylabel(r"mean affinity  $\langle A \rangle$  (nats step$^{-1}$)", fontsize=8.5)

_med_h = plt.Line2D([0], [0], color="white", lw=2.0,
                    path_effects=[pe.Stroke(linewidth=3.1, foreground="#222222"),
                                  pe.Normal()],
                    label="median, IQR  (n=%s)" % f"{T['n']:,}")
_corp_h = [plt.Line2D([0], [0], marker=C.CORPUS_MARKER[k], ls="none",
                      markersize=7.5 if C.CORPUS_MARKER[k] == "*" else 5.4,
                      markerfacecolor=C.CORPUS_COL[k], markeredgecolor="#111111",
                      markeredgewidth=0.7, label=C.CORPUS_LABEL.get(k, k))
           for k in C.CORPUS_ORDER]
leg = ax.legend(handles=[_med_h] + _corp_h, fontsize=5.9, frameon=True,
                loc="lower right", ncol=2, handletextpad=0.4, columnspacing=0.8,
                labelspacing=0.26, borderpad=0.35)
leg.get_frame().set_edgecolor("#222222")
leg.get_frame().set_linewidth(0.8)
leg.get_frame().set_facecolor("white")
leg.get_frame().set_alpha(1.0)
leg.set_zorder(8)
C.all_spines(ax)

axi = ax.inset_axes([0.135, 0.605, 0.315, 0.345])
axi.set_facecolor("#FFFFFF")
for k in C.CORPUS_ORDER:
    c = CENT[k]
    axi.plot(c["fifth"], c["arrow"], marker=C.CORPUS_MARKER[k], ls="none",
             markersize=8.0 if C.CORPUS_MARKER[k] == "*" else 5.6,
             markerfacecolor=C.CORPUS_COL[k], markeredgecolor="#FFFFFF",
             markeredgewidth=0.8, zorder=6,
             path_effects=[pe.withStroke(linewidth=1.0, foreground="#111111")])
    axi.errorbar(c["fifth"], c["arrow"],
                 xerr=[[c["fifth"] - c["fifth_ci"][0]], [c["fifth_ci"][1] - c["fifth"]]],
                 yerr=[[c["arrow"] - c["arrow_ci"][0]], [c["arrow_ci"][1] - c["arrow"]]],
                 fmt="none", ecolor="#111111", elinewidth=0.55, capsize=1.2,
                 capthick=0.55, zorder=8)
axi.set_xlim(0.163, 0.325)
axi.set_ylim(-0.008, 0.158)
axi.set_xticks([0.20, 0.25, 0.30]); axi.set_yticks([0.00, 0.05, 0.10, 0.15])
axi.tick_params(labelsize=5.0, length=1.6, pad=0.8, top=False, right=False)
axi.set_xlabel(""); axi.set_ylabel("")
C.all_spines(axi)
for s in axi.spines.values():
    s.set_linewidth(0.55)
ax.indicate_inset_zoom(axi, edgecolor="#111111", linewidth=0.6, alpha=0.85)

C.save(fig, "panel_4c_scaling")
print("wrote panel_4c_scaling | n=%d pearson=%+.4f spearman=%+.4f"
      % (T["n"], T["pearson"], T["spearman"]))
print("  xlim %.4f..%.4f  ylim %.4f..%.4f (data extent, 3%% pad)"
      % (*ax.get_xlim(), *ax.get_ylim()))
print("  centroids: " + "  ".join("%s(%.3f,%.3f)" % (k, CENT[k]["fifth"], CENT[k]["arrow"])
                                  for k in C.CORPUS_ORDER))
