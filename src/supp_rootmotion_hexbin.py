import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import _figure_style as C

C.style()
DATADIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dat")
T = json.load(open(os.path.join(DATADIR, "rootmotion_2x2.json")))
Z = np.load(os.path.join(DATADIR, "rootmotion_2x2.npz"), allow_pickle=True)
arrow = Z["arrow"]
CLASSES = T["classes"]
X = {k: Z["x_%d" % i] for i, k in enumerate(CLASSES)}

XLAB = {"descending fifth": "descending-fifth share",
        "ascending fifth":  "ascending-fifth share",
        "step motion":      "step-motion share",
        "third motion":     "third-motion share"}

fig, axes = plt.subplots(2, 2, figsize=(6.90, 5.60), sharex=True, sharey=True)
axes = axes.ravel()

_xhi = max(float(X[k].max()) for k in CLASSES)
_xpad, _ypad = 0.03 * _xhi, 0.03 * (arrow.max() - arrow.min())
hb = None
for ax, k in zip(axes, CLASSES):
    ax.set_facecolor("#EFEFF2")
    hb = ax.hexbin(X[k], arrow, gridsize=46, cmap="magma_r", bins="log",
                   mincnt=1, linewidths=0, zorder=2,
                   extent=(0.0, _xhi, arrow.min(), arrow.max()))
    hb.set_rasterized(True)

    b = T["bins"][k]
    ax.fill_between(b["x"], b["q25"], b["q75"], facecolor="white", alpha=0.26,
                    edgecolor="white", linewidth=0.9, zorder=3)
    ax.plot(b["x"], b["med"], color="white", lw=1.9, zorder=4,
            path_effects=[pe.Stroke(linewidth=3.0, foreground="#222222"), pe.Normal()])
    ax.scatter(b["x"], b["med"], s=11, color="white", edgecolor="#222222",
               linewidth=0.55, zorder=5)
    ax.axhline(0.0, color="#555555", lw=0.8, ls=(0, (4, 2)), zorder=3.5)

    r = T["class_corr"][k]["pearson"]
    ax.text(0.965, 0.945, "$r=%+.2f$" % r, transform=ax.transAxes, ha="right",
            va="top", fontsize=7.4,
            bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                      edgecolor="#222222", linewidth=0.7))
    ax.set_xlabel(XLAB[k], fontsize=8.2)
    C.all_spines(ax)

for ax in axes:
    ax.set_xlim(-_xpad, _xhi + _xpad)
    ax.set_ylim(arrow.min() - _ypad, arrow.max() + _ypad)
fig.supylabel(r"mean affinity  $\langle A \rangle$  (nats step$^{-1}$)",
              fontsize=8.6, x=0.045)

cb = fig.colorbar(hb, ax=axes.tolist(), fraction=0.026, pad=0.016)
cb.set_label("pieces per cell", fontsize=7.4)
cb.ax.tick_params(labelsize=6.2)

C.save(fig, "panel_4c_rootmotion_2x2", aspect=C.SUPP_ASPECT)
print("wrote panel_4c_rootmotion_2x2 | n=%d" % T["n"])
for k in CLASSES:
    d = T["class_corr"][k]
    print("  %-18s pearson %+.4f [%+.4f,%+.4f]  spearman %+.4f [%+.4f,%+.4f]  share %.4f"
          % (k, d["pearson"], d["ci"][0], d["ci"][1],
             d["spearman"], d["sci"][0], d["sci"][1], d["mean_share"]))
print("  xlim %.3f..%.3f  ylim %.3f..%.3f (shared)"
      % (*axes[0].get_xlim(), *axes[0].get_ylim()))
