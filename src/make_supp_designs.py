import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots

import estimator as ha
import _figure_style as _FS

HERE    = os.path.dirname(os.path.abspath(__file__))
FIGDIR  = os.path.join(HERE, "figs")
DATADIR = os.path.join(os.path.dirname(HERE), "dat")
os.makedirs(FIGDIR, exist_ok=True)

CROWD_GENRES = ["pop", "rock", "country", "alternative", "punk", "metal",
                "rap", "soul", "jazz", "electronic", "reggae"]
ENGINE_ERAS  = ["Baroque", "Classical", "Romantic", "Modern"]


def set_style():
    plt.style.use(["science", "nature"])
    mpl.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, "font.size": 8,
        "axes.labelsize": 8, "legend.fontsize": 6,
        "xtick.labelsize": 6.5, "ytick.labelsize": 7, "text.usetex": False,
    })


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), dpi=300,
                    bbox_inches=_FS.fixed_aspect_bbox(fig))
    plt.close(fig)


def _modern_reference_pools():
    import zipfile, json, ntpath, posixpath
    import pandas as pd
    recs = pd.read_parquet(ha.ALL_RECORDS)
    bn = lambda p: ntpath.basename(p) if "\\" in p else posixpath.basename(p)
    ds_of = {bn(r.midi_path): r.dataset for r in recs.itertuples()}
    zf = zipfile.ZipFile(ha.SEQUENCES_ZIP)
    pools, seen = {}, set()
    for name in zf.namelist():
        if not name.endswith(".json") or name.startswith("__MACOSX"):
            continue
        try:
            d = json.loads(zf.read(name))
        except Exception:
            continue
        k = bn(d["midi_path"]); ds = ds_of.get(k)
        if ds not in ("POP909", "Jazz") or k in seen:
            continue
        seen.add(k)
        s = ha.drop_repeats([e[0] for e in d["sequence"] if e[0] != "REST"])
        if len(s) >= ha.MIN_LEN:
            pools.setdefault(ds, []).append(s)
    return pools


def compute():
    print("pooling six datasets (dedup, Polish dropped) ...")
    era_pool = ha.load_all_datasets(dedup=True, drop_polish=True)
    eras = [e for e in ha.ERA_ORDER if e in era_pool]

    R = {"trajectory": {}, "multi_order": {}, "engine": {}, "concentration": {},
         "enrichment": {}, "palindrome": {}, "irish": {}, "chordonomicon": {}}

    print("trajectory + bootstrap error bars ...")
    for e in eras:
        sig = ha.sigma_pooled(era_pool[e])
        lo, hi, sd = ha.boot_sigma_ci(era_pool[e], B=300, seed=hash(e) % 997)
        R["trajectory"][e] = {"sigma": sig, "lo": lo, "hi": hi, "sd": sd,
                              "n": len(era_pool[e])}

    print("multi-order sigma_k (root alphabet) ...")
    for e in eras:
        R["multi_order"][e] = ha.multi_order_sigma(era_pool[e], orders=(2, 3, 4))

    print("engine decomposition ...")
    for e in ENGINE_ERAS:
        R["engine"][e] = ha.collapse_engines(ha.engine_decomposition(era_pool[e])[0])

    print("concentration + enrichment ...")
    for e in ENGINE_ERAS:
        R["concentration"][e] = list(ha.edge_concentration(era_pool[e])["cum"])
    for e in eras:
        R["enrichment"][e] = ha.descending_fifth_share(era_pool[e])["enrichment"]

    print("palindrome test (Modern) ...")
    R["palindrome"] = ha.palindrome_test(era_pool["Modern"], seed=909)

    print("Irish harmony vs melody ...")
    irish = ha.load_irish_by_mode()
    for mc, d in irish.items():
        R["irish"][mc] = {"h": ha.sigma_pooled(d["harmonic"]),
                          "m": ha.sigma_pooled(d["melodic"])}

    print("Chordonomicon by genre (bootstrap error bars; sigma is a lower bound) ...")
    crowd = ha.load_chordonomicon_by_genre(CROWD_GENRES, sample=1500)
    for g, seqs in crowd.items():
        m = ha.sigma_pooled(seqs)
        sd = ha.boot_sigma_ci(seqs, B=150, seed=1)[2]
        R["chordonomicon"][g] = {"sigma": m, "err_lo": sd, "err_hi": sd}

    def _clean(o):
        if isinstance(o, dict):  return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, np.generic): return o.item()
        return o
    with open(os.path.join(DATADIR, "arc_results.json"), "w") as f:
        json.dump(_clean(R), f, indent=2)
    return R


def fig_trajectory(R):
    eras = [e for e in ha.ERA_ORDER if e in R["trajectory"]]
    t = R["trajectory"]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    y = [t[e]["sigma"] for e in eras]
    err = [t[e]["sd"] for e in eras]
    ax.bar(range(len(eras)), y, yerr=err, color=[ha.ERA_COL[e] for e in eras],
           width=0.72, capsize=2, error_kw=dict(lw=0.7), zorder=3)
    ax.set_xticks(range(len(eras)))
    ax.set_xticklabels([e.replace("-", "-\n") for e in eras], rotation=45, ha="right", fontsize=6)
    ax.set_ylabel(r"harmonic irreversibility $\sigma$ (nats/step)")
    save(fig, "fig_trajectory")


def fig_multiorder(R):
    eras = [e for e in ha.ERA_ORDER if e in R["multi_order"]]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for e in eras:
        ax.plot([2, 3, 4], R["multi_order"][e], "-o", color=ha.ERA_COL[e], ms=4, lw=1.6, label=e, zorder=3)
    ax.set_xlabel(r"word order $k$"); ax.set_ylabel(r"null-corrected $\sigma_k$ (nats)")
    ax.set_xticks([2, 3, 4]); ax.legend(frameon=False, loc="upper left")
    save(fig, "fig_multiorder")


def fig_engine_eras(R):
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    x = np.arange(len(ENGINE_ERAS)); bottom = np.zeros(len(ENGINE_ERAS))
    for eng, col in zip(ha.ENGINE_ORDER, ha.ENGINE_COL):
        vals = [R["engine"][e][eng] for e in ENGINE_ERAS]
        ax.bar(x, vals, bottom=bottom, color=col, width=0.72, label=eng, zorder=3)
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels(ENGINE_ERAS)
    ax.set_ylabel(r"$\sigma$ carried by engine (nats/step)")
    ax.legend(frameon=False, ncol=2, handlelength=0.9, columnspacing=0.8)
    save(fig, "fig_engine_eras")


def fig_palindrome(R):
    p = R["palindrome"]
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    vv = [p["real"], p["reversed"], p["palindromized"], p["shuffled"]]
    ax.bar(range(4), vv, color=[ha.ERA_COL["Modern"], "#b9a2cf", "#c9ccd1", "#9aa0a6"],
           width=0.66, zorder=3)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["real", "time-\nreversed", "palin-\ndromized", "shuffled\nnull"])
    ax.set_ylabel(r"$\sigma$ (nats/step)")
    save(fig, "fig_palindrome")


def fig_concentration(R):
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for e in ENGINE_ERAS:
        cum = np.array(R["concentration"][e])
        ax.plot(np.arange(1, len(cum) + 1) / len(cum), cum, lw=1.8, color=ha.ERA_COL[e], label=e, zorder=3)
    ax.plot([0, 1], [0, 1], "k:", lw=0.7)
    ax.set_xlabel("fraction of edges (ranked)"); ax.set_ylabel(r"cumulative share of $\sigma$")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(frameon=False, loc="lower right")
    save(fig, "fig_concentration")


def fig_enrichment(R):
    eras = [e for e in ha.ERA_ORDER if e in R["enrichment"]]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.bar(range(len(eras)), [R["enrichment"][e] for e in eras],
           color=[ha.ERA_COL[e] for e in eras], width=0.72, zorder=3)
    ax.axhline(1, color="k", lw=0.7, ls="--")
    ax.set_xticks(range(len(eras)))
    ax.set_xticklabels([e.replace("-", "-\n") for e in eras], rotation=45, ha="right", fontsize=6)
    ax.set_ylabel(r"descending-fifth enrichment")
    save(fig, "fig_enrichment")


def fig_irish(R):
    mm = ["Major (functional)", "Dorian/Mixolydian (modal)", "Minor"]
    labels = ["Major\n(functional)", "Dorian / Mixolydian\n(modal, non-functional)", "Minor\n(functional)"]
    fig, ax = plt.subplots(figsize=(3.8, 3.0)); xx = np.arange(3); w = 0.36
    ax.bar(xx - w / 2, [R["irish"][m]["h"] for m in mm], w, color=ha.QUAL_COL["major"], label="harmonic", zorder=3)
    ax.bar(xx + w / 2, [R["irish"][m]["m"] for m in mm], w, color="#C8860A", label="melodic", zorder=3)
    ax.set_xticks(xx); ax.set_xticklabels(labels, fontsize=5.6)
    ax.set_ylabel(r"$\sigma$ (nats/step)"); ax.legend(frameon=False)
    save(fig, "fig_irish")


def fig_chordonomicon(R):
    g = R["chordonomicon"]
    order = sorted(g, key=lambda k: -g[k]["sigma"])
    y   = [g[k]["sigma"] for k in order]
    err = [g[k]["err_lo"] for k in order]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.bar(range(len(order)), y, yerr=err, color=ha.ERA_COL["Modern"],
           width=0.72, capsize=2, error_kw=dict(lw=0.7), zorder=3)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel(r"harmonic irreversibility $\sigma$ (nats/step)")
    save(fig, "fig_chordonomicon")


def main():
    set_style()
    R = compute()
    for fn in (fig_trajectory, fig_multiorder, fig_engine_eras, fig_palindrome,
               fig_concentration, fig_enrichment, fig_irish, fig_chordonomicon):
        fn(R)
        print("wrote", fn.__name__)
    print(f"\nAll 8 figures -> {FIGDIR}")
    print("trajectory:", {e: round(R['trajectory'][e]['sigma'], 3) for e in R['trajectory']})


if __name__ == "__main__":
    main()
