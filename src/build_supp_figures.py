#!/usr/bin/env python3
"""Build every supplementary figure from the frozen data

Rebuild every SI panel from the frozen data.

Every figure in `Supp Figures/` must come from this script, so the SI set is
reproducible and carries no orphan files.  Panels are defined in
`suppfig_panels.py`; this module only reconstructs their inputs and renders.
"""
import os, sys, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PANELS = HERE
sys.path.insert(0, PANELS)
import _figure_style as C
import supp_panel_library as SP
import matplotlib.pyplot as plt

_p = os.path.dirname(os.path.abspath(__file__))          # walk up to the project root
while not all(os.path.isdir(os.path.join(_p, _m)) for _m in ("dat", "src")):
    _p = os.path.dirname(_p)
_AA = _p                                              # project root

DATA = os.environ.get("SI_DATA", os.path.join(_p, "dat"))   # frozen json/npz
FLOOR = json.load(open(os.path.join(DATA, "bach_matched_floor.json")))
OUT  = os.environ.get("SI_OUT", os.path.join(_p, "fig", "panels"))
# Panels are components: they land in fig/panels. fig/supp holds the finalized
# set that SI.tex includes, so a rebuild here cannot silently change the paper.
L = lambda nm: json.load(open(os.path.join(DATA, nm)))

ND = L("null_distributions.json"); AG = L("aggregation_audit.json")
RV = L("results_values.json");     IP = L("irish_paired.json")
DEC = L("decoupling.json");        PC = L("provenance_calibration.json")
CHM = L("chordonomicon_mechanism.json")
DR = np.load(os.path.join(DATA, "null_draws.npz"))

ORD8 = ["Renaissance", "Baroque", "Classical", "Romantic", "Ragtime",
        "CoCoPops", "POP909", "Jazz"]
ORD_A = ["Renaissance", "Baroque", "Classical", "Romantic", "Late-Rom.",
         "CoCoPops", "POP909", "Jazz"]
DA = {"Late-Rom.": "Ragtime", "CoCoPops": "Billboard pop", "POP909": "Mandopop"}
S = RV["sigma"]

rows = [dict(corpus=k, **ND["per_corpus"][k]) for k in ORD8 if k in ND["per_corpus"]]
DRAWS = {r["corpus"]: DR[r["corpus"]] for r in rows if r["corpus"] in DR.files}
NT = np.array([r["n_trans"] for r in rows], float)
MEANS = np.array([r["mean"] for r in rows], float)
SL = ND["floor_scaling"]["log10_slope"]
IC = float(np.mean(np.log10(MEANS) - SL * np.log10(NT)))   # intercept at the frozen slope

# The corpus level is taken from the type-based recomputation, not from the stored
# record: that file holds a state-based value at the corpus level while its own
# within-tune level is type-based, so plotting both from it mixed two estimators.
_IT = json.load(open(os.path.join(DATA, "irish_type_based.json")))
CO = {"functional_h": _IT["functional"]["sigma"], "modal_h": _IT["modal"]["sigma"],
      "ratio": _IT["ratio"]["value"], "n_functional": _IT["functional"]["n"],
      "n_modal": _IT["modal"]["n"]}
WT = json.load(open(os.path.join(DATA, "irish_within_tune.json")))
EMP = IP["levels"]["exact_melody_pairs"]
CH = {g: {"sigma": v["sigma"], "sd": v["sd"]} for g, v in CHM["per_genre"].items()}
ROB, ERAS4 = {}, ["Renaissance", "Baroque", "Classical", "Romantic"]
for key, v in PC["per_era_dataset"].items():
    era, ds = key.split("|")
    ROB[(era, ds)] = dict(n=v["n"], sigma=v["sigma"], ci=tuple(v["ci95"]))

# --- data for every panel -------------------------------------------------
# Per-setting Irish arrays, recomputed from irish_tunes.json through the project's
# own _abc_chords loader and compute_mechanism.sigma_invariant, the type-based
# estimator whose pooled values reproduce irish_type_based.json exactly. The four
# Irish panels need per-setting distributions and only summaries were frozen.
_PS = np.load(os.path.join(DATA, "irish_per_setting.npz"))
PT, NTV = _PS["pt"], _PS["ntv"]
PER_SET = {"F": _PS["F"], "M": _PS["M"]}
PDIFF, POOLED = _PS["pdiff"], float(_PS["pooled"])
SF, SM = float(_PS["sf"]), float(_PS["sm"])

MECH = CHM["per_genre"]
R_, P_ = (float(x) for x in CHM["pearson"])   # [r, p] across the eleven genres

_CR = L("composer_robustness.json")
JCI = _CR["jackknife"]
ERAS = [e for e in ERAS4 if e in JCI]

_RM = L("rootmotion_2x2.json")
PI = _RM["per_interval"]
IVN = {0: "0", 1: "m2", 2: "M2", 3: "m3", 4: "M3", 5: "5th dn", 6: "TT",
       7: "5th up", 8: "m6", 9: "M6", 10: "m7", 11: "M7"}
# Keyed by int: p_g1 does IVN[int(k)]. Short forms because the ticks are rotated 45
# degrees and the spelled-out names collided with each other and with the y axis.

C.style()
made, failed = [], []
# EVERY builder in supp_panel_library runs. Nothing is withheld from the supplement.
JOBS = [("p_a1_surrogates", (rows, DRAWS)), ("p_a2_floor", (rows, NT, SL, IC)),
        ("p_a3_floor_signal", (AG, ORD_A, DA)), ("p_a4_bias", (AG, S, ORD8)),
        ("p_a5_per_tune", (PT,)), ("p_a6_zero_kinds", (PT, NTV)),
        ("p_a7_resolvability", (PT, NTV)),
        ("p_e2_decoupling", (DEC, S)), ("p_e1_chordonomicon", (CH,)),
        ("p_e1b_chord_mechanism", (MECH, R_, P_)),
        ("p_b1_mode_levels", (CO, WT)), ("p_b1_setting_dist", (PER_SET, SF, SM)),
        ("p_b2_paired", (WT,)), ("p_b2_tune_differences", (PDIFF, POOLED)),
        ("p_d1_jackknife", (JCI, ERAS)), ("p_g1_rootmotion_sweep", (PI, IVN)),
        ("p_b3_exact_pairs", (EMP,)), ("p_f1_provenance", (FLOOR,))]
for fn, args in JOBS:
    plt.close("all")
    try:
        nm = getattr(SP, fn)(*args, OUT)
        made.append(nm)
    except Exception as e:
        failed.append((fn, f"{type(e).__name__}: {e}"))
print("MADE:", made)
print("FAILED:", failed)
