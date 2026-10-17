# Code audit

Three questions, answered against the 28 modules in `src/` as packaged: is the commentary
excessive, can every number and every input be traced to code, and does the code stand on
minimal libraries with the method written out rather than delegated?

## 1. Comment and docstring density

6,029 lines total: 3,183 code, 799 comment, 1,333 docstring, 714 blank. Prose is **40% of
non-blank lines**.

| Module | Lines | Code | Comment | Docstring | Prose share |
|---|---|---|---|---|---|
| `fig3c_fifth_share_vs_arrow.py` | 290 | 77 | 79 | 105 | 70% |
| `config.py` | 62 | 15 | 24 | 11 | 70% |
| `fig2b_directedness_and_null.py` | 155 | 43 | 31 | 61 | 68% |
| `supp_rootmotion_hexbin.py` | 170 | 49 | 20 | 80 | 67% |
| `fig3a_current_matrices.py` | 169 | 55 | 45 | 50 | 63% |
| `fig2c_null_network.py` | 86 | 28 | 6 | 37 | 60% |
| `fig3_concentration_matrices.py` | 120 | 44 | 8 | 50 | 56% |
| `irreversibility_v2.py` | 304 | 129 | 16 | 108 | 49% |
| `fig4_irish_loop.py` | 163 | 73 | 7 | 55 | 45% |
| `fig2c_networks.py` | 461 | 241 | 152 | 46 | 45% |
| `_figure_style.py` | 179 | 89 | 34 | 34 | 43% |
| `supp_networks.py` | 42 | 20 | 0 | 14 | 41% |
| `fig3b_force_and_current.py` | 109 | 57 | 10 | 29 | 40% |
| `fig4a_irish_matrices.py` | 210 | 108 | 33 | 40 | 40% |
| `supp_chordonomicon.py` | 36 | 18 | 1 | 11 | 40% |
| `fig4a_irish_ladder.py` | 186 | 96 | 15 | 48 | 39% |
| `fig4_irish_mode_contrast.py` | 197 | 103 | 15 | 52 | 39% |
| `fig2a_composer_sigma.py` | 260 | 137 | 43 | 45 | 39% |
| `estimator.py` | 499 | 281 | 55 | 114 | 37% |
| `fig1b_state_space.py` | 227 | 128 | 27 | 41 | 34% |
| `compute_mechanism.py` | 370 | 230 | 33 | 62 | 29% |
| `compute_corpus_sigma.py` | 121 | 73 | 8 | 22 | 29% |
| `build_supp_figures.py` | 114 | 72 | 19 | 8 | 27% |
| `supp_panel_library.py` | 877 | 590 | 94 | 126 | 27% |
| `supp_aligned_panels.py` | 147 | 95 | 5 | 23 | 22% |
| `make_supp_designs.py` | 267 | 177 | 17 | 28 | 20% |
| `canonical_pools.py` | 100 | 75 | 0 | 17 | 18% |
| `supp_engine_decomposition.py` | 108 | 80 | 2 | 16 | 18% |

The density is high but it is not filler. The long headers are decision records: what the
plotted quantity is and why it carries the name it does, which earlier version of the panel
was wrong and by how much, which exclusions were applied and which were dropped as
unjustified, and what the axis limits are taken from. `fig3c_fifth_share_vs_arrow.py`'s
105-line header, the longest in the repository, states the exact additive decomposition that
licenses its y axis (pooled sigma recovered to 1.9e-7, 0.049821296 against 0.049821104),
records that earlier limits cropped 58 pieces out of frame, and explains why the full corpus
is correct here where era comparisons require the restricted set. That is the material a
reviewer asks for; deleting it would make the module shorter and less reproducible.

Three observations worth acting on, none of them about volume:

- **Stale paths, now fixed.** 18 docstring references pointed at `../data/`, the pre-rename location; the code reads `dat/`. All 13 affected lines across 8 modules were corrected in this package. None was executable code.
- **One cited file is not in `dat/`.** `fig3c_fifth_share_vs_arrow.py` cites `dat/coverage.json` as where a pipeline bug was logged. That file is in `archive/unreferenced_data/`, not in the project data directory.
- **`config.py` is 70% prose and its paths are wrong.** `DATA_DIR`, `CACHE_DIR` and `FIG_DIR` resolve to `src/data`, `src/cache` and `src/figures`, not the project root, and the docstring describes a layout that no longer holds. No module uses them to reach data; the constants below them are live. The prose, not the code, is what misleads here.

## 2. Black boxes

Clean on every axis of opaque execution:

| Check | Result |
|---|---|
| Bare `except:` or `except ...: pass` | none |
| `eval` / `exec` | none |
| `pickle` / `joblib` model loads | none (seven `np.load` calls, 5 of them with `allow_pickle=True`, all on `.npz` files produced by this project) |
| Network calls at analysis time | none |
| Unseeded randomness | none — every sampling path takes `seed=` (0 or 1) or `config.RNG_SEED`; `shuffle_seqs` receives the generator from its caller |
| Undocumented magic numbers | none found: `MARKER_SCALE`'s eight multipliers carry the measurement that produced them (ink area against the circle at ms=14, 200 dpi), and `RING_R`, `ERA_COL_FRACTION` and similar are layout constants |
| Hardcoded results standing in for computation | none: the Table 1 values in `canonical_pools.py` (`T1S`) and `compute_mechanism.py` (`CANON`) are assertion targets, checked to 6e-4 and 5e-4, and `compute_mechanism.py` asserts its Pearson and Spearman to 1e-12 against stored values |

One real opacity in the code itself: `canonical_pools.py:69` writes its self-check to
`os.environ["OUTJ"]` with no default and no mention of the variable in the docstring, so
running the module with `OUTJ` unset raises `KeyError` after the computation has finished.

### The black box is upstream of the code, in the data

30 of the 43 deposited data files are read by the 28 modules. **Six are produced by code in this repository:**

| File | Produced by |
|---|---|
| `arc_results.json` | `make_supp_designs.py:152` |
| `corpus_centroids.json` | `compute_mechanism.py:351` |
| `final_data.json` | `compute_corpus_sigma.py:115` |
| `rootmotion_2x2.json`, `rootmotion_2x2.npz` | `compute_mechanism.py:296-297` |
| `transfer_matrix_canonical.npz` | `supp_engine_decomposition.py:65` |

The other 24 arrive frozen. They split in two:

**Produced by scripts that exist but were not packaged (3).** `all_records_usable.parquet`,
and the per-dataset parquet and sequence caches behind it, come from the extraction layer
named in §1 of `DOCUMENTATION.md`: `pipeline.py`, `datasets/{kernscores,asap,giantmidi,pop909,jazz}.py`,
`merge_datasets.py`, `extraction.py`, `transitions.py`, `irreversibility.py`, `era.py`,
`state_space.py`. Every one of those is in the project under `archive/harmony-epr/` and
`archive/src 2/`. `merge_datasets.py:199` builds the usable filename with
`out.replace(".parquet", "_usable.parquet")`, which is why no literal search finds its
producer. Restoring that layer would close the corpora-to-sequences gap.

**No producer anywhere (21).** `mechanism.json`, `mechanism_ci.json`, `null_models.json`,
`null_distributions.json`, `null_draws.npz`, `corpus_directedness.json`, `results_values.json`,
`decoupling.json`, `emergent_full.json`, `emergent_full.npz`, `aggregation_audit.json`,
`bach_matched_floor.json`, `chordonomicon_mechanism.json`, `provenance_calibration.json`,
`irish_tunes.json`, `irish_examples.json`, `irish_expansion.json`, `irish_paired.json`,
`irish_per_setting.npz`, `irish_type_based.json`, `irish_within_tune.json`. No write site for
any of them exists in the project's 221 Python files — 28 in `src/`, 193 under `archive/` — nor
in the 26 modules inside `archive/harmony-epr.zip`. The four notebooks in the tree belong to
the ASAP dataset. These files are the repository's only genuine black box: a reader can plot
from them and can check them against the assertions, but cannot rebuild them.

`DATA.md` carries this classification per file.

## 3. Dependency surface — what is delegated, and to what

Every import across the 28 modules, with what it is used for:

| Import | Modules | Used for | Verdict |
|---|---|---|---|
| `numpy` | all | the arithmetic itself — counts, matrices, currents, affinities | the from-scratch layer |
| `matplotlib` | all figure modules | drawing | rendering |
| `scipy` | 2 (`compute_mechanism`, `supp_panel_library`) | `pearsonr` and `spearmanr`, 12 invocations: 11 in `compute_mechanism.py` (lines 218, 241-242, 272-273, 278-279, 281, 283, 293-294) and 1 at `supp_panel_library.py:613` | two correlation coefficients, nothing else |
| `pandas` | 6 | `read_parquet` (7 calls), `read_csv` (1) | I/O only — no `groupby`, `merge`, `apply` or `agg` anywhere |
| `scienceplots` | 3 | the matplotlib style sheet | styling |
| `networkx` | 1 (`irreversibility_v2.flux_peel`) | `find_cycle` while peeling flux | graph bookkeeping; the thermodynamics around it is explicit |
| `verovio`, `cairosvg`, `pypdf`, `pypdfium2` | 1 (`fig4_irish_mode_contrast`) | engraving two music excerpts to vector PDF | rendering a figure illustration |
| stdlib: `os`, `re`, `glob`, `sys`, `json`, `zipfile`, `collections`, `pathlib`, `ntpath`/`posixpath` | — | paths, parsing, the sequence archive, cross-platform basenames | — |

Three things are notable by their absence. **No `seaborn`**, so no statistical plot carries a
hidden bandwidth or confidence band. **No `sklearn`, `statsmodels`, `gaussian_kde`,
`curve_fit`, spline or lowess** — a search for all of them returns nothing; the only fit in
the repository is `np.polyfit(x, y, 1)` in one supplementary panel, a least-squares line.
And **no `music21` or `pretty_midi`**: those belong to the extraction layer that turns scores
into chord sequences, which is not part of this package, so nothing here depends on a
musicological parser at analysis time.

`estimator.py` imported `zlib` without using it; the import was dropped in this package.

### The core is written out

The estimator is twelve lines of arithmetic over a count dictionary — no library stands
between the reader and sigma = ½ Σ A(t)J(t):

```python
def sigma_invariant(seqs):
    Ct = invariant_counts(seqs); N = sum(Ct.values())
    sig = jtot = 0.0
    for t, f in Ct.items():
        r = Ct.get(_rev(t), 0)
        if r > 0:
            J = (f - r) / N
            sig += 0.5 * np.log(f / r) * J
            if J > 0: jtot += J
    return sig, jtot
```

The matrix form in `irreversibility_v2.flux_peel` is the same identity written in numpy:
`P` from row-normalised counts, `pi = stationary(P)`, `flow = pi[:, None] * P`,
`J = flow - flow.T`, `A = log(flow) - log(flow).T` on bidirectional edges,
`sigma = 0.5 * sum(J * A)`. `stationary(P)` is the dominant left eigenvector via
`np.linalg.eig(P.T)` with a power-iteration fallback. Partialling out in
`compute_mechanism.py` is `np.linalg.lstsq` residuals, written in two lines rather than
handed to a regression package.

### Readability

148 functions, median length 13 lines. Only two exceed 120: `fig2c_networks.render` (202)
and `compute_mechanism.main` (145), both composition rather than algorithm —
a drawing routine and a script body. 22 lines exceed 100 characters, across 8 modules. The
symbol names are the field's own — `C` counts, `P` transition matrix, `pi` stationary
distribution, `J` current, `A` affinity — which is what makes the matrix lines legible at a
glance without reading the comment above them.

## Method

Comment and docstring lines were counted with `tokenize` and `ast` rather than by pattern, so
a `#` inside a string is not a comment and a module docstring is not code. Library-use counts
are invocations at distinct line numbers, not import statements. The producer map
was built by locating every write call (`json.dump`, `np.savez`, `np.savez_compressed`,
`to_parquet`, `open(..., 'w')`) and resolving its target through the filename literal at the
call site, through a variable bound to a path containing that literal, and — for the parquet
layer — by reading the construction by hand.
