# Harmony: entropy production of harmonic motion — analysis code

The analysis layer behind the measurement of the Schnakenberg entropy production rate of
harmonic motion in notated and annotated music corpora. Each work is reduced to its chord
sequence, bigram transitions are counted by transposition-invariant type, and the estimator
returns sigma in nats per chord change — a physical rate, not a comparative index. The eight
corpus values in `results_values.json` match the paper's Table 1 to three decimals and span
36.9-fold.

This repository is the code only — 28 modules. No data, no manuscript sources, no rendered
figures: every table and every plot is produced by running the module named for it, against
the data deposit described in `DATA.md`.

## Running it

```bash
pip install numpy scipy pandas matplotlib scienceplots networkx pyarrow
cd src
python fig3a_current_matrices.py     # any module runs directly, from src/
```

That is what the modules actually import — `pandas` only to read parquet, `networkx` only for
one cycle search. `fig4_irish_mode_contrast.py` additionally needs `verovio`, `cairosvg`,
`pypdf` and `pypdfium2` to engrave its music excerpts; no other module touches them.
(§1 of `DOCUMENTATION.md` lists `music21` and `pretty_midi` as well — those belong to the
extraction layer, which is not in this repository.)

Modules sit flat in `src/`, so `os.path.dirname(HERE)` is the project root in every one of
them: data is read from `<root>/dat` (see `DATA.md`), panels are written to `<root>/fig/panels` (created on
import of `_figure_style`), supplementary panels to `<root>/fig/supp`. Nothing manipulates
`sys.path` — `config.py` and `irreversibility_v2.py` are siblings in `src/`.

Every module is named for what it produces: `fig3a_current_matrices.py` writes Figure 3a,
`supp_rootmotion_hexbin.py` writes a supplementary panel, `compute_*.py` and `estimator.py`
are the analysis layer. `LAYOUT.md` records why — an earlier naming had three scripts each
claiming "Fig 4a".

### Analysis layer

| Module | What it does |
|---|---|
| `estimator.py` | The entropy-production estimator and corpus loaders |
| `canonical_pools.py` | Canonical eight-corpus pooling, asserted against Table 1 |
| `compute_corpus_sigma.py` | Compute pooled sigma for every corpus |
| `compute_mechanism.py` | Compute the mechanism decomposition on invariant transition types |
| `irreversibility_v2.py` | Corrected irreversibility estimators |
| `config.py` | Global configuration for music-thermo project |
| `_figure_style.py` | Shared figure style, palette and glyph registry for every panel |

### Main figures

| Module | What it does |
|---|---|
| `fig1b_state_space.py` | Figure 1b — the 60+1 chord state space, as vector pieces |
| `fig2a_composer_sigma.py` | Figure 2a — sigma per composer, grouped by era |
| `fig2b_directedness_and_null.py` | Figure 2b — per-work directedness against the shuffled null |
| `fig2c_networks.py` | Figure 2c — corpus networks on the circle of fifths |
| `fig2c_null_network.py` | Figure 2c inset — the symmetrized-null network |
| `fig3_concentration_matrices.py` | Figure 3 — four matrices, most diffuse to most concentrated |
| `fig3a_current_matrices.py` | Figure 3a — net-current matrices by root motion and quality pair |
| `fig3b_force_and_current.py` | Figure 3b — constant affinity, rising current |
| `fig3c_fifth_share_vs_arrow.py` | Figure 3c — per-work arrow against descending-fifth share |
| `fig4_irish_loop.py` | Figure 4 — weighted loop network for the Irish settings |
| `fig4_irish_mode_contrast.py` | Figure 4 — the Irish mode contrast, with engraved illustrations |
| `fig4a_irish_ladder.py` | Figure 4a — one tune under two harmonizations, bar for bar |
| `fig4a_irish_matrices.py` | Figure 4a — Irish net-current matrices by mode class |

### Supplementary figures

| Module | What it does |
|---|---|
| `build_supp_figures.py` | Build every supplementary figure from the frozen data |
| `make_supp_designs.py` | Regenerate the supplementary design set from the frozen estimator |
| `supp_panel_library.py` | Supplementary panel library — every supplementary panel builder |
| `supp_aligned_panels.py` | Supplementary panels: palindrome control, current concentration, fifth share |
| `supp_networks.py` | Supplementary network figures on the circle of fifths |
| `supp_engine_decomposition.py` | Supplementary — net current by harmonic mechanism |
| `supp_rootmotion_hexbin.py` | Supplementary — the arrow rides on the descending fifth alone |
| `supp_chordonomicon.py` | Supplementary — Chordonomicon modern-genre breadth |

## Data

No data files are in this repository. The modules read 43 files from `dat/` — 41 result files
(26.6 MB) plus two caches in `dat/output/`, `sequences.zip` (79.8 MB, the extracted
chord-symbol sequences for every usable work) and `all_records_usable.parquet` (3.9 MB,
22,948 per-piece records). `DATA.md` lists every one of them with its size, the modules that
read it, and whether code in this repository can rebuild it; it also carries the deposit DOI
and the directory layout to unpack into. Six of the 43 are written by modules here, so with
the deposit in place every result and every figure rebuilds without obtaining a source corpus.

32 of the 34 data filenames referenced in `src/` are covered by the deposit. The two that are
not are `chordonomicon_v2.csv` (264 MB, the published Chordonomicon dataset) and a
format-string fragment that is not a real filename.

`DOCUMENTATION.md` is the project reference: pipeline usage, dataset provenance and attrition,
the chord-extraction method, the full derivations from the Markov chain through the entropy
production rate and the observation-window curve, figure captions, findings and caveats.
`LAYOUT.md` describes the wider project directory, of which this repository carries the code.

## Audit

`CODE_AUDIT.md` reports the comment-to-code balance of every module and the provenance of
every data file the code reads. Headline: no bare excepts, no `eval`/`exec`, no pickled
models, no network calls, every sampling path seeded, and the Table 1 constants in
`canonical_pools.py` and `compute_mechanism.py` are assertion targets rather than values
standing in for computation. On libraries: `numpy` carries the arithmetic, `scipy` appears only as
`pearsonr` and `spearmanr`, 12 invocations across two modules, `pandas` only reads parquet, and there is no
`seaborn`, `sklearn`, `statsmodels`, KDE or curve fit anywhere — the estimator itself is
twelve lines of explicit arithmetic. Six of the thirty data files consumed by `src/` are produced by
code here; `DATA.md` gives the status of each.

## Known defects, carried over

- `dat/final_data.json` still uses the pre-rename corpus key `Late-Romantic` and has no `Ragtime` entry, which is why the trajectory panel that read it was retired rather than repaired. Re-exporting it under current corpus names is the fix.
- `config.py`'s `DATA_DIR`, `CACHE_DIR` and `FIG_DIR` are vestigial: they resolve to `src/data`, `src/cache` and `src/figures` rather than the project root. No module uses them to reach data — each computes the root from its own location — but the constants below them (the 61-state space, the pseudocount, `W_MAX`, bootstrap settings) are live and imported.
- `fig3c_fifth_share_vs_arrow.py` cites `dat/coverage.json`, which sits in the project's `archive/unreferenced_data/` rather than in `dat/`.
- `canonical_pools.py` writes its self-check to `os.environ["OUTJ"]` with no default, so running it with that variable unset raises `KeyError` after the computation completes.

## Provenance

`MANIFEST.csv` lists every file with its size, sha256 and the path it came from in the
project directory. Every file was verified byte-identical to its source after copying. Nine
modules were then edited, each recorded in the manifest: eight had stale `../data/` docstring
paths corrected to `dat/`, and `estimator.py` had an unused `zlib` import removed. No
computation was touched.
