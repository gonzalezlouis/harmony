# Data

No data files are in this repository. The 28 modules in `src/` read 43 files from `dat/`,
listed below with the modules that consume them; the files themselves are deposited
separately so the repository stays code-only.

## Getting the data

Download the deposit and unpack it so that `dat/` sits beside `src/`:

```
harmony/
  src/          <- this repository
  dat/          <- the deposit: 41 result files here
    output/     <- sequences.zip and all_records_usable.parquet
  fig/          <- written by the figure modules
```

Zenodo DOI: _to be minted — replace this line with the DOI once the deposit is published._

## `dat/` — 41 result files, 26.6 MB

| File | Size | Read by | Regeneration status |
|---|---|---|---|
| `aggregation_audit.json` | 2 kB | `build_supp_figures` | frozen — no producer in the project |
| `arc_results.json` | 93 kB | `make_supp_designs` | regenerable here |
| `bach_matched_floor.json` | 1 kB | `build_supp_figures` | frozen — no producer in the project |
| `chordonomicon_mechanism.json` | 2 kB | `build_supp_figures` | frozen — no producer in the project |
| `composer_robustness.json` | 8 kB | `build_supp_figures` | not read by any module |
| `composer_sigma.json` | 4 kB | `fig2a_composer_sigma` | not read by any module |
| `corpus_centroids.json` | 3 kB | `compute_mechanism`, `fig3c_fifth_share_vs_arrow` | regenerable here |
| `corpus_directedness.json` | 0.13 MB | `fig2b_directedness_and_null` | frozen — no producer in the project |
| `decoupling.json` | 2 kB | `build_supp_figures`, `supp_panel_library` | frozen — no producer in the project |
| `emergent_full.json` | 2 kB | `fig3c_fifth_share_vs_arrow` | frozen — no producer in the project |
| `emergent_full.npz` | 1.48 MB | `compute_mechanism`, `fig3c_fifth_share_vs_arrow`, `supp_rootmotion_hexbin` | frozen — no producer in the project |
| `engine_decomposition.json` | 3 kB | `supp_engine_decomposition` | not read by any module |
| `estimator_and_pooling_record.json` | 4 kB | — | not read by any module |
| `exact_pair_audit.json` | 1 kB | — | not read by any module |
| `fifth_demo.json` | 1 kB | — | not read by any module |
| `final_data.json` | 4 kB | `_figure_style`, `compute_corpus_sigma` | regenerable here |
| `harmony_vs_melody.json` | 1 kB | — | not read by any module |
| `irish_distributions.json` | 1 kB | — | not read by any module |
| `irish_examples.json` | 2 kB | `fig4_irish_mode_contrast`, `fig4a_irish_ladder` | frozen — no producer in the project |
| `irish_expansion.json` | 26 kB | `fig4_irish_loop`, `fig4_irish_mode_contrast` | frozen — no producer in the project |
| `irish_paired.json` | 3 kB | `build_supp_figures` | frozen — no producer in the project |
| `irish_per_setting.npz` | 0.11 MB | `build_supp_figures` | frozen — no producer in the project |
| `irish_per_setting_record.json` | 1 kB | — | not read by any module |
| `irish_tunes.json` | 24.10 MB | `estimator`, `fig4a_irish_matrices` | frozen — no producer in the project |
| `irish_type_based.json` | 1 kB | `build_supp_figures` | frozen — no producer in the project |
| `irish_within_tune.json` | 0 kB | `build_supp_figures` | frozen — no producer in the project |
| `mechanism.json` | 79 kB | `build_supp_figures`, `fig2c_networks`, `fig3_concentration_matrices`, `fig3a_current_matrices`, `fig3b_force_and_current` | frozen — no producer in the project |
| `mechanism_ci.json` | 16 kB | `fig3b_force_and_current` | frozen — no producer in the project |
| `null_distributions.json` | 3 kB | `build_supp_figures` | frozen — no producer in the project |
| `null_draws.npz` | 21 kB | `build_supp_figures` | frozen — no producer in the project |
| `null_models.json` | 9 kB | `fig2b_directedness_and_null`, `fig2c_null_network` | frozen — no producer in the project |
| `palindrome.json` | 1 kB | — | not read by any module |
| `provenance_calibration.json` | 4 kB | `build_supp_figures` | frozen — no producer in the project |
| `provenance_matched_audit.json` | 4 kB | — | not read by any module |
| `results_values.json` | 6 kB | `build_supp_figures` | frozen — no producer in the project |
| `rootmotion_2x2.json` | 11 kB | `build_supp_figures`, `compute_mechanism`, `supp_rootmotion_hexbin` | regenerable here |
| `rootmotion_2x2.npz` | 0.44 MB | `compute_mechanism`, `supp_rootmotion_hexbin` | regenerable here |
| `sparsity_and_smoothing.json` | 7 kB | — | not read by any module |
| `transfer_matrix.npz` | 23 kB | `fig3_concentration_matrices`, `fig3a_current_matrices`, `supp_engine_decomposition` | producer archived, not packaged |
| `transfer_matrix_canonical.npz` | 22 kB | `fig3_concentration_matrices`, `fig3a_current_matrices`, `supp_engine_decomposition` | regenerable here |
| `unified.json` | 5 kB | — | not read by any module |

## `dat/output/` — 2 caches, 83.7 MB

| File | Size | Read by | Regeneration status |
|---|---|---|---|
| `all_records_usable.parquet` | 3.86 MB | `compute_corpus_sigma`, `compute_mechanism`, `estimator`, `fig2c_networks` | producer archived, not packaged |
| `sequences.zip` | 79.84 MB | `compute_corpus_sigma`, `compute_mechanism`, `estimator`, `fig2c_networks` | producer archived, not packaged |

## What the statuses mean

**regenerable here** — a module in `src/` writes it: `arc_results.json` from
`make_supp_designs.py`, `corpus_centroids.json` and `rootmotion_2x2.json`/`.npz` from
`compute_mechanism.py`, `final_data.json` from `compute_corpus_sigma.py`,
`transfer_matrix_canonical.npz` from `supp_engine_decomposition.py`.

**producer archived, not packaged** — written by the extraction and merge layer named in §1 of
`DOCUMENTATION.md` (`pipeline.py`, `datasets/*.py`, `merge_datasets.py`), which lives in the
project's `archive/` and is not in this repository.

**frozen — no producer in the project** — 21 files with no write site in any of the project's
221 Python files. `CODE_AUDIT.md` documents how that was established.

**not read by any module** — 10 files that travelled with the data directory but which no
module in `src/` opens: `estimator_and_pooling_record.json`, `exact_pair_audit.json`,
`fifth_demo.json`, `harmony_vs_melody.json`, `irish_distributions.json`,
`irish_per_setting_record.json`, `palindrome.json`, `provenance_matched_audit.json`,
`sparsity_and_smoothing.json`, `unified.json`.

## Source corpora

The deposit does not contain the raw corpora either. Seven were used — 42,432 files, 24,011
usable: KernScores (kern.humdrum.org), GiantMIDI-Piano (github.com/bytedance/GiantMIDI-Piano),
ASAP score and performance (github.com/fosfrancesco/asap-dataset), MAESTRO v3
(magenta.tensorflow.org), POP909 (github.com/music-x-lab/POP909-Dataset) and Jazz iRb v1.0
(github.com/shanahdt/irealb). Section 3 of `DOCUMENTATION.md` gives the attrition, genre and
era breakdown for each. The Irish traditional material is ABC from thesession.org; the
Chordonomicon corpus is Kantarelis et al. 2024, arXiv:2410.22046.
