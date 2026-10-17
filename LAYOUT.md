# Project layout

```
src/     all code, flat, one module per figure or analysis step
dat/     inputs and computed results
  input/    raw corpora
  output/   extracted sequences and per-work records
  *.json    frozen analysis results the figures read
fig/
  main/     fig1-4, Illustrator source + exported PDF
  supp/     the 21 figures SI.tex includes
  panels/   rendered panel components that feed the Illustrator mosaics
archive/ superseded material, nothing deleted
SI.tex   the combined Methods and Supplementary Information
```

## Naming

Every module is named for what it produces. `fig3a_current_matrices.py` writes
Figure 3a; `supp_rootmotion_hexbin.py` writes a supplementary panel;
`compute_*.py` and `estimator.py` are the analysis layer. The old names carried a
superseded figure numbering in which three separate scripts all claimed "Fig 4a",
so a filename could not be trusted to say where its output went.

## How paths resolve

Modules sit flat in `src/`, so `os.path.dirname(HERE)` is the project root in
every one of them. Data is read from `<root>/dat`, panels are written to
`<root>/fig/panels`. The estimator's two dependency modules, `config.py` and
`irreversibility_v2.py`, are siblings in `src/` rather than a separate tree, so
no script manipulates `sys.path` to reach them.

Run any figure directly:

```
cd src && python fig3a_current_matrices.py
```

## Verified

All 21 runnable modules execute from `src/`. The eight corpus values in
`dat/results_values.json` match Table 1 to three decimals, spanning 36.9-fold.
All 21 `\includegraphics` targets in `SI.tex` resolve to `fig/supp/`. Every data
filename referenced in `src/` exists under `dat/`.

## Known defect

`dat/final_data.json` still carries the pre-rename corpus key `Late-Romantic`
and has no `Ragtime` entry, which is why the trajectory panel that read it was
archived rather than repaired. Re-exporting that file under current corpus names
is the fix.
