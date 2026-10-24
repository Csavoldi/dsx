# T16 paper

This is a research draft with measurements from the original T16 experiments, not a current runtime specification.
Its gain values were tuned on the reported eval, and the broader calib5 quant did not improve general-text KL.
See [current state](../../docs/current-state.md) for the later integration and unresolved decode exactness check.
A paper build does not validate these numerical claims.

Build: `python3 figs.py && pdflatex main && bibtex main && pdflatex main && pdflatex main` (TeX Live with psnfss, booktabs, natbib,
cleveref, pgf/TikZ; matplotlib for the figures). Every plotted number is a measurement recorded in the dsx repo or reported by
the DSX sessions on 2026-10-21/22; figs.py notes each source.
