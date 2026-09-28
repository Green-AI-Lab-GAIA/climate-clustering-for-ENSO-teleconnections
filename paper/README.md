# Nature-style paper — Amazonia ENSO regimes

- `main.tex` — manuscript (Springer Nature `sn-jnl` class, `sn-nature` reference style).
- `refs.bib` — bibliography. **Entry details were filled from general knowledge — verify DOIs/pages before submission.**
- `make_figures.py` — regenerates all figures + `stats.json` from the trained
  `temperature-era-amazonia-15-anom` run (mirrors `notebooks/analysis_significance.ipynb`,
  same seeds ⇒ identical numbers). Run from repo root: `.venv/bin/python paper/make_figures.py`.
- `stats.json` — every number quoted in the text, machine-checkable.
- `figures/` — PDFs used by the manuscript (+ PNG previews).
- `sn-jnl.cls`, `sn-nature.bst`, `bst/` — Springer Nature template files.

Compile (no system LaTeX needed):

```bash
tectonic main.tex          # or: pdflatex main && bibtex main && pdflatex main ×2
```
