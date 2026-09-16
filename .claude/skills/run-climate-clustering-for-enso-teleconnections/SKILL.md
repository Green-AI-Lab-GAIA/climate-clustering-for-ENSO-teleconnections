---
name: run-climate-clustering-for-enso-teleconnections
description: Build, run, and drive the climate-clustering ENSO project. Use when asked to run or start this repo, train a model, smoke-test the training pipeline, run inference or clustering analysis, execute the analysis notebooks, launch the Streamlit viewer, or take a screenshot of its UI.
---

Self-supervised (MSN) clustering of ERA5 daily fields over Amazonia, used to test ENSO teleconnections. There is no single "app": the unit has four surfaces — the dataset layer, the training entrypoint (`main_msn.py`), the analysis notebooks, and a Streamlit viewer. Drive all of them through **`.claude/skills/run-climate-clustering-for-enso-teleconnections/driver.py`**, which `chdir`s to the repo root for you and exposes one subcommand per surface.

All paths below are relative to the repo root. Always use the project venv — bare `python` is not on PATH.

## Prerequisites

The venv already exists and has everything in `requirements.txt`. No `apt-get` is needed and **there is no sudo on this machine**.

Screenshotting the Streamlit UI needs Playwright, which is *not* in `requirements.txt`. Install it into the venv explicitly — `pip install` after `source .venv/bin/activate` lands in the system Python here, not the venv:

```bash
.venv/bin/python -m pip install playwright
playwright install chromium
```

`playwright install-deps` is **not** required (and would need root anyway): the bundled `chromium-headless-shell` runs without extra system libraries.

## Data

Lives outside the repo at `../data/` (i.e. `/u00/livia.meinhardt/data/`):

- `../data/era5/*.nc` — 90 files: 45 `*-min-max-temp-*.nc` (`mn2t`,`mx2t`) and 45 `*-total-prep-*.nc` (`tp`).
- `../data/oni_index.xlsx` — NOAA ONI index.
- `../data/era5/.daily_cache/*.pt` — resampled daily tensors. **With the cache warm the dataset loads in 0.3 s; cold it re-reads the raw netCDF and takes ~40 minutes.** Never delete these casually.

Verify everything before doing real work:

```bash
.venv/bin/python .claude/skills/run-climate-clustering-for-enso-teleconnections/driver.py env
```

## Run (agent path)

Ordered cheapest → most expensive. Each is a real command that passes today.

```bash
D=.claude/skills/run-climate-clustering-for-enso-teleconnections/driver.py

# dataset layer only, no model  (~1 s warm cache)
.venv/bin/python $D data

# load a checkpoint and push 8 real days through the encoder  (~30 s)
.venv/bin/python $D infer

# ENSO stats from cached F tensors, no GPU  (~20 s)
.venv/bin/python $D clusters

# a REAL 2-epoch training run into checkpoint/smoke-run/  (~150 s on the A100)
.venv/bin/python $D train-smoke
```

`train-smoke` leaves ~820 MB behind (`checkpoint/smoke-run/` and `configs/_smoke.yaml`). `checkpoint` is gitignored so nothing gets committed, but clean up after yourself:

```bash
rm -rf checkpoint/smoke-run configs/_smoke.yaml
```

`--config` defaults to `checkpoint/temperature-era-amazonia-15/params-temperature-era-amazonia-15.yaml` everywhere; pass another run's dumped `params-*.yaml` to switch. `infer` also takes `--suffix` and `--n`.

**For a PR that touches `src/`, `infer` and `data` are the entry points you want** — they import and call the code directly instead of going through the full app. Recent changes land in `src/data_manager.py`, `src/deit.py`, `configs/`, and `notebooks/`.

### Streamlit viewer

```bash
D=.claude/skills/run-climate-clustering-for-enso-teleconnections/driver.py
.venv/bin/python $D app          # background; prints pid + URL, log in logs/streamlit.log
.venv/bin/python $D shot         # fills the sidebar, clicks every tab, writes PNGs
```

`shot` writes to `.claude/skills/run-climate-clustering-for-enso-teleconnections/shots/` (`app-loaded.png` plus one per tab) and **exits non-zero if the page rendered a Streamlit exception**, so it doubles as the UI smoke test. Limit tabs with `--tabs Embedding ENSO`.

Stop the server — note the `[_]` bracket, see Gotchas:

```bash
pkill -f "streamlit run streamlit[_]app.py"
```

### Full training run

Hours, not minutes. Background it:

```bash
mkdir -p checkpoint/<tag> logs
.venv/bin/python main_msn.py --fname configs/<config>.yaml --devices cuda:0 > logs/<tag>.log 2>&1 &
```

Progress: `grep -c "avg. loss" logs/<tag>.log` counts finished epochs; `grep "val loss" logs/<tag>.log | tail -1` shows the latest validation.

### Analysis notebooks

They `os.chdir('..')` in their first cell, so they only work from `notebooks/`. `nbconvert` already uses the notebook's directory as cwd:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 notebooks/analysis_significance_temp_prep.ipynb
```

`notebooks/cluster_samples.ipynb` is run-agnostic — it discovers every run from `checkpoint/*/params-*.yaml` and is selected with env vars, so it needs no edit for a new run:

```bash
RUN=temperature-era-amazonia-15-anom N_SAMPLES=6 PICK=representative \
  .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 notebooks/cluster_samples.ipynb
```

It writes one PNG per cluster plus a per-variable contact sheet to `figures/cluster_samples/<RUN>/`. Other knobs: `PICK=representative|spread|random`, `SEED`, `MODEL_SUFFIX`, `VABS_PCT` (lower it when a heavy-tailed channel like `tp` looks washed out).

Check for failures afterwards — `nbconvert` exits 0 on some partial runs:

```bash
.venv/bin/python -c "
import nbformat; nb=nbformat.read('notebooks/analysis_significance_temp_prep.ipynb', as_version=4)
print([(i,o.ename) for i,c in enumerate(nb.cells) if c.cell_type=='code' for o in c.get('outputs',[]) if o.output_type=='error'] or 'no errors')"
```

## Gotchas

- **The Streamlit app is broken out of the box.** Its sidebar defaults to `checkpoint/temperature-run2/params-temperature-exp-2-c30.yaml`, which does not exist here → `FileNotFoundError` before any tab renders. (`.gitignore` has a negated rule for one file under `checkpoint/temperature-run2/`, so the default targets a checkpoint meant to be committed but absent from this working copy — it is not a typo you should "fix".) Overwrite *Config file* and *Results path*; `driver.py shot` does this for you.
- **Only two results dirs have t-SNE caches** (`results/temperature-era-amazonia-15`, `results/temperature-exp-2-c30`). Point the viewer at any other and `load_state` recomputes t-SNE over ~16k embeddings on load. The driver defaults to `results/temperature-era-amazonia-15` for this reason.
- **`dataset.imgs` is native resolution (94×120) and will not go through the encoder.** You get `RuntimeError: The size of tensor a (36) must match the size of tensor b (26)` from the positional embedding. Index the dataset instead — `dataset[i]` applies the Resize/CenterCrop/Normalize transform. See `cmd_infer` in the driver.
- **`inf.load_model` defaults to `-latest.pth.tar`.** For early-stopped runs that is the *worst* epoch, not the best. Pass `model_suffix='-best.pth.tar'` explicitly (e.g. `temp-prep-era-amazonia-15-anom` stopped at epoch 95 but peaked at 75).
- **`pkill -f "streamlit run streamlit_app.py"` kills your own shell**, because the pattern matches the shell's own command line. Exit code 144 with no other output is this. Use `streamlit[_]app.py`.
- **ERA5 files must be globbed per variable.** `../data/era5/` holds two downloads; a bare `*.nc` glob concatenates both into one time axis with each variable NaN over the other's files. `Era5Dataset.VAR_FILE_GLOB` handles this — add an entry there (plus `DAILY_AGG`, and `VAR_SCALE` if units need converting) for any new variable.
- **Model variants hardcode `in_chans`.** `deit_small_temperature` is 2-channel, `deit_small_temp_prep` is 3-channel. A new variable count needs a new factory in `src/deit.py`.
- The daily cache key covers the file list + variables + spatial crop, but **not** `deseason_standardize` — anomalies are applied after loading, so raw and deseasonalised runs share one cache entry.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: checkpoint/temperature-run2/...` in the browser | Shipped Streamlit default. Set Config file + Results path, or use `driver.py shot`. |
| `RuntimeError: size of tensor a (36) must match ... (26)` | You fed `dataset.imgs` to the encoder. Use `dataset[i]`. |
| `ModuleNotFoundError: No module named 'playwright'` | `pip` inside the activated venv installed to system Python. Use `.venv/bin/python -m pip install playwright`. |
| Dataset load hangs for tens of minutes | Cold daily cache. Expected once (~40 min); it is written to `../data/era5/.daily_cache/`. |
| Shell dies with exit 144 during `pkill` | Self-matching pattern — use `streamlit[_]app.py`. |
| `driver.py app` reports the port never opened | Read `logs/streamlit.log`; another server may still hold 8501. `pkill` it or pass `--port`. |
