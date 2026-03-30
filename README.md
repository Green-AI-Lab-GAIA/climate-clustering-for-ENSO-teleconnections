# Climate Clustering Model

Self-supervised clustering of Brazilian climate data using **Masked Siamese Networks (MSN)**. The model learns spatial representations of temperature and precipitation patterns without labels, discovering meaningful climate regions and their relationship to large-scale phenomena such as El Niño/La Niña.

## Overview

The project adapts the MSN framework — originally developed for image recognition — to gridded climate data. A Vision Transformer (DeiT) is trained via multi-view self-supervision to produce embeddings that are subsequently clustered into prototypes. These prototypes capture recurring climate patterns across Brazil.


## Repository Structure

```
climate-clustering/
├── configs/                  # YAML experiment configurations
│   └── temp.yaml             # Temperature experiment (30 prototypes)
├── src/
│   ├── msn_train.py          # MSN training loop
│   ├── inference.py          # Embedding extraction and dimensionality reduction
│   ├── data_manager.py       # Dataset class and multi-view transforms
│   ├── losses.py             # MSN loss + me-max + entropy regularization
│   ├── deit.py               # Vision Transformer (DeiT) backbone
│   ├── utils.py              # Distributed training utilities
│   ├── sgd.py                # SGD optimizer
│   ├── el_nino.py            # ONI-based El Niño/La Niña classification
│   └── data/
│       ├── load_variables.py # ERA5 NetCDF loading
│       ├── download_data.py  # Copernicus CDS download
├── notebooks/
│   ├── analysis.ipynb        # Main analysis: clustering, t-SNE, ENSO correlation
│   └── region_pre_analysis.ipynb  # Exploratory data analysis
├── checkpoint/               # Saved model weights and training logs
│   └── temperature-run2/
│       ├── *.pth.tar         # Checkpoints at epochs 10, 50, 100, 200, 300
│       └── *_r0.csv          # Per-iteration loss logs
├── results/                  # Cached inference outputs
│   └── temperature-exp-2-c30/
│       ├── E.pt, E_val.pt    # Encoder embeddings (128-dim)
│       ├── F.pt, F_val.pt    # Prototype logits
│       └── tsne_E.pt         # t-SNE projections
├── main_msn.py               # Training entry point
└── requirements.txt
```

## Data

This experiment uses **BR-DWGD** (Brazilian Daily Weather Gridded Data), a high-resolution gridded dataset for Brazil. ERA5 reanalysis is also supported via `src/data/download_data.py`, but was not used in this experiment.

| Variable | Description | File pattern |
|---|---|---|
| Tmin | Daily minimum temperature | `Tmin_*.nc` |
| Tmax | Daily maximum temperature | `Tmax_*.nc` |

- Spatial domain: approximately lat −22° to −7°, lon −57.5° to −43°
- Temporal coverage: 1961–2024
- validation years: 1981 and 2000 

Raw NetCDF files are expected under `../data/raw/` relative to the project root. BR-DWGD data can be obtained from the [project page](https://sites.google.com/site/alexandrecandidoxavierufes/brazilian-daily-weather-gridded-data).

## Model

**Backbone:** `deit_small_temperature` — a DeiT-Small Vision Transformer adapted to multi-channel climate inputs (surface variables + static fields). The encoder maps each spatial window to a 128-dimensional embedding via a 3-layer MLP projection head.

**Loss:**
```
L = L_MSN + λ_memax · L_memax + λ_ent · L_entropy
```

- `L_MSN`: cross-entropy between focal-crop predictions and sharpened random-crop targets
- `L_memax`: mean-entropy maximization to ensure balanced prototype usage
- `L_entropy`: optional entropy penalty

**Training configuration** (see `configs/temp.yaml`):

| Hyperparameter | Value |
|---|---|
| Prototypes | 30 |
| Output dimension | 128 |
| Epochs | 300 |
| Batch size | 512 |
| Optimizer | AdamW |
| Peak LR | 1e-3 |
| Warmup epochs | 15 |
| Weight decay | 0.04 → 0.4 |
| Momentum (EMA) | 0.996 → 1.0 |
| Gradient clip | 3.0 |

## Usage

### Training

```bash
python main_msn.py configs/temp.yaml
```

Multi-GPU training is handled via PyTorch distributed. Edit the config file to adjust data paths, hyperparameters, or the number of prototypes.

### Inference

```python
from src.inference import get_model_results

E, F = get_model_results(
    encoder=encoder,
    target_encoder=target_encoder,
    prototypes=prototypes,
    data_loader=loader,
    results_dir="results/temperature-exp-2-c30",
    tag="train"
)
```

Results are cached as `.pt` files and reloaded automatically on subsequent calls.

### Analysis

Open `notebooks/analysis.ipynb` to:
- Load trained embeddings and prototype assignments
- Visualize clusters with t-SNE / PCA / UMAP
- Map clusters back to geographical coordinates
- Correlate cluster time series with the ONI index

## Dependencies

Install with:

```bash
pip install -r requirements.txt
```

Key packages: `torch`, `torchvision`, `xarray`, `numpy`, `pandas`, `scikit-learn`, `timm`, `einops`, `cdsapi`, `matplotlib`, `plotly`, `seaborn`.

## References

- Assran et al. (2022). *Masked Siamese Networks for Label-Efficient Learning.* ECCV 2022.
- Touvron et al. (2021). *Training data-efficient image transformers.* ICML 2021.
- Xavier et al. (2016). *Daily gridded meteorological variables in Brazil (1980–2013).* International Journal of Climatology. BR-DWGD dataset: https://sites.google.com/site/alexandrecandidoxavierufes/brazilian-daily-weather-gridded-data
