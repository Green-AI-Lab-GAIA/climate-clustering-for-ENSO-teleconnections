"""
Pre-computation script — run ONCE locally before deploying to Streamlit Cloud.

Requires access to the full NetCDF climate dataset (../data/raw/).
Produces a small app_data/ folder (~30 MB) that the Streamlit app loads instead.

Usage (from project root):
    python scripts/precompute_app_data.py

Outputs:
    app_data/df_clusters.parquet   – dates + cluster assignments for all samples
    app_data/top_samples.pt        – top-N images per cluster (for prototype plots)
    app_data/quantiles.pt          – pre-computed per-cluster quantiles (for Q-Q plots)
    app_data/oni_index.xlsx        – ONI index copy (from ../data/oni_index.xlsx)
    app_data/params.yaml           – model config copy
"""
import os
import sys
import shutil

import numpy as np
import pandas as pd
import torch
import yaml

# Make sure the project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import src.inference as inf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_FILE = "checkpoint/temperature-run2/params-temperature-exp-2-c30.yaml"
SAVE_PATH   = "results/temperature-exp-2-c30"
ONI_SRC     = "../data/oni_index.xlsx"   # path to source ONI file
OUT_DIR     = "app_data"
TOP_N       = 5      # top samples per cluster to save for prototype images
N_QUANTILES = 200    # quantile resolution for Q-Q plots

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load dataset (needs NetCDF files) and cluster assignments (from .pt files)
# ---------------------------------------------------------------------------
print("Loading dataset and cluster assignments...")
params, dataset = inf.read_data(CONFIG_FILE, validation=True)

F     = torch.load(os.path.join(SAVE_PATH, "F.pt"),     weights_only=False)
F_val = torch.load(os.path.join(SAVE_PATH, "F_val.pt"), weights_only=False)

cluster_prob, cluster_id = torch.max(F,     dim=1)
val_prob,     val_id     = torch.max(F_val, dim=1)
n_train = len(dataset.time)

# ---------------------------------------------------------------------------
# 2. Build and save df_clusters.parquet
# ---------------------------------------------------------------------------
print("Building dataframe...")
df = pd.concat([
    pd.DataFrame({
        "cluster_id":   cluster_id.cpu().numpy(),
        "cluster_prob": cluster_prob.cpu().numpy(),
        "sample_type":  "train",
        "date":         pd.to_datetime(dataset.time),
        "train_idx":    np.arange(n_train),
    }),
    pd.DataFrame({
        "cluster_id":   val_id.cpu().numpy(),
        "cluster_prob": val_prob.cpu().numpy(),
        "sample_type":  "val",
        "date":         pd.to_datetime(dataset.val_time),
        "train_idx":    np.full(len(dataset.val_time), -1, dtype=int),
    }),
]).sort_values("date").reset_index(drop=True)

out_parquet = os.path.join(OUT_DIR, "df_clusters.parquet")
df.to_parquet(out_parquet, index=False)
print(f"  df_clusters.parquet  {os.path.getsize(out_parquet)/1e6:.1f} MB")

# ---------------------------------------------------------------------------
# 3. Save top-N images per cluster  (prototype visualisation)
# ---------------------------------------------------------------------------
print(f"Saving top-{TOP_N} images per cluster...")
train_imgs = dataset.imgs        # (N_train, nvars, H, W) raw float tensor
nvars      = train_imgs.shape[1]
N_PROTO    = params["criterion"]["num_proto"]
df_train   = df[df["sample_type"] == "train"]

top_images = {}
for cid in range(N_PROTO):
    rows = (
        df_train[df_train["cluster_id"] == cid]
        .nlargest(TOP_N, "cluster_prob")
    )
    imgs = []
    for _, row in rows.iterrows():
        idx = int(row["train_idx"])
        if 0 <= idx < len(train_imgs):
            imgs.append(train_imgs[idx])
    if imgs:
        top_images[cid] = torch.stack(imgs)

var_stats = {
    v: (float(train_imgs[:, v].min()), float(train_imgs[:, v].max()))
    for v in range(nvars)
}

out_top = os.path.join(OUT_DIR, "top_samples.pt")
torch.save({
    "images":    top_images,   # {cid: tensor(k, nvars, H, W)}
    "var_stats": var_stats,
    "vars_names": params["data"]["surf_vars"],
    "top_n":     TOP_N,
}, out_top)
print(f"  top_samples.pt       {os.path.getsize(out_top)/1e6:.1f} MB")

# ---------------------------------------------------------------------------
# 4. Pre-compute quantiles for Q-Q plots  (uses all train samples per cluster)
# ---------------------------------------------------------------------------
print("Pre-computing quantiles...")
q = np.linspace(0.01, 0.99, N_QUANTILES)

global_q = {
    v: np.quantile(train_imgs[:, v].numpy().ravel(), q)
    for v in range(nvars)
}

cluster_q = {}
for cid in range(N_PROTO):
    idx = df_train[df_train["cluster_id"] == cid]["train_idx"].values
    idx = idx[idx >= 0]
    if len(idx) == 0:
        cluster_q[cid] = {v: np.full(N_QUANTILES, np.nan) for v in range(nvars)}
    else:
        cluster_q[cid] = {
            v: np.quantile(train_imgs[idx, v].numpy().ravel(), q)
            for v in range(nvars)
        }

out_quant = os.path.join(OUT_DIR, "quantiles.pt")
torch.save({"q": q, "global": global_q, "clusters": cluster_q}, out_quant)
print(f"  quantiles.pt         {os.path.getsize(out_quant)/1e6:.2f} MB")

# ---------------------------------------------------------------------------
# 5. Copy ONI index and params.yaml
# ---------------------------------------------------------------------------
print("Copying ONI index and params...")
out_oni = os.path.join(OUT_DIR, "oni_index.xlsx")
shutil.copy(ONI_SRC, out_oni)
print(f"  oni_index.xlsx       {os.path.getsize(out_oni)/1e6:.2f} MB")

out_params = os.path.join(OUT_DIR, "params.yaml")
shutil.copy(CONFIG_FILE, out_params)
print(f"  params.yaml          {os.path.getsize(out_params)/1e3:.1f} KB")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total = sum(
    os.path.getsize(os.path.join(OUT_DIR, f))
    for f in os.listdir(OUT_DIR)
)
print(f"\nDone! Total app_data/ size: {total/1e6:.1f} MB")
print("Commit app_data/ to git and deploy.")
