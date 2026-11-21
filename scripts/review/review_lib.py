"""Shared helpers for the results review (run from the repo root)."""
import os
import sys
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, '.')
from src.data_manager import Era5Dataset
from src.el_nino import read_enso_data
import src.enso_stats as es

RUNS = {
    # label: (results dir, checkpoint params yaml)
    'T15 (paper, A100)':        ('results/temperature-era-amazonia-15-anom', None),
    'T15':                      ('results/temperature-era-amazonia-15-anom-100ep',
                                 'checkpoint/temperature-era-amazonia-15-anom-100ep/params-temperature-era-amazonia-15-anom-100ep.yaml'),
    'T30':                      ('results/temperature-era-amazonia-30-anom-100ep',
                                 'checkpoint/temperature-era-amazonia-30-anom-100ep/params-temperature-era-amazonia-30-anom-100ep.yaml'),
    'TP30 raw (early stop)':    ('results/temp-prep-era-amazonia-30-anom',
                                 'checkpoint/temp-prep-era-amazonia-30-anom/params-temp-prep-era-amazonia-30-anom.yaml'),
    'TP30 raw':                 ('results/temp-prep-era-amazonia-30-anom-100ep',
                                 'checkpoint/temp-prep-era-amazonia-30-anom-100ep/params-temp-prep-era-amazonia-30-anom-100ep.yaml'),
    'TP30 cbrt':                ('results/temp-prep-cbrt-era-amazonia-30-anom-100ep',
                                 'checkpoint/temp-prep-cbrt-era-amazonia-30-anom-100ep/params-temp-prep-cbrt-era-amazonia-30-anom-100ep.yaml'),
}

# A100 runs (Nov 2026, FGV HPC; scripts/review/infer_a100.py, merge_overclusters.py). p10_monthly is
# monthly (one label per month) and is analysed separately (a100_monthly.py).
for _r in ('t15_s0', 't15_s1', 't15_s2', 't30_s0', 't30_s1', 't30_s2', 't30_nomean', 't30_nomean_200ep',
           'p30_daily_raw', 'p30_daily_cbrt', 'p30_roll30', 't60_b1024', 't100_b1024',
           't60_b1024-m15', 't60_b1024-m30', 't100_b1024-m15', 't100_b1024-m30'):
    _base = _r.split('-')[0]
    RUNS['A100 ' + _r] = (f'results/a100-{_r}', f'checkpoint/a100-{_base}/params-a100-{_base}.yaml')

DATA_KW = dict(lat_lim=[-18, 5.5], lon_lim=[-74, -44], split_val=True, val_years=(2023, 2024, 2025))


def load_fields(vars_=('mn2t', 'mx2t', 'tp'), deseason=True, var_transform=None):
    """(train_imgs, val_imgs, train_time, val_time) at native resolution."""
    ds = Era5Dataset(transform=None, surf_vars=list(vars_), deseason_standardize=deseason,
                     var_transform=var_transform, **DATA_KW)
    return ds.imgs, ds.validation_imgs, pd.DatetimeIndex(ds.time), pd.DatetimeIndex(ds.val_time)


def run_assignments(results_dir):
    """Cluster id per day (train then val) and number of prototypes."""
    F = torch.load(os.path.join(results_dir, 'F.pt'), weights_only=False)
    Fv = torch.load(os.path.join(results_dir, 'F_val.pt'), weights_only=False)
    return F.argmax(1).cpu().numpy(), Fv.argmax(1).cpu().numpy(), F.shape[1], F, Fv


def oni_table():
    oni = read_enso_data()
    oni.index.name = 'date_period'
    return oni


def monthly_from(cid_tr, cid_v, t_tr, t_v, k, oni):
    df = pd.DataFrame({'cluster_id': np.concatenate([cid_tr, cid_v]),
                       'date': list(t_tr) + list(t_v)})
    return es.build_monthly(df, oni, k)


def _detrend(x):
    t = np.arange(len(x), dtype=float)
    X = np.c_[np.ones_like(t), t]
    if x.ndim == 1:
        return x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
    return x - X @ np.linalg.lstsq(X, x, rcond=None)[0]


def exact_rotation_stats(monthly, k, min_shift=1, detrend=False):
    """Exact circular-rotation null over ALL shifts with circular distance >= min_shift.

    Returns dict with omnibus stat/p, per-cluster corr, raw p, FWER p, and how many
    shifts were admissible. p uses (1 + #exceed) / (1 + #shifts)."""
    Fm = monthly[[f'f{j}' for j in range(k)]].values.astype(float)
    o = monthly['ONI'].values.astype(float)
    if detrend:
        Fm, o = _detrend(Fm), _detrend(o)
    T = len(o)
    shifts = np.array([s for s in range(1, T) if min(s, T - s) >= min_shift])
    obs = es._corr(Fm, o)
    null = np.stack([es._corr(Fm, np.roll(o, s)) for s in shifts])     # (S, K)
    stat = (obs ** 2).sum()
    null_stat = (null ** 2).sum(1)
    n = len(shifts)
    p_omni = (1 + (null_stat >= stat).sum()) / (1 + n)
    p_raw = (1 + (np.abs(null) >= np.abs(obs)).sum(0)) / (1 + n)
    p_fwer = (1 + (np.abs(null).max(1)[:, None] >= np.abs(obs)).sum(0)) / (1 + n)
    return dict(stat=stat, p_omni=p_omni, corr=obs, p_raw=p_raw, p_fwer=p_fwer, n_shifts=n,
                exceed_shifts=shifts[null_stat >= stat])
