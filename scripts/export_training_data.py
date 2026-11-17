"""Export the raw daily ERA5 Amazonia tensor to plain .npy files for another machine.

The slow, download-dependent part of the pipeline is reading ~43 GB of hourly
netCDF and aggregating it to daily fields over the training crop. This script
writes the result of that step only (raw values: K for mn2t/mx2t, mm/day for tp),
one file per variable, so a machine can train with `data.npy_dir: <out_dir>` and
no netCDF. Everything after it (var_transform, temporal_agg, train/val split,
per-month z-score, remove_spatial_mean) still runs in Era5Dataset, so climatology
is always estimated on the configured train years.

    .venv/bin/python scripts/export_training_data.py \
        --config configs/temp_prep_era_anom_30_100ep.yaml --out ../data/era5_npy/amazonia_daily

Output: <var>.npy (T, H, W) float32, times.npy (datetime64[D]), lat.npy, lon.npy,
meta.json (vars, units, crop, sha256 of every file), and SHA256SUMS.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_manager import Era5Dataset  # noqa: E402

UNITS = {'mn2t': 'K (daily min of hourly 2 m temperature minimum)',
         'mx2t': 'K (daily max of hourly 2 m temperature maximum)',
         'tp': 'mm/day (daily sum of hourly total precipitation)'}


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='a training config; its surf_vars and crop are exported')
    ap.add_argument('--vars', nargs='+', default=['mn2t', 'mx2t', 'tp'],
                    help='variables to export (default: all three, so one export serves every experiment)')
    ap.add_argument('--data-path', default='../data/era5/*.nc')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        data_cfg = yaml.safe_load(f)['data']
    lat_lim, lon_lim = data_cfg['lat_limit'], data_cfg['lon_limit']

    # Raw daily tensor exactly as training reads it (uses/creates the .daily_cache).
    ds = Era5Dataset(transform=None, data_path=args.data_path, surf_vars=args.vars,
                     lat_lim=lat_lim, lon_lim=lon_lim, split_val=False,
                     deseason_standardize=False, var_transform=None)
    x = ds.imgs.numpy()  # (T, V, H, W)
    times = np.asarray(ds.time.values, dtype='datetime64[D]')

    # Grid coordinates (the daily cache does not store them).
    import xarray as xr
    files = ds._resolve_var_files(args.data_path, args.vars)
    with xr.open_dataset(files[args.vars[0]][0]) as nc:
        nc = ds._crop(nc, lat_lim, lon_lim)
        lat, lon = nc.latitude.values.astype(np.float32), nc.longitude.values.astype(np.float32)
    assert (len(lat), len(lon)) == x.shape[2:], f'grid {len(lat)}x{len(lon)} vs tensor {x.shape[2:]}'

    os.makedirs(args.out, exist_ok=True)
    written = {}
    for i, var in enumerate(args.vars):
        p = os.path.join(args.out, f'{var}.npy')
        np.save(p, np.ascontiguousarray(x[:, i], dtype=np.float32))
        written[p] = x[:, i]
    for name, arr in (('times', times), ('lat', lat), ('lon', lon)):
        np.save(os.path.join(args.out, f'{name}.npy'), arr)

    names = sorted(f for f in os.listdir(args.out) if f.endswith('.npy'))
    sums = {f: sha256(os.path.join(args.out, f)) for f in names}
    meta = {
        'vars': list(args.vars), 'units': {v: UNITS.get(v, '') for v in args.vars},
        'shape': {'T': int(x.shape[0]), 'H': int(x.shape[2]), 'W': int(x.shape[3])},
        'first_day': str(times[0]), 'last_day': str(times[-1]),
        'lat_limit': lat_lim, 'lon_limit': lon_lim,
        'lat_range': [float(lat.min()), float(lat.max())], 'lon_range': [float(lon.min()), float(lon.max())],
        'source': 'ERA5 hourly single levels, aggregated by Era5Dataset (DAILY_AGG, VAR_SCALE)',
        'per_var_mean': {v: float(x[:, i].mean()) for i, v in enumerate(args.vars)},
        'sha256': sums,
    }
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(args.out, 'SHA256SUMS'), 'w') as f:
        f.writelines(f'{h}  {n}\n' for n, h in sums.items())
    print(json.dumps({k: meta[k] for k in ('vars', 'shape', 'first_day', 'last_day', 'per_var_mean')}, indent=2))
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
