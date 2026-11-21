"""p10_monthly (one label per calendar month of tp sums): ENSO test + baselines on the same months.

Occupancy is one-hot per month, so the per-cluster statistic is the point-biserial corr of the
label indicator with ONI; same exact rotation null (|shift| >= 24) as the daily runs.
Baselines: quantile bins of the monthly domain-mean precip anomaly, and k-means (K=10) on EOFs (90 %).
usage: .venv/bin/python -I scripts/review/a100_monthly.py scripts/review
"""
import sys
sys.path.insert(0, sys.argv[1])
from review_lib import *
import src.inference as inf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

params, ds = inf.read_data('checkpoint/a100-p10_monthly/params-a100-p10_monthly.yaml', validation=True)
from src.data_manager import Era5Dataset
raw = Era5Dataset(transform=None, surf_vars=['tp'], deseason_standardize=True, npy_dir=params['data']['npy_dir'],
                  temporal_agg={'kind': 'monthly'}, lat_lim=params['data']['lat_limit'],
                  lon_lim=params['data']['lon_limit'], split_val=True, val_years=(2023, 2024, 2025))
Z = torch.cat([raw.imgs, raw.validation_imgs])[:, 0].reshape(-1, raw.imgs.shape[-2] * raw.imgs.shape[-1]).numpy()
times = pd.DatetimeIndex(list(raw.time) + list(raw.val_time))
F = torch.load('results/a100-p10_monthly/F.pt', weights_only=False).cpu()
Fv = torch.load('results/a100-p10_monthly/F_val.pt', weights_only=False).cpu()
cid_msn = torch.cat([F.argmax(1), Fv.argmax(1)]).numpy()
k = F.shape[1]
oni = oni_table()
level = Z.mean(1)
print(f'months {len(times)} ({times[0]:%Y-%m}..{times[-1]:%Y-%m}), K={k}')


def report(name, cid):
    df = pd.DataFrame({'cluster_id': cid, 'date': times})
    m = es.build_monthly(df, oni, k)
    r = exact_rotation_stats(m, k, min_shift=24)
    lv = pd.Series(level, index=times.to_period('M')).reindex(m.index).values
    eta = sum((cid == j).sum() * (level[cid == j].mean() - level.mean()) ** 2 for j in range(k)) / ((level - level.mean()) ** 2).sum()
    gm = Z.mean(0); tot = ((Z - gm) ** 2).sum()
    eta_f = sum((cid == j).sum() * ((Z[cid == j].mean(0) - gm) ** 2).sum() for j in range(k)) / tot
    sig = ' '.join(f'{j}({r["corr"][j]:+.2f})' for j in np.where(r['p_fwer'] < 0.05)[0])
    print(f'{name:22s} eta2 field {eta_f*100:5.1f}%  eta2 level {eta*100:5.1f}%  omni S {r["stat"]:.2f} p={r["p_omni"]:.3f}  '
          f'nFWER {int((r["p_fwer"] < 0.05).sum())}  max|r| {np.abs(r["corr"]).max():.2f}  sig: {sig}')
    return m


m = report('MSN p10_monthly', cid_msn)
lv = pd.Series(level, index=times.to_period('M')).reindex(m.index).values
print(f'corr(monthly domain-mean precip anomaly, ONI) = {np.corrcoef(lv, m["ONI"].values)[0, 1]:+.2f}')
report('bins of level', np.searchsorted(np.quantile(level, np.linspace(0, 1, k + 1)[1:-1]), level))
pcs = PCA(0.90, random_state=0).fit_transform(Z)
report('k-means on EOFs', KMeans(k, n_init=20, random_state=0).fit_predict(pcs))
