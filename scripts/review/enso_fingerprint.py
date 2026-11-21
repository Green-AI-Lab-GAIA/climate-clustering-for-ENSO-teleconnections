"""Does ENSO have a spatial fingerprint over Amazonia beyond the regional mean? (no clustering)

Monthly means of the per-pixel, per-calendar-month z-scored anomalies (tp: cbrt), linear trend removed per
pixel (fitted on train months). For each pixel (i, j) and variable v:
  raw      : r_ij      = partial corr( X_ij(month), ONI | TNA )
  pattern  : R_ij      = X_ij - (a_ij + b_ij * L(month)),  L = domain-mean temperature anomaly of the month
             r_ij^pat  = partial corr( R_ij, ONI | TNA )        -> ENSO signal that a regional mean cannot carry
ONI, TNA, L detrended too. Field significance: exact circular rotation of ONI (|shift| >= 24 months);
statistics = fraction of area with |r| above the nominal 5 % threshold, and max |r|; p = share of rotations
at least as large. Done for all months and per season (DJF, MAM, JJA, SON) with ONI at lag 0 and lag 3
(ONI leading). Precip pattern also removes the precip domain mean.
usage: .venv/bin/python -I scripts/review/enso_fingerprint.py scripts/review <out_dir>
"""
import sys, os
sys.path.insert(0, sys.argv[1])
from review_lib import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = sys.argv[2]; os.makedirs(out, exist_ok=True)
X, Xv, t_tr, t_v = load_fields(('mn2t', 'mx2t', 'tp'), var_transform={'tp': 'cbrt'})
XA = torch.cat([X, Xv]).numpy(); dates = pd.DatetimeIndex(list(t_tr) + list(t_v))
per = dates.to_period('M'); months = per.unique()
C, H, W = XA.shape[1:]
M = np.stack([XA[per == p].mean(0) for p in months])                 # (months, C, H, W)
ntr = int((months <= pd.Period(t_tr[-1], 'M')).sum())
tt = np.arange(len(months)) / 120.0
D = np.c_[np.ones(len(months)), tt]
Mf = M.reshape(len(months), -1)
coef = np.linalg.lstsq(D[:ntr], Mf[:ntr], rcond=None)[0]
Mf = Mf - D @ coef
M = Mf.reshape(M.shape)
print(f'removed trend (sigma/decade, domain mean): Tmin {coef[1].reshape(C, H, W)[0].mean():+.2f} '
      f'Tmax {coef[1].reshape(C, H, W)[1].mean():+.2f} P {coef[1].reshape(C, H, W)[2].mean():+.2f}', flush=True)

oni = oni_table()
oni_s = oni.reindex(months)['ONI'].astype(float).values if 'ONI' in oni else oni.reindex(months).iloc[:, 0].astype(float).values


def read_psl(path):
    rows = []
    for line in open(path):
        p = line.split()
        if len(p) == 13 and p[0].isdigit():
            rows += [(pd.Period(f'{p[0]}-{m:02d}', 'M'), float(v)) for m, v in enumerate(p[1:], 1)]
    s = pd.Series(dict(rows)); return s[s > -99]


tna_s = read_psl('../data/indices/tna.data').reindex(months).values
ok = ~np.isnan(oni_s) & ~np.isnan(tna_s)
res = lambda y, L: y - L @ np.linalg.lstsq(L, y, rcond=None)[0]
L0 = np.c_[np.ones(len(months)), tt]
oni_d, tna_d = res(np.nan_to_num(oni_s), L0), res(np.nan_to_num(tna_s), L0)
LT = M[:, :2].mean((1, 2, 3)); LP = M[:, 2].mean((1, 2))
T = M[:, :2].mean(1)                                                  # (months, H, W) mean of Tmin, Tmax
fields = {'T': (T, LT), 'P': (M[:, 2], LP)}


def corr_cols(A, b):
    A = A - A.mean(0); b = b - b.mean()
    return (A * b[:, None]).sum(0) / np.sqrt((A ** 2).sum(0) * (b ** 2).sum() + 1e-12)


def analyse(mask, lag, label):
    """returns dict var -> (raw r map, pattern r map, stats)"""
    o = np.roll(oni_d, lag)                                          # ONI leading by `lag` months
    idx = np.where(mask & ok & (np.arange(len(months)) >= lag))[0]
    out_ = {}
    for v, (F, Lv) in fields.items():
        Ff = F.reshape(len(months), -1)[idx]
        Lc = np.c_[np.ones(len(idx)), tna_d[idx]]
        raw = res(Ff, Lc)
        pat = res(Ff, np.c_[Lc, LT[idx]] if v == 'T' else np.c_[Lc, LT[idx], LP[idx]])
        oc = res(o[idx], Lc)
        ocp = res(o[idx], np.c_[Lc, LT[idx]] if v == 'T' else np.c_[Lc, LT[idx], LP[idx]])
        n = len(idx); rcrit = 1.96 / np.sqrt(n)
        rr, rp = corr_cols(raw, oc), corr_cols(pat, ocp)
        lvl_r = corr_cols(res(Lv[idx], Lc)[:, None], oc)[0]
        stats = {}
        for nm, A, ob in (('raw', raw, oc), ('pattern', pat, ocp)):
            r_ = corr_cols(A, ob)
            area, mx = (np.abs(r_) > rcrit).mean(), np.abs(r_).max()
            null_a, null_m = [], []
            for s in range(len(months)):
                if min(s, len(months) - s) < 24: continue
                obr = np.roll(o, lag + s)[idx]
                obr = res(obr, Lc) if nm == 'raw' else res(obr, np.c_[Lc, LT[idx]] if v == 'T' else np.c_[Lc, LT[idx], LP[idx]])
                rn = corr_cols(A, obr); null_a.append((np.abs(rn) > rcrit).mean()); null_m.append(np.abs(rn).max())
            null_a, null_m = np.array(null_a), np.array(null_m)
            stats[nm] = dict(area=area, p_area=(1 + (null_a >= area).sum()) / (1 + len(null_a)), max=mx,
                             p_max=(1 + (null_m >= mx).sum()) / (1 + len(null_m)),
                             mean_r2=(r_ ** 2).mean())
        out_[v] = (rr.reshape(H, W), rp.reshape(H, W), stats, lvl_r, n)
    return out_


seasons = {'all': np.ones(len(months), bool)}
for nm, ms in (('DJF', (12, 1, 2)), ('MAM', (3, 4, 5)), ('JJA', (6, 7, 8)), ('SON', (9, 10, 11))):
    seasons[nm] = np.isin(months.month, ms)
lines = ['| season | ONI lag | var | n months | r(level, ONI) | raw: area sig (p) | raw: mean r2 | pattern: area sig (p) | pattern: max abs r (p) | pattern: mean r2 |',
         '|---|---|---|---|---|---|---|---|---|---|']
maps = {}
for sn, mask in seasons.items():
    for lag in (0, 3):
        a = analyse(mask, lag, sn)
        for v, (rr, rp, st, lr, n) in a.items():
            maps[(sn, lag, v)] = (rr, rp)
            lines.append(f'| {sn} | {lag} | {v} | {n} | {lr:+.2f} | {st["raw"]["area"]:.0%} ({st["raw"]["p_area"]:.3f}) | '
                         f'{st["raw"]["mean_r2"]:.3f} | {st["pattern"]["area"]:.0%} ({st["pattern"]["p_area"]:.3f}) | '
                         f'{st["pattern"]["max"]:.2f} ({st["pattern"]["p_max"]:.3f}) | {st["pattern"]["mean_r2"]:.3f} |')
        print(lines[-2], lines[-1], sep='\n', flush=True)

for lag in (0, 3):
    fig, ax = plt.subplots(4, 5, figsize=(17, 12), constrained_layout=True)
    for j, sn in enumerate(seasons):
        for i, (v, kind) in enumerate((('T', 0), ('T', 1), ('P', 0), ('P', 1))):
            im = ax[i, j].imshow(maps[(sn, lag, v)][kind], cmap='RdBu_r' if v == 'T' else 'BrBG', vmin=-0.6, vmax=0.6)
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            ax[i, j].set_title(f'{sn}: {v} {"raw" if kind == 0 else "minus regional mean"}', fontsize=9)
            if j == 4: fig.colorbar(im, ax=ax[i, j], shrink=0.8, label='partial r with ONI | TNA')
    fig.suptitle(f'ENSO fingerprint over Amazonia (monthly, detrended), ONI lag {lag}: raw vs after removing the regional mean')
    fig.savefig(f'{out}/fingerprint_lag{lag}.png', dpi=100); plt.close(fig)
open(f'{out}/fingerprint.md', 'w').write('\n'.join(lines) + '\n')
