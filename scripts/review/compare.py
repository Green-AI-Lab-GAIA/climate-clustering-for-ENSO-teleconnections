"""Same metrics for every partition (MSN runs + baselines), on the same fields/days.

Dispersion (train days, native 94x120 grid, deseasonalised sigma units):
  eta2_Tmin/Tmax/P : between-cluster / total variance per channel (P = cbrt-z precip)
  eta2_level       : eta2 of the daily domain-mean temperature anomaly (the 1-D "level")
  eta2_pattern     : eta2 of the temperature field after removing each day's domain mean
  pc_med / pc_neg  : centred pattern correlation of each day with its own cluster
                     composite (temperature) - median, and % of days with corr < 0
  misfit           : % of days whose field is closer (Euclidean) to another composite
ENSO (monthly occupancy, all days, rotations restricted to |shift| >= 24 months):
  omni_S, omni_p, nFWER, max|r|, nFWER_detr (linear trend removed from occupancy & ONI)
usage: compare.py <scratch_dir> <review_dir> <out_csv> [labels...]
"""
import sys, os, glob
sys.path.insert(0, sys.argv[1])
from review_lib import *

rev, out_csv = sys.argv[2], sys.argv[3]
only = set(sys.argv[4:])

X, Xv, t_tr, t_v = load_fields(('mn2t', 'mx2t', 'tp'), var_transform={'tp': 'cbrt'})
T = X[:, :2].reshape(len(X), -1)                       # temperature, flattened (n, 2*H*W)
P = X[:, 2].reshape(len(X), -1)
oni = oni_table()


def eta2(A, cid, k):
    gm = A.mean(0)
    tot = ((A - gm) ** 2).sum().item()
    bet = 0.0
    for j in range(k):
        m = torch.as_tensor(cid == j)
        if m.any():
            bet += int(m.sum()) * ((A[m].mean(0) - gm) ** 2).sum().item()
    return bet / tot


def sources():
    for name, (d, _) in RUNS.items():
        if os.path.exists(f'{d}/F.pt'):
            ctr, cv, k, *_ = run_assignments(d)
            yield 'MSN ' + name, ctr, cv, k
    for f in sorted(glob.glob(f'{rev}/baseline_*.npz')):
        z = np.load(f)
        yield os.path.basename(f)[9:-4], z['cid_tr'], z['cid_v'], int(z['k'])


rows = []
for name, ctr, cv, k in sources():
    if only and name not in only:
        continue
    off = len(t_tr) - len(ctr)                         # the paper run starts in 1981
    assert off in (0, 366), (name, off)
    Tn, Pn, tt = T[off:], P[off:], t_tr[off:]
    cid = np.asarray(ctr)
    level = Tn.mean(1, keepdim=True)
    comp = torch.stack([Tn[torch.as_tensor(cid == j)].mean(0) for j in range(k)])   # (k, d)
    # pattern correlation of each day with its own composite
    a = Tn - Tn.mean(1, keepdim=True); c = comp - comp.mean(1, keepdim=True)
    own = c[torch.as_tensor(cid)]
    pc = (a * own).sum(1) / (a.norm(dim=1) * own.norm(dim=1) + 1e-9)
    # Euclidean misfit
    d2 = (Tn ** 2).sum(1, keepdim=True) - 2 * Tn @ comp.T + (comp ** 2).sum(1)[None]
    misfit = (d2.argmin(1).numpy() != cid).mean()
    m = monthly_from(ctr, cv, tt, t_v, k, oni)
    r = exact_rotation_stats(m, k, min_shift=24)
    rd = exact_rotation_stats(m, k, min_shift=24, detrend=True)
    # Partial test: does regime occupancy carry ENSO information BEYOND the month's
    # domain-mean temperature level? Residualise occupancy and ONI on the monthly
    # mean level (+ intercept), rotate ONI (|shift| >= 24), re-residualise each time.
    lvl_day = pd.Series(np.concatenate([level[:, 0].numpy(), Xv[:, :2].mean((1, 2, 3)).numpy()]),
                        index=pd.DatetimeIndex(list(tt) + list(t_v)).to_period('M'))
    lvl_m = lvl_day.groupby(level=0).mean().reindex(m.index).values
    Fm = m[[f'f{j}' for j in range(k)]].values.astype(float)
    Lx = np.c_[np.ones(len(lvl_m)), lvl_m]
    res = lambda y: y - Lx @ np.linalg.lstsq(Lx, y, rcond=None)[0]
    Fr, o = res(Fm), m['ONI'].values.astype(float)
    Tm = len(o)
    obs_p = es._corr(Fr, res(o))
    shifts = [s for s in range(1, Tm) if min(s, Tm - s) >= 24]
    nullp = np.stack([es._corr(Fr, res(np.roll(o, s))) for s in shifts])
    pf_part = (1 + (np.abs(nullp).max(1)[:, None] >= np.abs(obs_p)).sum(0)) / (1 + len(shifts))
    stat_p = (obs_p ** 2).sum(); p_omni_part = (1 + ((nullp ** 2).sum(1) >= stat_p).sum()) / (1 + len(shifts))
    rows.append(dict(
        r_level_oni=float(np.corrcoef(lvl_m, o)[0, 1]),
        omni_p_partial=p_omni_part, nFWER_partial=int((pf_part < 0.05).sum()),
        max_r_partial=float(np.abs(obs_p).max()),
        partition=name, K=k, days=len(cid) + len(cv),
        size_min=np.bincount(cid, minlength=k).min() / len(cid) * 100,
        eta2_Tmin=eta2(Tn[:, :Tn.shape[1] // 2], cid, k) * 100,
        eta2_Tmax=eta2(Tn[:, Tn.shape[1] // 2:], cid, k) * 100,
        eta2_P=eta2(Pn, cid, k) * 100,
        eta2_level=eta2(level, cid, k) * 100,
        eta2_pattern=eta2(Tn - level, cid, k) * 100,
        pc_med=float(pc.median()), pc_neg=float((pc < 0).float().mean()) * 100,
        misfit=misfit * 100,
        omni_S=r['stat'], omni_p=r['p_omni'], nFWER=int((r['p_fwer'] < 0.05).sum()),
        max_r=float(np.abs(r['corr']).max()), nFWER_detr=int((rd['p_fwer'] < 0.05).sum()),
        fwer_clusters=' '.join(f'{j}({r["corr"][j]:+.2f})' for j in np.where(r['p_fwer'] < 0.05)[0]),
    ))
    print(f'done {name}', flush=True)

df = pd.DataFrame(rows).set_index('partition')
df.to_csv(out_csv)
pd.set_option('display.width', 250, 'display.max_columns', 30)
print(df.drop(columns=['fwer_clusters']).round(2).to_string())
print(df['fwer_clusters'].to_string())
