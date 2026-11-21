"""Temperature regimes x precipitation regimes vs ENSO.

For each (temperature partition, precipitation partition) pair:
  1. dependence of the two daily labelings (NMI, Cramer's V): does rain add a new axis?
  2. how well monthly occupancies predict ONI (ridge, 10 contiguous-block CV folds, R^2), for
     levels only (domain-mean T and P anomaly), T occupancy, P occupancy, T+P, T+P+levels
  3. combined regimes (day = (temperature cluster A, precip cluster B)): monthly occupancy of every
     frequent pair vs ONI with the exact rotation null (|shift| >= 24, FWER over pairs); for the
     significant pairs, r of A alone and B alone, the partial r of the pair after removing A and B
     occupancy (rotation p), and the lift in strong El Nino / strong La Nina months (|ONI| >= 1.5)
  4. composite maps (Tmax and precip anomaly) of the strongest El Nino and La Nina pairs
usage: .venv/bin/python -I scripts/review/joint_tp.py scripts/review <out_dir> "<T>|<P>" ...
  T/P: a RUNS label (e.g. "A100 t15_s0") or a baseline npz name (e.g. kmeans_T_15)
"""
import sys, os, re
sys.path.insert(0, sys.argv[1])
from review_lib import *
from sklearn.metrics import normalized_mutual_info_score
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = sys.argv[2]; os.makedirs(out, exist_ok=True)
REV = 'docs/review_2026-11'
X, Xv, t_tr, t_v = load_fields(('mn2t', 'mx2t', 'tp'), var_transform={'tp': 'cbrt'})
dates = pd.DatetimeIndex(list(t_tr) + list(t_v))
XA = torch.cat([X, Xv])                                          # (days, 3, H, W)
lvlT = XA[:, :2].mean((1, 2, 3)).numpy(); lvlP = XA[:, 2].mean((1, 2)).numpy()
oni = oni_table()
MIN_SHIFT = 24


def labels(name):
    if name in RUNS:
        ctr, cv, k, *_ = run_assignments(RUNS[name][0])
    else:
        z = np.load(f'{REV}/baseline_{name}.npz'); ctr, cv, k = z['cid_tr'], z['cid_v'], int(z['k'])
    c = np.r_[np.asarray(ctr), np.asarray(cv)]
    assert len(c) == len(dates), (name, len(c), len(dates))
    return c, k


def monthly_occ(cid, k):
    return monthly_from(cid[:len(t_tr)], cid[len(t_tr):], t_tr, t_v, k, oni)


def occ(m, k):
    return m[[f'f{j}' for j in range(k)]].values.astype(float)


def cramers_v(a, b):
    ct = pd.crosstab(a, b).values.astype(float)
    e = ct.sum(1, keepdims=True) * ct.sum(0, keepdims=True) / ct.sum()
    chi2 = ((ct - e) ** 2 / e).sum()
    return np.sqrt(chi2 / ct.sum() / (min(ct.shape) - 1))


def cv_r2(P, y, folds=10):
    pred = np.zeros_like(y)
    for tr, te in KFold(folds, shuffle=False).split(P):        # contiguous blocks of months
        mdl = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(P[tr], y[tr])
        pred[te] = mdl.predict(P[te])
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(), np.corrcoef(pred, y)[0, 1]


def rot_partial(f, o, C):
    """corr(f, o) after residualising both on C (+ intercept); exact rotation p (|shift| >= MIN_SHIFT)."""
    L = np.c_[np.ones(len(o)), C]
    res = lambda y: y - L @ np.linalg.lstsq(L, y, rcond=None)[0]
    fr = res(f)
    r = es._corr(fr[:, None], res(o))[0]
    T = len(o)
    null = np.array([es._corr(fr[:, None], res(np.roll(o, s)))[0] for s in range(1, T) if min(s, T - s) >= MIN_SHIFT])
    return r, (1 + (np.abs(null) >= abs(r)).sum()) / (1 + len(null))


report = []
for spec in sys.argv[3:]:
    nT, nP = spec.split('|')
    cT, kT = labels(nT); cP, kP = labels(nP)
    tag = re.sub(r'[^A-Za-z0-9]+', '_', f'{nT}__{nP}').strip('_')
    L = [f'### {nT}  x  {nP}', '']
    # 1. dependence
    L.append(f'daily labels: NMI = {normalized_mutual_info_score(cT, cP):.3f}, Cramer V = {cramers_v(cT, cP):.3f} '
             f'(0 = independent, 1 = one determines the other)')
    # 2. ONI prediction
    mT, mP = monthly_occ(cT, kT), monthly_occ(cP, kP)
    assert (mT.index == mP.index).all()
    o = mT['ONI'].values.astype(float)
    lv = pd.DataFrame({'T': lvlT, 'P': lvlP}, index=dates.to_period('M')).groupby(level=0).mean().reindex(mT.index).values
    sets = {'levels (domain-mean T, P)': lv, 'T occupancy': occ(mT, kT), 'P occupancy': occ(mP, kP),
            'T + P occupancy': np.c_[occ(mT, kT), occ(mP, kP)],
            'T + P occupancy + levels': np.c_[occ(mT, kT), occ(mP, kP), lv]}
    L.append('ONI from monthly predictors (ridge, 10 contiguous CV folds): ' + '; '.join(
        f'{k} R2={cv_r2(v, o)[0]:.2f}' for k, v in sets.items()))
    # 3. combined regimes
    pair = cT * kP + cP
    freq = np.bincount(pair, minlength=kT * kP) / len(pair)
    keep = np.where(freq >= 0.003)[0]                           # pairs with >= 0.3 % of days (~50 days)
    remap = -np.ones(kT * kP, int); remap[keep] = np.arange(len(keep))
    pid = remap[pair]
    dfp = pd.DataFrame({'cluster_id': pid, 'date': dates})
    mp = es.build_monthly(dfp[dfp.cluster_id >= 0], oni, len(keep)).reindex(mT.index).fillna(0.0)
    mp['ONI'] = o
    r = exact_rotation_stats(mp, len(keep), min_shift=MIN_SHIFT)
    rT = es._corr(occ(mT, kT), o); rP = es._corr(occ(mP, kP), o)
    strong_en, strong_ln = o >= 1.5, o <= -1.5
    Fp = occ(mp, len(keep))
    L.append(f'combined regimes: {len(keep)} pairs with >= 0.3 % of days ({freq[keep].sum():.0%} of days); '
             f'FWER-significant pairs: {int((r["p_fwer"] < 0.05).sum())}; strong El Nino months {strong_en.sum()}, '
             f'strong La Nina months {strong_ln.sum()}')
    L.append('')
    L.append('| pair (T, P) | days % | r pair | p FWER | r T alone | r P alone | partial r (pair beyond T, P) | p | lift strong EN | lift strong LN |')
    L.append('|---|---|---|---|---|---|---|---|---|---|')
    sig = np.where(r['p_fwer'] < 0.05)[0]
    top = sorted(set(sig) | set(np.argsort(r['corr'])[:3]) | set(np.argsort(r['corr'])[-3:]), key=lambda j: -r['corr'][j])
    rows = []
    for j in top:
        a, b = divmod(keep[j], kP)
        pr, pp = rot_partial(Fp[:, j], o, np.c_[occ(mT, kT)[:, a], occ(mP, kP)[:, b]])
        liftE = Fp[strong_en, j].mean() / Fp[:, j].mean(); liftL = Fp[strong_ln, j].mean() / Fp[:, j].mean()
        rows.append(dict(T=a, P=b, r=r['corr'][j], p=r['p_fwer'][j], rT=rT[a], rP=rP[b], pr=pr, pp=pp))
        L.append(f'| T{a} + P{b} | {freq[keep[j]]*100:.1f} | {r["corr"][j]:+.2f} | {r["p_fwer"][j]:.3f} | {rT[a]:+.2f} | '
                 f'{rP[b]:+.2f} | {pr:+.2f} | {pp:.3f} | {liftE:.1f}x | {liftL:.1f}x |')
    # 4. composites of the two strongest El Nino and La Nina pairs
    ext = [rows[0], rows[1], rows[-2], rows[-1]]
    fig, ax = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for i, rr in enumerate(ext):
        m_ = torch.as_tensor((cT == rr['T']) & (cP == rr['P']))
        for row, (ch, nm, cmap) in enumerate(((1, 'Tmax anomaly (sigma)', 'RdBu_r'), (2, 'precip anomaly (cbrt sigma)', 'BrBG'))):
            im = ax[row, i].imshow(XA[m_, ch].mean(0).numpy(), cmap=cmap, vmin=-1.5, vmax=1.5)
            ax[row, i].set_xticks([]); ax[row, i].set_yticks([])
            if row == 0:
                ax[row, i].set_title(f'T{rr["T"]} + P{rr["P"]}: r={rr["r"]:+.2f} (T {rr["rT"]:+.2f}, P {rr["rP"]:+.2f})\n'
                                     f'n={int(m_.sum())} days', fontsize=9)
            if i == 3: fig.colorbar(im, ax=ax[row, i], label=nm, shrink=0.8)
    fig.suptitle(f'{nT} x {nP}: strongest El Nino (left) and La Nina (right) combined regimes')
    fig.savefig(f'{out}/joint_{tag}.png', dpi=110); plt.close(fig)
    L += ['', f'figure: joint_{tag}.png', '']
    print('\n'.join(L), flush=True)
    report += L

open(f'{out}/joint_tp.md', 'w').write('\n'.join(report) + '\n')
