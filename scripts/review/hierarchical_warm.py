"""Two-level regimes: level first (domain-mean temperature anomaly), then pattern within the extreme days.

Level 1: warm days = top QUANT of the daily domain-mean temperature anomaly (thresholds from train days);
         cold days = bottom QUANT.
Level 2: within warm (cold) days, K sub-regimes from the *pattern*:
  eof : k-means on EOFs of [temperature with the day's domain mean removed, cbrt-z precip] (full field)
  msn : k-means on the self-supervised t30_nomean embeddings (pattern-only encoder, cosine)
For each sub-regime: share of days, mean level, monthly frequency vs ONI and vs TNA (exact rotation null,
|shift| >= 24, FWER over the K sub-regimes), the same after linear detrending, the years with most days,
and composite maps (Tmax anomaly incl. level, precip anomaly).
DETREND=1 removes a linear trend per pixel and channel (fitted on train days) before everything, so
"warm" means warm for its time, not "recent" (the z-scored anomalies still contain the warming trend).
partial r: corr with ONI controlling TNA and vice versa (monthly, after detrending if DETREND=1).
usage: [DETREND=1] .venv/bin/python -I scripts/review/hierarchical_warm.py scripts/review <out_dir> [K] [QUANT]
"""
import sys, os
sys.path.insert(0, sys.argv[1])
from review_lib import *
from sklearn.cluster import KMeans
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = sys.argv[2]; os.makedirs(out, exist_ok=True)
K = int(sys.argv[3]) if len(sys.argv) > 3 else 4
QUANT = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
X, Xv, t_tr, t_v = load_fields(('mn2t', 'mx2t', 'tp'), var_transform={'tp': 'cbrt'})
XA = torch.cat([X, Xv]); ntr = len(X)
dates = pd.DatetimeIndex(list(t_tr) + list(t_v))
DETREND = os.environ.get('DETREND') == '1'
if DETREND:
    tt = torch.as_tensor((dates - dates[0]).days.values / 3652.5, dtype=torch.float32)   # decades
    D = torch.stack([torch.ones_like(tt), tt], 1)
    coef = torch.linalg.lstsq(D[:ntr], XA[:ntr].reshape(ntr, -1)).solution               # (2, C*H*W)
    XA = (XA.reshape(len(XA), -1) - D @ coef).reshape(XA.shape)
    trend = coef[1].reshape(XA.shape[1:])
    print(f'removed linear trend: domain-mean Tmin {trend[0].mean():+.2f}, Tmax {trend[1].mean():+.2f}, '
          f'precip {trend[2].mean():+.2f} sigma/decade')
    out = out.rstrip('/') + '_detrended'; os.makedirs(out, exist_ok=True)
lvl = XA[:, :2].mean((1, 2, 3)).numpy()
oni = oni_table()


def read_psl(path):
    rows = []
    for line in open(path):
        p = line.split()
        if len(p) == 13 and p[0].isdigit():
            rows += [(pd.Period(f'{p[0]}-{m:02d}', 'M'), float(v)) for m, v in enumerate(p[1:], 1)]
    s = pd.Series(dict(rows)); return s[s > -99]


tna = read_psl('../data/indices/tna.data')
EMB = os.environ.get('EMB', 'a100-t30_nomean')            # which self-supervised run gives the embedding
EOFVARS = os.environ.get('EOFVARS', 'TP')                 # TP: [T pattern, precip]; T: temperature pattern only
E = torch.cat([torch.load(f'results/{EMB}/E.pt', weights_only=False),
               torch.load(f'results/{EMB}/E_val.pt', weights_only=False)]).float().cpu().numpy()
if EMB != 'a100-t30_nomean' or EOFVARS != 'TP':
    out = out.rstrip('/') + f'_{EMB}_{EOFVARS}'; os.makedirs(out, exist_ok=True)
assert len(E) == len(XA)


def eof_features(idx):
    T = XA[idx, :2]; T = T - T.mean((2, 3), keepdim=True)
    A = torch.cat([T.reshape(len(idx), -1)] + ([XA[idx, 2].reshape(len(idx), -1)] if EOFVARS == 'TP' else []), 1)
    A = A - A[: (idx < ntr).sum()].mean(0, keepdim=True)
    U, S, V = torch.pca_lowrank(A[torch.as_tensor(idx < ntr)], q=40, center=False, niter=4)
    var = (S ** 2).cumsum(0) / (A[torch.as_tensor(idx < ntr)] ** 2).sum()
    npc = min(int((var < 0.8).sum()) + 1, len(var))
    return (A @ V[:, :npc]).numpy(), npc, float(var[npc - 1])


def test(sub, k, index_name):
    """monthly frequency of each sub-regime (share of ALL days) vs an index; exact rotation null."""
    cid = np.full(len(dates), k); cid[sel] = sub
    m = monthly_from(cid[:ntr], cid[ntr:], t_tr, t_v, k + 1, oni)
    if index_name == 'TNA':
        m['ONI'] = tna.reindex(m.index).values
        m = m.dropna(subset=['ONI'])
    r = exact_rotation_stats(m, k, min_shift=24)
    rd = exact_rotation_stats(m, k, min_shift=24, detrend=True)
    return r, rd


def partials(sub, k):
    """(r ONI | TNA, r TNA | ONI) per sub-regime, monthly, linear trend removed from all series."""
    cid = np.full(len(dates), k); cid[sel] = sub
    m = monthly_from(cid[:ntr], cid[ntr:], t_tr, t_v, k + 1, oni)
    m['TNA'] = tna.reindex(m.index).values; m = m.dropna(subset=['TNA'])
    F = m[[f'f{j}' for j in range(k)]].values.astype(float)
    t = np.arange(len(m)); L0 = np.c_[np.ones(len(m)), t]
    res = lambda y, L: y - L @ np.linalg.lstsq(L, y, rcond=None)[0]
    o, a = res(m['ONI'].values, L0), res(m['TNA'].values, L0); F = res(F, L0)
    pO = es._corr(res(F, np.c_[L0, a]), res(o, np.c_[L0, a]))
    pT = es._corr(res(F, np.c_[L0, o]), res(a, np.c_[L0, o]))
    return pO, pT, float(np.corrcoef(o, a)[0, 1])


def composition(sub, k, min_days=5):
    """Second-level question: GIVEN an extreme month, which kind? Share of each sub-regime among the
    month's extreme days (months with >= min_days extreme days), vs detrended ONI and TNA. Rotation null:
    the index is rotated over the full monthly series (|shift| >= 24), then restricted to the same months."""
    cid = np.full(len(dates), k); cid[sel] = sub
    m = monthly_from(cid[:ntr], cid[ntr:], t_tr, t_v, k + 1, oni)
    m['TNA'] = tna.reindex(m.index).values; m = m.dropna(subset=['TNA'])
    days = pd.Series(1, index=dates.to_period('M')).groupby(level=0).sum().reindex(m.index).values
    F = m[[f'f{j}' for j in range(k)]].values * days[:, None]           # extreme days per sub-regime
    n = F.sum(1); keep = n >= min_days
    S = F[keep] / n[keep, None]
    t = np.arange(len(m)); L0 = np.c_[np.ones(len(m)), t]
    res = lambda y, L: y - L @ np.linalg.lstsq(L, y, rcond=None)[0]
    o, a = res(m['ONI'].values, L0), res(m['TNA'].values, L0)
    out = {}
    for nm, x, other in (('ONI', o, a), ('TNA', a, o)):
        def stat(xx, oo):
            Lk = np.c_[np.ones(keep.sum()), oo[keep]]
            return es._corr(res(S, Lk), res(xx[keep], Lk))       # partial on the other index
        obs = stat(x, other)
        T = len(x)
        null = np.stack([stat(np.roll(x, sh), other) for sh in range(1, T) if min(sh, T - sh) >= 24])
        p = (1 + (np.abs(null).max(1)[:, None] >= np.abs(obs)).sum(0)) / (1 + len(null))
        out[nm] = (obs, p)
    return out, int(keep.sum())


lines = []
for side in ('warm', 'cold'):
    q = np.quantile(lvl[:ntr], 1 - QUANT if side == 'warm' else QUANT)
    sel = np.where(lvl >= q if side == 'warm' else lvl <= q)[0]
    yrs = pd.Series(dates[sel].year).value_counts()
    lines += [f'## {side} days: {"top" if side == "warm" else "bottom"} {QUANT:.0%} of the domain-mean T anomaly '
              f'(threshold {q:+.2f} sigma, {len(sel)} days); years with most days: '
              + ', '.join(f'{y} ({n})' for y, n in yrs.head(8).items()), '']
    Z, npc, v = eof_features(sel)
    En = E[sel] / np.linalg.norm(E[sel], axis=1, keepdims=True)
    for method, F in (('eof', Z), ('msn', En)):
        tr = sel < ntr
        km = KMeans(K, n_init=20, random_state=0).fit(F[tr])
        sub = km.predict(F)
        # order sub-regimes by ONI correlation for readability
        r, rd = test(sub, K, 'ONI'); rt, rtd = test(sub, K, 'TNA')
        order = np.argsort(-r['corr']); rank = np.empty(K, int); rank[order] = np.arange(K)
        sub = rank[sub]
        r, rd = test(sub, K, 'ONI'); rt, rtd = test(sub, K, 'TNA')
        pO, pT, rOT = partials(sub, K)
        comp, nmon = composition(sub, K)
        desc = f'k-means on {npc} EOFs ({v:.0%} var) of [T minus daily domain mean{", precip" if EOFVARS == "TP" else ""}]' if method == 'eof' \
            else f'k-means on MSN {EMB} embeddings (self-supervised, pattern-only)'
        lines += [f'### {side} / {method}: {desc}', '', f'(corr ONI-TNA, detrended monthly: {rOT:+.2f})', '',
                  '| sub | days % | mean level | r ONI (p FWER) | r ONI detr. | r TNA (p FWER) | r TNA detr. | r ONI given TNA | r TNA given ONI | top years |',
                  '|---|---|---|---|---|---|---|---|---|---|']
        for j in range(K):
            m_ = sub == j
            ty = pd.Series(dates[sel][m_].year).value_counts().head(5)
            lines.append(f'| {side[0].upper()}{j} | {m_.mean()*100:.0f} | {lvl[sel][m_].mean():+.2f} | '
                         f'{r["corr"][j]:+.2f} ({r["p_fwer"][j]:.3f}) | {rd["corr"][j]:+.2f} | '
                         f'{rt["corr"][j]:+.2f} ({rt["p_fwer"][j]:.3f}) | {rtd["corr"][j]:+.2f} | {pO[j]:+.2f} | {pT[j]:+.2f} | '
                         + ', '.join(f'{y}:{n}' for y, n in ty.items()) + ' |')
        lines += ['', f'Composition among {side} days ({nmon} months with >= 5 {side} days): partial r of each sub-regime share '
                  'with ONI (controlling TNA) and with TNA (controlling ONI), detrended; p = FWER over sub-regimes', '',
                  '| sub | share r ONI given TNA (p) | share r TNA given ONI (p) |', '|---|---|---|']
        for j in range(K):
            lines.append(f'| {side[0].upper()}{j} | {comp["ONI"][0][j]:+.2f} ({comp["ONI"][1][j]:.3f}) | '
                         f'{comp["TNA"][0][j]:+.2f} ({comp["TNA"][1][j]:.3f}) |')
        lines.append('')
        fig, ax = plt.subplots(2, K, figsize=(3.6 * K, 6.2), constrained_layout=True)
        for j in range(K):
            m_ = torch.as_tensor(sub == j)
            for row, (ch, cmap, lab) in enumerate(((1, 'RdBu_r', 'Tmax anomaly (sigma)'), (2, 'BrBG', 'precip anomaly (cbrt sigma)'))):
                im = ax[row, j].imshow(XA[sel][m_, ch].mean(0).numpy(), cmap=cmap, vmin=-2, vmax=2)
                ax[row, j].set_xticks([]); ax[row, j].set_yticks([])
                if row == 0:
                    ax[row, j].set_title(f'{side[0].upper()}{j}: {int(m_.sum())} d, level {lvl[sel][sub == j].mean():+.2f}\n'
                                         f'r ONI {r["corr"][j]:+.2f}, r TNA {rt["corr"][j]:+.2f}', fontsize=9)
                if j == K - 1: fig.colorbar(im, ax=ax[row, j], label=lab, shrink=0.8)
        fig.suptitle(f'{side} days ({QUANT:.0%}), sub-regimes by pattern ({method})')
        fig.savefig(f'{out}/{side}_{method}_K{K}.png', dpi=100); plt.close(fig)


open(f'{out}/hierarchical_K{K}.md', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
