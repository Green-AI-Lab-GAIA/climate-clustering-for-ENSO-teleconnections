"""Mask-only MSN (tp30_mask, tp60_mask) vs crop MSN (tp30_detr) vs PCA k-means: representativeness (REVIEW §14.15).

Same field and metrics as k_selection.py: z~ (Tmin, Tmax, cbrt tp, per-pixel z per calendar month, per-pixel trend
removed, train fit), full 94x120 field, all N days (train + 2023-2025).
Clusterings compared
  PCA k-means K        : k-means on 60 EOF scores (seeds 0-2)
  MSN-crop native 30   : tp30_detr_s0-2, argmax over the 30 prototypes
  MSN-mask native K    : tp30_mask / tp60_mask, argmax over prototypes ('best' = lowest val loss, 'final' = last epoch)
  MSN-mask k-means K   : k-means on the L2-normalised mask embedding
  MSN-mask 60 -> K     : the 60 native clusters merged into K groups, two ways
       proto : k-means on the 60 prototype vectors (embedding space), each day takes its prototype's group
       comp  : Ward clustering of the 60 cluster mean maps (field space, weighted by size)
Metrics (no ENSO information): eta2, coherence (median day-to-own-map r; share r > 0.5), pat (median max centred
pattern r with another map; 1 = copies), sizes; year-split ARI for k-means variants (k-means refit on even vs odd
train years). ENSO (secondary): ONI R2 (ridge CV on monthly frequencies), n clusters linked to ONI | TNA (FWER).
Also a per-cluster table and maps for the chosen MSN K~10 grouping and PCA K = 10.
usage: .venv/bin/python -I scripts/review/mask_eval.py scripts/review <out_dir>
"""
import sys, os
sys.path.insert(0, sys.argv[1])
from review_lib import *
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from scipy.optimize import linear_sum_assignment
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = sys.argv[2]; os.makedirs(out, exist_ok=True)
X, Xv, t_tr, t_v = load_fields(('mn2t', 'mx2t', 'tp'), var_transform={'tp': 'cbrt'})
XA = torch.cat([X, Xv]); ntr = len(X); N = len(XA); del X, Xv
dates = pd.DatetimeIndex(list(t_tr) + list(t_v))
tt = torch.as_tensor((dates - dates[0]).days.values / 3652.5, dtype=torch.float32)
D = torch.stack([torch.ones_like(tt), tt], 1)
coef = torch.linalg.lstsq(D[:ntr], XA[:ntr].reshape(ntr, -1)).solution
A = XA.reshape(N, -1) - D @ coef; del XA
mu_all = A.mean(0, keepdim=True); SST = float(((A - mu_all) ** 2).sum())
# physical anomalies for tables / maps
R, Rv, _, _ = load_fields(('mn2t', 'mx2t', 'tp'), deseason=False)
RA = torch.cat([R, Rv]); del R, Rv
mon = torch.as_tensor(dates.month.values) - 1
clim = torch.stack([RA[:ntr][mon[:ntr] == m].mean(0) for m in range(12)])
AN = RA - clim[mon]; del RA
cf = torch.linalg.lstsq(D[:ntr], AN[:ntr].reshape(ntr, -1)).solution
AN = (AN.reshape(N, -1) - D @ cf).reshape(AN.shape)


def read_psl(path):
    rows = []
    for line in open(path):
        p = line.split()
        if len(p) == 13 and p[0].isdigit():
            rows += [(pd.Period(f'{p[0]}-{m:02d}', 'M'), float(v)) for m, v in enumerate(p[1:], 1)]
    s = pd.Series(dict(rows)); return s[s > -99]


oni_s = oni_table()['ONI'].astype(float); tna_s = read_psl('../data/indices/tna.data')
per = dates.to_period('M'); last = min(oni_s.index.max(), tna_s.index.max())
months = per.unique(); months = months[months <= last]; nm_ = len(months)
mi = months.get_indexer(per); okd = mi >= 0
days_in = np.bincount(mi[okd], minlength=nm_).astype(float)
res = lambda y, L: y - L @ np.linalg.lstsq(L, y, rcond=None)[0]
L0 = np.c_[np.ones(nm_), np.arange(nm_)]
oni0 = oni_s.reindex(months).values
oni_d = res(oni0, L0); tna_d = res(tna_s.reindex(months).values, L0)


def cv_r2(P, y, folds=10):
    pred = np.zeros_like(y)
    for tr, te in KFold(folds, shuffle=False).split(P):
        pred[te] = RidgeCV(alphas=np.logspace(-3, 4, 15)).fit(P[tr], y[tr]).predict(P[te])
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def raw_freq(lab, k):
    return np.stack([np.bincount(mi[okd][lab[okd] == j], minlength=nm_) for j in range(k)], 1) / days_in[:, None]


def enso(lab, k):
    F0 = raw_freq(lab, k); F = res(F0, L0)
    Lc = np.c_[np.ones(nm_), tna_d]; Fr = res(F, Lc)
    stat = lambda x: es._corr(Fr, res(x, Lc))
    obs = stat(oni_d)
    null = np.stack([stat(np.roll(oni_d, s)) for s in range(1, nm_) if min(s, nm_ - s) >= 24])
    p = (1 + (np.abs(null).max(1)[:, None] >= np.abs(obs)).sum(0)) / (1 + len(null))
    m = F0.mean(0).clip(1e-9)
    return dict(r2=cv_r2(F, oni_d), r=obs, p=p, nsig=int((p < 0.05).sum()),
                liftEN=F0[oni0 >= 0.5].mean(0) / m, liftLN=F0[oni0 <= -0.5].mean(0) / m)


def field_stats(lab, k):
    idx = torch.as_tensor(lab)
    cnt = torch.bincount(idx, minlength=k).float()
    means = torch.zeros(k, A.shape[1]).index_add_(0, idx, A) / cnt[:, None].clamp_min(1)
    ssb = float((cnt[:, None] * (means - mu_all) ** 2).sum())
    Mc = means - means.mean(1, keepdim=True); Mc = Mc / Mc.norm(dim=1, keepdim=True).clamp_min(1e-9)
    cors = []
    for a in range(0, N, 2000):
        Ad = A[a:a + 2000]; Ad = Ad - Ad.mean(1, keepdim=True)
        cors.append(((Ad / Ad.norm(dim=1, keepdim=True).clamp_min(1e-9)) * Mc[idx[a:a + 2000]]).sum(1))
    cors = torch.cat(cors).numpy()
    V = means.numpy().reshape(k, 3, -1)[:, 1:].reshape(k, -1)
    V = V - V.mean(1, keepdims=True); V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    P = V @ V.T; np.fill_diagonal(P, -1)
    sz = cnt.numpy() / N * 100
    return dict(eta2=ssb / SST, coh=float(np.median(cors)), coh05=float((cors > 0.5).mean()),
                pat=float(np.median(P.max(1))), min_size=sz.min(), max_size=sz.max(), means=means.numpy(), cors=cors)


def pca_scores(fit_idx, q=60):
    m = A[fit_idx].mean(0, keepdim=True)
    U, S, V = torch.pca_lowrank(A[fit_idx] - m, q=q, center=False, niter=4)
    return ((A - m) @ V).numpy()


def load_run(run):
    g = lambda f: torch.load(f'results/a100-{run}/{f}.pt', weights_only=False, map_location='cpu')
    E = torch.cat([g('E'), g('E_val')]).float().numpy(); F = torch.cat([g('F'), g('F_val')]).float().numpy()
    return E / np.linalg.norm(E, axis=1, keepdims=True), F


def protos(run):
    tag = 'latest' if run.endswith('_latest') else 'best'; base = run.replace('_latest', '')
    p = torch.load(f'checkpoint/a100-{base}/a100-{base}-{tag}-slim.pth.tar', map_location='cpu', weights_only=False)['prototypes']
    p = p.float().numpy(); return p / np.linalg.norm(p, axis=1, keepdims=True)


train_years = np.unique(dates[:ntr].year)
half = {h: np.where((np.arange(N) < ntr) & np.isin(dates.year, train_years[h::2]))[0] for h in (0, 1)}
Z = pca_scores(np.arange(ntr)); Zh = {h: pca_scores(half[h]) for h in (0, 1)}
MASK = {'tp30_mask': 'MSN-mask30 best (ep57)', 'tp30_mask_latest': 'MSN-mask30 final (ep100)',
        'tp60_mask': 'MSN-mask60 best (ep36)', 'tp60_mask_latest': 'MSN-mask60 final (ep174)'}
RUN = {r: load_run(r) for r in MASK}
CROP = {s: load_run(f'tp30_detr_s{s}') for s in (0, 1, 2)}
print('features ready', flush=True)


def split_ari(Fh, K, s):
    """year-split ARI of k-means on features Fh[h] (dict per half or a fixed array)."""
    L = []
    for h in (0, 1):
        f = Fh[h] if isinstance(Fh, dict) else Fh
        L.append(KMeans(K, n_init=5, random_state=s).fit(f[half[h]]).predict(f))
    return adjusted_rand_score(*L)


def group_proto(run, nat, K, s=0):
    g = KMeans(K, n_init=20, random_state=s).fit(protos(run)).labels_
    return g[nat]


def group_comp(nat, k0, K):
    st = field_stats(nat, k0); M = st['means']; w = np.bincount(nat, minlength=k0)
    # Ward on composites, replicating each composite in proportion to its size is costly; use sqrt(w) scaling instead
    g = AgglomerativeClustering(K, linkage='ward').fit(M * np.sqrt(w / w.mean())[:, None]).labels_
    return g[nat]


rows, labels = [], {}


def add(name, family, K, lab, split=np.nan, seed=0):
    st = field_stats(lab, K); en = enso(lab, K)
    labels[(name, K, seed)] = lab
    rows.append(dict(method=name, family=family, K=K, seed=seed, eta2=st['eta2'], coh=st['coh'], coh05=st['coh05'],
                     pat=st['pat'], min_size=st['min_size'], max_size=st['max_size'], split_ari=split,
                     oni_r2=en['r2'], nsig=en['nsig']))
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)


for K in (10, 30, 60):
    for s in (0, 1, 2):
        add('PCA k-means', 'PCA', K, KMeans(K, n_init=10, random_state=s).fit(Z[:ntr]).predict(Z), split_ari(Zh, K, s), s)
for s in (0, 1, 2):
    add('MSN-crop native', 'MSN-crop', 30, CROP[s][1].argmax(1), seed=s)
    add('MSN-crop k-means', 'MSN-crop', 10, KMeans(10, n_init=10, random_state=0).fit(CROP[s][0][:ntr]).predict(CROP[s][0]),
        split_ari(CROP[s][0], 10, 0), s)
for r, nm in MASK.items():
    E, F = RUN[r]; k0 = F.shape[1]; nat = F.argmax(1)
    add(f'{nm} native', 'MSN-mask', k0, nat)
    for K in (10, 30):
        for s in (0, 1, 2):
            add(f'{nm} k-means', 'MSN-mask', K, KMeans(K, n_init=10, random_state=s).fit(E[:ntr]).predict(E), split_ari(E, K, s), s)
    for K in ((10, 15, 20) if k0 == 60 else (10, 15)):
        for s in (0, 1, 2):
            add(f'{nm} {k0}->K proto', 'MSN-mask', K, group_proto(r, nat, K, s), seed=s)
        add(f'{nm} {k0}->K comp', 'MSN-mask', K, group_comp(nat, k0, K))

df = pd.DataFrame(rows); df.to_csv(f'{out}/mask_eval.csv', index=False, float_format='%.4f')


def seed_ari(name, K):
    L = [labels[k] for k in labels if k[0] == name and k[1] == K]
    if len(L) < 2: return np.nan
    return np.mean([adjusted_rand_score(L[i], L[j]) for i in range(len(L)) for j in range(i + 1, len(L))])


g = df.groupby(['method', 'family', 'K'], sort=False).mean(numeric_only=True).reset_index()
g['seed_ari'] = [seed_ari(m, K) for m, K in zip(g.method, g.K)]
g.to_csv(f'{out}/mask_eval_mean.csv', index=False, float_format='%.4f')

# probe: ONI from monthly means of the features
def monthly_mean(Fe):
    M = np.stack([np.bincount(mi[okd], Fe[okd, c], nm_) for c in range(Fe.shape[1])], 1) / days_in[:, None]
    return res(M, L0)


probe = {'PCA (60 EOFs)': cv_r2(monthly_mean(Z), oni_d), 'MSN-crop s0': cv_r2(monthly_mean(CROP[0][0]), oni_d)}
for r, nm in MASK.items():
    probe[nm] = cv_r2(monthly_mean(RUN[r][0]), oni_d)

lines = ['# Mask-only MSN vs crop MSN vs PCA k-means (detrended Tmin, Tmax, cbrt tp; full field)', '',
         'Means over seeds where there are seeds (k-means seeds for PCA / k-means on a fixed embedding / prototype grouping;'
         ' 3 trained models for MSN-crop native). eta2 = share of field variance explained by the cluster means; '
         'coh = median spatial r of a day with its cluster map; coh>0.5 = share of days with r > 0.5; pat = median '
         'max pattern r with another map (1 = copies); split ARI = k-means refit on even vs odd train years; seed ARI '
         '= agreement across seeds; ONI R2 = out-of-sample R2 from monthly frequencies; n ONI = clusters linked to '
         'ONI | TNA (FWER).', '',
         '| method | K | eta2 | coh | coh>0.5 | pat | min–max size % | split ARI | seed ARI | ONI R2 | n ONI |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
f2 = lambda v, f: '–' if pd.isna(v) else f.format(v)
for _, x in g.iterrows():
    lines.append(f'| {x.method} | {int(x.K)} | {x.eta2:.3f} | {x.coh:.2f} | {x.coh05:.0%} | {x.pat:.2f} | '
                 f'{x.min_size:.1f}–{x.max_size:.1f} | {f2(x.split_ari, "{:.2f}")} | {f2(x.seed_ari, "{:.2f}")} | '
                 f'{x.oni_r2:.3f} | {x.nsig:.1f} |')
lines += ['', '## Probe: ONI R2 from monthly means of the features (no clustering)', '', '| features | ONI R2 |', '|---|---|']
lines += [f'| {k} | {v:.3f} |' for k, v in probe.items()]
open(f'{out}/mask_eval.md', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines), flush=True)

# ---- figure: metric vs K by method family
fig, ax = plt.subplots(1, 5, figsize=(24, 4.4), constrained_layout=True)
sty = {'PCA k-means': ('#1f77b4', 'o-'), 'MSN-crop native': ('#7f7f7f', 's'), 'MSN-crop k-means': ('#7f7f7f', 'o'),
       'MSN-mask60 final (ep174) native': ('#d62728', '*'), 'MSN-mask30 final (ep100) native': ('#d62728', 'P'),
       'MSN-mask60 final (ep174) k-means': ('#ff7f0e', 'o-'), 'MSN-mask60 final (ep174) 60->K proto': ('#2ca02c', 'o-'),
       'MSN-mask60 final (ep174) 60->K comp': ('#9467bd', 'o-')}
for i, (col, t) in enumerate((('eta2', 'field variance explained (η²)'), ('coh', 'median day-to-map r'),
                              ('coh05', 'share of days with r > 0.5'), ('pat', 'max pattern r between maps (lower = more distinct)'),
                              ('oni_r2', 'ONI R² out of sample (secondary)'))):
    for m, (c, mk) in sty.items():
        gg = g[g.method == m].sort_values('K')
        if len(gg): ax[i].plot(gg.K, gg[col], mk, color=c, label=m, ms=9 if '*' in mk else 6)
    ax[i].set_xscale('log'); ax[i].set_xticks([10, 15, 20, 30, 60]); ax[i].set_xticklabels([10, 15, 20, 30, 60])
    ax[i].set_xlabel('K'); ax[i].set_title(t, fontsize=10)
ax[0].legend(fontsize=7)
fig.suptitle('Mask-only MSN (final epoch) vs PCA k-means vs crop MSN')
fig.savefig(f'{out}/mask_eval.png', dpi=100); plt.close(fig)


# ---- per-cluster table + maps: PCA K=10 vs best MSN-mask K=10 grouping
def describe(lab, K, tag, title):
    en = enso(lab, K); st = field_stats(lab, K)
    rows_ = []
    for j in range(K):
        dj = torch.as_tensor(np.where(lab == j)[0]); cl = clim[mon[dj]]; dd = dates[lab == j]
        sea = np.array([np.isin(dd.month, ms).mean() for ms in ((12, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11))])
        rows_.append(dict(cluster=j, days=(lab == j).mean() * 100, Tmin=AN[dj, 0].mean().item(), Tmax=AN[dj, 1].mean().item(),
                          P_pct=100 * AN[dj, 2].sum().item() / cl[:, 2].sum().item(), coh=np.median(st['cors'][lab == j]),
                          season=['DJF', 'MAM', 'JJA', 'SON'][sea.argmax()], season_share=sea.max() * 100,
                          r_oni=en['r'][j], p=en['p'][j], liftEN=en['liftEN'][j], liftLN=en['liftLN'][j]))
    d = pd.DataFrame(rows_).sort_values('Tmax', ascending=False).reset_index(drop=True)
    d.to_csv(f'{out}/clusters_{tag}.csv', index=False, float_format='%.3f')
    L = [f'### {title}', '', '| # | days % | Tmin °C | Tmax °C | P % | coh | peak season | r ONI (p) | lift EN | lift LN |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for i, x in d.iterrows():
        b = '**' if x.p < 0.05 else ''
        L.append(f'| {i} | {x.days:.1f} | {x.Tmin:+.2f} | {x.Tmax:+.2f} | {x.P_pct:+.0f} | {x.coh:.2f} | {x.season} ({x.season_share:.0f} %) | '
                 f'{b}{x.r_oni:+.2f} ({x.p:.3f}){b} | {x.liftEN:.2f} | {x.liftLN:.2f} |')
    if K > 15: return L, d
    fig, ax = plt.subplots(2, K, figsize=(2.3 * K, 5.0), constrained_layout=True)
    for i, x in d.iterrows():
        dj = torch.as_tensor(np.where(lab == int(x.cluster))[0])
        im1 = ax[0, i].imshow(AN[dj, 1].mean(0).numpy(), cmap='RdBu_r', vmin=-3, vmax=3)
        im2 = ax[1, i].imshow((100 * AN[dj, 2].sum(0) / clim[mon[dj], 2].sum(0).clamp_min(1e-3)).numpy(), cmap='BrBG', vmin=-80, vmax=80)
        star = '*' if x.p < 0.05 else ''
        ax[0, i].set_title(f'#{i} {x.days:.1f}% {x.season}\nTx {x.Tmax:+.1f}°C P {x.P_pct:+.0f}%\ncoh {x.coh:.2f} rONI {x.r_oni:+.2f}{star}', fontsize=8)
        for a in ax[:, i]: a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im1, ax=ax[0], location='left', shrink=0.8, label='Tmax °C')
    fig.colorbar(im2, ax=ax[1], location='left', shrink=0.8, label='P % of clim.')
    fig.suptitle(title); fig.savefig(f'{out}/maps_{tag}.png', dpi=90); plt.close(fig)
    return L, d


best = 'MSN-mask60 final (ep174) 60->K comp'   # the user's idea: merge the 60 equal-size prototypes into 10 groups
pick = [k for k in labels if k[0] == best and k[1] == 10][0]
L1, d1 = describe(labels[('PCA k-means', 10, 0)], 10, 'pca10', 'PCA k-means, K = 10 (seed 0)')
L2, d2 = describe(labels[pick], 10, 'msnmask10', f'{best}, K = 10 (seed {pick[2]})')
fin = labels[('MSN-mask60 final (ep174) native', 60, 0)]
L3, _ = describe(fin, 60, 'msnmask60_native', 'MSN-mask60 final, native 60 prototypes')
# matching PCA10 <-> chosen MSN10
Ca = field_stats(labels[('PCA k-means', 10, 0)], 10)['means']; Cb = field_stats(labels[pick], 10)['means']
cc = np.corrcoef(Ca, Cb)[:10, 10:]; r_, c_ = linear_sum_assignment(-cc)
lines += ['', f'## Per-cluster tables (ordered by Tmax); MSN-mask K = 10 shown: {best}', ''] + L1 + [''] + L2 + [
    '', f'PCA10 ↔ MSN10 matching (Hungarian on mean maps): r = {", ".join(f"{v:.2f}" for v in sorted(cc[r_, c_])[::-1])}; '
        f'daily ARI {adjusted_rand_score(labels[("PCA k-means", 10, 0)], labels[pick]):.2f}', ''] + L3
open(f'{out}/mask_eval.md', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines[-40:]), flush=True)
