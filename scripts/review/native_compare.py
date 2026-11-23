"""MSN native prototypes vs EOF k-means at the same K, on the same detrended T + P anomalies.

MSN: tp30_detr_s{0,1,2}, cluster = argmax over its 30 prototypes (no k-means on the embedding).
EOF: k-means (K = 30, seeds 0-2) on 60 EOFs of z~ (Tmin, Tmax, cbrt tp; per-pixel trend removed; train fit).
Per cluster: share of days, Tmin / Tmax anomaly (degC), precip anomaly (mm/day and % of the calendar-month
climatology), peak season, monthly frequency vs ONI | TNA (detrended, exact rotation |shift| >= 24, FWER over
clusters), lift in El Nino / La Nina months. Cross-seed reproducibility (Hungarian match on composites, ARI);
MSN <-> EOF cluster matching. Figures: map grids per method (Tmax degC, P %), ordered by r with ONI; a
summary panel.
usage: .venv/bin/python -I scripts/review/native_compare.py scripts/review <out_dir> [K]
"""
import sys, os
sys.path.insert(0, sys.argv[1])
from review_lib import *
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = sys.argv[2]; os.makedirs(out, exist_ok=True)
K = int(sys.argv[3]) if len(sys.argv) > 3 else 30
X, Xv, t_tr, t_v = load_fields(('mn2t', 'mx2t', 'tp'), var_transform={'tp': 'cbrt'})
XA = torch.cat([X, Xv]); ntr = len(X); N = len(XA); del X, Xv
dates = pd.DatetimeIndex(list(t_tr) + list(t_v))
tt = torch.as_tensor((dates - dates[0]).days.values / 3652.5, dtype=torch.float32)
D = torch.stack([torch.ones_like(tt), tt], 1)
coef = torch.linalg.lstsq(D[:ntr], XA[:ntr].reshape(ntr, -1)).solution
XA = (XA.reshape(N, -1) - D @ coef).reshape(XA.shape)
# physical anomalies (detrended): Tmin, Tmax in degC, tp in mm/day
R, Rv, _, _ = load_fields(('mn2t', 'mx2t', 'tp'), deseason=False)
RA = torch.cat([R, Rv]); del R, Rv
assert len(RA) == N
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
per = dates.to_period('M')
last = min(oni_s.index.max(), tna_s.index.max())
months = per.unique(); months = months[months <= last]; nm_ = len(months)
mi = months.get_indexer(per); okd = mi >= 0
days_in = np.bincount(mi[okd], minlength=nm_).astype(float)
res = lambda y, L: y - L @ np.linalg.lstsq(L, y, rcond=None)[0]
L0 = np.c_[np.ones(nm_), np.arange(nm_)]
oni0 = oni_s.reindex(months).values; oni_d = res(oni0, L0); tna_d = res(tna_s.reindex(months).values, L0)


def enso(lab):
    F = np.stack([np.bincount(mi[okd][lab[okd] == j], minlength=nm_) for j in range(K)], 1) / days_in[:, None]
    Lc = np.c_[np.ones(nm_), tna_d]
    Fr = res(res(F, L0), Lc)
    stat = lambda x: es._corr(Fr, res(x, Lc))
    obs = stat(oni_d)
    null = np.stack([stat(np.roll(oni_d, s)) for s in range(1, nm_) if min(s, nm_ - s) >= 24])
    p = (1 + (np.abs(null).max(1)[:, None] >= np.abs(obs)).sum(0)) / (1 + len(null))
    return obs, p, F[oni0 >= 0.5].mean(0) / F.mean(0), F[oni0 <= -0.5].mean(0) / F.mean(0)


def composites(lab):
    return np.stack([XA[torch.as_tensor(lab == j)].mean(0).numpy() for j in range(K)])


def align(Cref, C):
    """permutation p such that cluster p[j] of C matches cluster j of Cref (max composite correlation)."""
    cc = np.corrcoef(Cref.reshape(K, -1), C.reshape(K, -1))[:K, K:]
    r, c = linear_sum_assignment(-cc)
    return c, cc[r, c]


def pattern_corr(C):
    V = C[:, 1:].reshape(K, -1); V = V - V.mean(1, keepdims=True)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V @ V.T


# labels
A = XA.reshape(N, -1); mu = A[:ntr].mean(0, keepdim=True)
U, S, V = torch.pca_lowrank(A[:ntr] - mu, q=60, center=False, niter=4)
var = float((S ** 2).sum() / ((A[:ntr] - mu) ** 2).sum())
Z = ((A - mu) @ V).numpy(); del A
labs = {'EOF k-means': [], 'MSN prototypes': []}
for s in (0, 1, 2):
    km = KMeans(K, n_init=20, random_state=s).fit(Z[:ntr]); labs['EOF k-means'].append(km.predict(Z))
    F = torch.cat([torch.load(f'results/a100-tp30_detr_s{s}/F.pt', weights_only=False, map_location='cpu'),
                   torch.load(f'results/a100-tp30_detr_s{s}/F_val.pt', weights_only=False, map_location='cpu')])
    labs['MSN prototypes'].append(F.argmax(1).numpy())
print(f'60 EOFs explain {var:.0%}', flush=True)

lines = [f'# MSN native prototypes vs EOF k-means, K = {K} (detrended Tmin, Tmax, cbrt tp; full field)', '',
         'r = partial corr of the monthly frequency with ONI | TNA (detrended), p = rotation FWER over clusters; '
         'lift = frequency in El Nino (ONI >= 0.5) / La Nina (<= -0.5) months over the mean; stab = composite r with '
         'the matched cluster of seeds 1 and 2 (min); pat = max centred pattern r with another cluster of the same method.', '']
summ, comp, info = {}, {}, {}
for m, L in labs.items():
    Cs = [composites(l) for l in L]
    stab = []
    for i in (1, 2):
        pmt, r_ = align(Cs[0], Cs[i]); stab.append(r_)
    ari = [adjusted_rand_score(L[0], L[i]) for i in (1, 2)]
    lab = L[0]; C = Cs[0]
    r, p, le, ll = enso(lab)
    P = pattern_corr(C); np.fill_diagonal(P, -1)
    rows = []
    for j in range(K):
        dj = torch.as_tensor(np.where(lab == j)[0])
        cl = clim[mon[dj]]
        dd = dates[lab == j]
        sea = np.array([np.isin(dd.month, ms).mean() for ms in ((12, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11))])
        rows.append(dict(cluster=j, days=(lab == j).mean() * 100, Tmin=AN[dj, 0].mean().item(), Tmax=AN[dj, 1].mean().item(),
                         P_mmd=AN[dj, 2].mean().item(), P_pct=100 * AN[dj, 2].sum().item() / cl[:, 2].sum().item(),
                         season=['DJF', 'MAM', 'JJA', 'SON'][sea.argmax()], season_share=sea.max() * 100,
                         r_oni=r[j], p_fwer=p[j], lift_EN=le[j], lift_LN=ll[j], stab=min(stab[0][j], stab[1][j]),
                         pat=P[j].max()))
    df = pd.DataFrame(rows).sort_values('r_oni', ascending=False).reset_index(drop=True)
    df['name'] = [f'{"E" if m.startswith("EOF") else "M"}{i:02d}' for i in range(K)]
    summ[m], comp[m] = df, C
    info[m] = dict(ari=ari, sizes=(df.days.min(), df.days.max()), nsig=int((df.p_fwer < 0.05).sum()),
                   nsig_pos=int(((df.p_fwer < 0.05) & (df.r_oni > 0)).sum()), pat=np.median(df.pat),
                   stab=np.median(df.stab), Tsd=df.Tmax.std(), Psd=df.P_pct.std())
    lines += [f'## {m}', '', f'seed ARI (labels): {ari[0]:.2f}, {ari[1]:.2f}; cluster sizes {df.days.min():.1f}–{df.days.max():.1f} % of days', '',
              '| cluster | days % | Tmin °C | Tmax °C | P mm/d | P % | peak season | r ONI (p FWER) | lift EN | lift LN | stab | pat |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for _, x in df.iterrows():
        sig = '**' if x.p_fwer < 0.05 else ''
        lines.append(f'| {x["name"]} | {x.days:.1f} | {x.Tmin:+.2f} | {x.Tmax:+.2f} | {x.P_mmd:+.2f} | {x.P_pct:+.0f} | '
                     f'{x.season} ({x.season_share:.0f} %) | {sig}{x.r_oni:+.2f} ({x.p_fwer:.3f}){sig} | {x.lift_EN:.2f} | '
                     f'{x.lift_LN:.2f} | {x.stab:.2f} | {x.pat:.2f} |')
    lines.append('')
    df.to_csv(f'{out}/{"eof" if m.startswith("EOF") else "msn"}_K{K}.csv', index=False, float_format='%.3f')

# MSN <-> EOF matching
pm, rr = align(comp['EOF k-means'], comp['MSN prototypes'])
lines += ['## Summary', '', '| | EOF k-means | MSN prototypes |', '|---|---|---|']
for key, lab_, fmt in (('nsig', 'clusters linked to ONI (FWER p < 0.05)', '{}'), ('nsig_pos', '... of which El Niño-favoured', '{}'),
                       ('sizes', 'cluster size range (% of days)', '{0[0]:.1f}–{0[1]:.1f}'), ('ari', 'label ARI vs seeds 1, 2', '{0[0]:.2f}, {0[1]:.2f}'),
                       ('stab', 'median composite r with matched cluster of other seeds', '{:.2f}'),
                       ('pat', 'median max pattern r with another cluster (1 = copies)', '{:.2f}'),
                       ('Tsd', 'spread of cluster Tmax anomalies (sd, °C)', '{:.2f}'), ('Psd', 'spread of cluster P anomalies (sd, %)', '{:.0f}')):
    lines.append(f'| {lab_} | {fmt.format(info["EOF k-means"][key])} | {fmt.format(info["MSN prototypes"][key])} |')
lines += ['', f'MSN ↔ EOF cluster matching (Hungarian on composites): median r {np.median(rr):.2f}, '
          f'{(rr >= 0.8).sum()} of {K} pairs with r ≥ 0.8; daily-label ARI between methods '
          f'{adjusted_rand_score(labs["EOF k-means"][0], labs["MSN prototypes"][0]):.2f}', '',
          'figures: maps_eof_K{K}.png, maps_msn_K{K}.png, summary_K{K}.png'.format(K=K)]
open(f'{out}/native_compare_K{K}.md', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines[-16:]), flush=True)


# map grids: each cluster = Tmax (degC) over P (%), ordered by r ONI
def grid(m, tag):
    df, ncol = summ[m], 10
    nrow = int(np.ceil(K / ncol))
    fig, ax = plt.subplots(2 * nrow, ncol, figsize=(2.15 * ncol, 2.0 * 2 * nrow + 0.8), constrained_layout=True)
    lab = labs[m][0]
    for i, x in df.iterrows():
        j = int(x.cluster); rr_, cc_ = 2 * (i // ncol), i % ncol
        dj = torch.as_tensor(np.where(lab == j)[0])
        tmap = AN[dj, 1].mean(0).numpy()
        pmap = (100 * AN[dj, 2].sum(0) / clim[mon[dj], 2].sum(0).clamp_min(1e-3)).numpy()
        im1 = ax[rr_, cc_].imshow(tmap, cmap='RdBu_r', vmin=-3, vmax=3)
        im2 = ax[rr_ + 1, cc_].imshow(pmap, cmap='BrBG', vmin=-80, vmax=80)
        star = '*' if x.p_fwer < 0.05 else ''
        ax[rr_, cc_].set_title(f'{x["name"]} {x.days:.1f}% {x.season}\nTx {x.Tmax:+.1f}°C  P {x.P_pct:+.0f}%\n'
                               f'r ONI {x.r_oni:+.2f}{star}  EN {x.lift_EN:.1f}× LN {x.lift_LN:.1f}×', fontsize=7.5)
        for a in (ax[rr_, cc_], ax[rr_ + 1, cc_]): a.set_xticks([]); a.set_yticks([])
    for i in range(len(df), nrow * ncol):
        for d in (0, 1): ax[2 * (i // ncol) + d, i % ncol].axis('off')
    fig.colorbar(im1, ax=ax, location='left', shrink=0.5, label='Tmax anomaly (°C), upper map of each pair')
    fig.colorbar(im2, ax=ax, location='right', shrink=0.5, label='precip anomaly (% of climatology), lower map')
    fig.suptitle(f'{m}, K = {K}: mean maps (top Tmax °C, bottom precip %), ordered by r with ONI (* FWER p < 0.05)')
    fig.savefig(f'{out}/maps_{tag}_K{K}.png', dpi=90); plt.close(fig)


grid('EOF k-means', 'eof'); grid('MSN prototypes', 'msn')

# summary panel
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
for m, c in (('EOF k-means', '#1f77b4'), ('MSN prototypes', '#d62728')):
    df = summ[m]
    sig = df.p_fwer < 0.05
    ax[0].scatter(df.Tmax[~sig], df.P_pct[~sig], s=df.days[~sig] * 25, facecolors='none', edgecolors=c)
    ax[0].scatter(df.Tmax[sig], df.P_pct[sig], s=df.days[sig] * 25, color=c, label=f'{m} (filled: ONI-linked)')
    ax[1].scatter(df.Tmax, df.r_oni, s=df.days * 25, color=c, alpha=0.7, label=m)
    ax[2].plot(np.sort(df.days)[::-1], 'o-', color=c, label=m)
ax[0].set_xlabel('cluster mean Tmax anomaly (°C)'); ax[0].set_ylabel('cluster mean precip anomaly (%)')
ax[0].axhline(0, color='grey', lw=0.5); ax[0].axvline(0, color='grey', lw=0.5); ax[0].legend(fontsize=8)
ax[0].set_title('what each cluster is (size = share of days)')
ax[1].set_xlabel('cluster mean Tmax anomaly (°C)'); ax[1].set_ylabel('r with ONI | TNA'); ax[1].axhline(0, color='grey', lw=0.5)
ax[1].set_title('ENSO link vs temperature'); ax[1].legend(fontsize=8)
ax[2].set_xlabel('cluster rank'); ax[2].set_ylabel('% of days'); ax[2].set_title('cluster sizes'); ax[2].legend(fontsize=8)
fig.suptitle(f'EOF k-means vs MSN prototypes (K = {K}, detrended T + P anomalies)')
fig.savefig(f'{out}/summary_K{K}.png', dpi=100); plt.close(fig)
