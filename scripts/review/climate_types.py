"""Climate types first, ENSO second (REVIEW §14.9).

Input field: z~ = per-pixel, per-calendar-month z-score (tp: cbrt) minus a per-pixel linear trend fitted on
train days. No daily-mean removal. Channels Tmin, Tmax, P on the full 94x120 grid.

Step 1 (no ENSO information): cluster the days into types, for three day sets
  all  : every day, K = K_ALL
  warm : top 20 % of the daily domain-mean T anomaly L(d) (train threshold), K = K_EXT
  cold : bottom 20 %,                                                        K = K_EXT
with two feature sets
  eof  : PCA of z~ (train days, up to 80 % variance, <= 60 PCs) + k-means (seeds 0,1,2)
  msn  : the self-supervised embeddings of the full-field runs t30_s0/s1/s2 (trained on z, with trend;
         central 61 % crop), L2-normalised + k-means
Step 2, describe each type: share of days, mean T level and P level, peak season, mean run length (days),
composites (Tmax, P). Distinctness: centred pattern correlation between type composites (1 = same shape,
scaled copies). Stability: types matched across seeds (Hungarian on composite correlation), ARI of labels.
Step 3, ENSO: monthly frequency of each type (days of type / days in month), detrended, vs ONI and RONI at
lag 0 and 3 (index leading), partial on TNA; per season; lift in El Nino / La Nina months (index >= 0.5 /
<= -0.5); secondary: partial r also controlling the month's mean level (how much is intensity).
p-values: exact circular rotation of the index (|shift| >= 24 months), FWER over the K types.
Composites also in physical units: Tmax anomaly in degC and precipitation anomaly in mm/day and in % of the
calendar-month climatology (raw fields minus the train climatology, per-pixel linear trend removed).
env: SEASON=DJF|MAM|JJA|SON clusters only that season's days (tests on that season's months);
     EMB=<prefix> MSN runs results/<prefix>{0,1,2} (default a100-t30_s).
usage: [SEASON=..] [EMB=..] .venv/bin/python -I scripts/review/climate_types.py scripts/review <out_dir> [K_ALL] [K_EXT]
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
K_ALL = int(sys.argv[3]) if len(sys.argv) > 3 else 8
K_EXT = int(sys.argv[4]) if len(sys.argv) > 4 else 4
QUANT = 0.2
SEASON = os.environ.get('SEASON', '')
EMB = os.environ.get('EMB', 'a100-t30_s')
SEAS_MONTHS = {'DJF': (12, 1, 2), 'MAM': (3, 4, 5), 'JJA': (6, 7, 8), 'SON': (9, 10, 11)}
if SEASON or EMB != 'a100-t30_s':
    out = out.rstrip('/') + (f'_{SEASON}' if SEASON else '') + (f'_{EMB}' if EMB != 'a100-t30_s' else '')
    os.makedirs(out, exist_ok=True)
X, Xv, t_tr, t_v = load_fields(('mn2t', 'mx2t', 'tp'), var_transform={'tp': 'cbrt'})
XA = torch.cat([X, Xv]); ntr = len(X); N = len(XA)
dates = pd.DatetimeIndex(list(t_tr) + list(t_v))
tt = torch.as_tensor((dates - dates[0]).days.values / 3652.5, dtype=torch.float32)
D = torch.stack([torch.ones_like(tt), tt], 1)
coef = torch.linalg.lstsq(D[:ntr], XA[:ntr].reshape(ntr, -1)).solution
XA = (XA.reshape(N, -1) - D @ coef).reshape(XA.shape)
del X, Xv
# physical units: Tmax (degC) and tp (mm/day) anomalies vs the train calendar-month climatology, detrended
R, Rv, _, _ = load_fields(('mx2t', 'tp'), deseason=False)
RA = torch.cat([R, Rv]); del R, Rv
assert len(RA) == N
mon = torch.as_tensor(dates.month.values) - 1
clim = torch.stack([RA[:ntr][mon[:ntr] == m].mean(0) for m in range(12)])          # (12, 2, H, W)
AN = RA - clim[mon]
cf = torch.linalg.lstsq(D[:ntr], AN[:ntr].reshape(ntr, -1)).solution
AN = (AN.reshape(N, -1) - D @ cf).reshape(AN.shape)
del RA
lvlT = XA[:, :2].mean((1, 2, 3)).numpy(); lvlP = XA[:, 2].mean((1, 2)).numpy()
per = dates.to_period('M')


def read_psl(path):
    rows = []
    for line in open(path):
        p = line.split()
        if len(p) == 13 and p[0].isdigit():
            rows += [(pd.Period(f'{p[0]}-{m:02d}', 'M'), float(v)) for m, v in enumerate(p[1:], 1)]
    s = pd.Series(dict(rows)); return s[s > -99]


def read_roni(path):
    cen = {'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4, 'AMJ': 5, 'MJJ': 6, 'JJA': 7, 'JAS': 8, 'ASO': 9, 'SON': 10,
           'OND': 11, 'NDJ': 12}
    rows = {}
    for line in open(path):
        p = line.split()
        if len(p) == 3 and p[0] in cen:
            rows[pd.Period(f'{p[1]}-{cen[p[0]]:02d}', 'M')] = float(p[2])
    return pd.Series(rows)


idx = {'ONI': oni_table()['ONI'].astype(float), 'RONI': read_roni('../data/indices/roni.txt')}
tna_s = read_psl('../data/indices/tna.data')
last = min(idx['ONI'].index.max(), idx['RONI'].index.max(), tna_s.index.max())
months = per.unique(); months = months[months <= last]; nm_ = len(months)   # months with every index
mi = months.get_indexer(per)                                         # month index of each day (-1: after `last`)
days_in = np.bincount(mi[mi >= 0], minlength=nm_).astype(float)
tna = tna_s.reindex(months).values
t_m = np.arange(nm_, dtype=float)
res = lambda y, L: y - L @ np.linalg.lstsq(L, y, rcond=None)[0]
L0 = np.c_[np.ones(nm_), t_m]
tna_d = res(tna, L0)
mlev = res(np.bincount(mi[mi >= 0], lvlT[mi >= 0], nm_) / days_in, L0)                 # month's mean T level (detrended)
seasons = {'all': np.ones(nm_, bool)}
for s_, ms in SEAS_MONTHS.items():
    seasons[s_] = np.isin(months.month, ms)
BASE = seasons[SEASON] if SEASON else seasons['all']               # months entering every test


def rot_test(F, x, mask, C):
    """partial corr of each column of F with x on months `mask`, controlling C (+ intercept); rotation FWER."""
    Lc = np.c_[np.ones(mask.sum()), C[mask]] if C is not None else np.ones((mask.sum(), 1))
    Fr = res(F[mask], Lc)
    stat = lambda xx: es._corr(Fr, res(xx[mask], Lc))
    obs = stat(x)
    null = np.stack([stat(np.roll(x, s)) for s in range(1, nm_) if min(s, nm_ - s) >= 24])
    return obs, (1 + (np.abs(null).max(1)[:, None] >= np.abs(obs)).sum(0)) / (1 + len(null))


def eof_feats(sel):
    A = XA[sel].reshape(len(sel), -1)
    tr = torch.as_tensor(sel < ntr)
    mu = A[tr].mean(0, keepdim=True); A = A - mu
    U, S, V = torch.pca_lowrank(A[tr], q=60, center=False, niter=4)
    var = (S ** 2).cumsum(0) / (A[tr] ** 2).sum()
    npc = min(int((var < 0.8).sum()) + 1, len(var))
    return (A @ V[:, :npc]).numpy(), npc, float(var[npc - 1])


def msn_feats(seed):
    E = torch.cat([torch.load(f'results/{EMB}{seed}/E.pt', weights_only=False),
                   torch.load(f'results/{EMB}{seed}/E_val.pt', weights_only=False)]).float().cpu().numpy()
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def composites(lab, sel, k):
    C = np.stack([XA[sel][torch.as_tensor(lab == j)].mean(0).numpy() for j in range(k)])   # (k, 3, H, W)
    return C


def pattern_corr(C):
    """centred pattern correlation between type composites (Tmax and P stacked)."""
    V = C[:, 1:].reshape(len(C), -1); V = V - V.mean(1, keepdims=True)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V @ V.T


def run_length(lab, sel):
    """mean length (days) of consecutive-day runs of the same type."""
    out_ = []
    for j in range(lab.max() + 1):
        d = np.sort(sel[lab == j]); br = np.where(np.diff(d) != 1)[0]
        out_.append(len(d) / (len(br) + 1) if len(d) else 0)
    return np.array(out_)


def fit(set_name, sel, k, method, seed):
    if method == 'eof':
        Z = FE[set_name]
        km = KMeans(k, n_init=20, random_state=seed).fit(Z[sel < ntr]); return km.predict(Z)
    E = msn_feats(seed)[sel]
    km = KMeans(k, n_init=20, random_state=0).fit(E[sel < ntr]); return km.predict(E)


inseason = np.isin(dates.month, SEAS_MONTHS[SEASON]) if SEASON else np.ones(N, bool)
q_hi, q_lo = np.quantile(lvlT[:ntr][inseason[:ntr]], [1 - QUANT, QUANT])
sets = {'all': (np.where(inseason)[0], K_ALL), 'warm': (np.where(inseason & (lvlT >= q_hi))[0], K_EXT),
        'cold': (np.where(inseason & (lvlT <= q_lo))[0], K_EXT)}
FE, NPC = {}, {}
for s_, (sel, _) in sets.items():
    Z, npc, v = eof_feats(sel); FE[s_] = Z; NPC[s_] = (npc, v)
    print(f'{s_}: {len(sel)} days, {npc} PCs ({v:.0%} var)', flush=True)

lines = ['Climate types first, ENSO second (detrended anomalies, no daily-mean removal).',
         f'Warm/cold = top/bottom {QUANT:.0%} of the daily domain-mean T anomaly (thresholds {q_hi:+.2f} / {q_lo:+.2f} sigma).',
         'freq = days of the type / days in month; r = partial corr with the index | TNA, detrended, rotation FWER p; '
         'lift = mean freq in El Nino (index >= 0.5) or La Nina (<= -0.5) months / overall; "+lvl" also controls the '
         "month's mean T level. Types ordered by r with ONI (lag 0). Seasons: share of the type's days in DJF/MAM/JJA/SON.", '']
summary = []
for s_, (sel, k) in sets.items():
    for method in ('eof', 'msn'):
        labs = [fit(s_, sel, k, method, sd) for sd in (0, 1, 2)]
        Cs = [composites(l, sel, k) for l in labs]
        # align seeds 1, 2 to seed 0 by composite pattern correlation
        for i in (1, 2):
            V0 = Cs[0].reshape(k, -1); Vi = Cs[i].reshape(k, -1)
            cc = np.corrcoef(V0, Vi)[:k, k:]
            r_, c_ = linear_sum_assignment(-cc)
            mp = np.empty(k, int); mp[c_] = r_; labs[i] = mp[labs[i]]; Cs[i] = Cs[i][c_]
        stab = [np.diag(np.corrcoef(Cs[0].reshape(k, -1), Cs[i].reshape(k, -1))[:k, k:]) for i in (1, 2)]
        ari = [adjusted_rand_score(labs[0], labs[i]) for i in (1, 2)]
        lab = labs[0]
        Fm = np.stack([np.bincount((lambda v: v[v >= 0])(mi[sel][lab == j]), minlength=nm_) for j in range(k)], 1) / days_in[:, None]
        Fm = res(Fm, L0)
        R = {}
        for iname, ser in idx.items():
            x0 = ser.reindex(months).values.astype(float)
            assert not np.isnan(x0).any(), iname
            xd = res(x0, L0)
            for lag in (0, 3):
                x = np.roll(xd, lag); mask = BASE.copy(); mask[:lag] = False
                R[(iname, lag)] = rot_test(Fm, x, mask, tna_d[:, None])
            R[(iname, 'lvl')] = rot_test(Fm, xd, BASE, np.c_[tna_d, mlev])
            for sn in ('DJF', 'MAM', 'JJA', 'SON'):
                R[(iname, sn)] = rot_test(Fm, xd, seasons[sn], tna_d[:, None]) if (not SEASON or sn == SEASON) \
                    else (np.full(k, np.nan), np.full(k, np.nan))
            raw = np.stack([np.bincount((lambda v: v[v >= 0])(mi[sel][lab == j]), minlength=nm_) for j in range(k)], 1) / days_in[:, None]
            R[(iname, 'liftEN')] = raw[(x0 >= 0.5) & BASE].mean(0) / raw[BASE].mean(0)
            R[(iname, 'liftLN')] = raw[(x0 <= -0.5) & BASE].mean(0) / raw[BASE].mean(0)
        order = np.argsort(-R[('ONI', 0)][0]); rank = np.empty(k, int); rank[order] = np.arange(k)
        P = pattern_corr(Cs[0])
        rl = run_length(lab, sel)
        name = lambda j: f'{s_[0].upper()}{rank[j]}'
        desc = (f'k-means on {NPC[s_][0]} EOFs ({NPC[s_][1]:.0%} var) of z~ (Tmin, Tmax, P), seeds 0-2' if method == 'eof'
                else f'k-means on MSN {EMB}0/1/2 embeddings (one per seed)')
        lines += [f'## {s_} days / {method}: {desc}', '',
                  f'stability vs seeds 1, 2: ARI {ari[0]:.2f}, {ari[1]:.2f}; composite r per type (min over seeds) shown as "stab"', '',
                  '| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | '
                  'r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |',
                  '|' + '---|' * 18]
        for j in order:
            m_ = lab == j; dd = dates[sel][m_]
            sea = [np.isin(dd.month, ms).mean() * 100 for ms in ((12, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11))]
            other = np.delete(P[j], j); jo = np.delete(np.arange(k), j)[np.argmax(other)]
            dj = sel[m_]
            u_t = AN[dj, 0].mean().item(); u_p = AN[dj, 1].mean().item()
            u_pc = 100 * AN[dj, 1].sum().item() / clim[mon[dj], 1].sum().item()
            f = lambda key: f'{R[key][0][j]:+.2f} ({R[key][1][j]:.3f})'
            sr = ' / '.join(f'{R[("ONI", sn)][0][j]:+.2f}{"*" if R[("ONI", sn)][1][j] < 0.05 else ""}' for sn in ('DJF', 'MAM', 'JJA', 'SON'))
            lines.append(f'| {name(j)} | {m_.mean() * 100:.0f} | {lvlT[sel][m_].mean():+.2f} | {lvlP[sel][m_].mean():+.2f} | '
                         f'{u_t:+.2f} | {u_p:+.2f} | {u_pc:+.0f} | '
                         + '/'.join(f'{v:.0f}' for v in sea) + f' | {rl[j]:.1f} | {min(stab[0][j], stab[1][j]):.2f} | '
                         f'{other.max():+.2f} ({name(jo)}) | {f(("ONI", 0))} | {f(("ONI", 3))} | {f(("ONI", "lvl"))} | '
                         f'{f(("RONI", 0))} | {sr} | {R[("ONI", "liftEN")][j]:.2f} | {R[("ONI", "liftLN")][j]:.2f} |')
            summary.append(dict(set=s_, method=method, type=name(j), days=m_.mean(), T=lvlT[sel][m_].mean(),
                                P=lvlP[sel][m_].mean(), Tmax_C=u_t, P_mmd=u_p, P_pct=u_pc, r_oni=R[('ONI', 0)][0][j], p_oni=R[('ONI', 0)][1][j],
                                r_oni_lvl=R[('ONI', 'lvl')][0][j], p_oni_lvl=R[('ONI', 'lvl')][1][j],
                                r_roni=R[('RONI', 0)][0][j], p_roni=R[('RONI', 0)][1][j],
                                stab=min(stab[0][j], stab[1][j]), max_pattern_r=other.max(), run=rl[j],
                                liftEN=R[('ONI', 'liftEN')][j], liftLN=R[('ONI', 'liftLN')][j]))
        lines += ['', '* = p < 0.05 (FWER over types) in that season', '', f'figure: types_{s_}_{method}.png', '']
        # composite figure: Tmax and P per type, ordered
        fig, ax = plt.subplots(2, k, figsize=(3.2 * k, 5.8), constrained_layout=True)
        for c, j in enumerate(order):
            for row, (ch, cmap, lab_) in enumerate(((1, 'RdBu_r', 'Tmax anomaly (sigma)'), (2, 'BrBG', 'precip anomaly (cbrt sigma)'))):
                im = ax[row, c].imshow(Cs[0][j, ch], cmap=cmap, vmin=-2, vmax=2)
                ax[row, c].set_xticks([]); ax[row, c].set_yticks([])
                if row == 0:
                    ax[row, c].set_title(f'{name(j)}: {(lab == j).mean() * 100:.0f}% d, T {lvlT[sel][lab == j].mean():+.2f}\n'
                                         f'r ONI {R[("ONI", 0)][0][j]:+.2f} (p {R[("ONI", 0)][1][j]:.2f})', fontsize=9)
                if c == k - 1: fig.colorbar(im, ax=ax[row, c], label=lab_, shrink=0.8)
        fig.suptitle(f'{s_} days, climate types ({method}); detrended anomalies')
        fig.savefig(f'{out}/types_{s_}_{method}.png', dpi=95); plt.close(fig)
        fig, ax = plt.subplots(2, k, figsize=(3.2 * k, 5.8), constrained_layout=True)
        for c, j in enumerate(order):
            dj = sel[lab == j]
            tmap = AN[dj, 0].mean(0).numpy()
            pmap = (100 * AN[dj, 1].sum(0) / clim[mon[dj], 1].sum(0).clamp_min(1e-3)).numpy()
            for row, (mp_, cmap, lim, lab_) in enumerate(((tmap, 'RdBu_r', 3, 'Tmax anomaly (degC)'),
                                                          (pmap, 'BrBG', 80, 'precip anomaly (% of climatology)'))):
                im = ax[row, c].imshow(mp_, cmap=cmap, vmin=-lim, vmax=lim)
                ax[row, c].set_xticks([]); ax[row, c].set_yticks([])
                if row == 0:
                    ax[row, c].set_title(f'{name(j)}: {(lab == j).mean() * 100:.0f}% d, Tmax {AN[dj, 0].mean():+.1f} C\n'
                                         f'P {100 * AN[dj, 1].sum() / clim[mon[dj], 1].sum():+.0f}%, r ONI {R[("ONI", 0)][0][j]:+.2f}', fontsize=9)
                if c == k - 1: fig.colorbar(im, ax=ax[row, c], label=lab_, shrink=0.8)
        fig.suptitle(f'{s_} days{" (" + SEASON + ")" if SEASON else ""}, climate types ({method}), physical units (detrended)')
        fig.savefig(f'{out}/types_{s_}_{method}_units.png', dpi=95); plt.close(fig)
        print('\n'.join(lines[-(k + 9):]), flush=True)
        fig = None

open(f'{out}/climate_types.md', 'w').write('\n'.join(lines) + '\n')
pd.DataFrame(summary).to_csv(f'{out}/climate_types.csv', index=False, float_format='%.3f')
