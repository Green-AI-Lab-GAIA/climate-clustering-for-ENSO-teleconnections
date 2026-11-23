"""PCA vs MSN embeddings as the representation for climate types, over K; how to choose K.

Primary criteria = how well the cluster maps represent Amazonian climate (no ENSO information):
  eta2, coherence (day-to-own-map spatial correlation), distinctness (pattern r between maps), stability.
ENSO usefulness (ONI R2, n ONI) is secondary.

Data: detrended z~ (Tmin, Tmax, cbrt tp), full field. Representations:
  PCA : 60 EOF scores (PCA fitted on train days)
  MSN : tp30_detr_s{0,1,2} target-encoder embedding (384-d, L2-normalised); same input fields
For K in KS, k-means (3 seeds; for MSN one trained model per seed) and four criteria, all measured the same way
for both representations:
  eta2   : share of the daily field variance (z~, all pixels and channels) explained by the cluster means
  split  : year-split stability: k-means (and the PCA itself) fitted on even train years vs odd train years,
           both applied to every day, ARI between the two labelings (MSN embedding is fixed: the network saw all
           years, so this only tests the clustering step for MSN)
  oni_r2 : out-of-sample R^2 of ONI predicted from the monthly cluster frequencies (ridge, 10 contiguous folds,
           detrended), and the number of clusters linked to ONI | TNA (rotation FWER)
  pat    : median over clusters of the max centred pattern r with another cluster (1 = copies)
  coh    : coherence = spatial correlation of each day's anomaly field (Tmin, Tmax, P; centred per day) with the
           mean map of its cluster; median over days and share of days with r > 0.5 (well represented)
Dimensions test: PCA with 20 / 60 / 200 EOFs at K = 10 and 30.
Probe (no clustering): out-of-sample R^2 of ONI from monthly means of the features themselves.
usage: .venv/bin/python -I scripts/review/k_selection.py scripts/review <out_dir>
"""
import sys, os
sys.path.insert(0, sys.argv[1])
from review_lib import *
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = sys.argv[2]; os.makedirs(out, exist_ok=True)
KS = [4, 6, 8, 10, 12, 15, 20, 30, 40]
X, Xv, t_tr, t_v = load_fields(('mn2t', 'mx2t', 'tp'), var_transform={'tp': 'cbrt'})
XA = torch.cat([X, Xv]); ntr = len(X); N = len(XA); del X, Xv
dates = pd.DatetimeIndex(list(t_tr) + list(t_v))
tt = torch.as_tensor((dates - dates[0]).days.values / 3652.5, dtype=torch.float32)
D = torch.stack([torch.ones_like(tt), tt], 1)
coef = torch.linalg.lstsq(D[:ntr], XA[:ntr].reshape(ntr, -1)).solution
A = XA.reshape(N, -1) - D @ coef; del XA                                   # (N, 33840) detrended z~
mu_all = A.mean(0, keepdim=True); SST = float(((A - mu_all) ** 2).sum())


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
oni_d = res(oni_s.reindex(months).values, L0); tna_d = res(tna_s.reindex(months).values, L0)


def cv_r2(P, y, folds=10):
    pred = np.zeros_like(y)
    for tr, te in KFold(folds, shuffle=False).split(P):
        pred[te] = RidgeCV(alphas=np.logspace(-3, 4, 15)).fit(P[tr], y[tr]).predict(P[te])
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def monthly_freq(lab, k):
    F = np.stack([np.bincount(mi[okd][lab[okd] == j], minlength=nm_) for j in range(k)], 1) / days_in[:, None]
    return res(F, L0)


def n_fwer(F):
    Lc = np.c_[np.ones(nm_), tna_d]; Fr = res(F, Lc)
    stat = lambda x: es._corr(Fr, res(x, Lc))
    obs = stat(oni_d)
    null = np.stack([stat(np.roll(oni_d, s)) for s in range(1, nm_) if min(s, nm_ - s) >= 24])
    p = (1 + (np.abs(null).max(1)[:, None] >= np.abs(obs)).sum(0)) / (1 + len(null))
    return int((p < 0.05).sum())


def eta2_and_pat(lab, k):
    idx = torch.as_tensor(lab)
    cnt = torch.bincount(idx, minlength=k).float()
    sums = torch.zeros(k, A.shape[1]).index_add_(0, idx, A)
    means = sums / cnt[:, None].clamp_min(1)
    ssb = float((cnt[:, None] * (means - mu_all) ** 2).sum())
    Mc = means - means.mean(1, keepdim=True); Mc = Mc / Mc.norm(dim=1, keepdim=True).clamp_min(1e-9)
    cors = []
    for a in range(0, N, 2000):
        Ad = A[a:a + 2000]; Ad = Ad - Ad.mean(1, keepdim=True)
        cors.append(((Ad / Ad.norm(dim=1, keepdim=True).clamp_min(1e-9)) * Mc[idx[a:a + 2000]]).sum(1))
    cors = torch.cat(cors).numpy()
    global last_coh; last_coh = cors
    V = means.numpy().reshape(k, 3, -1)[:, 1:].reshape(k, -1)             # Tmax + P composites
    V = V - V.mean(1, keepdims=True); V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    P = V @ V.T; np.fill_diagonal(P, -1)
    return ssb / SST, float(np.median(P.max(1))), float(np.median(cors)), float((cors > 0.5).mean())


def pca_scores(fit_idx, q=60):
    m = A[fit_idx].mean(0, keepdim=True)
    U, S, V = torch.pca_lowrank(A[fit_idx] - m, q=q, center=False, niter=4)
    return ((A - m) @ V).numpy()


train_years = np.unique(dates[:ntr].year)
half = {h: np.where((np.arange(N) < ntr) & np.isin(dates.year, train_years[h::2]))[0] for h in (0, 1)}
Z_full = pca_scores(np.arange(ntr))
Z_half = {h: pca_scores(half[h]) for h in (0, 1)}
E = {}
for s in (0, 1, 2):
    e = torch.cat([torch.load(f'results/a100-tp30_detr_s{s}/E.pt', weights_only=False, map_location='cpu'),
                   torch.load(f'results/a100-tp30_detr_s{s}/E_val.pt', weights_only=False, map_location='cpu')]).float().numpy()
    E[s] = e / np.linalg.norm(e, axis=1, keepdims=True)
print('features ready', flush=True)

# probe: ONI from monthly means of the features (no clustering)
def monthly_mean(F):
    M = np.stack([np.bincount(mi[okd], F[okd, c], nm_) for c in range(F.shape[1])], 1) / days_in[:, None]
    return res(M, L0)


lvl = A.reshape(N, 3, -1)[:, :2].mean((1, 2)).numpy()[:, None]
probe = {'domain-mean T level only': cv_r2(monthly_mean(lvl), oni_d),
         'PCA (60 EOFs)': cv_r2(monthly_mean(Z_full), oni_d),
         'PCA (10 EOFs)': cv_r2(monthly_mean(Z_full[:, :10]), oni_d)}
for s in (0, 1, 2):
    probe[f'MSN seed {s} (384-d)'] = cv_r2(monthly_mean(E[s]), oni_d)
    Ec = E[s] - E[s][:ntr].mean(0); U_, S_, Vt = np.linalg.svd(Ec[:ntr], full_matrices=False)
    probe[f'MSN seed {s}, 60 PCs of the embedding'] = cv_r2(monthly_mean(Ec @ Vt[:60].T), oni_d)
print(probe, flush=True)

rows = []
for K in KS:
    for rep in ('PCA', 'MSN'):
        for s in (0, 1, 2):
            F_ = Z_full if rep == 'PCA' else E[s]
            km = KMeans(K, n_init=10, random_state=s).fit(F_[:ntr]); lab = km.predict(F_)
            e2, pat, coh, coh05 = eta2_and_pat(lab, K)
            Fm = monthly_freq(lab, K)
            r2 = cv_r2(Fm, oni_d)
            nsig = n_fwer(Fm)
            # year-split stability
            labs_h = []
            for h in (0, 1):
                Fh = Z_half[h] if rep == 'PCA' else E[s]
                kh = KMeans(K, n_init=5, random_state=s).fit(Fh[half[h]]); labs_h.append(kh.predict(Fh))
            split = adjusted_rand_score(labs_h[0], labs_h[1])
            rows.append(dict(K=K, rep=rep, seed=s, eta2=e2, pat=pat, coh=coh, coh05=coh05, oni_r2=r2, nsig=nsig, split_ari=split,
                             min_size=np.bincount(lab, minlength=K).min() / N * 100))
            print(rows[-1], flush=True)
df = pd.DataFrame(rows); df.to_csv(f'{out}/k_selection.csv', index=False, float_format='%.4f')
# seed ARI (between k-means seeds for PCA, between MSN models)
seed_ari = []
for K in KS:
    for rep in ('PCA', 'MSN'):
        L = []
        for s in (0, 1, 2):
            F_ = Z_full if rep == 'PCA' else E[s]
            L.append(KMeans(K, n_init=10, random_state=s).fit(F_[:ntr]).predict(F_))
        seed_ari.append(dict(K=K, rep=rep, seed_ari=np.mean([adjusted_rand_score(L[0], L[1]), adjusted_rand_score(L[0], L[2]),
                                                               adjusted_rand_score(L[1], L[2])])))
sa = pd.DataFrame(seed_ari)
g = df.groupby(['K', 'rep']).mean(numeric_only=True).reset_index().merge(sa, on=['K', 'rep'])
g.to_csv(f'{out}/k_selection_mean.csv', index=False, float_format='%.4f')

lines = ['# PCA vs MSN embeddings over K (detrended Tmin, Tmax, cbrt tp; full field)', '',
         'Means over 3 seeds. eta2 = share of daily field variance explained by the cluster means; split ARI = '
         'agreement of k-means fitted on even vs odd train years (PCA refitted per half; MSN embedding fixed); seed '
         'ARI = agreement across seeds (k-means seeds for PCA, independently trained models for MSN); ONI R2 = '
         'out-of-sample R2 of ONI from monthly cluster frequencies; n ONI = clusters linked to ONI | TNA (FWER); '
         'pat = median max pattern r between clusters (1 = copies).', '',
         'coh = median spatial correlation of a day with its cluster map; coh>0.5 = share of days with r > 0.5.', '',
         '| K | rep | eta2 | coh | coh>0.5 | pat | split ARI | seed ARI | min size % | ONI R2 (CV) | n ONI |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
for _, x in g.iterrows():
    lines.append(f'| {int(x.K)} | {x.rep} | {x.eta2:.3f} | {x.coh:.2f} | {x.coh05:.0%} | {x.pat:.2f} | {x.split_ari:.2f} | '
                 f'{x.seed_ari:.2f} | {x.min_size:.1f} | {x.oni_r2:.3f} | {x.nsig:.1f} |')
lines += ['', '## Probe: ONI from monthly means of the features (no clustering; ridge, 10 contiguous folds, detrended)', '',
          '| features | ONI R2 (CV) |', '|---|---|'] + [f'| {k} | {v:.3f} |' for k, v in probe.items()]
open(f'{out}/k_selection.md', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))

# dimensions test (PCA)
dims_rows = []
for q in (20, 60, 200):
    Zq = pca_scores(np.arange(ntr), q=q)
    for K in (10, 30):
        lab = KMeans(K, n_init=10, random_state=0).fit(Zq[:ntr]).predict(Zq)
        e2, pat, coh, coh05 = eta2_and_pat(lab, K)
        dims_rows.append(f'| PCA {q} EOFs | {K} | {e2:.3f} | {coh:.2f} | {coh05:.0%} | {pat:.2f} |')
for K in (10, 30):
    for s in (0,):
        lab = KMeans(K, n_init=10, random_state=0).fit(E[s][:ntr]).predict(E[s])
        e2, pat, coh, coh05 = eta2_and_pat(lab, K)
        dims_rows.append(f'| MSN 384-d (seed 0) | {K} | {e2:.3f} | {coh:.2f} | {coh05:.0%} | {pat:.2f} |')
    F = torch.cat([torch.load('results/a100-tp30_detr_s0/F.pt', weights_only=False, map_location='cpu'),
                   torch.load('results/a100-tp30_detr_s0/F_val.pt', weights_only=False, map_location='cpu')])
nat = F.argmax(1).numpy(); e2, pat, coh, coh05 = eta2_and_pat(nat, 30)
dims_rows.append(f'| MSN native prototypes (seed 0) | 30 | {e2:.3f} | {coh:.2f} | {coh05:.0%} | {pat:.2f} |')
lines += ['', '## Does dimensionality matter?', '', '| features | K | eta2 | coh | coh>0.5 | pat |', '|---|---|---|---|---|---|'] + dims_rows
open(f'{out}/k_selection.md', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines[-12:]), flush=True)

fig, ax = plt.subplots(1, 7, figsize=(30, 4.2), constrained_layout=True)
for rep, c in (('PCA', '#1f77b4'), ('MSN', '#d62728')):
    gg = g[g.rep == rep]; dd = df[df.rep == rep]
    for i, (col, lab) in enumerate((('eta2', 'field variance explained (η²)'), ('coh', 'coherence: median day-to-map r'),
                                    ('coh05', 'share of days with r > 0.5'), ('split_ari', 'year-split stability (ARI)'),
                                    ('oni_r2', 'ONI R² out of sample'), ('nsig', 'clusters linked to ONI (FWER)'),
                                    ('pat', 'max pattern r between clusters'))):
        ax[i].plot(gg.K, gg[col], 'o-', color=c, label=rep)
        ax[i].scatter(dd.K, dd[col], color=c, s=8, alpha=0.4)
        ax[i].set_xlabel('K'); ax[i].set_title(lab); ax[i].set_xscale('log'); ax[i].set_xticks(KS); ax[i].set_xticklabels(KS)
ax[0].legend()
fig.suptitle('Choosing K: PCA (60 EOFs) vs MSN embedding, k-means, 3 seeds (dots) and mean (line)')
fig.savefig(f'{out}/k_selection.png', dpi=100); plt.close(fig)
