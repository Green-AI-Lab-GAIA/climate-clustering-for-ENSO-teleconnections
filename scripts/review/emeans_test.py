"""e-means (Bessa et al. 2026, arXiv 2611.09501) vs Lloyd k-means on the climate-type features (REVIEW §14.16).

e-means = mini-batch k-means with EMA cluster statistics (Alg. 1):
  N_j <- g N_j + (1-g) n_j,  S_j <- g S_j + (1-g) s_j,  c_j = S_j / N_j,  a(x) = argmin_j ||x - c_j||^2
  step a = 1-g on a cosine schedule a(t) = a0 (eps + (1-eps) (1 + cos(pi t / M)) / 2), eps = 0.1, M = T * floor(n/B)
  init: random data points, N_j = 1e-10; dead-cluster revival every 5 batches (N_j < 0.1 mean -> split the largest)
  finalisation: centroid = mean of the points assigned to it during the last epoch.
The paper's batch / step-size laws were fitted for n >= 1e5, so a0 and B are scanned here (n = 15,706 train days).
Features: 60 EOF scores of z~ (as k_selection.py) and the MSN mask30 (final) embedding. K = 10, 30.
Metrics: train inertia (k-means objective, relative to the best Lloyd run), eta2, coherence, pattern r, seed ARI,
ARI with Lloyd.
usage: .venv/bin/python -I scripts/review/emeans_test.py scripts/review <out_dir>
"""
import sys, os
sys.path.insert(0, sys.argv[1])
from review_lib import *
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

out = sys.argv[2]; os.makedirs(out, exist_ok=True)
X, Xv, t_tr, t_v = load_fields(('mn2t', 'mx2t', 'tp'), var_transform={'tp': 'cbrt'})
XA = torch.cat([X, Xv]); ntr = len(X); N = len(XA); del X, Xv
dates = pd.DatetimeIndex(list(t_tr) + list(t_v))
tt = torch.as_tensor((dates - dates[0]).days.values / 3652.5, dtype=torch.float32)
D = torch.stack([torch.ones_like(tt), tt], 1)
coef = torch.linalg.lstsq(D[:ntr], XA[:ntr].reshape(ntr, -1)).solution
A = XA.reshape(N, -1) - D @ coef; del XA
mu_all = A.mean(0, keepdim=True); SST = float(((A - mu_all) ** 2).sum())


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
    return ssb / SST, float(np.median(cors)), float(np.median(P.max(1)))


def emeans(Xtr, k, a0, B, T, seed, eps=0.1):
    rng = np.random.default_rng(seed)
    n, d = Xtr.shape
    c = Xtr[rng.choice(n, k, replace=False)].astype(np.float64)
    Nj = np.full(k, 1e-10); Sj = Nj[:, None] * c
    nb = n // B; M = T * nb; t = 0
    for ep in range(T):
        perm = rng.permutation(n)
        last = ep == T - 1
        if last: lsum = np.zeros((k, d)); lcnt = np.zeros(k)
        for b in range(nb):
            xb = Xtr[perm[b * B:(b + 1) * B]]
            cen = Sj / Nj[:, None]
            a_ = ((xb ** 2).sum(1)[:, None] - 2 * xb @ cen.T + (cen ** 2).sum(1)[None]).argmin(1)
            nj = np.bincount(a_, minlength=k).astype(float)
            sj = np.zeros((k, d)); np.add.at(sj, a_, xb)
            al = a0 * (eps + (1 - eps) * 0.5 * (1 + np.cos(np.pi * t / M)))
            Nj = (1 - al) * Nj + al * nj; Sj = (1 - al) * Sj + al * sj
            if last: lsum += sj; lcnt += nj
            t += 1
            if t % 5 == 0:                                   # dead-cluster revival: split the largest cluster
                dead = np.where(Nj < 0.1 * Nj.mean())[0]
                for j in dead:
                    donor = Nj.argmax(); cd = Sj[donor] / Nj[donor]
                    pert = 1e-3 * rng.standard_normal(d) * (np.abs(cd).mean() + 1e-9)
                    Nj[donor] /= 2; Nj[j] = Nj[donor]
                    Sj[donor] = Nj[donor] * (cd + pert); Sj[j] = Nj[j] * (cd - pert)
    cen = np.where(lcnt[:, None] > 0, lsum / np.maximum(lcnt, 1)[:, None], Sj / Nj[:, None])
    return cen


def assign(Xa, cen):
    return ((Xa ** 2).sum(1)[:, None] - 2 * Xa @ cen.T + (cen ** 2).sum(1)[None]).argmin(1)


def inertia(Xtr, cen):
    lab = assign(Xtr, cen); return float(((Xtr - cen[lab]) ** 2).sum())


m = A[:ntr].mean(0, keepdim=True)
U, S, V = torch.pca_lowrank(A[:ntr] - m, q=60, center=False, niter=4)
FEAT = {'PCA 60 EOFs': ((A - m) @ V).numpy().astype(np.float64)}
g = lambda f: torch.load(f'results/a100-tp30_mask_latest/{f}.pt', weights_only=False, map_location='cpu')
E = torch.cat([g('E'), g('E_val')]).float().numpy().astype(np.float64)
FEAT['MSN-mask30 embedding'] = E / np.linalg.norm(E, axis=1, keepdims=True)

rows = []
for fname, Fa in FEAT.items():
    Ftr = Fa[:ntr]
    for K in (10, 30):
        lloyd = [KMeans(K, n_init=10, random_state=s).fit(Ftr) for s in (0, 1, 2)]
        ref_in = min(l.inertia_ for l in lloyd)
        L_lab = [l.predict(Fa) for l in lloyd]
        cfgs = [('Lloyd (sklearn, n_init 10)', None)] + [(f'e-means a0={a0} B={B} T={T}', (a0, B, T))
                                                       for a0 in (0.03, 0.1, 0.3) for B in (256, 1024) for T in (3, 10)]
        for name, cfg in cfgs:
            labs, ins = [], []
            for s in (0, 1, 2):
                if cfg is None:
                    lab = L_lab[s]; ins.append(lloyd[s].inertia_)
                else:
                    cen = emeans(Ftr, K, *cfg, seed=s); lab = assign(Fa, cen); ins.append(inertia(Ftr, cen))
                labs.append(lab)
            st = [field_stats(l, K) for l in labs]
            rows.append(dict(features=fname, K=K, method=name, rel_inertia=np.mean(ins) / ref_in,
                             eta2=np.mean([x[0] for x in st]), coh=np.mean([x[1] for x in st]), pat=np.mean([x[2] for x in st]),
                             seed_ari=np.mean([adjusted_rand_score(labs[i], labs[j]) for i, j in ((0, 1), (0, 2), (1, 2))]),
                             ari_lloyd=np.mean([adjusted_rand_score(labs[s], L_lab[s]) for s in (0, 1, 2)]),
                             min_size=min(np.bincount(l, minlength=K).min() for l in labs) / N * 100))
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)

df = pd.DataFrame(rows); df.to_csv(f'{out}/emeans.csv', index=False, float_format='%.4f')
lines = ['# e-means vs Lloyd k-means (means over 3 seeds)', '',
         'rel inertia = train k-means objective / best of the 3 Lloyd runs (1 = as good as Lloyd; < 1 = better); '
         'eta2, coh (median day-to-map r), pat (median max pattern r between maps) as in k_selection; ARI Lloyd = '
         'daily-label agreement with the Lloyd run of the same seed.', '',
         '| features | K | method | rel inertia | eta2 | coh | pat | seed ARI | ARI vs Lloyd | min size % |',
         '|---|---|---|---|---|---|---|---|---|---|']
for _, x in df.iterrows():
    lines.append(f'| {x.features} | {x.K} | {x.method} | {x.rel_inertia:.4f} | {x.eta2:.3f} | {x.coh:.3f} | {x.pat:.2f} | '
                 f'{x.seed_ari:.2f} | {x.ari_lloyd:.2f} | {x.min_size:.1f} |')
open(f'{out}/emeans.md', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
