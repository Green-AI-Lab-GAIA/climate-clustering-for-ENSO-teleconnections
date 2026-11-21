"""Classical baselines with the same K, fitted on TRAIN days, applied to val days.

  bins  : K quantile bins of the domain-mean temperature anomaly (a 1-D index)
  kmeans: k-means on the leading EOFs (PCA) of the anomaly fields (classical regimes)

Writes <out>/baseline_<name>.npz with cid_tr, cid_v, k.
  P      : precipitation only (cbrt-z tp); bins use the domain-mean precip anomaly
  Tnomean: temperature with each day's domain mean removed (pattern only, like t30_nomean)
usage: baselines.py <scratch_dir> <out_dir> [tags...]   (default T TPcbrt; also P, Tnomean)
"""
import sys, os, time
sys.path.insert(0, sys.argv[1])
from review_lib import *
from sklearn.cluster import KMeans

out = sys.argv[2]; os.makedirs(out, exist_ok=True)
torch.manual_seed(0)

TAGS = {'T': ('mn2t', 'mx2t'), 'TPcbrt': ('mn2t', 'mx2t', 'tp'), 'P': ('tp',), 'Tnomean': ('mn2t', 'mx2t')}
for tag in (sys.argv[3:] or ['T', 'TPcbrt']):
    chans = TAGS[tag]
    vt = {'tp': 'cbrt'} if 'tp' in chans else None
    X, Xv, _, _ = load_fields(chans, var_transform=vt)
    n, V, H, W = X.shape
    # 1-D index: domain mean of the temperature channels (of the precip channel for P)
    nc = 1 if tag == 'P' else 2
    idx_tr = X[:, :nc].mean((1, 2, 3)).numpy(); idx_v = Xv[:, :nc].mean((1, 2, 3)).numpy()
    if tag == 'Tnomean':                       # same input as t30_nomean (after the z-score)
        X = X - X.mean((2, 3), keepdim=True); Xv = Xv - Xv.mean((2, 3), keepdim=True)

    # EOFs on the full fields (train days), all channels in sigma units
    A = X.reshape(n, -1)
    mu = A.mean(0, keepdim=True)
    t0 = time.time()
    U, S, Vt = torch.pca_lowrank(A - mu, q=60, center=False, niter=4)
    var = (S ** 2) / ((A - mu) ** 2).sum()
    npc = int((var.cumsum(0) < 0.90).sum()) + 1
    npc = min(npc, 60)
    Z = ((A - mu) @ Vt[:, :npc]).numpy()
    Zv = ((Xv.reshape(len(Xv), -1) - mu) @ Vt[:, :npc]).numpy()
    print(f'[{tag}] EOFs: {npc} PCs explain {float(var[:npc].sum()):.1%} '
          f'(first 3: {[round(float(v), 3) for v in var[:3]]}) [{time.time() - t0:.0f}s]', flush=True)

    for k in (15, 30):
        if tag == 'Tnomean' and k == 15: continue
        q = np.quantile(idx_tr, np.linspace(0, 1, k + 1)[1:-1])
        if tag != 'Tnomean': np.savez(f'{out}/baseline_bins_{tag}_{k}.npz', cid_tr=np.digitize(idx_tr, q),
                 cid_v=np.digitize(idx_v, q), k=k)
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z)
        np.savez(f'{out}/baseline_kmeans_{tag}_{k}.npz', cid_tr=km.labels_, cid_v=km.predict(Zv), k=k)
        sizes = np.bincount(km.labels_, minlength=k) / len(km.labels_) * 100
        print(f'[{tag}] K={k}: kmeans sizes {sizes.min():.1f}-{sizes.max():.1f}% | bins equal-size', flush=True)
    del X, Xv, A
