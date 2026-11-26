"""Embedding check (REVIEW §14.18): does a cluster's mean map sit at its center in the representation?

For each clustering, cluster k has members S_k, a center c_k in representation space (MSN: prototype p_k;
EOF k-means: centroid mu_k) and a composite (mean map) m_k = mean of the member fields.
The representation f maps a field to its feature (MSN: target encoder, L2-normalised; EOF: projection on
the 60 EOFs). Both methods use the same field: the model input (Tmin, Tmax, cbrt tp; z per pixel and
calendar month; per-pixel trend removed), train days only.

Per cluster:
  self      : is f(m_k) nearest to c_k among all centers? (rank of own center)
  d_rel     : d(f(m_k), c_k) / median_{x in S_k} d(f(x), c_k)   (< 1: the mean map is more central than a
              typical member; MSN d = 1 - cos, EOF d = Euclidean). For EOF f is linear, f(m_k) = mu_k, d_rel = 0.
  cos_mean  : MSN only, cos(f(m_k), mean_{x in S_k} f(x)) (encoder of the mean vs mean of the encodings)
  r_top20   : field-space pattern r between m_k and the mean map of the 20 days nearest c_k in the
              representation (symmetric for both methods: what the center "looks like").
  r_medoid  : same with the single nearest day.
usage: .venv/bin/python -I scripts/review/embed_check.py scripts/review <out_dir> [runs...]
       runs default: tp30_mask tp30_detr_s0 (any a100-<run> with results/ E,F and a best-slim checkpoint)
"""
import sys, os
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, '.')
import numpy as np, pandas as pd, torch
from sklearn.cluster import KMeans
import src.inference as inf

out = sys.argv[2]; os.makedirs(out, exist_ok=True)
runs = sys.argv[3:] or ['tp30_mask', 'tp30_detr_s0']
TOP = 20


def pat_r(a, b):
    a = a - a.mean(-1, keepdims=True); b = b - b.mean(-1, keepdims=True)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def composites(X, lab, K):
    idx = torch.as_tensor(lab)
    cnt = torch.bincount(idx, minlength=K).float()
    return (torch.zeros(K, X.shape[1]).index_add_(0, idx, X) / cnt[:, None].clamp_min(1)).numpy(), cnt.numpy()


def top_maps(X, score, K):
    """score: (N, K) similarity to each center (higher = nearer). Mean map of TOP nearest days and the medoid."""
    o = np.argsort(-score, axis=0)
    top = np.stack([X[o[:TOP, k]].mean(0).numpy() for k in range(K)])
    med = np.stack([X[o[0, k]].numpy() for k in range(K)])
    return top, med


def summarise(name, K, cnt, rank, d_rel, r_top, r_med, cos_mean=None):
    rows = pd.DataFrame(dict(method=name, k=np.arange(K), size_pct=cnt / cnt.sum() * 100, own_rank=rank,
                             d_rel=d_rel, r_top20=r_top, r_medoid=r_med,
                             cos_mean=np.nan if cos_mean is None else cos_mean))
    s = dict(method=name, K=K, self_assigned=float((rank == 1).mean()), median_rank=float(np.median(rank)),
             d_rel=float(np.median(d_rel)), d_rel_gt1=float((d_rel > 1).mean()),
             r_top20=float(np.median(r_top)), r_medoid=float(np.median(r_med)),
             cos_mean=np.nan if cos_mean is None else float(np.median(cos_mean)))
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}, flush=True)
    return rows, s


per, summ = [], []
X_ref = None
for run in runs:
    params, ds = inf.read_data(f'checkpoint/a100-{run}/params-a100-{run}.yaml', validation=True)
    enc, prot = inf.load_model(params, model_suffix='-best-slim.pth.tar')
    imgs = ds.imgs                                      # (N, C, H, W) model-input field, train days
    N = len(imgs); size = params['data'].get('rand_size', 128)
    tf = lambda x: torch.nn.functional.interpolate(x, size=(size, size), mode='bilinear', antialias=True)
    assert params['data'].get('eval_full_field', False), 'composites assume the full-field eval transform'
    E = torch.load(f'results/a100-{run}/E.pt', weights_only=False).float().cpu()
    F = torch.load(f'results/a100-{run}/F.pt', weights_only=False).float().cpu()
    assert len(E) == N, (len(E), N)
    P = prot.float().cpu(); K = P.shape[0]; lab = F.argmax(1).numpy()
    X = imgs.reshape(N, -1).float()
    M, cnt = composites(X, lab, K)
    with torch.no_grad():
        Mi = torch.as_tensor(M).reshape(K, *imgs.shape[1:])
        # check the transform reproduces the cached encodings
        e0 = torch.nn.functional.normalize(enc(tf(imgs[:8].float()).cuda()).float().cpu())
        print(run, 're-encode check, min cos with cached E:', float((e0 * E[:8]).sum(1).min()), flush=True)
        fM = torch.nn.functional.normalize(enc(tf(Mi).cuda()).float().cpu())
        # amplitude control: a mean map is smoother / weaker than any day. Rescale each composite's
        # anomaly to the median member norm and encode again.
        nrm = X.norm(dim=1).numpy()
        g = np.array([np.median(nrm[lab == k]) if (lab == k).any() else 1. for k in range(K)])
        Ms = Mi * torch.as_tensor(g / np.linalg.norm(M, axis=1).clip(1e-9), dtype=Mi.dtype)[:, None, None, None]
        fMs = torch.nn.functional.normalize(enc(tf(Ms).cuda()).float().cpu())
    S = fM @ P.T                                          # (K, K) composite vs every prototype
    Ss = fMs @ P.T
    rank_s = (Ss > Ss.diag()[:, None]).sum(1).numpy() + 1
    print(run, f'amplitude ratio composite/member median {np.median(np.linalg.norm(M, axis=1) / g):.2f}; '
          f'self-assigned after rescaling {(rank_s == 1).mean():.2f}; composites land on '
          f'{len(set(S.argmax(1).tolist()))} distinct prototypes (rescaled {len(set(Ss.argmax(1).tolist()))})',
          flush=True)
    rank = (S > S.diag()[:, None]).sum(1).numpy() + 1
    d_c = 1 - S.diag().numpy()
    d_x = 1 - (E * P[lab]).sum(1).numpy()
    d_rel = d_c / np.array([np.median(d_x[lab == k]) if (lab == k).any() else np.nan for k in range(K)])
    Em = torch.nn.functional.normalize(torch.as_tensor(composites(E, lab, K)[0]))
    cos_mean = (fM * Em).sum(1).numpy()
    top, med = top_maps(X, (E @ P.T).numpy(), K)
    r, s = summarise(f'MSN {run} (prototypes)', K, cnt, rank, d_rel, pat_r(M, top), pat_r(M, med), cos_mean)
    per.append(r); summ.append(s)
    if X_ref is None:
        X_ref = X
    del enc
    torch.cuda.empty_cache()

# EOF k-means on the same field (first run's input; all runs here share it), K = 30, 60 EOFs
X = X_ref; N = len(X)
mu = X.mean(0, keepdim=True)
U, Sv, V = torch.pca_lowrank(X - mu, q=60, center=False, niter=4)
Z = ((X - mu) @ V).numpy()
for K in (30, 10):
    km = KMeans(K, n_init=10, random_state=0).fit(Z); lab = km.labels_; C = km.cluster_centers_
    M, cnt = composites(X, lab, K)
    fM = (torch.as_tensor(M) - mu).numpy() @ V.numpy()
    D = np.linalg.norm(fM[:, None] - C[None], axis=2)
    rank = (D < D.diagonal()[:, None]).sum(1) + 1
    d_x = np.linalg.norm(Z - C[lab], axis=1)
    d_rel = D.diagonal() / np.array([np.median(d_x[lab == k]) for k in range(K)])
    dist = np.linalg.norm(Z[:, None] - C[None], axis=2)
    top, med = top_maps(X, -dist, K)
    r, s = summarise(f'EOF k-means (centroids)', K, cnt, rank, d_rel, pat_r(M, top), pat_r(M, med))
    per.append(r); summ.append(s)

pd.concat(per).to_csv(f'{out}/embed_check_clusters.csv', index=False, float_format='%.4f')
df = pd.DataFrame(summ); df.to_csv(f'{out}/embed_check.csv', index=False, float_format='%.4f')
with open(f'{out}/embed_check.md', 'w') as f:
    f.write(df.to_string(index=False, float_format='%.3f'))
print(df.to_string())
