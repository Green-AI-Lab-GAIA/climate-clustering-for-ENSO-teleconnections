"""Do two independently trained models find the same regimes and the same ENSO links?

usage: replicate.py <scratch_dir> <label A> <label B>
Matches clusters by composite pattern correlation (Hungarian), reports assignment
agreement (ARI, % same matched cluster) on the common days, and the ONI correlation
of each matched pair.
"""
import sys
sys.path.insert(0, sys.argv[1])
from review_lib import *
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

A, B = sys.argv[2], sys.argv[3]
X, Xv, t_tr, t_v = load_fields(('mn2t', 'mx2t'))
oni = oni_table()
info = {}
for lab in (A, B):
    ctr, cv, k, *_ = run_assignments(RUNS[lab][0])
    off = len(t_tr) - len(ctr)
    full = np.r_[np.full(off, -1), ctr]                  # align on the full day axis
    m = monthly_from(ctr, cv, t_tr[off:], t_v, k, oni)
    r = exact_rotation_stats(m, k, min_shift=24)
    info[lab] = dict(cid=full, cv=cv, k=k, corr=r['corr'], fwer=r['p_fwer'] < 0.05)

common = (info[A]['cid'] >= 0) & (info[B]['cid'] >= 0)
ca, cb = info[A]['cid'][common], info[B]['cid'][common]
Xc = X[torch.as_tensor(common)].mean(1).reshape(int(common.sum()), -1)   # mean T anomaly
ka, kb = info[A]['k'], info[B]['k']
compA = torch.stack([Xc[torch.as_tensor(ca == j)].mean(0) for j in range(ka)])
compB = torch.stack([Xc[torch.as_tensor(cb == j)].mean(0) for j in range(kb)])
za = compA - compA.mean(1, keepdim=True); zb = compB - compB.mean(1, keepdim=True)
C = (za @ zb.T / (za.norm(dim=1)[:, None] * zb.norm(dim=1)[None])).numpy()   # (ka, kb)
# also match on uncentred similarity (level matters): cosine of raw composites
Cu = (compA @ compB.T / (compA.norm(dim=1)[:, None] * compB.norm(dim=1)[None])).numpy()
ri, ci = linear_sum_assignment(-Cu)
mapping = dict(zip(ri, ci))
ari = adjusted_rand_score(ca, cb)
same = np.mean([mapping.get(a) == b for a, b in zip(ca, cb)]) if ka == kb else float('nan')
print(f'{A} vs {B}: common train days {common.sum()} | ARI={ari:.3f} | same matched cluster: {same:.1%}')
print(f'matched pairs (cosine of composites; centred pattern corr; ONI r A vs B; FWER A/B):')
rows = []
for a, b in sorted(mapping.items(), key=lambda ab: -info[A]['corr'][ab[0]]):
    rows.append((a, b, Cu[a, b], C[a, b], info[A]['corr'][a], info[B]['corr'][b], info[A]['fwer'][a], info[B]['fwer'][b]))
    print(f'  A C{a:<2} <-> B C{b:<2}  cos={Cu[a, b]:+.2f}  patt={C[a, b]:+.2f}  ONI r {info[A]["corr"][a]:+.2f} vs {info[B]["corr"][b]:+.2f}'
          f'  {"*" if info[A]["fwer"][a] else " "}/{"*" if info[B]["fwer"][b] else " "}')
ra = np.array([r_[4] for r_ in rows]); rb = np.array([r_[5] for r_ in rows])
print(f'correlation of per-regime ONI effects across the two models (matched): {np.corrcoef(ra, rb)[0, 1]:.2f}')
fa = {a for a, *_ in rows if info[A]['fwer'][a]}
both = sum(1 for a, b, *_ in rows if info[A]['fwer'][a] and info[B]['fwer'][b])
print(f'FWER-significant in A: {len(fa)}, in B: {int(info[B]["fwer"].sum())}, matched pairs significant in both: {both}')
